"""The per-orbit generation loop, and the scoring pass over what it wrote.

Replaces the three near-duplicate try-on loops in the notebook, which differed only in their
folder, their garment photograph and their reference frame — and which saved the final image only.

Nothing here imports torch, diffusers or CatVTON. The notebook passes ``two_phase_tryon`` in as
``tryon_fn``, so the model lives in the notebook and the orchestration lives under version control.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import metrics as M
from .runio import RunConfig, RunWriter, load_run

#: RunConfig defaults, for manifests written before a field existed: such a run behaved as the default.
_CONFIG_DEFAULTS = {f.name: f.default for f in fields(RunConfig) if f.default is not MISSING}

__all__ = ["run_orbit", "ensure_orbit", "score_run", "report_run", "compare_runs",
           "infer_view_of", "backfill_view_of", "count_decoder_attn1", "generating_stage",
           "parse_finals", "body_distortion", "ensure_garment_masks", "garment_lookup",
           "colour_by_class", "outside_change_by_class", "GarmentMismatchError",
           "GENERATING_STAGE", "VIEW_ORDER"]

#: Generation order over the orientation classes. **This order is load-bearing** (thesis 4.4.2).
#: A reference pass clears the key/value bank and refills it, so every frame that consumes a bank
#: must be generated before the next reference pass runs. Lateral frames are each generated as
#: their own reference pass (Section 4.5: they receive no injection), which clears the bank, so
#: they must all come first — running them between a reference and its targets would wipe it.
VIEW_ORDER = ("side", "front", "back")


def _load_person(path, resolution):
    from PIL import Image  # local: keeps the module importable without PIL at import time

    return Image.open(path).convert("RGB").resize(tuple(resolution))


class GarmentMismatchError(RuntimeError):
    """A scorer was handed a garment photograph other than the one a run was conditioned on."""


def _check_garment_for(run_dir, garment_for, names: Sequence[str], views: dict[str, str],
                       tolerance: float = 2.0) -> None:
    """Refuse to score a run against a photograph it was not conditioned on.

    A ``garment_for`` written in the notebook closes over notebook variables, and those hold whichever
    garment's setup cell ran last. A detail ratio or fidelity computed against another garment's
    photograph is a plausible-looking number that is simply wrong, and nothing downstream can tell.
    Each class's supplied photograph is compared with ``garment/<view>.png``, which :func:`run_orbit`
    wrote from the photograph it actually used; runs written before garments were stored are not
    checked. Raises rather than returning a row error, so a table is never printed without the run.
    """
    import numpy as np
    from PIL import Image

    root = Path(run_dir) / "garment"
    checked: set[str] = set()
    for n in names:
        view = views.get(Path(n).stem)
        if view is None or view in checked:
            continue
        checked.add(view)
        stored = root / f"{view}.png"
        if not stored.exists():
            continue
        supplied = M._open(garment_for(n)).convert("RGB")
        reference = Image.open(stored).convert("RGB")
        if supplied.size != reference.size:
            supplied = supplied.resize(reference.size)
        diff = float(np.abs(np.asarray(supplied, np.float32) - np.asarray(reference, np.float32)).mean())
        if diff > tolerance:
            raise GarmentMismatchError(
                f"{Path(run_dir).name}: the {view} photograph from garment_for is not the one this run "
                f"was conditioned on (mean difference {diff:.1f} on 0-255). Score against the run's own "
                f"photographs with garment_lookup(run_dir).")


def run_orbit(
    frames_by_view: dict[str, Sequence[str]],
    garments: dict[str, "object"],
    references: dict[str, str],
    tryon_fn: Callable[..., dict],
    config: RunConfig,
    writer: RunWriter,
    parse_steps_for: int = 2,
    skip_existing: bool = False,
    configure_fn: Callable[[RunConfig], None] | None = None,
    on_frame: Callable[[str, str, dict], None] | None = None,
    verbose: bool = True,
) -> dict:
    """Generate one full orbit under `config` and write every artefact through `writer`.

    Parameters
    ----------
    frames_by_view
        ``{"front": [...], "back": [...], "side": [...]}`` of person-image paths. Reference frames
        are generated separately and should not appear here.
    garments
        One conditioning photograph per view (``eq:garment-selection``).
    references
        ``{"front": path, "back": path}``. Lateral frames have no reference by design.
    tryon_fn
        The notebook's ``two_phase_tryon``. Must accept ``(person_img, garment_img,
        is_ref_pass=..., guidance_scale=..., coarse_steps=..., fine_steps=..., mask_dilate=...,
        seed=..., mask_variant=...)`` and return a dict keyed by the stage names of
        :data:`vton2d.runio.STAGES`, optionally with ``parse_steps``.
    parse_steps_for
        Save the mask progression for this many frames per view class. Source material for
        ``fig:mask-stages``; saving it for every frame is wasteful.
    configure_fn
        Called once with `config` before any frame is generated, to install the attention
        processors that realise ``config.injection`` and ``config.window``. Required for anything
        but the released setting: without it those fields would reach the manifest without
        reaching the model.
    skip_existing
        Generate only the frames whose ``final`` image is not already in the run directory. Use
        with ``RunWriter(..., resume=True)`` after a Colab runtime dies mid-orbit. The reference
        frame of a class is regenerated whenever any of its targets are, because its pass is what
        fills the key/value bank the targets read from and that bank does not survive a restart.

    Returns a summary dict; the manifest is written before returning.
    """
    if config.injection not in ("decoder", "all", "none"):
        raise ValueError(f"unknown injection mode {config.injection!r}")

    if configure_fn is not None:
        # The notebook installs the attention processors for this configuration. Done once per
        # run rather than per frame, because the layer set is fixed at installation time.
        configure_fn(config)
    elif config.injection != "decoder" or tuple(config.window) != (5, 45):
        # Without a configure_fn the processor keeps whatever it was last given, so a config
        # asking for anything but the released setting would be recorded in the manifest and
        # silently not applied. That is the one failure this refuses to allow.
        raise ValueError(
            f"injection={config.injection!r}, window={tuple(config.window)} differ from the "
            f"released setting, but no configure_fn was supplied to apply them. Pass "
            f"configure_fn=... (see install_reference_attention in the notebook), or the manifest "
            f"would claim a configuration that was never run."
        )

    generated: list[str] = []
    n_generated = 0          # frames actually put through the model
    per_view: dict[str, int] = {}
    view_of: dict[str, str] = {}   # frame stem -> orientation class, for boundary analysis

    def call(person_path: str, view: str, is_ref: bool, saved_steps: int) -> int:
        person_img = _load_person(person_path, config.resolution)
        guidance = (config.ref_guidance_scale if (is_ref and view in tuple(config.ref_guidance_views))
                    else config.guidance_scale)
        artefacts = tryon_fn(
            person_img,
            garments[view],
            is_ref_pass=is_ref,
            guidance_scale=guidance,
            coarse_steps=config.coarse_steps,
            fine_steps=config.fine_steps,
            mask_dilate=config.mask_dilate,
            seed=config.seed,
            mask_variant=config.mask_variant,
        )
        name = os.path.basename(person_path)
        writer.save_frame(name, artefacts)
        if saved_steps < parse_steps_for and artefacts.get("parse_steps"):
            writer.save_parse_steps(name, artefacts["parse_steps"])
            saved_steps += 1
        nonlocal n_generated
        n_generated += 1
        generated.append(Path(name).stem)
        view_of[Path(name).stem] = view
        if on_frame is not None:
            on_frame(view, name, artefacts)
        if verbose:
            tag = "reference" if is_ref else "target"
            print(f"  [{view}/{tag}] {name}")
        return saved_steps

    def note(person_path: str, view: str) -> None:
        """Record a frame that is already on disk, without regenerating it.

        Registered with the writer as well: the manifest is written from the writer's list, so a
        reused frame that only reached `generated` would be on disk but missing from the manifest.
        """
        stem = Path(person_path).stem
        view_of[stem] = view
        writer.register(stem)
        if stem not in generated:
            generated.append(stem)

    missing = lambda path: not writer.has("final", os.path.basename(path))

    for view in VIEW_ORDER:
        targets = list(frames_by_view.get(view, []))
        reference = references.get(view)
        if not targets and not reference:
            continue
        if view not in garments:
            raise KeyError(f"no garment photograph supplied for view {view!r}")

        todo = [p for p in targets if missing(p)] if skip_existing else list(targets)
        done = [p for p in targets if p not in todo]
        if verbose:
            have = f", {len(done)} already present" if done else ""
            print(f"\n=== {view}: {len(todo)} target frame(s) to generate{have}"
                  f"{', 1 reference' if reference else ', no reference'} ===")

        writer.save_garment(view, garments[view])
        saved_steps = 0
        for path in done:
            note(path, view)

        if view == "side":
            # No injection: every lateral frame is generated as its own reference pass, so each
            # one can be resumed independently of the others.
            for path in todo:
                saved_steps = call(path, view, True, saved_steps)
            per_view[view] = len(targets)
            continue

        if reference is None:
            raise KeyError(f"view {view!r} has target frames but no reference frame")

        if todo or not skip_existing or missing(reference):
            # The reference is regenerated whenever any target of this class still has to be
            # made: its pass is what fills the key/value bank those targets read from, and the
            # bank does not survive between sessions. One extra frame, and skipping it would
            # silently produce targets with an empty bank.
            saved_steps = call(reference, view, True, saved_steps)
        else:
            note(reference, view)
        inject = config.injection != "none"
        for path in todo:
            saved_steps = call(path, view, not inject, saved_steps)
        per_view[view] = len(targets) + 1

    writer.write_manifest(extra={
        "per_view": per_view,
        "view_order": list(VIEW_ORDER),
        # Per-frame orientation class. Needed to tell which adjacent pairs sit at a class
        # boundary, which is what Section 6.3 predicts the consistency minima coincide with.
        "view_of": view_of,
    })
    if verbose:
        print(f"\nWrote {len(generated)} frames to {writer.root}")
    return {"run_dir": writer.root, "frames": generated, "per_view": per_view,
            "n_generated": n_generated}


def ensure_orbit(
    frames_by_view: dict[str, Sequence[str]],
    garments: dict[str, "object"],
    references: dict[str, str],
    tryon_fn: Callable[..., dict],
    config: RunConfig,
    runs_root: str | os.PathLike,
    force: bool = False,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Generate a run only to the extent that it is not already on disk.

    Makes the notebook cell safe to re-run from the top, which matters because a Colab runtime
    can die at any point in an hour-long orbit. Three outcomes:

    * nothing on disk        -> generate everything
    * partially generated    -> generate only what is missing, keeping what is there
    * already complete       -> generate nothing, just report

    `force=True` deletes the existing run and regenerates it from scratch. That is the only path
    that destroys work, and it never happens by accident.
    """
    expected = []
    for view in VIEW_ORDER:
        targets = list(frames_by_view.get(view, []))
        if not targets and view not in references:
            continue
        expected += [Path(p).stem for p in targets]
        if view != "side" and references.get(view):
            expected.append(Path(references[view]).stem)

    root = Path(runs_root) / config.run_id
    present = {p.stem for p in (root / "final").glob("*.png")} if (root / "final").is_dir() else set()
    outstanding = [f for f in expected if f not in present]

    if not force and (root / "manifest.json").exists() and not outstanding:
        if verbose:
            print(f"run already complete: {len(present)} frame(s) in {root}\n"
                  f"nothing to generate. Pass force=True to regenerate from scratch "
                  f"(this DELETES the existing run).")
        return {"run_dir": root, "frames": sorted(present), "generated": 0, "reused": len(present)}

    if verbose and present and not force:
        print(f"resuming: {len(present)} frame(s) already present, {len(outstanding)} to generate")

    writer = RunWriter(runs_root, config, overwrite=force, resume=not force)
    summary = run_orbit(frames_by_view, garments, references, tryon_fn, config, writer,
                        skip_existing=not force, verbose=verbose, **kwargs)
    summary["generated"] = summary.get("n_generated", len(outstanding))
    summary["reused"] = max(0, len(summary["frames"]) - summary["generated"])
    return summary


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_run(
    run_dir: str | os.PathLike,
    garment_for: Callable[[str], "object"] | None = None,
    frames: Iterable[str] | None = None,
    masks_stage: str = "cloth_mask",
    verbose: bool = True,
    garment_mask_for: Callable[[str], "object"] | None = None,
) -> dict:
    """Score a run written by :func:`run_orbit`, reading from disk.

    Frames are ordered by filename, which is extraction order and therefore orbit order — the
    consistency measure compares *adjacent views*, so this ordering is part of the measurement,
    not a convenience.

    Returns consistency (Section 5.4), the detail statistic that must accompany it, and the width
    error for each of the three masks (Section 5.7).

    `garment_mask_for` maps a frame to the garment region of its photograph (:func:`garment_lookup`).
    Without it the detail statistic is normalised over the backdrop heuristic, and
    ``result["garment_region"]`` records which of the two was used.
    """
    run = load_run(run_dir)
    path = run["path"]
    names = sorted(frames if frames is not None else run["frames"])
    if not names:
        raise ValueError(f"no frames recorded in {run_dir}")

    finals = [path("final", n) for n in names]
    masks = [path(masks_stage, n) for n in names]
    missing = [p for p in finals + masks if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} artefact(s) missing, first is {missing[0]}. "
            f"Was this run written before the pipeline persisted its intermediates?"
        )

    result: dict = {"run_dir": Path(run_dir), "config": run["config"], "n_frames": len(names)}
    result["consistency"] = M.consistency_series(finals, masks)
    result["frames"] = names

    # Boundary analysis. Section 6.3 predicts that consistency is worst where the orientation
    # class changes, because both the injected appearance and the conditioning photograph switch
    # there at once. Splitting the pairs is what turns that prediction into a measurement.
    views = run["manifest"].get("extra", {}).get("view_of") or infer_view_of(run_dir)
    if views:
        result["view_of"] = views
        result["boundaries"] = _boundary_split(result["consistency"]["distances"], names, views)

    # The detail statistic: never report consistency without it (Section 5.2.3).
    result["garment_region"] = None
    if garment_for is not None:
        _check_garment_for(run_dir, garment_for, names, views or {})
        # Which part of the photograph normalises the ratio. Recorded because the two regions differ
        # in kind on an on-model or non-white photograph (vton2d.metrics.garment_region).
        result["garment_region"] = ("stored garment region" if garment_mask_for is not None
                                    else "backdrop heuristic")
        ratios = []
        for n in names:
            try:
                ratios.append(M.detail_statistic(
                    path("final", n), path(masks_stage, n), garment_for(n),
                    garment_mask_for(n) if garment_mask_for is not None else None))
            except ValueError:
                pass
        if ratios:
            import numpy as np

            result["detail"] = {
                "mean": float(np.mean(ratios)),
                "min": float(np.min(ratios)),
                "n_frames": len(ratios),
            }
    else:
        result["detail"] = None

    # Width error for each mask stage (Section 5.2.4). Frames without a torso are skipped.
    widths = {}
    for stage in ("agnostic", "cloth_mask", "comp_mask"):
        per_frame = []
        for n in names:
            mask_path, dp_path = path(stage, n), path("densepose", n)
            if not (Path(mask_path).exists() and Path(dp_path).exists()):
                continue
            per_frame.append(M.width_error(mask_path, dp_path))
        if per_frame:
            widths[stage] = M.aggregate_width_error(per_frame)
    result["width_error"] = widths

    if verbose:
        c = result["consistency"]
        print(f"consistency  mean {c['mean']:.4f}   max {c['max']:.4f} "
              f"(pair {c['argmax_pair']})   over {c['n_frames']} frames")
        if result.get("detail"):
            print(f"detail       mean {result['detail']['mean']:.3f}   min {result['detail']['min']:.3f}")
        else:
            print("detail       not computed - pass garment_for=... ; consistency alone is not "
                  "evidence (Section 5.2.3)")
        for stage, w in widths.items():
            print(f"width {stage:<11} median {w['median_pct']:+.1f}%   "
                  f"q75 {w['q75_pct']:+.1f}%   over {w['n_frames']} frames")
    return result


def infer_view_of(run_dir: str | os.PathLike) -> dict[str, str]:
    """Recover each frame's orientation class from the DensePose maps saved in a run.

    Applies the same 2:1 dominance test the pipeline used at generation time, so a run written
    before ``run_orbit`` recorded ``view_of`` can still be analysed. Returns ``{}`` when the run
    has no ``densepose/`` stage.
    """
    root = Path(run_dir)
    if not (root / "densepose").is_dir():
        return {}
    out = {}
    for path in sorted((root / "densepose").glob("*.png")):
        out[path.stem] = M.view_from_densepose(path)
    return out


def backfill_view_of(run_dir: str | os.PathLike, verbose: bool = True) -> dict[str, str]:
    """Infer the orientation classes of an existing run and write them into its manifest."""
    from .runio import update_manifest

    views = infer_view_of(run_dir)
    if not views:
        raise FileNotFoundError(f"no densepose/ stage in {run_dir}; cannot infer orientations")
    update_manifest(run_dir, view_of=views)
    if verbose:
        counts: dict[str, int] = {}
        for v in views.values():
            counts[v] = counts.get(v, 0) + 1
        print(f"wrote view_of for {len(views)} frames: " +
              "  ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    return views


def _boundary_split(distances, names: Sequence[str], view_of: dict[str, str]) -> dict:
    """Split adjacent-pair distances into those crossing an orientation class and those inside one."""
    import numpy as np

    edges, interior = [], []
    for i in range(len(distances)):
        a, b = view_of.get(names[i]), view_of.get(names[i + 1])
        (edges if (a and b and a != b) else interior).append(i)

    def stat(idx):
        if not idx:
            return None
        d = np.asarray([distances[i] for i in idx])
        return {"n": len(idx), "mean": float(d.mean()), "max": float(d.max())}

    b, it = stat(edges), stat(interior)
    return {
        "boundary": b,
        "interior": it,
        "ratio": (b["mean"] / it["mean"]) if (b and it and it["mean"] > 0) else None,
        "boundary_pairs": [(names[i], names[i + 1], float(distances[i])) for i in edges],
    }


def report_run(run_dir: str | os.PathLike, garment_for=None, save: bool = True,
               garment_mask_for=None) -> dict:
    """Print every statistic a run supports, in one block, and return them.

    Reads only the run directory, so it works in a fresh session with no model loaded. By default
    the numbers are also written to ``scores.json`` inside the run, so a result survives the
    Colab runtime that produced it and can be quoted without being recomputed.
    """
    from .runio import load_run, save_scores

    run = load_run(run_dir)
    cfg = run["config"]
    man = run["manifest"]
    res = score_run(run_dir, garment_for=garment_for, verbose=False,
                    garment_mask_for=garment_mask_for)
    W = 64

    def rule(title=""):
        print("-" * W if not title else f"\n{'-' * W}\n {title}\n{'-' * W}")

    print("=" * W)
    print(f" RUN  {cfg.get('run_id', Path(run_dir).name)}")
    print("=" * W)
    print(f" subject / garment  {cfg.get('subject','?')} / {cfg.get('garment','?')}")
    print(f" steps T1 / T2      {cfg.get('coarse_steps')} / {cfg.get('fine_steps')}")
    ref_views = cfg.get("ref_guidance_views", _CONFIG_DEFAULTS["ref_guidance_views"])
    print(f" guidance           {cfg.get('guidance_scale')}  (reference passes of "
          f"{', '.join(ref_views) if ref_views else 'no class'} at {cfg.get('ref_guidance_scale')})")
    print(f" mask variant       {cfg.get('mask_variant')}   dilation {cfg.get('mask_dilate')} px")
    print(f" injection          {cfg.get('injection')}   window {cfg.get('window')}")
    print(f" seed / resolution  {cfg.get('seed')} / {tuple(cfg.get('resolution', ()))}")
    print(f" written / commit   {man.get('written_utc','?')} / {str(man.get('commit',''))[:8]}")
    if cfg.get("notes"):
        print(f" notes              {cfg['notes']}")

    views = res.get("view_of", {})
    counts: dict[str, int] = {}
    for v in views.values():
        counts[v] = counts.get(v, 0) + 1
    tally = "   ".join(f"{k} {n}" for k, n in sorted(counts.items())) if counts else "unknown"
    print(f"\n FRAMES  {res['n_frames']}    {tally}")

    c = res["consistency"]
    rule("CROSS-VIEW CONSISTENCY   (Section 5.2.3, lower is better)")
    print(f" mean {c['mean']:.4f}    median {c['median']:.4f}    "
          f"p90 {c['p90']:.4f}    max {c['max']:.4f}")
    i, j = c["argmax_pair"]
    fa, fb = res["frames"][i], res["frames"][j]
    edge = ""
    if views:
        edge = f"  ({views.get(fa,'?')} -> {views.get(fb,'?')})"
    print(f" worst pair  {fa} -> {fb}{edge}")
    if c["skipped_frames"]:
        print(f" skipped {len(c['skipped_frames'])} frame(s) with an empty mask")

    if res.get("boundaries"):
        bs = res["boundaries"]
        b, it, ratio = bs["boundary"], bs["interior"], bs["ratio"]
        print()
        if b:
            print(f" boundary pairs  n={b['n']:<3} mean {b['mean']:.4f}   max {b['max']:.4f}")
        if it:
            print(f" interior pairs  n={it['n']:<3} mean {it['mean']:.4f}   max {it['max']:.4f}")
        if ratio:
            verdict = ("supports Section 6.3" if ratio > 1.25 else
                       "does NOT support Section 6.3" if ratio < 1.05 else "inconclusive")
            print(f" ratio           {ratio:.2f}x  -> {verdict}")
        if b:
            print(" every class boundary:")
            for a, bb, d in bs["boundary_pairs"]:
                print(f"   {a} -> {bb}   {d:.4f}   "
                      f"({views.get(a,'?')} -> {views.get(bb,'?')})")

    rule("GARMENT DETAIL   (Section 5.2.3, ~1.0 = photograph-level texture)")
    if res.get("detail"):
        d = res["detail"]
        print(f" mean {d['mean']:.3f}    min {d['min']:.3f}    over {d['n_frames']} frames")
        print(f" normalised over the {res['garment_region']} of the photograph")
        print(" -> " + ("well clear of the degenerate case; the consistency figure is usable"
                        if d["mean"] > 0.6 else
                        "LOW: check whether the garment is being erased rather than stabilised"))
    else:
        print(" not computed. Pass garment_for=... — consistency alone is not evidence,")
        print(" because a flat, textureless garment scores a perfect 0 (Section 5.2.3).")

    rule("MASK WIDTH vs DensePose torso   (Section 5.2.4, + means too wide)")
    if res["width_error"]:
        print(f" {'mask':<12}{'median':>9}{'q75':>9}{'frames':>9}")
        for stage in ("agnostic", "cloth_mask", "comp_mask"):
            w = res["width_error"].get(stage)
            if w:
                print(f" {stage:<12}{w['median_pct']:>+8.1f}%{w['q75_pct']:>+8.1f}%"
                      f"{w['n_frames']:>9}")
        print(f" -> this run generates with {generating_stage(cfg)}")
    else:
        print(" no width errors: the run has no densepose/ stage")

    if save:
        path = save_scores(run_dir, res)
        print(f"\n saved to {path}")
    print()
    return res


def compare_runs(
    run_dirs: Sequence[str | os.PathLike],
    baseline: str | os.PathLike | None = None,
    garment_for=None,
    axes: Sequence[str] = ("coarse_steps", "fine_steps"),
    frames: Iterable[str] | None = None,
    common_frames: bool = True,
    mask_stage: str = "cloth_mask",
    lpips: bool = False,
    verbose: bool = True,
    garment_mask_for=None,
) -> list[dict]:
    """Tabulate several runs side by side: one row per run, one column per statistic.

    Built for a sweep. `axes` names the config fields that vary, so the table leads with them.

    `baseline` is a run every other run is compared against, which is what makes a sweep readable:

    * **mask IoU** against the baseline's cloth-specific mask isolates the shape half of the $T_1$
      question (Section 4.3.2). The budget also moves garment detail, which the detail column
      reports; scoring $T_1$ on the final image alone would confound it with $T_2$.
    * **LPIPS / SSIM** against the baseline's final frame answers the $T_2$ question, since the
      refined pass output is the delivered artefact (Section 4.3.5). Off by default: it needs the
      optional `lpips` package and is much slower than the rest.

    Only frames present in both a run and the baseline are compared, so a sweep may be run on a
    subset of the orbit. By default (`common_frames`) **every run is scored over the frames they
    all share**, so that a sweep on a 14-frame arc can be tabulated against a 58-frame reference
    corpus without the two rows measuring different things. Pass `frames` to fix the set
    explicitly, or `common_frames=False` to score each run over everything it holds.
    """
    import numpy as np

    from .runio import load_run

    run_dirs = list(run_dirs)
    base = load_run(baseline) if baseline is not None else None
    base_frames = set(base["frames"]) if base else set()

    shared_set: set[str] | None = None
    if frames is not None:
        shared_set = {Path(f).stem for f in frames}
    elif common_frames:
        every = [set(load_run(d)["frames"]) for d in run_dirs]
        if base is not None:
            every.append(base_frames)
        shared_set = set.intersection(*every) if every else None
        if shared_set is not None and verbose and any(len(s) != len(shared_set) for s in every):
            print(f"scoring every run over the {len(shared_set)} frame(s) they share\n")

    rows = []
    for run_dir in run_dirs:
        run = load_run(run_dir)
        cfg = run["config"]
        try:
            res = score_run(run_dir, garment_for=garment_for, verbose=False,
                            frames=sorted(shared_set) if shared_set else None,
                            garment_mask_for=garment_mask_for)
        except (ValueError, FileNotFoundError) as exc:
            rows.append({"run_id": cfg.get("run_id"), "error": str(exc)[:60]})
            continue

        row: dict = {"run_id": cfg.get("run_id", Path(run_dir).name)}
        for axis in axes:
            row[axis] = cfg.get(axis, _CONFIG_DEFAULTS.get(axis))
        row["n"] = res["n_frames"]
        row["consistency_mean"] = res["consistency"]["mean"]
        row["consistency_max"] = res["consistency"]["max"]
        row["detail_mean"] = (res.get("detail") or {}).get("mean")

        # Garment colour fidelity (thesis eq:fidelity). The only column that can see colour bleed:
        # bleed raises gradient energy like retained texture does, so detail cannot. Scored over
        # `mask_stage` in every row, so the region is identical and only the rendered pixels differ.
        if garment_for is not None:
            fid_frames = sorted(shared_set) if shared_set else run["frames"]
            fids = []
            for n in fid_frames:
                try:
                    fids.append(M.garment_fidelity(
                        run["path"]("final", n), run["path"](mask_stage, n), garment_for(n),
                        garment_mask_for(n) if garment_mask_for is not None else None))
                except (ValueError, FileNotFoundError):
                    pass
            if fids:
                row["fidelity_mean"] = float(np.mean(fids))
        # Measure each arm on the mask that actually generated it, not on a fixed stage: in the
        # mask ablation the generating mask is a different artefact in every arm.
        stage = generating_stage(cfg)
        row["gen_mask"] = stage
        width = res["width_error"].get(stage) or {}
        row["width_median_pct"] = width.get("median_pct")
        row["width_q75_pct"] = width.get("q75_pct")
        if res.get("boundaries") and res["boundaries"]["ratio"]:
            row["boundary_ratio"] = res["boundaries"]["ratio"]

        if base is not None and str(run_dir) != str(baseline):
            shared = sorted((set(run["frames"]) & base_frames) if shared_set is None
                            else shared_set)
            if shared:
                ious = [
                    M.mask_iou(run["path"](mask_stage, n), base["path"](mask_stage, n))
                    for n in shared
                ]
                row["mask_iou_vs_base"] = float(np.nanmean(ious))
                row["mask_iou_min"] = float(np.nanmin(ious))
                if lpips:
                    vals = [
                        M.pairwise_lpips_ssim(run["path"]("final", n), base["path"]("final", n))
                        for n in shared
                    ]
                    got = [v["lpips"] for v in vals if v["lpips"] is not None]
                    if got:
                        row["lpips_vs_base"] = float(np.mean(got))
        rows.append(row)

    if verbose:
        _print_table(rows, axes)
    return rows


#: Which mask actually drives generation, per mask variant. The width error of thesis Section 5.2.4
#: is only comparable across the arms of the mask ablation if each arm is measured on the mask that
#: produced its frames, which is a different stage in each arm.
GENERATING_STAGE = {
    "single_pass": "agnostic",       # no refinement: the wide cloth-agnostic mask generates
    "no_composition": "cloth_mask",  # refined mask, used directly on the original frame
    "erosion": "comp_mask",          # the contracted mask is stored in the comp_mask slot
    "composition": "comp_mask",      # refined then dilated, the released pipeline
}


def generating_stage(config: dict | RunConfig) -> str:
    """The artefact stage holding the mask that generated a run's frames."""
    variant = (config.get("mask_variant") if isinstance(config, dict) else config.mask_variant)
    return GENERATING_STAGE.get(variant, "comp_mask")


def _print_table(rows: Sequence[dict], axes: Sequence[str]) -> None:
    """Fixed-width table of :func:`compare_runs` rows."""
    if not rows:
        print("no runs to compare")
        return
    def axis_text(v):
        # Config values arrive from JSON, so a window tuple comes back as a list, and a plain
        # "{:>5}" format raises on it.
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(str(x) for x in v) + "]"
        return "" if v is None else str(v)

    axis_cols = []
    for a in axes:
        label = a.replace("coarse_steps", "T1").replace("fine_steps", "T2")
        width = max([len(label)] + [len(axis_text(r.get(a))) for r in rows]) + 2
        axis_cols.append((a, label, width, None))

    cols = axis_cols + [
        ("n", "n", 4, "{:>4}"),
        ("consistency_mean", "consist", 9, "{:>9.4f}"),
        ("consistency_max", "worst", 8, "{:>8.4f}"),
        ("detail_mean", "detail", 8, "{:>8.3f}"),
        ("fidelity_mean", "fidelity", 10, "{:>10.4f}"),
        ("gen_mask", "genmask", 12, "{:>12}"),
        ("width_median_pct", "width%", 8, "{:>+8.1f}"),
        ("width_q75_pct", "q75%", 8, "{:>+8.1f}"),
        ("mask_iou_vs_base", "maskIoU", 9, "{:>9.3f}"),
        ("mask_iou_min", "IoU min", 9, "{:>9.3f}"),
        ("lpips_vs_base", "LPIPS", 8, "{:>8.4f}"),
        ("boundary_ratio", "bnd x", 7, "{:>7.2f}"),
    ]
    present = [c for c in cols if any(r.get(c[0]) is not None for r in rows)]
    header = "".join(f"{label:>{w}}" for _, label, w, _ in present)
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"  {r['run_id']}: {r['error']}")
            continue
        line = ""
        for key, _, w, fmt in present:
            v = r.get(key)
            if fmt is None:
                line += f"{axis_text(v):>{w}}"
            else:
                line += fmt.format(v) if v is not None else " " * w
        print(line)
    print("\nconsist/worst: lower is better | detail ~1.0 = photograph-level texture")
    print("width%: composition mask against the DensePose torso, + means too wide")
    print("maskIoU: agreement of the cloth-specific mask with the baseline run")
    print("fidelity: colour distance to the conditioning photograph, lower is closer")


def parse_finals(
    run_dir: str | os.PathLike,
    parse_fn: Callable[["object"], "object"],
    stage: str = "final_densepose",
    skip_existing: bool = True,
    verbose: bool = True,
) -> int:
    """Run a parser over a run's delivered frames and store the result as a new stage.

    The runs hold DensePose for every *input* frame but not for the frames the pipeline produced,
    so nothing on disk can answer a question about the rendered body. This adds that side. It is a
    parsing pass, not a generation pass — no diffusion — so it is cheap to apply to every run that
    already exists.

    `parse_fn` takes a PIL image and returns the parse to store, e.g.
    ``lambda img: automasker(img)["densepose"]``.
    """
    from PIL import Image

    root = Path(run_dir)
    out = root / stage
    out.mkdir(parents=True, exist_ok=True)
    finals = sorted((root / "final").glob("*.png"))
    n = 0
    for path in finals:
        target = out / path.name
        if skip_existing and target.exists():
            continue
        parsed = parse_fn(Image.open(path).convert("RGB"))
        parsed.convert("L").save(target)
        n += 1
    if verbose:
        print(f"{root.name}: parsed {n} frame(s) into {stage}/ "
              f"({len(finals) - n} already present)")
    return n


def body_distortion(
    run_dir: str | os.PathLike,
    frames: Iterable[str] | None = None,
    stage: str = "final_densepose",
    verbose: bool = True,
) -> dict:
    """Compare the rendered subject's torso against the captured one, over a run.

    Requires :func:`parse_finals` to have been run first. Answers the geometric questions the
    colour measures cannot: whether the pipeline renders a body of the wrong width, and by how
    much (thesis §4.3.7, §4.4.4).
    """
    from .runio import load_run

    run = load_run(run_dir)
    path = run["path"]
    names = sorted(frames if frames is not None else run["frames"])
    per_frame = []
    for n in names:
        gen, ref = path(stage, n), path("densepose", n)
        if not (Path(gen).exists() and Path(ref).exists()):
            continue
        per_frame.append(M.body_width_shift(gen, ref))
    result = M.aggregate_body_shift(per_frame)
    result["run_dir"] = Path(run_dir)
    result["config"] = run["config"]
    if verbose:
        if result["n_frames"]:
            print(f"{Path(run_dir).name:<46} body width  signed {result['median_pct']:+6.1f}%   "
                  f"absolute {result['abs_median_pct']:5.1f}%   worst {result['worst_abs_pct']:5.1f}%"
                  f"   over {result['n_frames']} frames")
        else:
            print(f"{Path(run_dir).name}: no comparable frames — run parse_finals first")
    return result


# ---------------------------------------------------------------------------
# garment photographs and colour
# ---------------------------------------------------------------------------

def ensure_garment_masks(
    run_dir: str | os.PathLike,
    parse_fn: Callable[["object"], "object"] | None,
    views: Sequence[str] = ("front", "side", "back"),
    refresh: bool = False,
    verbose: bool = True,
) -> dict[str, str]:
    """Store the garment region of each conditioning photograph a run was generated from.

    :func:`run_orbit` saves the photographs to ``garment/<view>.png``; this adds
    ``garment/<view>_mask.png`` beside each, so colour measures against the photograph can be
    scored from the run directory over a region that means the same thing on every garment
    (:func:`vton2d.metrics.garment_region`). A parsing pass with no diffusion, so it is cheap to
    apply to runs that already exist.

    `parse_fn` takes the PIL photograph and returns either ``(mask, source)``, as
    :func:`vton2d.metrics.catalogue_garment_region` does, or a bare upper-clothes parse (or ``None``)
    for :func:`vton2d.metrics.garment_region` to choose from. Returns ``{view: source}``, with
    ``"present"`` for a region already on disk. ``refresh`` recomputes those as well; it rewrites only
    the ``garment/<view>_mask.png`` files and never touches a generated frame.
    """
    from PIL import Image

    root = Path(run_dir) / "garment"
    out: dict[str, str] = {}
    for view in views:
        photo_path, mask_path = root / f"{view}.png", root / f"{view}_mask.png"
        if not photo_path.exists():
            continue
        if mask_path.exists() and not refresh:
            out[view] = "present"
            continue
        photo = Image.open(photo_path).convert("RGB")
        parse = parse_fn(photo) if parse_fn is not None else None
        if isinstance(parse, tuple):
            mask, source = M.as_mask(parse[0]), parse[1]
        else:
            mask, source = M.garment_region(photo, parse, return_source=True)
        Image.fromarray(mask).save(mask_path)
        out[view] = source
        if verbose:
            print(f"  {Path(run_dir).name}/garment/{view}_mask.png   {source}, "
                  f"{float((mask > 0).mean()):.0%} of the photograph")
    if not out:
        raise FileNotFoundError(f"no garment photographs in {root}; was this run written by run_orbit?")
    return out


def garment_lookup(run_dir: str | os.PathLike, require_masks: bool = False):
    """``(garment_for, garment_mask_for)`` read from a run directory, for the scoring functions.

    Both map a frame name to a file for that frame's orientation class: the photograph it was
    conditioned on (``eq:garment-selection``) and the garment region stored by
    :func:`ensure_garment_masks`. Reading them from the run rather than from notebook variables
    means a frame cannot be scored against a different garment from the one that generated it.

    With ``require_masks`` a missing region raises instead of letting the scorers fall back to the
    backdrop heuristic, which is the right setting for any comparison across garments.
    """
    run = load_run(run_dir)
    root = Path(run_dir) / "garment"
    views = run["manifest"].get("extra", {}).get("view_of") or infer_view_of(run_dir)
    classes = set(views.values()) - {"unknown"}
    if not classes:
        raise ValueError(f"no orientation classes for {run_dir}; run backfill_view_of first")
    photos = {v: root / f"{v}.png" for v in classes if (root / f"{v}.png").exists()}
    masks = {v: root / f"{v}_mask.png" for v in photos if (root / f"{v}_mask.png").exists()}
    if classes - set(photos):
        raise FileNotFoundError(f"{run_dir}: no conditioning photograph for {sorted(classes - set(photos))}")
    if require_masks and set(masks) != set(photos):
        raise FileNotFoundError(f"{run_dir}: no garment region for {sorted(set(photos) - set(masks))}; "
                                f"run ensure_garment_masks first")

    def view(frame):
        v = views.get(Path(frame).stem)
        if v not in photos:
            raise ValueError(f"frame {frame!r} has no usable orientation class in {run_dir}")
        return v

    return (lambda frame: photos[view(frame)]), (lambda frame: masks.get(view(frame)))


def colour_by_class(
    run_dir: str | os.PathLike,
    frames: Iterable[str] | None = None,
    mask_stage: str = "cloth_mask",
    verbose: bool = True,
    stage: str = "final",
    washed_margin: float = 64.0,
) -> dict:
    """Colour against the conditioning photograph, per orientation class (thesis §6.2).

    ``n_washed`` counts the frames whose garment median value exceeds the photograph's by more than
    ``washed_margin``. Unlike the shares it does not depend on the spread of the photograph's colours,
    which for a plain black garment is narrow enough that a rendering still close to black exceeds it.

    ``stage="coarse"`` scores the Phase-1 output over the same region instead of the delivered frame,
    which is what §6.2's statement about the refinement pass on dorsal frames needs.

    For every frame, :func:`vton2d.metrics.colour_shift` and :func:`vton2d.metrics.garment_fidelity`
    against the photograph of that frame's class, both over the stored garment region, which must
    exist (:func:`ensure_garment_masks`). Aggregated per class as medians.

    Read within a garment, between classes. Capture lighting differs from catalogue lighting, so no
    class sits at zero, and a light garment has no headroom to lighten; what places a failure on the
    dorsal views is the dorsal class departing from the frontal class of the *same* garment, which
    ``dorsal_minus_frontal`` reports.
    """
    import numpy as np

    run = load_run(run_dir)
    path = run["path"]
    garment_for, mask_for = garment_lookup(run_dir, require_masks=True)
    views = run["manifest"].get("extra", {}).get("view_of") or infer_view_of(run_dir)
    names = sorted(frames if frames is not None else run["frames"])

    rows = []
    for n in names:
        image, mask = path(stage, n), path(mask_stage, n)
        if not (Path(image).exists() and Path(mask).exists()):
            continue
        try:
            shift = M.colour_shift(image, mask, garment_for(n), mask_for(n))
            fid = M.garment_fidelity(image, mask, garment_for(n), mask_for(n))
        except ValueError:
            continue
        rows.append({"frame": Path(n).stem, "view": views.get(Path(n).stem), "fidelity": fid, **shift})

    keys = ("d_value", "d_saturation", "fidelity", "lighter_share", "greyer_share",
            "value", "ref_value", "saturation", "ref_saturation")
    by_class = {}
    for cls in ("front", "side", "back"):
        sel = [r for r in rows if r["view"] == cls]
        if sel:
            by_class[cls] = {"n": len(sel), **{k: float(np.median([r[k] for r in sel])) for k in keys}}
            # Washout can switch on for whole frames rather than shift every frame a little, and a
            # class median hides that: count the frames where most of the garment left the range of
            # the photograph's colours.
            by_class[cls]["n_mostly_lighter"] = int(sum(r["lighter_share"] >= 0.5 for r in sel))
            by_class[cls]["lighter_max"] = float(max(r["lighter_share"] for r in sel))
            by_class[cls]["n_washed"] = int(sum(r["d_value"] > washed_margin for r in sel))
    penalty = None
    if "front" in by_class and "back" in by_class:
        penalty = {k: by_class["back"][k] - by_class["front"][k]
                   for k in ("d_value", "d_saturation", "fidelity", "lighter_share", "greyer_share")}

    result = {"run_dir": Path(run_dir), "garment": run["config"].get("garment"), "stage": stage,
              "frames": rows, "by_class": by_class, "dorsal_minus_frontal": penalty}
    if verbose:
        print(f"{Path(run_dir).name}   ({stage}, {len(rows)} frames, garment {result['garment']})")
        print(f"  {'class':<7}{'n':>4}{'dV':>8}{'dS':>8}{'fidelity':>10}{'lighter':>9}{'greyer':>8}"
              f"{'mostly lighter':>17}{'washed':>8}   V frame/photo   S frame/photo")
        for cls, c in by_class.items():
            print(f"  {cls:<7}{c['n']:>4}{c['d_value']:>+8.1f}{c['d_saturation']:>+8.1f}"
                  f"{c['fidelity']:>10.3f}{c['lighter_share']:>9.1%}{c['greyer_share']:>8.1%}"
                  f"{c['n_mostly_lighter']:>6} (max {c['lighter_max']:>4.0%}){c['n_washed']:>8}"
                  f"   {c['value']:5.0f} / {c['ref_value']:<5.0f}"
                  f"  {c['saturation']:5.0f} / {c['ref_saturation']:<5.0f}")
        if penalty:
            print(f"  {'back-front':<11}{penalty['d_value']:>+8.1f}{penalty['d_saturation']:>+8.1f}"
                  f"{penalty['fidelity']:>+10.3f}{penalty['lighter_share']:>+9.1%}"
                  f"{penalty['greyer_share']:>+8.1%}")
        print("  lighter / greyer: share of garment pixels lighter than the photograph's lightest 5 %,"
              " greyer than its greyest 5 %")
        print("  mostly lighter: frames in which over half the garment is lighter than that, and the"
              " largest share in any frame")
        print(f"  washed: frames whose garment median value exceeds the photograph's by more than"
              f" {washed_margin:.0f}; saturation is nan where the photograph's garment is near-black")
    return result


def outside_change_by_class(
    run_dir: str | os.PathLike,
    frames: Iterable[str] | None = None,
    margin: int = 15,
    verbose: bool = True,
) -> dict:
    """What the refinement pass changed outside the composition mask, per orientation class (§6.2).

    :func:`vton2d.metrics.outside_mask_change` between the delivered frame and the pass's own input
    (``composite``, which outside ``comp_mask`` is the captured frame), beyond ``margin`` px of the
    mask. At the released guidance this is the autoencoder round trip alone; §6.2's account of raised
    guidance predicts it grows. Runs without a composite (the single-pass arm) yield no classes.
    """
    import numpy as np

    run = load_run(run_dir)
    path = run["path"]
    views = run["manifest"].get("extra", {}).get("view_of") or infer_view_of(run_dir)
    names = sorted(frames if frames is not None else run["frames"])

    rows = []
    for n in names:
        final, comp, mask = path("final", n), path("composite", n), path("comp_mask", n)
        if not all(Path(p).exists() for p in (final, comp, mask)):
            continue
        try:
            change = M.outside_mask_change(final, comp, mask, margin=margin)
        except ValueError:
            continue
        rows.append({"frame": Path(n).stem, "view": views.get(Path(n).stem), **change})

    keys = ("grain_ratio", "abs_diff", "d_saturation")
    by_class = {}
    for cls in ("front", "side", "back"):
        sel = [r for r in rows if r["view"] == cls]
        if sel:
            by_class[cls] = {"n": len(sel), **{k: float(np.median([r[k] for r in sel])) for k in keys}}

    result = {"run_dir": Path(run_dir), "margin": margin, "frames": rows, "by_class": by_class}
    if verbose:
        print(f"{Path(run_dir).name}   (outside comp_mask + {margin} px, {len(rows)} frames)")
        print(f"  {'class':<7}{'n':>4}{'grain':>9}{'|diff|':>9}{'dS':>8}")
        for cls, c in by_class.items():
            print(f"  {cls:<7}{c['n']:>4}{c['grain_ratio']:>9.3f}{c['abs_diff']:>9.2f}"
                  f"{c['d_saturation']:>+8.1f}")
        print("  grain: output over input gradient energy, 1.0 = unchanged; |diff| on 0-255")
    return result


def count_decoder_attn1(unet, verbose: bool = True) -> dict:
    """Count the ``attn1`` layers in the U-Net decoder and record their feature resolutions.

    Settles Q5 in ``OPEN_QUESTIONS.md`` and fills the ``\\todo`` at Section 4.4.4. Read off the
    backbone actually in use rather than assumed from a stock Stable Diffusion topology — the
    thesis is written about CatVTON's U-Net, and its attention set is what the layer restriction
    of ``sec:lfma-layers`` selects.
    """
    rows = []
    for name, module in unet.named_modules():
        if name.endswith("attn1") and name.startswith("up_blocks"):
            heads = getattr(module, "heads", None)
            dim = getattr(getattr(module, "to_q", None), "in_features", None)
            rows.append({"name": name, "heads": heads, "inner_dim": dim})
    encoder = sum(
        1 for name, _ in unet.named_modules()
        if name.endswith("attn1") and name.startswith("down_blocks")
    )
    mid = sum(
        1 for name, _ in unet.named_modules()
        if name.endswith("attn1") and name.startswith("mid_block")
    )
    out = {"decoder": rows, "n_decoder": len(rows), "n_encoder": encoder, "n_mid": mid}
    if verbose:
        print(f"attn1 layers - decoder {len(rows)}, encoder {encoder}, mid {mid}")
        for r in rows:
            print(f"  {r['name']:<52} heads={r['heads']} dim={r['inner_dim']}")
        print("\nFeature resolutions are not stored on the module; to record them for Section 4.4.4, "
              "hook these layers during one denoising call and log the sequence length n_l, then "
              "n_l = (H/s) * (W/s) gives the downsampling factor s for the working resolution.")
    return out
