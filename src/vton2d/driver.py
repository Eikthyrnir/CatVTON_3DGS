"""The per-orbit generation loop, and the scoring pass over what it wrote.

Replaces the three near-duplicate try-on loops in the notebook, which differed only in their
folder, their garment photograph and their reference frame — and which saved the final image only.

Nothing here imports torch, diffusers or CatVTON. The notebook passes ``two_phase_tryon`` in as
``tryon_fn``, so the model lives in the notebook and the orchestration lives under version control.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import metrics as M
from .runio import RunConfig, RunWriter, load_run

__all__ = ["run_orbit", "ensure_orbit", "score_run", "report_run", "compare_runs",
           "infer_view_of", "backfill_view_of", "count_decoder_attn1", "generating_stage",
           "parse_finals", "body_distortion", "GENERATING_STAGE", "VIEW_ORDER"]

#: Generation order over the orientation classes. **This order is load-bearing** (thesis 4.4.2).
#: A reference pass clears the key/value bank and refills it, so every frame that consumes a bank
#: must be generated before the next reference pass runs. Lateral frames are each generated as
#: their own reference pass (Section 4.5: they receive no injection), which clears the bank, so
#: they must all come first — running them between a reference and its targets would wipe it.
VIEW_ORDER = ("side", "front", "back")


def _load_person(path, resolution):
    from PIL import Image  # local: keeps the module importable without PIL at import time

    return Image.open(path).convert("RGB").resize(tuple(resolution))


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
        guidance = config.ref_guidance_scale if (is_ref and view == "front") else config.guidance_scale
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
        """Record a frame that is already on disk, without regenerating it."""
        stem = Path(person_path).stem
        view_of[stem] = view
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
) -> dict:
    """Score a run written by :func:`run_orbit`, reading from disk.

    Frames are ordered by filename, which is extraction order and therefore orbit order — the
    consistency measure compares *adjacent views*, so this ordering is part of the measurement,
    not a convenience.

    Returns consistency (Section 5.4), the detail statistic that must accompany it, and the width
    error for each of the three masks (Section 5.7).
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
    if garment_for is not None:
        ratios = []
        for n in names:
            try:
                ratios.append(M.detail_statistic(path("final", n), path(masks_stage, n), garment_for(n)))
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


def report_run(run_dir: str | os.PathLike, garment_for=None, save: bool = True) -> dict:
    """Print every statistic a run supports, in one block, and return them.

    Reads only the run directory, so it works in a fresh session with no model loaded. By default
    the numbers are also written to ``scores.json`` inside the run, so a result survives the
    Colab runtime that produced it and can be quoted without being recomputed.
    """
    from .runio import load_run, save_scores

    run = load_run(run_dir)
    cfg = run["config"]
    man = run["manifest"]
    res = score_run(run_dir, garment_for=garment_for, verbose=False)
    W = 64

    def rule(title=""):
        print("-" * W if not title else f"\n{'-' * W}\n {title}\n{'-' * W}")

    print("=" * W)
    print(f" RUN  {cfg.get('run_id', Path(run_dir).name)}")
    print("=" * W)
    print(f" subject / garment  {cfg.get('subject','?')} / {cfg.get('garment','?')}")
    print(f" steps T1 / T2      {cfg.get('coarse_steps')} / {cfg.get('fine_steps')}")
    print(f" guidance           {cfg.get('guidance_scale')}  (frontal reference "
          f"{cfg.get('ref_guidance_scale')})")
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
) -> list[dict]:
    """Tabulate several runs side by side: one row per run, one column per statistic.

    Built for a sweep. `axes` names the config fields that vary, so the table leads with them.

    `baseline` is a run every other run is compared against, which is what makes a sweep readable:

    * **mask IoU** against the baseline's cloth-specific mask answers the $T_1$ question, because
      the only property of the coarse pass used downstream is the *shape* it produces
      (Section 4.3.2). Scoring $T_1$ on the final image instead would confound it with $T_2$.
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
                            frames=sorted(shared_set) if shared_set else None)
        except (ValueError, FileNotFoundError) as exc:
            rows.append({"run_id": cfg.get("run_id"), "error": str(exc)[:60]})
            continue

        row: dict = {"run_id": cfg.get("run_id", Path(run_dir).name)}
        for axis in axes:
            row[axis] = cfg.get(axis)
        row["n"] = res["n_frames"]
        row["consistency_mean"] = res["consistency"]["mean"]
        row["consistency_max"] = res["consistency"]["max"]
        row["detail_mean"] = (res.get("detail") or {}).get("mean")
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
    cols = [
        (a, a.replace("coarse_steps", "T1").replace("fine_steps", "T2"), 5, "{:>5}") for a in axes
    ] + [
        ("n", "n", 4, "{:>4}"),
        ("consistency_mean", "consist", 9, "{:>9.4f}"),
        ("consistency_max", "worst", 8, "{:>8.4f}"),
        ("detail_mean", "detail", 8, "{:>8.3f}"),
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
            line += fmt.format(v) if v is not None else " " * w
        print(line)
    print("\nconsist/worst: lower is better | detail ~1.0 = photograph-level texture")
    print("width%: composition mask against the DensePose torso, + means too wide")
    print("maskIoU: agreement of the cloth-specific mask with the baseline run")


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
