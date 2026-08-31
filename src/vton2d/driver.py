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

__all__ = ["run_orbit", "score_run", "count_decoder_attn1", "VIEW_ORDER"]

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

    Returns a summary dict; the manifest is written before returning.
    """
    if config.injection == "all":
        raise NotImplementedError(
            "injection='all' needs the layer set lifted out of ReferenceAttentionProcessor into "
            "the config; it is not honoured yet and would silently run the 'decoder' variant. "
            "Wire the layer restriction before running the layer ablation (experiment C2)."
        )
    if config.injection not in ("decoder", "none"):
        raise ValueError(f"unknown injection mode {config.injection!r}")
    if tuple(config.window) != (5, 45):
        raise NotImplementedError(
            f"window={tuple(config.window)} is recorded in the manifest but not yet applied: the "
            "injection window lives in ReferenceAttentionProcessor. Wire it before the sweep (E7)."
        )

    generated: list[str] = []
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
        generated.append(Path(name).stem)
        view_of[Path(name).stem] = view
        if on_frame is not None:
            on_frame(view, name, artefacts)
        if verbose:
            tag = "reference" if is_ref else "target"
            print(f"  [{view}/{tag}] {name}")
        return saved_steps

    for view in VIEW_ORDER:
        targets = list(frames_by_view.get(view, []))
        reference = references.get(view)
        if not targets and not reference:
            continue
        if view not in garments:
            raise KeyError(f"no garment photograph supplied for view {view!r}")
        if verbose:
            print(f"\n=== {view}: {len(targets)} target frame(s)"
                  f"{', 1 reference' if reference else ', no reference'} ===")

        writer.save_garment(view, garments[view])
        saved_steps = 0

        if view == "side":
            # No injection: every lateral frame is generated as its own reference pass.
            for path in targets:
                saved_steps = call(path, view, True, saved_steps)
            per_view[view] = len(targets)
            continue

        if reference is None:
            raise KeyError(f"view {view!r} has target frames but no reference frame")
        # Reference first: fills the bank that the targets below consume.
        saved_steps = call(reference, view, True, saved_steps)
        inject = config.injection != "none"
        for path in targets:
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
    return {"run_dir": writer.root, "frames": generated, "per_view": per_view}


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
