"""Run directories, artefact persistence and manifests.

The pipeline used to save only the final try-on image: ``two_phase_tryon`` returned the masks and
the refinement steps, and the calling loops displayed them and dropped them. That made the two 2D
metrics of thesis Section 5.2 impossible to compute after the fact, and made ``fig:mask-stages``
impossible to draw. This module fixes that by giving every run its own directory, one
subdirectory per artefact stage, and a manifest recording the configuration that produced it.

Layout of a run directory::

    <root>/<run_id>/
        manifest.json          the RunConfig, verbatim, plus provenance
        garment/<view>.png     the conditioning photographs actually used
        agnostic/<frame>.png   M^(0), the cloth-agnostic mask
        densepose/<frame>.png  DensePose part map (needed for the width error)
        coarse/<frame>.png     I^(1), the Phase-1 try-on
        cloth_mask/<frame>.png M^(1), the cloth-specific mask
        comp_mask/<frame>.png  M_comp, the dilated composition mask
        composite/<frame>.png  I~, the body-preserving composite
        final/<frame>.png      I^(2), the delivered frame
        parse_steps/<frame>/NN_<name>.png   mask progression, first frames only
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Artefact stages written once per frame. Order is the order they are produced in.
STAGES = (
    "agnostic",
    "densepose",
    "coarse",
    "cloth_mask",
    "comp_mask",
    "composite",
    "final",
)

#: Stages that hold a single-channel mask or label map; saved as PIL mode "L".
MASK_STAGES = ("agnostic", "densepose", "cloth_mask", "comp_mask")


@dataclass
class RunConfig:
    """Everything that distinguishes one 2D run from another.

    Anything that changes a pixel belongs here, because the manifest is what makes a results
    table reproducible (thesis Section 5.3). Defaults are the released configuration of
    ``tab:config``; a sweep overrides one field at a time.
    """

    run_id: str
    subject: str = ""
    garment: str = ""

    # --- generation -------------------------------------------------------
    coarse_steps: int = 50          # T_1
    fine_steps: int = 50            # T_2
    guidance_scale: float = 2.5     # target frames and the dorsal reference
    ref_guidance_scale: float = 5.0 # frontal reference only; see tab:config
    mask_dilate: int = 14           # rho, in px
    seed: int = 42
    resolution: tuple[int, int] = (768, 1024)

    # --- mask variant (experiment E8) -------------------------------------
    # "composition"  released pipeline: TPMR + dilation + paste-back + refine
    # "erosion"      superseded variant: TPMR + contraction, no paste-back
    # "single_pass"  no TPMR at all: the Phase-1 output with the wide mask
    mask_variant: str = "composition"

    # --- LF-MA (experiments E7, layer ablation) ---------------------------
    # "decoder"  attn1 in up_blocks, the proposed pipeline
    # "all"      every attn1 layer, the unrestricted baseline
    # "none"     no injection at all
    injection: str = "decoder"
    window: tuple[int, int] = (5, 45)   # [T_lo, T_hi] of `coarse_steps`

    # --- bookkeeping ------------------------------------------------------
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def variant_of(self, run_id: str, **overrides: Any) -> "RunConfig":
        """Return a copy with `run_id` and any overridden fields. Used to build sweeps."""
        data = asdict(self)
        data.update(overrides)
        data["run_id"] = run_id
        return RunConfig(**data)


def _git_commit(start: Path) -> str:
    """Best-effort commit hash of the repository containing `start`. Never raises."""
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            try:
                out = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=10,
                )
                if out.returncode == 0:
                    return out.stdout.strip()
            except Exception:
                return "unknown"
            return "unknown"
    return "unknown"


class RunWriter:
    """Creates a run directory and writes artefacts into it.

    Refuses to write into a directory that already holds a manifest unless ``overwrite=True``.
    Two configurations sharing an output directory is the failure this class exists to prevent:
    the second silently overwrites the first and the resulting table cannot be trusted.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        config: RunConfig,
        overwrite: bool = False,
        resume: bool = False,
    ):
        self.config = config
        self.root = Path(root) / config.run_id
        self.resumed = False
        manifest = self.root / "manifest.json"
        if manifest.exists():
            if resume:
                self.resumed = True
            elif overwrite:
                # Destructive, and on a full orbit that is an hour of GPU time. Reached only
                # because a caller asked for it explicitly.
                shutil.rmtree(self.root)
            else:
                raise FileExistsError(
                    f"{self.root} already holds a run.\n"
                    f"  - to add only what is missing, pass resume=True (nothing is deleted)\n"
                    f"  - to regenerate from scratch, pass overwrite=True (DELETES the run)\n"
                    f"  - or give this configuration a different run_id"
                )
        for stage in STAGES:
            (self.root / stage).mkdir(parents=True, exist_ok=True)
        (self.root / "garment").mkdir(parents=True, exist_ok=True)
        self._counts = {stage: 0 for stage in STAGES}
        self._frames: list[str] = []
        if self.resumed:
            try:
                self._frames = list(json.loads(manifest.read_text(encoding="utf-8"))
                                    .get("frames", []))
            except Exception:
                self._frames = []

    def has(self, stage: str, name: str) -> bool:
        """Whether this run already holds `stage` for `name`."""
        return (self.root / stage / f"{self._stem(name)}.png").exists()

    # -- writing -----------------------------------------------------------

    @staticmethod
    def _stem(name: str) -> str:
        return Path(name).stem

    def save(self, stage: str, name: str, image) -> Path:
        """Write one artefact. `image` is a PIL image; masks are coerced to mode L."""
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
        if image is None:
            return self.root / stage / f"{self._stem(name)}.png"
        img = image.convert("L") if stage in MASK_STAGES else image.convert("RGB")
        path = self.root / stage / f"{self._stem(name)}.png"
        img.save(path)
        self._counts[stage] += 1
        return path

    def save_frame(self, name: str, artefacts: dict) -> None:
        """Write every stage present in `artefacts` (the dict returned by ``two_phase_tryon``)."""
        for stage in STAGES:
            if artefacts.get(stage) is not None:
                self.save(stage, name, artefacts[stage])
        stem = self._stem(name)
        if stem not in self._frames:
            self._frames.append(stem)

    def save_garment(self, view: str, image) -> Path:
        path = self.root / "garment" / f"{view}.png"
        image.convert("RGB").save(path)
        return path

    def save_parse_steps(self, name: str, images, labels=None) -> None:
        """Write the mask progression for one frame. Source material for ``fig:mask-stages``."""
        if not images:
            return
        out = self.root / "parse_steps" / self._stem(name)
        out.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            label = ""
            if labels and i < len(labels):
                label = "_" + str(labels[i])
            img.convert("L" if img.mode in ("L", "1") else "RGB").save(out / f"{i:02d}{label}.png")

    def write_manifest(self, extra: dict | None = None) -> Path:
        payload = {
            "config": asdict(self.config),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": _git_commit(Path(__file__).resolve()),
            "frames": self._frames,
            "counts": dict(self._counts),
        }
        if extra:
            payload["extra"] = extra
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def load_run(run_dir: str | os.PathLike) -> dict:
    """Read a run written by :class:`RunWriter`.

    Returns ``{"dir", "config", "manifest", "frames", "path"}`` where ``path(stage, frame)``
    gives the file for one artefact. Scoring reads from disk rather than from a live kernel, so
    a run can be scored in a later session, or after the metric is changed.
    """
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {root}; is this a run directory?")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames") or sorted(p.stem for p in (root / "final").glob("*.png"))

    def path(stage: str, frame: str) -> Path:
        return root / stage / f"{Path(frame).stem}.png"

    return {
        "dir": root,
        "config": manifest.get("config", {}),
        "manifest": manifest,
        "frames": frames,
        "path": path,
    }


def update_manifest(run_dir: str | os.PathLike, **extra: Any) -> dict:
    """Merge keys into a run's ``manifest["extra"]`` and rewrite it.

    For backfilling information onto a run that was produced before the code recorded it. The
    config block is never touched: what a run was configured with is a historical fact.
    """
    root = Path(run_dir)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("extra", {}).update(extra)
    manifest.setdefault("amended_utc", [])
    manifest["amended_utc"].append({
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "keys": sorted(extra),
    })
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_stage(
    run_dir: str | os.PathLike,
    dest: str | os.PathLike,
    stage: str = "final",
    suffix: str = ".jpg",
) -> int:
    """Copy one stage of a run into a flat directory, e.g. the 3DGS input folder.

    The run directory is the archive; the flat copy is what a downstream tool consumes. Keeping
    them separate is what lets several configurations coexist without overwriting each other.
    """
    src = Path(run_dir) / stage
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.glob("*.png")):
        target = out / (path.stem + suffix)
        if suffix.lower() in (".jpg", ".jpeg"):
            from PIL import Image

            Image.open(path).convert("RGB").save(target, quality=95)
        else:
            shutil.copyfile(path, target)
        n += 1
    return n


def list_runs(root: str | os.PathLike) -> list[dict]:
    """Summarise every run under `root`, newest first. Handy for a notebook overview cell."""
    out = []
    for manifest_path in sorted(Path(root).glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = manifest.get("config", {})
        out.append({
            "run_id": cfg.get("run_id", manifest_path.parent.name),
            "dir": manifest_path.parent,
            "frames": len(manifest.get("frames", [])),
            "written_utc": manifest.get("written_utc", ""),
            "mask_variant": cfg.get("mask_variant"),
            "injection": cfg.get("injection"),
            "window": cfg.get("window"),
            "guidance_scale": cfg.get("guidance_scale"),
            "coarse_steps": cfg.get("coarse_steps"),
        })
    return sorted(out, key=lambda r: r["written_utc"], reverse=True)
