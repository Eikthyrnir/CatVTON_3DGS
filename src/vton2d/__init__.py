"""2D evaluation support for the cascaded try-on pipeline.

Everything here is called from ``src/CatVTON_3DGS_pipeline.ipynb``; nothing in this package
imports CatVTON, diffusers or torch at module level, so it can be imported (and the metrics
exercised) on a machine with no GPU.

Modules
-------
runio    run directories, artefact persistence, manifests
metrics  the 2D instruments of thesis Section 5.2
driver   the per-orbit generation loop that replaces the three duplicated notebook loops

Usage from the notebook::

    import sys; sys.path.append('/content/CatVTON_3DGS/src')
    from vton2d import RunConfig, RunWriter, run_orbit, score_run
"""

from .runio import (STAGES, RunConfig, RunWriter, load_run, list_runs, export_stage,
                    update_manifest, save_scores, load_scores)
from .metrics import (
    consistency_series,
    consistency_pair,
    detail_statistic,
    gradient_magnitude_mean,
    hsv_histogram,
    pairwise_lpips_ssim,
    width_error,
    view_from_densepose,
    mask_iou,
    garment_fidelity,
)
from .driver import (run_orbit, ensure_orbit, score_run, report_run, compare_runs,
                     infer_view_of, backfill_view_of, count_decoder_attn1, generating_stage)

__all__ = [
    "STAGES",
    "RunConfig",
    "RunWriter",
    "load_run",
    "list_runs",
    "export_stage",
    "update_manifest",
    "save_scores",
    "load_scores",
    "view_from_densepose",
    "mask_iou",
    "garment_fidelity",
    "report_run",
    "compare_runs",
    "generating_stage",
    "infer_view_of",
    "backfill_view_of",
    "consistency_series",
    "consistency_pair",
    "detail_statistic",
    "gradient_magnitude_mean",
    "hsv_histogram",
    "pairwise_lpips_ssim",
    "width_error",
    "run_orbit",
    "ensure_orbit",
    "score_run",
    "count_decoder_attn1",
]

#: Stamped into every ``scores.json`` by :func:`vton2d.runio.save_scores`.
#:
#: **Bump this only when a metric's definition changes** — the histogram binning, the distance,
#: the gradient normalisation, the width-error rows, or which mask a variant is scored on. It marks
#: results as incomparable, so bumping it for an added function or a docstring would falsely
#: condemn numbers that are still valid.
#:
#: 0.1.0  consistency (HSV 8x8x8, Bhattacharyya), detail statistic, width error, mask IoU.
#:        Unchanged since the first reference corpus, so every score file to date is comparable.
__version__ = "0.1.0"
