"""The 2D instruments of thesis Section 5.2.

Three measures, each answering a question the others cannot:

* :func:`consistency_series`  — cross-view consistency (Section 5.2.3, ``eq:consistency``), RQ1
* :func:`detail_statistic`    — the guard against the degenerate case, reported *with* the above
* :func:`width_error`         — mask width against the DensePose torso (Section 5.2.4,
  ``eq:width-error``), RQ3

Only numpy, OpenCV and PIL are required. LPIPS and SSIM are imported lazily, so the rest of the
module works without them.

**The consistency measure is never reported alone.** A garment rendered as a uniform, textureless
patch scores perfectly, because a blur is extremely consistent from view to view. Section 5.2.3
exists to close that hole: a variant is credited only when it is both consistent *and* retains
garment detail, so every call site pairs the two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "as_bgr",
    "as_mask",
    "hsv_histogram",
    "consistency_pair",
    "consistency_series",
    "gradient_magnitude_mean",
    "detail_statistic",
    "width_error",
    "pairwise_lpips_ssim",
]

HSV_BINS = (8, 8, 8)  # Section 5.2.3, fixed by the thesis text
TORSO_LABELS = (1, 2)  # DensePose: 1 = torso back, 2 = torso front


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------

def _open(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image


def as_bgr(image) -> np.ndarray:
    """Return an ``HxWx3`` uint8 BGR array from a path, PIL image or array."""
    rgb = np.asarray(_open(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def as_mask(mask, threshold: int = 127) -> np.ndarray:
    """Return an ``HxW`` uint8 array in {0, 255} from a path, PIL image or array."""
    arr = np.asarray(_open(mask).convert("L"))
    return ((arr > threshold).astype(np.uint8)) * 255


def _resize_mask_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Section 5.2.3 — cross-view consistency
# ---------------------------------------------------------------------------

def hsv_histogram(image, mask, bins: Sequence[int] = HSV_BINS) -> np.ndarray:
    """Joint HSV histogram over the masked region, normalised to unit mass.

    Section 5.2.3: ``8 x 8 x 8`` bins, accumulated over the pixels inside the cloth-specific mask
    ``M^(1)``. A histogram rather than a pixel-wise difference is deliberate — between adjacent
    views the garment genuinely moves, and a pixel-wise comparison would report legitimate
    parallax as inconsistency.
    """
    bgr = as_bgr(image)
    m = _resize_mask_to(as_mask(mask), bgr.shape[:2])
    if not m.any():
        raise ValueError("empty mask: no pixels to accumulate")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], m, list(bins), [0, 180, 0, 256, 0, 256]
    )
    total = hist.sum()
    if total <= 0:
        raise ValueError("degenerate histogram: zero total mass")
    return (hist / total).astype(np.float32)


def consistency_pair(image_a, mask_a, image_b, mask_b, bins: Sequence[int] = HSV_BINS) -> float:
    """Bhattacharyya distance between two masked frames. Bounded in ``[0, 1]``; lower is better."""
    ha = hsv_histogram(image_a, mask_a, bins)
    hb = hsv_histogram(image_b, mask_b, bins)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def consistency_series(images: Iterable, masks: Iterable, bins: Sequence[int] = HSV_BINS) -> dict:
    """Consistency over an ordered sequence of frames.

    `images` and `masks` must be in orbit order, one mask per frame. Returns the per-adjacent-pair
    distances together with the summary statistics Section 5.4 reports.

    The **maximum** matters as much as the mean: a single discontinuity between two adjacent views
    is what becomes geometry in the reconstruction, and a good mean hides it.
    """
    images = list(images)
    masks = list(masks)
    if len(images) != len(masks):
        raise ValueError(f"{len(images)} images but {len(masks)} masks")
    if len(images) < 2:
        raise ValueError("need at least two frames to measure consistency between views")

    hists, kept, skipped = [], [], []
    for i, (img, msk) in enumerate(zip(images, masks)):
        try:
            hists.append(hsv_histogram(img, msk, bins))
            kept.append(i)
        except ValueError:
            skipped.append(i)  # empty mask: the parser found no garment in this frame

    if len(hists) < 2:
        raise ValueError(f"only {len(hists)} frames had a non-empty mask")

    d = np.array([
        cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_BHATTACHARYYA)
        for i in range(len(hists) - 1)
    ], dtype=np.float64)

    return {
        "distances": d,
        "mean": float(d.mean()),
        "max": float(d.max()),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "argmax_pair": (kept[int(d.argmax())], kept[int(d.argmax()) + 1]),
        "n_frames": len(hists),
        "skipped_frames": skipped,
    }


# ---------------------------------------------------------------------------
# Section 5.2.3 — the detail statistic that guards it
# ---------------------------------------------------------------------------

def gradient_magnitude_mean(image, mask=None) -> float:
    """Mean Sobel gradient magnitude over the masked region of a greyscale view of `image`."""
    bgr = as_bgr(image)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if mask is None:
        return float(mag.mean())
    m = _resize_mask_to(as_mask(mask), mag.shape[:2]) > 0
    if not m.any():
        raise ValueError("empty mask: no pixels to average")
    return float(mag[m].mean())


def _garment_foreground(image, white_threshold: int = 245) -> np.ndarray:
    """Rough foreground of a catalogue photograph: everything that is not near-white backdrop.

    Product photographs are shot on a white sweep, and including that flat background in the
    denominator would depress the reference gradient and inflate every ratio. Falls back to the
    whole frame when the heuristic finds almost nothing, so an on-model or dark-background
    photograph still yields a usable number.
    """
    bgr = as_bgr(image)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg = (grey < white_threshold).astype(np.uint8) * 255
    if fg.mean() < 25:  # under ~10 % of the frame: the heuristic failed
        return np.full(grey.shape, 255, dtype=np.uint8)
    return fg


def detail_statistic(image, mask, garment_image, garment_mask=None) -> float:
    """Garment detail retained, relative to the conditioning photograph.

    Section 5.2.3: the mean gradient magnitude inside ``M^(1)``, normalised by that of the
    conditioning garment photograph. Roughly 1.0 means the rendered garment carries as much
    high-frequency structure as the photograph it was conditioned on; a value falling towards 0
    alongside an improving consistency score is the degenerate case — the garment is being erased,
    not stabilised.

    `garment_mask` defaults to a near-white-backdrop heuristic over the photograph.
    """
    if garment_mask is None:
        garment_mask = _garment_foreground(garment_image)
    reference = gradient_magnitude_mean(garment_image, garment_mask)
    if reference <= 0:
        raise ValueError("conditioning photograph has no gradient to normalise by")
    return gradient_magnitude_mean(image, mask) / reference


# ---------------------------------------------------------------------------
# Section 5.2.4 — mask width against the DensePose torso
# ---------------------------------------------------------------------------

def width_error(mask, densepose, torso_labels: Sequence[int] = TORSO_LABELS) -> dict | None:
    """Relative width error of `mask` against the DensePose torso box (``eq:width-error``).

    For each row ``y`` spanned by the torso, ``W_M(y)`` is the horizontal extent of the mask and
    ``W_B(y)`` that of the torso box. Returns the median and the upper quartile as percentages —
    garment-shape bias is a systematic excess rather than a symmetric error, and the tail is what
    produces the visible artefact. The median is preferred to the mean because a single row
    crossing a sleeve produces an outlier that no amount of correct masking removes.

    Returns ``None`` when DensePose found no torso in the frame, which is a real outcome on
    strongly lateral views rather than an error.
    """
    m = as_mask(mask)
    dp = np.asarray(_open(densepose).convert("L"))
    dp = _resize_mask_to(dp, m.shape[:2]) if dp.shape[:2] != m.shape[:2] else dp

    torso = np.isin(dp, list(torso_labels))
    if not torso.any():
        return None

    rows = np.flatnonzero(torso.any(axis=1))
    cols = np.flatnonzero(torso.any(axis=0))
    box_width = float(cols[-1] - cols[0] + 1)
    if box_width <= 0:
        return None

    errors, n_empty = [], 0
    for y in range(int(rows[0]), int(rows[-1]) + 1):
        set_px = np.flatnonzero(m[y] > 0)
        if set_px.size == 0:
            n_empty += 1
            continue
        w = float(set_px[-1] - set_px[0] + 1)
        errors.append((w - box_width) / box_width)

    if not errors:
        return None
    e = np.asarray(errors, dtype=np.float64)
    return {
        "median_pct": float(np.median(e) * 100.0),
        "q75_pct": float(np.percentile(e, 75) * 100.0),
        "mean_pct": float(e.mean() * 100.0),
        "n_rows": int(e.size),
        "n_rows_empty": int(n_empty),
        "torso_box_width_px": box_width,
    }


def aggregate_width_error(per_frame: Iterable[dict | None]) -> dict:
    """Combine per-frame :func:`width_error` results over an orbit, ignoring frames without a torso."""
    kept = [r for r in per_frame if r]
    if not kept:
        return {"median_pct": float("nan"), "q75_pct": float("nan"), "n_frames": 0}
    med = np.array([r["median_pct"] for r in kept])
    q75 = np.array([r["q75_pct"] for r in kept])
    return {
        "median_pct": float(np.median(med)),
        "q75_pct": float(np.median(q75)),
        "median_spread_pct": float(med.std()),
        "n_frames": len(kept),
    }


# ---------------------------------------------------------------------------
# Section 5.2.5 — perceptual distance between two variants
# ---------------------------------------------------------------------------

def pairwise_lpips_ssim(image_a, image_b, lpips_net: str = "alex") -> dict:
    """LPIPS and SSIM between two variants generating the *same* frame.

    Section 5.2.5 is explicit that this is only meaningful between two variants, never between a
    variant and an absent ground truth: no photograph exists of the subject wearing the target
    garment. Neither image is treated as correct; the number says how far apart they are.

    Both backends are optional. Missing ones come back as ``None`` with the reason recorded.
    """
    out: dict = {"lpips": None, "ssim": None, "notes": []}

    a_bgr, b_bgr = as_bgr(image_a), as_bgr(image_b)
    if a_bgr.shape != b_bgr.shape:
        b_bgr = cv2.resize(b_bgr, (a_bgr.shape[1], a_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        out["notes"].append("second image resized to match the first")

    try:
        from skimage.metrics import structural_similarity

        out["ssim"] = float(structural_similarity(
            cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY),
        ))
    except ImportError:
        out["notes"].append("scikit-image not installed: SSIM skipped")

    try:
        import torch
        import lpips as lpips_lib

        if not hasattr(pairwise_lpips_ssim, "_net"):
            pairwise_lpips_ssim._net = {}
        if lpips_net not in pairwise_lpips_ssim._net:
            pairwise_lpips_ssim._net[lpips_net] = lpips_lib.LPIPS(net=lpips_net)
        net = pairwise_lpips_ssim._net[lpips_net]

        def to_tensor(bgr: np.ndarray) -> "torch.Tensor":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(rgb).permute(2, 0, 1)[None]

        with torch.no_grad():
            out["lpips"] = float(net(to_tensor(a_bgr), to_tensor(b_bgr)).item())
    except ImportError:
        out["notes"].append("lpips or torch not installed: LPIPS skipped")

    return out
