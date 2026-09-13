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
    "aggregate_width_error",
    "view_from_densepose",
    "mask_iou",
    "garment_fidelity",
    "foreground_fraction",
    "garment_region",
    "catalogue_garment_region",
    "colour_shift",
    "outside_mask_change",
    "body_width_shift",
    "aggregate_body_shift",
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


def body_width_shift(
    densepose_generated,
    densepose_original,
    torso_labels: Sequence[int] = TORSO_LABELS,
) -> dict | None:
    """How much the subject's torso changed width between a generated frame and its capture.

    The geometric counterpart to :func:`width_error`. That one asks whether a *mask* is wider than
    the body; this one asks whether the *rendered body* is wider than the one photographed, by
    comparing the DensePose torso of the generated frame against the DensePose torso of the frame
    it was generated from.

    Two questions in the thesis need it and neither of the colour measures can answer them. §4.3.7
    asks whether the refinement pass moves the body inside the composition mask. §4.4.4 asserts
    that unrestricted injection produces "individually distorted" frames — a claim about geometry,
    which a histogram over the garment region cannot detect, since a warped frame can be perfectly
    colour-consistent.

    Rows are compared only where both frames report a torso, so a frame whose parse fails on one
    side contributes nothing rather than a spurious extreme. Returns ``None`` when they share no
    torso rows, which happens on strongly lateral views.
    """
    dp_gen = np.asarray(_open(densepose_generated).convert("L"))
    dp_ref = np.asarray(_open(densepose_original).convert("L"))
    if dp_gen.shape[:2] != dp_ref.shape[:2]:
        dp_gen = _resize_mask_to(dp_gen, dp_ref.shape[:2])

    torso_gen = np.isin(dp_gen, list(torso_labels))
    torso_ref = np.isin(dp_ref, list(torso_labels))
    if not (torso_gen.any() and torso_ref.any()):
        return None

    rows = np.flatnonzero(torso_gen.any(axis=1) & torso_ref.any(axis=1))
    if rows.size == 0:
        return None

    shifts = []
    for y in rows:
        g = np.flatnonzero(torso_gen[y])
        r = np.flatnonzero(torso_ref[y])
        w_ref = float(r[-1] - r[0] + 1)
        if w_ref <= 0:
            continue
        shifts.append((float(g[-1] - g[0] + 1) - w_ref) / w_ref)

    if not shifts:
        return None
    s = np.asarray(shifts, dtype=np.float64)
    return {
        "median_pct": float(np.median(s) * 100.0),
        "q75_pct": float(np.percentile(s, 75) * 100.0),
        "abs_median_pct": float(np.median(np.abs(s)) * 100.0),
        "n_rows": int(s.size),
    }


def aggregate_body_shift(per_frame: Iterable[dict | None]) -> dict:
    """Combine :func:`body_width_shift` over an orbit, ignoring frames without a shared torso.

    ``abs_median_pct`` is the one to read for distortion: a body that is too wide on some frames
    and too narrow on others averages to nothing under the signed median while being badly wrong
    on every frame.
    """
    kept = [r for r in per_frame if r]
    if not kept:
        return {"median_pct": float("nan"), "abs_median_pct": float("nan"), "n_frames": 0}
    return {
        "median_pct": float(np.median([r["median_pct"] for r in kept])),
        "q75_pct": float(np.median([r["q75_pct"] for r in kept])),
        "abs_median_pct": float(np.median([r["abs_median_pct"] for r in kept])),
        "worst_abs_pct": float(np.max([r["abs_median_pct"] for r in kept])),
        "n_frames": len(kept),
    }


def garment_fidelity(image, mask, garment_image, garment_mask=None, bins=HSV_BINS) -> float:
    """Colour distance between the rendered garment and the photograph it was conditioned on.

    The instrument §4.3.6's argument needs and the width error cannot supply. That section adopted
    body-preserving composition because running the refinement pass on the *original* frame lets
    the garment being replaced bleed into the generated one — a navy target desaturating toward the
    cream shirt underneath. A mask-width measure is blind to that, and the detail statistic is
    actively misleading, since bleed-through raises gradient energy exactly as retained texture does.

    Bhattacharyya distance between the masked HSV histogram of the frame and that of the
    conditioning photograph. Lower means the rendered garment's colours sit closer to the garment
    it is supposed to be. It never reaches zero — lighting, pose and drape all differ from a
    catalogue shot — so it is comparative between variants, like every other measure here.
    """
    if garment_mask is None:
        garment_mask = _garment_foreground(garment_image)
    return float(cv2.compareHist(
        hsv_histogram(image, mask, bins),
        hsv_histogram(garment_image, garment_mask, bins),
        cv2.HISTCMP_BHATTACHARYYA,
    ))


def foreground_fraction(image, white_threshold: int = 245) -> float:
    """Share of a catalogue photograph that the near-white backdrop heuristic keeps as garment.

    The heuristic behind the ``garment_mask`` default of :func:`detail_statistic` and
    :func:`garment_fidelity` assumes a white sweep. On a grey studio wall, an indoor room or a
    concrete backdrop it keeps close to the whole frame, and a region that is the whole photograph
    no longer excludes the backdrop that ``eq:fidelity`` says it excludes. A value near 1.0 is the
    signal to pass a parsed garment region instead (:func:`garment_region`).
    """
    grey = cv2.cvtColor(as_bgr(image), cv2.COLOR_BGR2GRAY)
    return float((grey < white_threshold).mean())


def garment_region(garment_image, parse_mask=None, min_fraction: float = 0.02,
                   return_source: bool = False):
    """The garment's pixels in its catalogue photograph, as an ``HxW`` mask in {0, 255}.

    Prefers `parse_mask`, the upper-clothes parse of the photograph, which excludes the model's
    skin, hair and trousers as well as the backdrop. Falls back to the near-white backdrop
    heuristic when no parse is given or the parser found almost nothing, which is what happens on
    a flat-lay packshot with no person in it — the case the heuristic was built for.

    The choice matters as soon as garments are compared with each other. The heuristic keeps skin
    and trousers on an on-model photograph and keeps everything on a non-white backdrop, so the
    region it yields differs in kind from garment to garment, and a colour measure compared across
    garments would compare those differences rather than the rendering.

    With ``return_source`` returns ``(mask, "parsed" | "backdrop heuristic")``.
    """
    if parse_mask is not None:
        shape = as_bgr(garment_image).shape[:2]
        m = _resize_mask_to(as_mask(parse_mask), shape)
        if (m > 0).mean() >= min_fraction:
            return (m, "parsed") if return_source else m
    m = _garment_foreground(garment_image)
    return (m, "backdrop heuristic") if return_source else m


def colour_shift(image, mask, garment_image, garment_mask) -> dict:
    """How far the rendered garment's value and saturation sit from its photograph's.

    The direct reading of the washout of §6.2: a garment rendered lighter and less saturated than
    the one requested. Medians of the HSV value and saturation channels (OpenCV's 0-255 scale)
    inside the frame's garment mask and inside the photograph's garment region; ``d_value`` and
    ``d_saturation`` are frame minus photograph, so washout shows as ``d_value > 0`` together with
    ``d_saturation < 0``.

    It complements :func:`garment_fidelity` rather than replacing it. Fidelity is a histogram
    distance, which saturates once a flat colour crosses a bin boundary and does not say which way
    the colour moved; these two numbers keep the direction and are not quantised. Hue is left out,
    being undefined for the near-black and near-grey pixels a dark garment consists of. Capture
    lighting differs from catalogue lighting, so neither number sits at zero for a good result:
    they are read between orientation classes of the same garment.

    Medians describe a shift of the whole garment and cannot see a patch: a washed-out region over a
    third of the back leaves them where they were. ``lighter_share`` and ``greyer_share`` cover that
    case, as the share of the rendered garment lighter than the lightest 5 % of the photograph's
    garment and greyer than its greyest 5 %. The photograph's own extremes include its print, so a
    pixel counted there lies outside every colour the garment actually has.

    `garment_mask` has no default. The backdrop heuristic is wrong for most on-model photographs
    (:func:`garment_region`), and a colour statistic over skin, trousers and wall would describe
    those instead of the garment.
    """
    def channels(img, msk):
        bgr = as_bgr(img)
        m = _resize_mask_to(as_mask(msk), bgr.shape[:2]) > 0
        if not m.any():
            raise ValueError("empty mask: no pixels to take a median over")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return hsv[..., 2][m].astype(np.float64), hsv[..., 1][m].astype(np.float64)

    v, s = channels(image, mask)
    v_ref, s_ref = channels(garment_image, garment_mask)
    out = {"value": float(np.median(v)), "saturation": float(np.median(s)),
           "ref_value": float(np.median(v_ref)), "ref_saturation": float(np.median(s_ref))}
    out["d_value"] = out["value"] - out["ref_value"]
    out["d_saturation"] = out["saturation"] - out["ref_saturation"]
    out["lighter_share"] = float((v > np.percentile(v_ref, 95)).mean())
    out["greyer_share"] = float((s < np.percentile(s_ref, 5)).mean())
    return out


LIP_UPPER_CLOTHES = 5   # SCHP, LIP label set
ATR_UPPER_CLOTHES = 4   # SCHP, ATR label set


def _label_map(labels, shape: tuple[int, int]) -> np.ndarray:
    """A parser's label map as uint8 indices. No mode conversion: a palette image holds indices."""
    arr = np.asarray(_open(labels))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return _resize_mask_to(arr.astype(np.uint8), shape)


def catalogue_garment_region(photo, densepose, schp_lip, schp_atr, min_fraction: float = 0.02,
                             min_person: float = 0.01) -> tuple[np.ndarray, str]:
    """The garment region of a catalogue photograph, from the parses the automasker produces anyway.

    Decided in order:

    * **No person** (DensePose covers under ``min_person`` of the frame): a flat-lay packshot. The
      human parsers are outside their domain there and label part of the garment at best, while
      the near-white backdrop heuristic is exactly right on a packshot's white sweep.
    * **Both parsers agree** on upper-clothes over at least ``min_fraction`` of the frame: their
      intersection. Taking the larger of the two instead systematically picks whichever parser
      over-segments, and on an on-model photograph that means bare arms labelled as sleeve, which
      makes a colour statistic over the region partly a statistic of skin.
    * Otherwise the larger single parse, and failing that the backdrop heuristic.

    Returns ``(mask, source)``, the mask in {0, 255} at the photograph's resolution.
    """
    shape = as_bgr(photo).shape[:2]
    if (_label_map(densepose, shape) > 0).mean() < min_person:
        return _garment_foreground(photo), "no person: backdrop heuristic"
    lip = _label_map(schp_lip, shape) == LIP_UPPER_CLOTHES
    atr = _label_map(schp_atr, shape) == ATR_UPPER_CLOTHES
    both = lip & atr
    if both.mean() >= min_fraction:
        return both.astype(np.uint8) * 255, "parsed: LIP and ATR agree"
    single = lip if lip.sum() >= atr.sum() else atr
    if single.mean() >= min_fraction:
        return single.astype(np.uint8) * 255, "parsed: one parser only"
    return _garment_foreground(photo), "backdrop heuristic"


def outside_mask_change(image, reference, mask, margin: int = 15) -> dict:
    """What a generated frame changed outside the region it was asked to repaint.

    §6.2 places the cost of raised guidance on the whole canvas: the autoencoder decodes every pixel,
    so overshoot appears as grain and saturation in the face and background as well as the garment.
    ``two_phase_tryon`` stores the pipeline's output as the delivered frame without pasting it back,
    so the delivered frame carries any such change. Measured against the pass's own input, beyond
    ``margin`` px of the mask so the feathered seam is not counted:

    * ``abs_diff``      mean absolute RGB difference, 0-255
    * ``grain_ratio``   mean gradient magnitude of the output over that of the input; 1.0 = unchanged
    * ``d_saturation``  median HSV saturation of the output minus that of the input
    """
    a_bgr, b_bgr = as_bgr(image), as_bgr(reference)
    if a_bgr.shape != b_bgr.shape:
        raise ValueError(f"image {a_bgr.shape} and reference {b_bgr.shape} differ in size")
    m = _resize_mask_to(as_mask(mask), a_bgr.shape[:2])
    k = 2 * margin + 1
    outside = cv2.dilate(m, np.ones((k, k), np.uint8)) == 0
    if not outside.any():
        raise ValueError("no pixels outside the mask and its margin")
    region = outside.astype(np.uint8) * 255
    diff = np.abs(a_bgr.astype(np.float32) - b_bgr.astype(np.float32))[outside].mean()
    g_out, g_in = gradient_magnitude_mean(image, region), gradient_magnitude_mean(reference, region)
    if g_in > 1e-6:
        grain = g_out / g_in
    else:
        # A flat input region has no gradient to normalise by: flat in both is unchanged, and any
        # structure added to it is unbounded relative to none.
        grain = 1.0 if g_out <= 1e-6 else float("inf")
    s_a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)[..., 1][outside]
    s_b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)[..., 1][outside]
    return {"abs_diff": float(diff), "grain_ratio": float(grain),
            "d_saturation": float(np.median(s_a)) - float(np.median(s_b))}


def mask_iou(mask_a, mask_b) -> float:
    """Intersection over union of two masks.

    The instrument for the $T_1$ sweep of Section 4.3.2. The cloth-specific mask is what the coarse
    pass hands to the refinement, so its agreement across budgets isolates the shape half of that
    trade-off; the grid showed the budget also moves garment detail, which this cannot see.
    """
    a = as_mask(mask_a) > 0
    b = _resize_mask_to(as_mask(mask_b), a.shape[:2]) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def view_from_densepose(
    densepose,
    front_label: int = 2,
    back_label: int = 1,
    dominance: float = 2.0,
) -> str:
    """Orientation class of a frame from its DensePose part map (``eq:view-label``).

    The same 2:1 dominance test the pipeline uses at generation time: front torso against back
    torso, and *side* when neither dominates. Recomputing it from the saved part map means the
    orientation classes can be recovered from a run directory alone, without re-running the
    parser or keeping the notebook's ``body_params`` alive.
    """
    dp = np.asarray(_open(densepose).convert("L"))
    front = int(np.sum(dp == front_label))
    back = int(np.sum(dp == back_label))
    if front + back == 0:
        return "unknown"
    if front > back * dominance:
        return "front"
    if back > front * dominance:
        return "back"
    return "side"


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
