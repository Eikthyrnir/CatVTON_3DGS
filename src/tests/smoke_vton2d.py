"""End-to-end smoke test of vton2d with a fake try-on function. No GPU, no CatVTON.

Exercises the exact contract run_orbit expects of the notebook's two_phase_tryon, then scores the
run the way the notebook's cells do. Run from anywhere:  python src/tests/smoke_vton2d.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vton2d import RunConfig, RunWriter, run_orbit, score_run, load_run, list_runs, export_stage

W, H = 96, 128
rng = np.random.default_rng(7)


def make_person(i):
    a = np.full((H, W, 3), 40, np.uint8)
    a[30:90, 25:70] = rng.integers(0, 255, (60, 45, 3), dtype=np.uint8)
    return Image.fromarray(a)


_call = {"n": 0}
# Generation order is side x3, front x4 (ref + 3), back x4 (ref + 3): mirror it so the fake
# DensePose maps carry real orientation classes and the boundary split has something to find.
_ORDER = ["side"] * 3 + ["front"] * 4 + ["back"] * 4


def fake_tryon(person_img, garment_img, is_ref_pass=True, coarse_steps=50, fine_steps=50,
               seed=42, mask_dilate=14, guidance_scale=2.5, mask_variant="composition"):
    """Mimics the notebook's two_phase_tryon contract exactly."""
    view = _ORDER[min(_call["n"], len(_ORDER) - 1)]
    _call["n"] += 1
    agnostic = np.zeros((H, W), np.uint8); agnostic[25:95, 18:78] = 255
    cloth = np.zeros((H, W), np.uint8);    cloth[30:90, 25:70] = 255
    comp = np.zeros((H, W), np.uint8);     comp[28:92, 22:73] = 255
    dp = np.zeros((H, W), np.uint8)
    if view == "front":
        dp[30:90, 28:66] = 2                      # front torso dominates
    elif view == "back":
        dp[30:90, 28:66] = 1                      # back torso dominates
    else:
        dp[30:60, 28:66] = 1; dp[60:90, 28:66] = 2   # neither dominates -> side
    out = {
        "agnostic": Image.fromarray(agnostic),
        "densepose": Image.fromarray(dp),
        "coarse": person_img,
        "cloth_mask": Image.fromarray(cloth),
        "comp_mask": None if mask_variant == "single_pass" else Image.fromarray(comp),
        "composite": None if mask_variant == "single_pass" else person_img,
        "final": person_img,
        "parse_steps": [Image.fromarray(cloth), Image.fromarray(comp)],
    }
    return out


root = Path(tempfile.mkdtemp(prefix="vton2d_smoke_"))
frames = {v: [f"{v}_{i:03d}.jpg" for i in range(3)] for v in ("front", "back", "side")}
garments = {v: make_person(0) for v in ("front", "back", "side")}
references = {"front": "front_ref.jpg", "back": "back_ref.jpg"}
# The photograph each frame is conditioned on: the class is the prefix of the frame's name.
garment_by_name = lambda n: garments[Path(n).stem.split("_")[0]]

# run_orbit opens person paths, so give it real files
work = root / "in"; work.mkdir()
for v, names in frames.items():
    for n in names:
        make_person(0).save(work / n)
for n in references.values():
    make_person(0).save(work / n)
frames = {v: [str(work / n) for n in names] for v, names in frames.items()}
references = {k: str(work / v) for k, v in references.items()}

cfg = RunConfig(run_id="smoke", subject="s", garment="g", resolution=(W, H))
writer = RunWriter(root / "runs", cfg)
summary = run_orbit(frames, garments, references, fake_tryon, cfg, writer, parse_steps_for=1)
print("\nframes generated:", len(summary["frames"]), summary["per_view"])

run = load_run(writer.root)
print("manifest config keys:", sorted(run["config"])[:6], "...")
print("commit recorded:", run["manifest"]["commit"][:12])

res = score_run(writer.root, garment_for=garment_by_name)

# --- orientation backfill on a run whose manifest predates view_of -------------------
from vton2d import backfill_view_of, infer_view_of, report_run, update_manifest
import json as _json

man_path = writer.root / "manifest.json"
man = _json.loads(man_path.read_text())
man["extra"].pop("view_of", None)          # simulate an older run
man_path.write_text(_json.dumps(man, indent=2))
print("\nsimulated an old manifest; view_of present:", "view_of" in man["extra"])

views = backfill_view_of(writer.root)
man = _json.loads(man_path.read_text())
assert "view_of" in man["extra"], "backfill did not write view_of"
assert man["amended_utc"], "amendment not recorded"
print("inferred classes:", sorted(set(views.values())))

print()
report_run(writer.root, garment_for=garment_by_name)

n = export_stage(writer.root, root / "flat", stage="final")
print(f"\nexported {n} finals")
print("runs listed:", [r["run_id"] for r in list_runs(root / "runs")])

# the guard: a second writer on the same run_id must refuse
try:
    RunWriter(root / "runs", cfg)
    print("FAIL: overwrite guard did not fire")
except FileExistsError:
    print("ok: overwrite guard fired")

# sweep ergonomics
sweep = [cfg.variant_of(f"smoke__T1_{t}", coarse_steps=t) for t in (10, 20, 30)]
print("sweep ids:", [c.run_id for c in sweep], "steps:", [c.coarse_steps for c in sweep])
# --- the T1 x T2 grid: several runs tabulated against a baseline --------------------
from vton2d import compare_runs

grid = []
for t1 in (10, 30, 50):
    for t2 in (30, 50):
        gc = cfg.variant_of(f"grid_T1_{t1}_T2_{t2}", coarse_steps=t1, fine_steps=t2)
        _call["n"] = 0                      # replay the same view order for each config
        gw = RunWriter(root / "runs", gc)
        run_orbit(frames, garments, references, fake_tryon, gc, gw,
                  parse_steps_for=0, verbose=False)
        grid.append(gw.root)

print("\n--- compare_runs (grid vs the 11-frame reference corpus) ---------")
rows = compare_runs(grid + [writer.root], baseline=writer.root,
                    garment_for=garment_by_name)
assert len(rows) == 7, rows
ns = {r["n"] for r in rows if "n" in r}
assert len(ns) == 1, f"rows scored over different frame counts: {ns}"
assert rows[0]["coarse_steps"] == 10 and rows[0]["fine_steps"] == 30
assert "mask_iou_vs_base" in rows[0], "baseline comparison missing"

# --- ensure_orbit: idempotent re-runs and partial resume ---------------------------
from vton2d import ensure_orbit

print("\n--- ensure_orbit: first call on a fresh run_id -------------------")
ec = cfg.variant_of("ensure_demo")
_call["n"] = 0
s1 = ensure_orbit(frames, garments, references, fake_tryon, ec, root / "runs",
                  parse_steps_for=0, verbose=False)
print(f"generated {s1['generated']}, reused {s1['reused']}")
assert s1["generated"] == 11 and s1["reused"] == 0, s1

print("\n--- ensure_orbit: second call, nothing to do --------------------")
_call["n"] = 0
s2 = ensure_orbit(frames, garments, references, fake_tryon, ec, root / "runs",
                  parse_steps_for=0)
assert s2["generated"] == 0 and s2["reused"] == 11, s2

print("\n--- ensure_orbit: partial resume after losing 2 frames ----------")
run_root = root / "runs" / "ensure_demo"
lost = [p.name for p in sorted((run_root / "final").glob("front_00*.png"))[:2]]
for n in lost:
    (run_root / "final" / n).unlink()
print("deleted:", lost)
_call["n"] = 0
s3 = ensure_orbit(frames, garments, references, fake_tryon, ec, root / "runs",
                  parse_steps_for=0, verbose=True)
print(f"generated {s3['generated']}, reused {s3['reused']}")
# 3, not 2: the two lost targets plus the front reference, which has to be regenerated because
# its pass is what fills the key/value bank those targets read from.
assert s3["generated"] == 3, s3
assert all((run_root / "final" / n).exists() for n in lost), "lost frames were not restored"
assert len(list((run_root / "final").glob("*.png"))) == 11, "frame count changed"
print("both restored, run intact")

# --- configure_fn: injection / window must reach the model, or refuse ---------------
print("\n--- configure_fn contract ---------------------------------------")
seen = []
def configure_from(cfg_):
    seen.append((cfg_.injection, tuple(cfg_.window)))

for mode, window in (("all", (5, 45)), ("none", (5, 45)), ("decoder", (0, 50))):
    c = cfg.variant_of(f"cfg_{mode}_{window[0]}_{window[1]}", injection=mode, window=window)
    _call["n"] = 0
    s = ensure_orbit(frames, garments, references, fake_tryon, c, root / "runs",
                     parse_steps_for=0, configure_fn=configure_from, verbose=False)
    print(f"  injection={mode:<8} window={window}  generated {s['generated']}")
assert seen == [("all", (5, 45)), ("none", (5, 45)), ("decoder", (0, 50))], seen
print("  configure_fn received every configuration verbatim")

for mode, window in (("all", (5, 45)), ("decoder", (3, 40))):
    c = cfg.variant_of(f"refuse_{mode}_{window[0]}", injection=mode, window=window)
    try:
        ensure_orbit(frames, garments, references, fake_tryon, c, root / "runs",
                     parse_steps_for=0, verbose=False)
        raise AssertionError(f"{mode}/{window} was accepted without configure_fn")
    except ValueError as exc:
        assert "configure_fn" in str(exc), exc
print("  non-released configs are refused when configure_fn is absent")

c = cfg.variant_of("released_no_hook")
_call["n"] = 0
ensure_orbit(frames, garments, references, fake_tryon, c, root / "runs",
             parse_steps_for=0, verbose=False)
print("  the released config still runs without a hook")

# --- regression: an interrupted attempt, then a resume --------------------------------
# Reproduces the E7 bug: side frames written, no manifest (the attempt died), resume reuses them.
# Before the fix the new manifest listed only the frames generated in the second attempt.
print("\n--- interrupted attempt, then resume ------------------------------")
from vton2d import load_run, repair_frames
import json as _json

ic = cfg.variant_of("interrupted", window=(3, 45))
iw = RunWriter(root / "runs", ic)
_call["n"] = 0
for path in frames["side"]:
    iw.save_frame(Path(path).name, fake_tryon(make_person(0), garments["side"]))
assert not (iw.root / "manifest.json").exists(), "simulated interruption must leave no manifest"
print(f"  interrupted: {len(list((iw.root / 'final').glob('*.png')))} finals on disk, no manifest")

_call["n"] = 3                                  # next calls map to front, then back
s = ensure_orbit(frames, garments, references, fake_tryon, ic, root / "runs",
                 parse_steps_for=0, configure_fn=configure_from, verbose=False)
listed = _json.loads((iw.root / "manifest.json").read_text())["frames"]
print(f"  resumed: generated {s['generated']}, manifest lists {len(listed)} frames")
assert len(listed) == 11, f"manifest lists {len(listed)} of 11 frames — reused frames dropped"
assert load_run(iw.root)["unlisted_frames"] == [], "load_run still finds unlisted frames"
print("  every reused frame is in the manifest")

# --- load_run reconciles a truncated manifest; repair_frames fixes the file -------------
mp = iw.root / "manifest.json"
m = _json.loads(mp.read_text())
m["frames"] = [f for f in m["frames"] if not f.startswith("side_")]
mp.write_text(_json.dumps(m, indent=2))
r = load_run(iw.root)
assert len(r["frames"]) == 11, f"load_run returned {len(r['frames'])} frames from a truncated manifest"
assert len(r["unlisted_frames"]) == 3, r["unlisted_frames"]
print(f"  truncated manifest: load_run still sees {len(r['frames'])} frames, flags {len(r['unlisted_frames'])} unlisted")
added = repair_frames(iw.root, verbose=False)
assert len(added) == 3 and len(_json.loads(mp.read_text())["frames"]) == 11
assert repair_frames(iw.root, verbose=False) == [], "repair must be idempotent"
print("  repair_frames restored the file, and is idempotent")

# --- the table accepts a list-valued axis (JSON turns the window tuple into a list) -----
print()
rows = compare_runs([root / "runs" / "cfg_all_5_45", root / "runs" / "cfg_decoder_0_50", iw.root],
                    axes=("injection", "window"), garment_for=garment_by_name)
assert len(rows) == 3 and all("error" not in r for r in rows), rows
print("  list-valued window axis renders")

# --- garment regions, colour shift and colour by class (E2) ------------------------------
print("\n--- garment regions, colour_shift, colour_by_class ----------------")
from vton2d import (colour_shift, garment_region, foreground_fraction, ensure_garment_masks,
                    garment_lookup, colour_by_class)

# A navy garment photographed on a grey wall: the backdrop heuristic keeps the whole frame.
photo = np.full((H, W, 3), 200, np.uint8)
photo[30:90, 25:70] = (20, 30, 90)
assert foreground_fraction(photo) > 0.99, "a grey backdrop should defeat the heuristic"
parse = np.zeros((H, W), np.uint8); parse[30:90, 25:70] = 255
m, src = garment_region(photo, parse, return_source=True)
assert src == "parsed" and int((m > 0).sum()) == 60 * 45, (src, int((m > 0).sum()))
_, src = garment_region(photo, np.zeros((H, W), np.uint8), return_source=True)
assert src == "backdrop heuristic", src
print("  parse preferred; an empty parse falls back to the heuristic")

# Washed out: the frame renders the navy lighter and greyer than the photograph.
washed = photo.copy(); washed[30:90, 25:70] = (70, 75, 110)
cs = colour_shift(washed, parse, photo, parse)
assert cs["d_value"] > 0 and cs["d_saturation"] < 0, cs
assert cs["lighter_share"] == 1.0 and cs["greyer_share"] == 1.0, cs
same = colour_shift(photo, parse, photo, parse)
assert same["d_value"] == 0 and same["d_saturation"] == 0, same
assert same["lighter_share"] == 0.0 and same["greyer_share"] == 0.0, same
# A washed-out patch over a third of the garment: medians unmoved, the shares see it.
patch = photo.copy(); patch[30:50, 25:70] = (200, 205, 215)
ps = colour_shift(patch, parse, photo, parse)
assert ps["d_value"] == 0 and 0.30 < ps["lighter_share"] < 0.37, ps
# A near-black garment has no saturation to measure; its value still reads, and a grey render is washed.
black = np.full((H, W, 3), 200, np.uint8); black[30:90, 25:70] = (22, 20, 26)
grey = black.copy(); grey[30:90, 25:70] = (150, 150, 152)
bs = colour_shift(grey, parse, black, parse)
assert bs["d_value"] > 64 and bs["lighter_share"] == 1.0, bs
assert all(bs[k] != bs[k] for k in ("saturation", "ref_saturation", "d_saturation", "greyer_share")), bs
try:
    colour_shift(photo, np.zeros((H, W), np.uint8), photo, parse)
    raise AssertionError("an empty frame mask was accepted")
except ValueError:
    pass
print(f"  synthetic washout reads as dV {cs['d_value']:+.0f}, dS {cs['d_saturation']:+.0f}")

# Regions stored with a run, then read back for scoring.
try:
    garment_lookup(writer.root, require_masks=True)
    raise AssertionError("missing garment regions were not refused")
except FileNotFoundError:
    pass
got = ensure_garment_masks(writer.root, lambda img: parse, verbose=False)
assert set(got.values()) == {"parsed"}, got
again = ensure_garment_masks(writer.root, lambda img: parse, verbose=False)
assert set(again.values()) == {"present"}, again
gfor, mfor = garment_lookup(writer.root, require_masks=True)
assert Path(gfor("front_000")).name == "front.png" and Path(mfor("back_001")).name == "back_mask.png"
cb = colour_by_class(writer.root)
assert set(cb["by_class"]) == {"front", "side", "back"}, cb["by_class"].keys()
assert all(isinstance(c["n_washed"], int) for c in cb["by_class"].values()), cb["by_class"]
assert cb["dorsal_minus_frontal"] is not None
masked = score_run(writer.root, garment_for=gfor, garment_mask_for=mfor, verbose=False)
assert masked["garment_region"] == "stored garment region", masked["garment_region"]
assert score_run(writer.root, garment_for=gfor, verbose=False)["garment_region"] == "backdrop heuristic"
rows = compare_runs([writer.root], garment_for=gfor, garment_mask_for=mfor, verbose=False)
assert rows[0].get("fidelity_mean") is not None, rows
report_run(writer.root, garment_for=gfor, save=False, garment_mask_for=mfor)
print("  regions stored, looked up from the run, and threaded through every scorer")

from vton2d import GarmentMismatchError
for bad in (lambda n: garments["front"], lambda n: make_person(0)):
    try:
        score_run(writer.root, garment_for=bad, verbose=False)
        raise AssertionError("a run was scored against a photograph it was not conditioned on")
    except GarmentMismatchError:
        pass
try:
    compare_runs([writer.root], garment_for=lambda n: garments["side"], verbose=False)
    raise AssertionError("compare_runs swallowed a garment mismatch")
except GarmentMismatchError:
    pass
print("  scoring against another class's or another garment's photograph is refused")

coarse = colour_by_class(writer.root, stage="coarse", verbose=False)
assert coarse["stage"] == "coarse" and coarse["by_class"], coarse
refreshed = ensure_garment_masks(writer.root, lambda img: (parse, "custom rule"), refresh=True,
                                 verbose=False)
assert set(refreshed.values()) == {"custom rule"}, refreshed
print("  coarse stage scores; refresh rewrites regions from a (mask, source) parse")

# --- catalogue_garment_region: packshot, agreeing parsers, palette label maps ------------------
from vton2d import catalogue_garment_region

def palette(arr):
    """A mode-P image whose pixel values are the label indices, as SCHP returns them."""
    img = Image.new("P", (arr.shape[1], arr.shape[0]))
    img.putdata(arr.ravel().tolist())
    return img

packshot = np.full((H, W, 3), 250, np.uint8); packshot[30:90, 25:70] = (20, 30, 90)
empty = np.zeros((H, W), np.uint8)
m, src = catalogue_garment_region(packshot, empty, empty, empty)
assert src == "no person: backdrop heuristic" and int((m > 0).sum()) == 60 * 45, (src, int((m > 0).sum()))

person = np.zeros((H, W), np.uint8); person[20:110, 10:85] = 2
lip = np.zeros((H, W), np.uint8); lip[30:90, 25:70] = 5; lip[30:60, 10:25] = 5   # arm read as sleeve
atr = np.zeros((H, W), np.uint8); atr[30:90, 25:70] = 4; atr[30:60, 10:25] = 14  # arm read as arm
headless = lip.copy()
lip[5:20, 35:60] = 2                                                               # hair: a person
# DensePose finding a body in a flat-lay is not enough; with no hat, hair or face it is a packshot.
m, src = catalogue_garment_region(packshot, person, palette(headless), palette(atr))
assert src == "no person: backdrop heuristic" and int((m > 0).sum()) == 60 * 45, (src, int((m > 0).sum()))
m, src = catalogue_garment_region(photo, person, palette(lip), palette(atr))
assert src == "parsed: LIP and ATR agree" and int((m > 0).sum()) == 60 * 45, (src, int((m > 0).sum()))
m, src = catalogue_garment_region(photo, person, palette(lip), palette(empty))
assert src == "parsed: one parser only" and int((m > 0).sum()) == 60 * 45 + 30 * 15, src
print("  packshot -> heuristic; agreement drops the arm one parser called sleeve; palette maps read as indices")

# --- ref_guidance_views reaches the model -------------------------------------------------------
seen_guidance = []
def recording_tryon(person_img, garment_img, **kw):
    view = next(v for v, g in garments.items() if g is garment_img)
    seen_guidance.append((view, kw["is_ref_pass"], kw["guidance_scale"]))
    return fake_tryon(person_img, garment_img, **kw)

for tag, views_raised in (("default", ("front",)), ("back_raised", ("front", "back")), ("none", ())):
    seen_guidance.clear()
    gc = cfg.variant_of(f"guidance_{tag}", ref_guidance_views=views_raised)
    run_orbit(frames, garments, references, recording_tryon, gc, RunWriter(root / "runs", gc),
              parse_steps_for=0, verbose=False)
    refs = {(v, g) for v, is_ref, g in seen_guidance if is_ref and v != "side"}
    expected = {(v, 5.0 if v in views_raised else 2.5) for v in ("front", "back")}
    assert refs == expected, (tag, refs)
    assert all(g == 2.5 for v, is_ref, g in seen_guidance if not is_ref), (tag, seen_guidance)
print("  reference guidance follows ref_guidance_views; targets stay at guidance_scale")

# A manifest written before the field existed reads as the default in a table.
mp = root / "runs" / "guidance_none" / "manifest.json"
m = _json.loads(mp.read_text())
m["config"].pop("ref_guidance_views")
mp.write_text(_json.dumps(m, indent=2))
rows = compare_runs([mp.parent], axes=("ref_guidance_views",), verbose=False)
assert rows[0]["ref_guidance_views"] == ("front",), rows[0]
print("  an old manifest tabulates with the default reference guidance")

# --- outside_mask_change: what the pass did beyond the region it repainted ----------------------
from PIL import ImageFilter
from vton2d import outside_mask_change, outside_change_by_class

scene = Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(2))
region = np.zeros((H, W), np.uint8); region[40:80, 30:60] = 255
unchanged = outside_mask_change(scene, scene, region, margin=5)
assert unchanged["abs_diff"] == 0 and abs(unchanged["grain_ratio"] - 1) < 1e-6, unchanged
grainy = np.clip(np.asarray(scene).astype(int) + rng.integers(-40, 41, (H, W, 3)), 0, 255).astype(np.uint8)
changed = outside_mask_change(Image.fromarray(grainy), scene, region, margin=5)
assert changed["grain_ratio"] > 1.2 and changed["abs_diff"] > 5, changed
oc = outside_change_by_class(writer.root, verbose=False)
assert set(oc["by_class"]) == {"front", "side", "back"}, oc["by_class"]
assert all(abs(c["grain_ratio"] - 1) < 1e-6 for c in oc["by_class"].values()), oc["by_class"]
print(f"  grain outside the mask: unchanged 1.000, added noise {changed['grain_ratio']:.2f}")

# --- composition after refinement, seams, and full-orbit summaries ---------------------------
print("\n--- recomposition, boundary seam, orbit summaries ----------------")
from vton2d import boundary_seam, parse_finals, recompose_run, recomposition_report, orbit_summary

def alpha_paste(capture, tryon, mask):
    a = (np.asarray(mask.convert("L"), np.float32) / 255.0)[..., None]
    out = np.asarray(tryon.convert("RGB"), np.float32) * a + np.asarray(capture.convert("RGB"), np.float32) * (1 - a)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

assert abs(boundary_seam(scene, scene, region) - 1.0) < 1e-6
stepped = np.asarray(scene).astype(int); stepped[40:80, 30:60] += 60
stepped = Image.fromarray(stepped.clip(0, 255).astype(np.uint8))
assert boundary_seam(stepped, scene, region) > 1.5, boundary_seam(stepped, scene, region)
print(f"  seam: unchanged 1.000, hard step pasted in {boundary_seam(stepped, scene, region):.2f}")

capture_for = lambda frame: Image.open(work / f"{frame}.jpg").convert("RGB")
assert recompose_run(writer.root, alpha_paste, capture_for=capture_for, verbose=False) == 11
assert recompose_run(writer.root, alpha_paste, capture_for=capture_for, verbose=False) == 0
torso = np.zeros((H, W), np.uint8); torso[30:90, 28:66] = 2
parse_finals(writer.root, lambda img: Image.fromarray(torso), verbose=False)
parse_finals(writer.root, lambda img: Image.fromarray(torso), source="recomposed",
             stage="recomposed_densepose", verbose=False)
assert len(list((writer.root / "recomposed_densepose").glob("*.png"))) == 11
rec = recomposition_report(writer.root)
assert rec["overall"]["n"] == 11 and set(rec["by_class"]) == {"front", "side", "back"}, rec["overall"]
assert rec["overall"]["body_final"] == 0.0 and rec["overall"]["body_recomposed"] == 0.0, rec["overall"]

summary = orbit_summary(writer.root)
assert summary["n_frames"] == 11 and set(summary["body"]) == {"front", "side", "back"}, summary
print("  recomposition written and parsed; per-class seam and body report; orbit summary")

print("\n--- consistency over adjacent frames only -------------------------")
from vton2d.driver import _adjacent_only, _boundary_split

# An arc run: frames 0019-0022 plus two references sorted in at either end, one empty-mask frame.
names = ["0011", "0019", "0020", "0021", "0022", "0041"]
series = {"distances": np.array([0.9, 0.1, 0.8, 0.95]), "skipped_frames": [3],
          "mean": 0.0, "max": 0.0, "median": 0.0, "p90": 0.0, "argmax_pair": (0, 1), "n_frames": 5}
adj = _adjacent_only(series, names)
assert adj["pairs"] == [(1, 2)], adj["pairs"]               # 0020 -> 0022 bridges the empty frame
assert adj["dropped_pairs"] == [("0011", "0019"), ("0020", "0022"), ("0022", "0041")], adj["dropped_pairs"]
assert adj["n_pairs"] == 1 and abs(adj["mean"] - 0.1) < 1e-12 and adj["argmax_pair"] == (1, 2)
split = _boundary_split(adj["distances"], adj["pairs"], names, {"0019": "front", "0020": "side"})
assert split["boundary_pairs"] == [("0019", "0020", 0.1)] and split["interior"] is None, split
full = _adjacent_only({**series, "distances": np.array([0.9, 0.1, 0.2, 0.8, 0.95]),
                       "skipped_frames": []}, [f"{i:04d}" for i in range(6)])
assert full["n_pairs"] == 5 and not full["dropped_pairs"]  # a contiguous orbit loses nothing
print("  reference and gap pairs left out; boundary split follows the kept pairs")

print("\nSMOKE TEST PASSED")
