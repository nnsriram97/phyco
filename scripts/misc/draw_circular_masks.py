#!/usr/bin/env python3
# %% [markdown]
# # Hand-drawn circular masks for new scenes
#
# Draw circular masks over an object by clicking on the image. The saved file
# matches the segmentation format PhyCo consumes (uncompressed RLE inside a
# pickle), so you can point a batch JSON at it and apply physical-property maps
# to your own images — no SAM / external segmentation model required.
#
# This file is self-contained (only numpy / Pillow / matplotlib) and runs two
# ways:
#   - As a script:   PYTHONPATH=$(pwd) python scripts/misc/draw_circular_masks.py \
#                        --image assets/qualitative/restitution/rubber_duck_1.png
#   - Cell-by-cell in a Jupyter / VS Code notebook (the `# %%` markers below).
#
# **Drawing controls (in the figure window):**
# - Left click: first click sets the **center**, second click sets a point on
#   the **edge** (radius = distance between the two clicks).
# - Press `u`: undo the last circle (or cancel a pending center).
# - Press `r`: reset all circles.
# - Close the figure window when you are done.

# %% Enable an interactive matplotlib backend
# The `widget` backend needs `ipympl` (pip install ipympl). If it isn't
# installed we fall back to `qt`, then `tk`. If none work, use the "Manual
# override" cell below to type circle coordinates by hand.
def _enable_interactive_backend():
    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return  # plain Python script -- matplotlib will pick its own backend
    for name in ("widget", "qt", "tk", "notebook"):
        try:
            ip.run_line_magic("matplotlib", name)
            print(f"Using matplotlib backend: {name}")
            return
        except Exception:
            continue
    print(
        "No interactive matplotlib backend available. "
        "Install one with `pip install ipympl` (then restart the kernel), "
        "or use the Manual override cell below."
    )


_enable_interactive_backend()

# %% Imports
# (Targets Python 3.10+, so `X | Y` / `list[dict]` annotations work natively.)
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image


# %% Uncompressed-RLE helpers (PhyCo's mask format -- no external deps)
# These mirror the encoder/decoder the inference pipeline uses
# (examples/physprop_kubric_video2world.py:rle_to_mask). The mask is flattened
# in column-major (Fortran) order and stored as alternating run lengths that
# always start with a background run (a leading 0 if the first pixel is
# foreground).
def mask_to_uncompressed_rle(mask: np.ndarray) -> dict:
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).flatten(order="F")
    change_idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    borders = np.concatenate(([0], change_idx, [flat.size]))
    runs = np.diff(borders).tolist()
    counts = [] if not flat[0] else [0]
    counts.extend(runs)
    return {"size": [h, w], "counts": counts}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE (inverse of the above)."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    return mask.reshape(w, h).transpose()  # back to C order, shape (h, w)


# %% Configuration -- edit these for each run
# Paths are relative to the repo root (run with `PYTHONPATH=$(pwd)` from there).
# The defaults below point at a bundled asset so this runs out of the box; the
# `--image` / `--output-dir` CLI flags override them when run as a script.
IMAGE_PATH = "assets/qualitative/restitution/rubber_duck_1.png"  # input image
OUTPUT_DIR = "output/custom_masks"      # None -> same folder as the image
RESIZE_FACTOR = 1                       # set >1 to shrink the image first
SAVE_MASK_IMAGES = False                # also write per-circle PNGs
LOAD_EXISTING = False                   # if True, load any prior <stem>_masks.pkl
                                        # and append new circles to it


def _parse_cli_overrides():
    """Apply --image / --output-dir when run as a plain script (ignored in Jupyter)."""
    try:
        get_ipython()  # type: ignore[name-defined]
        return  # running under IPython/Jupyter -- keep the config above
    except NameError:
        pass
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="path to the input image")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to write <stem>_masks.pkl (default: output/custom_masks)",
    )
    parser.add_argument(
        "--resize-factor", type=int, default=None, help="shrink the image by this factor"
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="append to an existing <stem>_masks.pkl instead of starting fresh",
    )
    args, _ = parser.parse_known_args()
    global IMAGE_PATH, OUTPUT_DIR, RESIZE_FACTOR, LOAD_EXISTING
    if args.image is not None:
        IMAGE_PATH = args.image
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    if args.resize_factor is not None:
        RESIZE_FACTOR = args.resize_factor
    if args.load_existing:
        LOAD_EXISTING = True


_parse_cli_overrides()


# %% Load the image
def load_image(image_path: str | Path, resize_factor: int = 1) -> np.ndarray:
    image = Image.open(image_path)
    if resize_factor != 1:
        image = image.resize(
            (image.width // resize_factor, image.height // resize_factor)
        )
    image = np.array(image.convert("RGB"))
    print(f"Loaded image with shape: {image.shape}")
    return image


image = load_image(IMAGE_PATH, RESIZE_FACTOR)
H, W = image.shape[:2]


# %% (Optional) Load existing masks so new circles append to them
def _existing_pickle_path() -> Path:
    img_path = Path(IMAGE_PATH)
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else img_path.parent
    return out_dir / f"{img_path.stem}_masks.pkl"


loaded_entries: list[dict] = []
loaded_overlay: np.ndarray | None = None

if LOAD_EXISTING:
    _pkl = _existing_pickle_path()
    if _pkl.exists():
        with open(_pkl, "rb") as f:
            loaded_entries = pickle.load(f)
        print(f"Loaded {len(loaded_entries)} existing mask(s) from {_pkl}")

        # Build a single boolean overlay of all existing masks so we can show
        # them on the interactive canvas as a translucent tint.
        union = np.zeros((H, W), dtype=bool)
        for entry in loaded_entries:
            m = rle_to_mask(entry["segmentation_mask_rle"]["data"])
            if m.shape != (H, W):
                print(
                    f"  warning: existing mask shape {m.shape} != image {(H, W)}; "
                    "skipping in overlay (it will still be saved as-is)."
                )
                continue
            union |= m
        loaded_overlay = union
    else:
        print(f"No existing pickle at {_pkl} -- starting fresh.")


# %% Interactive drawing
# Stores finished circles as (cx, cy, r). Patches list mirrors it for undo.
circles: list[tuple[float, float, float]] = []
_pending_center: list[tuple[float, float] | None] = [None]
_patches: list = []
_center_markers: list = []

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
if loaded_overlay is not None and loaded_overlay.any():
    tint = np.zeros((H, W, 4), dtype=float)
    tint[..., 1] = 1.0  # green channel
    tint[..., 3] = loaded_overlay.astype(float) * 0.35  # alpha
    ax.imshow(tint)
ax.set_title(
    f"Existing: {len(loaded_entries)} mask(s) (green). "
    "Click center then edge to add. 'u' = undo, 'r' = reset. Close when done."
)
ax.set_axis_off()


def _redraw():
    fig.canvas.draw_idle()


def _on_click(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    x, y = float(event.xdata), float(event.ydata)

    if _pending_center[0] is None:
        _pending_center[0] = (x, y)
        marker = ax.plot(x, y, "r+", markersize=12, mew=2)[0]
        _center_markers.append(marker)
        print(f"  center pending at ({x:.1f}, {y:.1f}) -- click edge")
    else:
        cx, cy = _pending_center[0]
        r = float(np.hypot(x - cx, y - cy))
        circles.append((cx, cy, r))
        patch = Circle((cx, cy), r, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(patch)
        _patches.append(patch)
        _pending_center[0] = None
        print(
            f"  circle {len(circles)}: center=({cx:.1f}, {cy:.1f}), radius={r:.1f}"
        )
    _redraw()


def _on_key(event):
    if event.key == "u":
        if _pending_center[0] is not None:
            _pending_center[0] = None
            if _center_markers:
                _center_markers.pop().remove()
            print("  cancelled pending center")
        elif circles:
            circles.pop()
            _patches.pop().remove()
            if _center_markers:
                _center_markers.pop().remove()
            print(f"  undid last circle ({len(circles)} remaining)")
        _redraw()
    elif event.key == "r":
        circles.clear()
        while _patches:
            _patches.pop().remove()
        while _center_markers:
            _center_markers.pop().remove()
        _pending_center[0] = None
        print("  reset all circles")
        _redraw()


fig.canvas.mpl_connect("button_press_event", _on_click)
fig.canvas.mpl_connect("key_press_event", _on_key)
plt.show()


# %% (Optional) Manual override
# If the interactive backend is unavailable, comment out the cell above and
# fill in circles by hand here, e.g.:
# circles = [(320.0, 240.0, 80.0), (500.0, 410.0, 55.0)]
print(f"{len(circles)} circle(s) ready to save.")
for i, (cx, cy, r) in enumerate(circles):
    print(f"  segment_{i}: center=({cx:.1f}, {cy:.1f}), radius={r:.1f}")


# %% Convert circles to PhyCo-style mask entries
def circle_to_mask(cx: float, cy: float, r: float, h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def build_mask_entries(
    circles: list[tuple[float, float, float]],
    h: int,
    w: int,
    start_index: int = 0,
) -> list[dict]:
    entries: list[dict] = []
    for i, (cx, cy, r) in enumerate(circles):
        mask = circle_to_mask(cx, cy, r, h, w)
        rle = mask_to_uncompressed_rle(mask)
        area = int(mask.sum())
        ys, xs = np.where(mask)
        if ys.size == 0:
            bbox = [0, 0, 0, 0]
        else:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]  # XYWH

        entries.append(
            {
                "phrase": f"segment_{start_index + i}",
                "segmentation_mask_rle": {
                    "data": rle,
                    "mask_shape": rle["size"],
                },
                "area": area,
                "bbox": bbox,
                "predicted_iou": 1.0,
                "stability_score": 1.0,
            }
        )
    return entries


new_entries = build_mask_entries(circles, H, W, start_index=len(loaded_entries))
mask_entries = list(loaded_entries) + new_entries

# Re-number phrases so the merged file has segment_0 .. segment_{N-1}.
for i, entry in enumerate(mask_entries):
    entry["phrase"] = f"segment_{i}"

print(
    f"Built {len(mask_entries)} total mask entries "
    f"({len(loaded_entries)} existing + {len(new_entries)} new)."
)


# %% Save in PhyCo's segmentation format
def save_results(
    entries: list[dict],
    image_path: str | Path,
    output_dir: str | Path | None,
) -> tuple[Path, Path]:
    image_path = Path(image_path)
    out_dir = Path(output_dir) if output_dir else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = image_path.stem
    pickle_path = out_dir / f"{base_name}_masks.pkl"
    txt_path = out_dir / f"{base_name}_mask_ids.txt"

    with open(pickle_path, "wb") as f:
        pickle.dump(entries, f)
    print(f"Saved {len(entries)} masks to {pickle_path}")

    with open(txt_path, "w") as f:
        f.write("# Mask ID Information\n")
        f.write(f"# Total masks: {len(entries)}\n")
        f.write(f"# Generated from: {image_path.name} (hand-drawn circles)\n\n")
        f.write("# Selected as foreground: ALL\n")
        f.write(
            "foreground_mask_ids: "
            + " ".join(map(str, range(len(entries))))
            + "\n"
        )
        f.write("\n# All mask IDs and their areas:\n")
        for i, entry in enumerate(entries):
            f.write(
                f"mask_{i}: area={entry['area']}, "
                f"iou={entry['predicted_iou']:.3f}, "
                f"stability={entry['stability_score']:.3f}\n"
            )
    print(f"Saved mask ID information to {txt_path}")
    return pickle_path, txt_path


pickle_path, txt_path = save_results(mask_entries, IMAGE_PATH, OUTPUT_DIR)


# %% (Optional) Export each circle as its own PNG
if SAVE_MASK_IMAGES and mask_entries:
    img_dir = pickle_path.parent / f"{Path(IMAGE_PATH).stem}_mask_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(mask_entries):
        binary = rle_to_mask(entry["segmentation_mask_rle"]["data"])
        Image.fromarray(binary.astype(np.uint8) * 255).save(
            img_dir / f"mask_{i:04d}.png"
        )
    print(f"Wrote {len(mask_entries)} mask PNGs to {img_dir}")


# %% Sanity check: load the pickle back and overlay on the image
with open(pickle_path, "rb") as f:
    loaded = pickle.load(f)

rng = np.random.default_rng(0)
overlay = image.copy()
for entry in loaded:
    binary = rle_to_mask(entry["segmentation_mask_rle"]["data"])
    color = rng.integers(0, 255, size=3, dtype=np.uint8)
    overlay[binary] = (0.5 * overlay[binary] + 0.5 * color).astype(np.uint8)

plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.title(f"Loaded {len(loaded)} mask(s) from {pickle_path.name}")
plt.axis("off")
plt.show()

# %%
