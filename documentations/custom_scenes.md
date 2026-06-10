# Trying PhyCo on your own scenes

PhyCo conditions generation on **physical-property maps** that are painted onto
the objects in your input image. To run PhyCo on a *new* image you therefore
need a **segmentation mask** for the object(s) you want to control. The released
examples ship their masks in `assets/qualitative/masks/`, but for your own
images you can create them in under a minute by drawing circles over the
objects — no SAM / external segmentation model required.

This guide covers the full path: **image → masks → batch JSON → generated video.**

---

## 1. Draw circular masks

Use `scripts/misc/draw_circular_masks.py`. It opens your image in an interactive
window; click to drop circles over each object and it writes a mask file in the
exact format PhyCo's pipeline consumes (uncompressed RLE inside a pickle).

```bash
# run from the repo root
PYTHONPATH=$(pwd) python scripts/misc/draw_circular_masks.py \
  --image path/to/your_scene.png \
  --output-dir assets/qualitative/masks
```

**Drawing controls** (in the figure window):

| Action | Result |
| --- | --- |
| left click ×2 | first click = circle **center**, second click = a point on the **edge** (radius = distance between them) |
| `u` | undo the last circle (or cancel a pending center) |
| `r` | reset all circles |
| close window | finish and save |

Each circle becomes one segment, numbered `segment_0, segment_1, …` in click
order. Draw one circle per object you want to control (e.g. the ball and the
ground), plus optionally a background circle. When you close the window it writes:

```
<output-dir>/<stem>_masks.pkl       # the masks (referenced by the batch JSON)
<output-dir>/<stem>_mask_ids.txt     # human-readable list of mask ids + areas
```

> **No display?** (headless server / SSH without X-forwarding). Either run it
> locally, or open the file as a notebook (it has `# %%` cells) and use the
> **Manual override** cell to type circle coordinates by hand:
> `circles = [(cx, cy, r), ...]`. To append to an existing mask file instead of
> starting over, pass `--load-existing`.

The script can also be run cell-by-cell in Jupyter / VS Code — the `# %%`
markers define the cells, and a final cell overlays the saved masks back on the
image as a sanity check.

---

## 2. Note your mask ids

Open `<stem>_mask_ids.txt` (or read the printout) to see how many segments you
drew and their ids. You'll reference these ids in the batch JSON:

- **foreground** ids — the object(s) whose physics you control (the ball, the
  deformable duck, …).
- **background** id — the static surface / scene, if you drew one.

---

## 3. Point a batch JSON at your scene

The generation scripts read a **batch JSON** — a list of scene entries. Copy an
entry from the matching example (e.g.
`scripts/batch_jsons/benchmark/deformation_all-v1.json`) and edit these fields:

```jsonc
{
  "id": 1,
  "input_video": "assets/qualitative/your_scene.png",          // your image
  "prompt": "A short description of the scene and the motion.",  // text prompt
  "segmentation_pkl": "assets/qualitative/masks/your_scene_masks.pkl", // step 1 output
  "needed_segmentation_ids": [0, 1],   // every mask id this scene uses
  "fg_seg_ids": [1],                    // foreground object(s) to control
  "bg_seg_id": 0,                       // background / surface id
  "fg_object_types": ["soft"],          // per-fg type, e.g. "soft" / "rigid"
  "props_of_interest": ["deformable"],  // which physics branch(es) to apply
  "output_video": "your_scene_out.mp4",
  "neo_hookean_mu": "high",             // deformation stiffness (deformable scenes)
  "neo_hookean_damping": "high",
  "use_text_encoder": true,
  "is_kubric_generated": false
}
```

The property fields depend on what you're controlling:

| `props_of_interest` | Branch | Key knobs |
| --- | --- | --- |
| `deformable` | `controlnet_2` | `neo_hookean_mu`, `neo_hookean_damping` (`low`/`high`) |
| `friction` / `bounciness` | `controlnet_1` | friction / restitution level |
| `force` | `controlnet_3` | applied force / motion direction |

The cleanest starting point is to copy a single entry from the property-matched
example JSON, swap in your `input_video` / `segmentation_pkl` / ids, and adjust
the property values — the other fields can stay as-is.

---

## 4. Generate

Run the same launcher as the Quick-start in the [README](../README.md), pointing
`--batch_input_json` at your edited JSON:

```bash
DIT=checkpoints/phyco/phyco.pt

PYTHONPATH=$(pwd) bash scripts/launch/run_physprop_kubric_mpgu.sh \
  --batch_input_json scripts/batch_jsons/benchmark/your_scene.json \
  --post_dit_subfolder your_scene \
  --seeds 42,0,10,31 \
  --dit_path "$DIT" \
  --controlnet_branch_names controlnet_1,controlnet_2,controlnet_3 \
  --active_controlnets controlnet_1,controlnet_2,controlnet_3 \
  --controlnet_branch_scales 1.0,1.0,1.0 --dynamic_controlnet \
  --pipeline_config controlnet_multi_24fps_57frames \
  --conditioning_type image_blob --blob_type circle \
  --resolution 480 --fps 24 \
  --physprop_type friction_bounciness_deformable_force_move_dir \
  --num_splits 1 --split_index 0
```

Videos are written to `output/phyco/your_scene/`, one clip per seed. From here
it's the same workflow as the bundled examples — sweep the property values
(e.g. `neo_hookean_mu` low→high) to see the controllable physics.
