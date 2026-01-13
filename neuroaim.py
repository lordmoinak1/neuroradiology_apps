import os
import tempfile
from collections import defaultdict

import numpy as np
import streamlit as st
import nibabel as nib
from PIL import Image
from scipy.ndimage import label as cc_label


# ==============================
# Constants
# ==============================

MODALITY_MAP = {
    "-t1c": "T1C",
    "-t1n": "T1N",
    "-t2f": "T2F",
    "-t2w": "T2W",
}
MODALITY_ORDER = ["T1C", "T1N", "T2F", "T2W"]


# ==============================
# Utility functions
# ==============================

def load_nifti(file):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name
    try:
        img = nib.load(path)
        vol = img.get_fdata().astype(np.float32)
        affine = img.affine
    finally:
        os.remove(path)

    if vol.ndim > 3:
        vol = vol[..., 0]
    return vol, affine


def normalize_slice(slc):
    slc = np.nan_to_num(slc)
    vmin, vmax = slc.min(), slc.max()
    return (slc - vmin) / (vmax - vmin + 1e-8)


def get_slice(vol, axis, idx):
    slc = np.take(vol, idx, axis=axis)
    slc = normalize_slice(slc)
    return np.rot90(slc, k=1)  # MRI orientation unchanged from your last request


def resize_img(img, size=256):
    im = Image.fromarray((img * 255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im) / 255.0


def resize_mask(mask, size=256):
    im = Image.fromarray(mask.astype(np.uint8), "L")
    im = im.resize((size, size), Image.NEAREST)
    return np.asarray(im).astype(int)


def overlay_segmentation(img, mask, alpha=0.4):
    base = (img * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)
    tumor = mask > 0
    rgb[tumor] = (1 - alpha) * rgb[tumor] + alpha * np.array([255, 0, 0])
    return np.clip(rgb / 255.0, 0, 1)


def spacing_from_affine(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def total_volume_mm3(seg, affine):
    voxel_vol = np.prod(spacing_from_affine(affine))
    return int(np.sum(seg > 0)) * voxel_vol


def per_lesion_volumes(seg, affine):
    voxel_vol = np.prod(spacing_from_affine(affine))
    labeled, num = cc_label(seg > 0, structure=np.ones((3, 3, 3)))
    return {
        lab: np.sum(labeled == lab) * voxel_vol
        for lab in range(1, num + 1)
    }


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroTracker", layout="wide")
st.title("🧠 NeuroTracker — Longitudinal Quantitative Tumor Tracking")

files = st.file_uploader(
    "Upload all BraTS-GLI .nii.gz files (MRI + seg)",
    type=["gz"],
    accept_multiple_files=True,
)

if not files:
    st.stop()


# ==============================
# Group files by timepoint
# ==============================

timepoints = defaultdict(lambda: {"modalities": {}, "seg": None})

for f in files:
    name = f.name.lower()

    if name.endswith("-seg.nii.gz"):
        tp_id = name.replace("-seg.nii.gz", "")
        timepoints[tp_id]["seg"] = f
        continue

    for tag, mod in MODALITY_MAP.items():
        if tag in name:
            tp_id = name.split(tag)[0]
            timepoints[tp_id]["modalities"][mod] = f
            break


# ==============================
# Layout
# ==============================

viewer_col, metrics_col = st.columns([3, 1])
metrics_data = []


# ==============================
# LEFT: Viewer
# ==============================

with viewer_col:
    axis = 2

    for tp_idx, (_, data) in enumerate(sorted(timepoints.items()), start=1):
        mods = data["modalities"]
        seg_file = data["seg"]

        if any(m not in mods for m in MODALITY_ORDER):
            continue

        # Row: rotated label + MRIs
        label_col, *img_cols = st.columns([0.5, 1, 1, 1, 1])

        with label_col:
            st.markdown(
                f"""
                <div style="
                    transform: rotate(-90deg);
                    transform-origin: left top;
                    white-space: nowrap;
                    font-size: 18px;
                    font-weight: 600;
                    margin-top: 140px;
                ">
                    ⏱ Timepoint {tp_idx}
                </div>
                """,
                unsafe_allow_html=True,
            )

        volumes = {}
        for mod in MODALITY_ORDER:
            vol, affine = load_nifti(mods[mod])
            volumes[mod] = vol

        seg = seg_affine = None
        if seg_file:
            seg, seg_affine = load_nifti(seg_file)

        max_slices = min(v.shape[axis] for v in volumes.values())
        slice_idx = st.slider(
            f"Axial slice — Timepoint {tp_idx}",
            0,
            max_slices - 1,
            max_slices // 2,
            key=f"slice_{tp_idx}",
        )

        for col, mod in zip(img_cols, MODALITY_ORDER):
            with col:
                img = resize_img(get_slice(volumes[mod], axis, slice_idx))
                if seg is not None:
                    seg_slc = resize_mask(get_slice(seg, axis, slice_idx))
                    img = overlay_segmentation(img, seg_slc)
                st.image(img, caption=mod, use_column_width=True)

        if seg is not None:
            metrics_data.append(
                {
                    "tp": tp_idx,
                    "total": total_volume_mm3(seg, seg_affine),
                    "per_lesion": per_lesion_volumes(seg, seg_affine),
                }
            )


# ==============================
# RIGHT: Metrics Pane
# ==============================

with metrics_col:
    # st.subheader("📊 Metrics")

    for m in metrics_data:
        st.markdown(f"### ⏱ Timepoint {m['tp']}")
        st.markdown(f"- **Total volume:** {m['total']:,.2f} mm³")
        st.markdown(f"- **Lesions:** {len(m['per_lesion'])}")
        st.markdown("**Per-lesion volume (mm³):**")

        for lab, vol in sorted(
            m["per_lesion"].items(), key=lambda x: x[1], reverse=True
        ):
            st.markdown(f"• Lesion {lab}: {vol:,.2f}")

        st.markdown("---")
