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
    if vmax > vmin:
        slc = (slc - vmin) / (vmax - vmin)
    else:
        slc = np.zeros_like(slc)
    return slc


def get_slice(vol, axis, idx):
    slc = np.take(vol, idx, axis=axis)
    slc = normalize_slice(slc)
    return np.rot90(slc, k=2)  # 180° rotation


def resize_img(img, size=256):
    im = Image.fromarray((img * 255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im) / 255.0


def calculate_volume(seg, affine):
    voxels = int(np.sum(seg > 0))
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_volume = np.prod(spacing)
    return voxels * voxel_volume


def count_lesions(seg):
    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    _, num = cc_label(seg > 0, structure=structure)
    return num


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroINK Longitudinal Viewer", layout="wide")
st.title("🧠 NeuroINK — BraTS-GLI Longitudinal Viewer")

files = st.file_uploader(
    "Upload all BraTS-GLI .nii.gz files (all timepoints together)",
    type=["gz"],
    accept_multiple_files=True,
)

if not files:
    st.stop()


# ==============================
# Group files by timepoint ID
# ==============================

timepoints = defaultdict(dict)

for f in files:
    name = f.name.lower()

    # Extract timepoint ID (everything before modality tag)
    tp_id = None
    for tag in MODALITY_MAP:
        if tag in name:
            tp_id = name.split(tag)[0]
            modality = MODALITY_MAP[tag]
            timepoints[tp_id][modality] = f
            break

    if tp_id is None:
        st.warning(f"Ignored file (unknown modality): {f.name}")


# ==============================
# Layout: Viewer | Metrics
# ==============================

viewer_col, metrics_col = st.columns([3, 1])
metrics_data = []


# ==============================
# LEFT: Viewer
# ==============================

with viewer_col:
    axis = 2

    for tp_idx, (tp, mods) in enumerate(sorted(timepoints.items()), start=1):
        st.markdown("---")
        st.subheader(f"⏱ Timepoint {tp_idx}: {tp}")

        missing = [m for m in MODALITY_ORDER if m not in mods]
        if missing:
            st.error(f"Missing modalities: {', '.join(missing)}")
            continue

        volumes = {}
        affine_ref = None

        for mod in MODALITY_ORDER:
            vol, affine = load_nifti(mods[mod])
            volumes[mod] = vol
            affine_ref = affine

        # Look for segmentation: same prefix, no modality suffix
        seg_file = next(
            (
                f
                for f in files
                if f.name.lower().startswith(tp)
                and not any(tag in f.name.lower() for tag in MODALITY_MAP)
            ),
            None,
        )

        vol_mm3 = None
        lesions = None

        if seg_file:
            seg, seg_affine = load_nifti(seg_file)
            vol_mm3 = calculate_volume(seg, seg_affine)
            lesions = count_lesions(seg)

        metrics_data.append(
            {
                "timepoint": tp,
                "volume_mm3": vol_mm3,
                "lesions": lesions,
            }
        )

        max_slices = min(v.shape[axis] for v in volumes.values())
        slice_idx = st.slider(
            f"Axial slice — {tp}",
            0,
            max_slices - 1,
            max_slices // 2,
            key=f"slice_{tp}",
        )

        cols = st.columns(4)
        for col, mod in zip(cols, MODALITY_ORDER):
            with col:
                img = resize_img(get_slice(volumes[mod], axis, slice_idx))
                st.image(img, caption=mod, use_column_width=True)


# ==============================
# RIGHT: Metrics Pane
# ==============================

with metrics_col:
    st.subheader("📊 Timepoint Metrics")

    for i, m in enumerate(metrics_data, start=1):
        st.markdown(f"### ⏱ Timepoint {i}")
        st.markdown(f"`{m['timepoint']}`")

        if m["volume_mm3"] is not None:
            st.markdown(f"- **Volume:** {m['volume_mm3']:,.2f} mm³")
            st.markdown(f"- **Lesions:** {m['lesions']}")
        else:
            st.markdown("_No segmentation found_")

        st.markdown("---")
