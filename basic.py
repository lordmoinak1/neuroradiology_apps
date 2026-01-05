import os
import tempfile
from typing import Dict

import numpy as np
import streamlit as st
from PIL import Image
import nibabel as nib
from scipy.ndimage import label as cc_label


# ==============================
# Constants
# ==============================

MODALITY_MAP = {
    "_0000": "T1C",
    "_0001": "T1N",
    "_0002": "T2F",
    "_0003": "T2W",
}


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

    # Rotate 180 degrees (2 × 90° rotations)
    return np.rot90(slc, k=2)

def resize_img(img, size=256):
    im = Image.fromarray((img * 255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im) / 255.0


def resize_mask(mask, size=256):
    im = Image.fromarray(mask.astype(np.uint8), "L")
    im = im.resize((size, size), Image.NEAREST)
    return np.asarray(im).astype(int)


def overlay_segmentation(img, mask, colors, alpha=0.4):
    base = (img * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)

    for lab, col in colors.items():
        region = mask == lab
        rgb[region] = (1 - alpha) * rgb[region] + alpha * np.array(col)

    return np.clip(rgb / 255.0, 0, 1)


def calculate_volume(seg, affine):
    voxels = int(np.sum(seg > 0))
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_vol = np.prod(spacing)
    mm3 = voxels * voxel_vol
    ml = mm3 / 1000.0
    return voxels, mm3, ml


def count_lesions(seg):
    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    _, num = cc_label(seg > 0, structure=structure)
    return num


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroTrack", layout="wide")
st.title("🧠 NeuroTrack")

with st.sidebar:
    st.header("Upload BraTS MRI Sequences")

    files = st.file_uploader(
        "Upload *_0000/0001/0002/0003.nii.gz together",
        type=["gz"],
        accept_multiple_files=True,
        key="mri",
    )

    st.header("Segmentation (optional)")
    seg_file = st.file_uploader(
        "Upload segmentation (.nii.gz)",
        type=["gz"],
        key="seg",
    )


# ==============================
# Validate & load MRI sequences
# ==============================

if not files:
    st.info("👈 Upload BraTS MRI sequences to begin.")
    st.stop()

volumes: Dict[str, np.ndarray] = {}
affine_ref = None

for f in files:
    name = f.name.lower()
    matched = False

    for tag, modality in MODALITY_MAP.items():
        if tag in name:
            vol, affine = load_nifti(f)
            volumes[modality] = vol
            affine_ref = affine
            matched = True
            break

    if not matched:
        st.warning(f"Ignored file (unknown modality): {f.name}")

missing = [m for m in MODALITY_MAP.values() if m not in volumes]
if missing:
    st.error(f"Missing modalities: {', '.join(missing)}")
    st.stop()

modalities_order = ["T1C", "T1N", "T2F", "T2W"]


# ==============================
# Load segmentation
# ==============================

seg_volume = None
label_colors = None

if seg_file:
    seg_volume, seg_affine = load_nifti(seg_file)

    labels = np.unique(seg_volume.astype(int))
    labels = labels[labels > 0]

    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]

    label_colors = {
        int(l): palette[i % len(palette)]
        for i, l in enumerate(labels)
    }

    voxels, mm3, ml = calculate_volume(seg_volume, seg_affine)
    lesions = count_lesions(seg_volume)

    st.sidebar.subheader("Segmentation Metrics")
    # st.sidebar.write(f"Voxel count: {voxels:,}")
    st.sidebar.write(f"Volume: {mm3:,.2f} mm³")
    # st.sidebar.write(f"Volume: {ml:,.2f} ml")
    st.sidebar.write(f"Number of lesions: {lesions}")


# ==============================
# Axial Viewer
# ==============================

axis = 2
max_slices = min(v.shape[axis] for v in volumes.values())
slice_idx = st.slider("Axial slice", 0, max_slices - 1, max_slices // 2)

cols = st.columns(4)

for col, mod in zip(cols, modalities_order):
    with col:
        img = resize_img(get_slice(volumes[mod], axis, slice_idx))

        if seg_volume is not None and seg_volume.shape == volumes[mod].shape:
            seg_slc = resize_mask(
                get_slice(seg_volume, axis, slice_idx)
            )
            img = overlay_segmentation(img, seg_slc, label_colors)

        st.image(
            img,
            caption=f"{mod} — Axial {slice_idx}",
            use_column_width=True,
        )
