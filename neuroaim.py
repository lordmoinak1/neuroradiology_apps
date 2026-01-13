import os
import tempfile

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
    return np.rot90(slc, k=1)


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


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroTracker", layout="wide")
st.title("🧠 NeuroTracker — Single Timepoint Review")

files = st.file_uploader(
    "Upload ONE timepoint (MRI modalities + optional seg)",
    type=["gz"],
    accept_multiple_files=True,
)

if not files:
    st.stop()


# ==============================
# Parse uploaded files
# ==============================

modalities = {}
seg_file = None

for f in files:
    name = f.name.lower()

    if name.endswith("-seg.nii.gz"):
        seg_file = f
        continue

    for tag, mod in MODALITY_MAP.items():
        if tag in name:
            modalities[mod] = f
            break

if any(m not in modalities for m in MODALITY_ORDER):
    st.error("Missing one or more required modalities (T1C, T1N, T2F, T2W)")
    st.stop()


# ==============================
# Load data
# ==============================

volumes = {}
for mod in MODALITY_ORDER:
    vol, affine = load_nifti(modalities[mod])
    volumes[mod] = vol

seg = None
if seg_file:
    seg, _ = load_nifti(seg_file)


# ==============================
# Viewer
# ==============================

axis = 2
max_slices = min(v.shape[axis] for v in volumes.values())

slice_idx = st.slider(
    "Axial slice",
    0,
    max_slices - 1,
    max_slices // 2,
)

cols = st.columns(4)

for col, mod in zip(cols, MODALITY_ORDER):
    with col:
        img = resize_img(get_slice(volumes[mod], axis, slice_idx))
        if seg is not None:
            seg_slc = resize_mask(get_slice(seg, axis, slice_idx))
            img = overlay_segmentation(img, seg_slc)
        st.image(img, caption=mod, use_column_width=True)


# ==============================
# Radiologist Rating System
# ==============================

st.markdown("---")
st.subheader("🩺 Radiologist Assessment")

if "anatomy_rating" not in st.session_state:
    st.session_state.anatomy_rating = 2

anatomy_rating = st.radio(
    "Anatomy (1–3)",
    options=[1, 2, 3],
    index=st.session_state.anatomy_rating - 1,
    help="""
    1 = Poor anatomical clarity  
    2 = Adequate anatomy  
    3 = Excellent anatomical depiction
    """,
)

st.session_state.anatomy_rating = anatomy_rating

st.success(f"✔ Anatomy rating recorded: **{anatomy_rating} / 3**")
