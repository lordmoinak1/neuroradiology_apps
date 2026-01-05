import io
import os
import tempfile
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    nib = None


# ==============================
# Utility functions
# ==============================

def load_volume(file) -> Tuple[np.ndarray, dict]:
    """
    Load a 3D volume from NIfTI / NumPy / image.
    Returns: volume, metadata
    """
    filename = file.name.lower()
    meta = {"affine": None}

    if filename.endswith((".nii", ".nii.gz")):
        if nib is None:
            raise ImportError("Install nibabel for NIfTI support")

        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        try:
            img = nib.load(path)
            vol = img.get_fdata().astype(np.float32)
            meta["affine"] = img.affine
        finally:
            os.remove(path)

    elif filename.endswith(".npy"):
        vol = np.load(io.BytesIO(file.read())).astype(np.float32)

    elif filename.endswith(".npz"):
        data = np.load(io.BytesIO(file.read()))
        vol = data[list(data.keys())[0]].astype(np.float32)

    else:
        img = Image.open(io.BytesIO(file.read())).convert("L")
        vol = np.array(img, dtype=np.float32)[..., None]

    if vol.ndim == 2:
        vol = vol[..., None]
    elif vol.ndim > 3:
        vol = vol[..., 0]

    return vol, meta


def get_slice(vol, axis, idx):
    slc = np.take(vol, idx, axis=axis)
    slc = np.nan_to_num(slc)
    vmin, vmax = slc.min(), slc.max()
    if vmax > vmin:
        slc = (slc - vmin) / (vmax - vmin)
    else:
        slc = np.zeros_like(slc)
    return np.rot90(slc)


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


def calculate_segmentation_volume(seg, affine=None):
    voxel_count = int(np.sum(seg > 0))
    mm3 = ml = None

    if affine is not None:
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        voxel_vol = np.prod(spacing)
        mm3 = voxel_count * voxel_vol
        ml = mm3 / 1000.0

    return voxel_count, mm3, ml


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroINK", layout="wide")
st.title("🧠 NeuroINK — MRI Viewer")

with st.sidebar:
    st.header("MRI Sequences")

    t1c_file = st.file_uploader("Upload **T1C**", type=["nii", "nii.gz", "npy", "npz"])
    t2f_file = st.file_uploader("Upload **T2F**", type=["nii", "nii.gz", "npy", "npz"])
    t2w_file = st.file_uploader("Upload **T2W**", type=["nii", "nii.gz", "npy", "npz"])

    st.header("Segmentation (optional)")
    seg_file = st.file_uploader("Upload segmentation", type=["nii", "nii.gz", "npy", "npz"])


# ==============================
# Load MRI volumes
# ==============================

if not all([t1c_file, t2f_file, t2w_file]):
    st.info("👈 Upload T1C, T2F, and T2W to begin.")
    st.stop()

t1c, _ = load_volume(t1c_file)
t2f, _ = load_volume(t2f_file)
t2w, _ = load_volume(t2w_file)

volumes = {
    "T1C": t1c,
    "T2F": t2f,
    "T2W": t2w,
}

# ==============================
# Load segmentation
# ==============================

seg_volume = None
label_colors = None
seg_affine = None

if seg_file:
    seg_volume, meta = load_volume(seg_file)
    seg_affine = meta.get("affine")

    labels = np.unique(seg_volume.astype(int))
    labels = labels[labels > 0]

    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    label_colors = {
        int(l): palette[i % len(palette)]
        for i, l in enumerate(labels)
    }

    voxels, mm3, ml = calculate_segmentation_volume(seg_volume, seg_affine)

    st.sidebar.subheader("Segmentation Volume")
    st.sidebar.write(f"Voxel count: {voxels:,}")
    if mm3 is not None:
        st.sidebar.write(f"Volume: {mm3:,.2f} mm³")
        st.sidebar.write(f"Volume: {ml:,.2f} ml")
    else:
        st.sidebar.info("Physical volume unavailable")


# ==============================
# Axial Viewer
# ==============================

axis = 2
max_slices = min(v.shape[axis] for v in volumes.values())

slice_idx = st.slider(
    "Axial slice index",
    0, max_slices - 1, max_slices // 2
)

cols = st.columns(3)

for col, (name, vol) in zip(cols, volumes.items()):
    with col:
        img = resize_img(get_slice(vol, axis, slice_idx))

        if seg_volume is not None and seg_volume.shape == vol.shape:
            seg_slc = resize_mask(get_slice(seg_volume, axis, slice_idx))
            img = overlay_segmentation(img, seg_slc, label_colors)

        st.image(img, caption=f"{name} — Axial {slice_idx}", use_column_width=True)
