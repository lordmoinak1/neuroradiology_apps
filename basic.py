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
    return np.rot90((slc - vmin) / (vmax - vmin + 1e-8))


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
    voxels = int(np.sum(seg > 0))
    mm3 = ml = None
    if affine is not None:
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        voxel_vol = np.prod(spacing)
        mm3 = voxels * voxel_vol
        ml = mm3 / 1000.0
    return voxels, mm3, ml


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroINK", layout="wide")
st.title("🧠 NeuroINK — MRI Viewer")

with st.sidebar:
    st.header("MRI Sequences")

    t1c_file = st.file_uploader("Upload T1C", type=["nii", "gz", "nii.gz", "npy", "npz"], key="t1c")
    t2f_file = st.file_uploader("Upload T2F", type=["nii", "gz", "nii.gz", "npy", "npz"], key="t2f")
    t2w_file = st.file_uploader("Upload T2W", type=["nii", "gz", "nii.gz", "npy", "npz"], key="t2w")

    st.header("Segmentation")
    seg_file = st.file_uploader("Upload segmentation", type=["nii", "nii.gz", "npy", "npz"], key="seg")


if not all([t1c_file, t2f_file, t2w_file]):
    st.info("Upload T1C, T2F, and T2W to begin.")
    st.stop()

t1c, _ = load_volume(t1c_file)
t2f, _ = load_volume(t2f_file)
t2w, _ = load_volume(t2w_file)

volumes = {"T1C": t1c, "T2F": t2f, "T2W": t2w}

seg_volume = None
label_colors = None

if seg_file:
    seg_volume, meta = load_volume(seg_file)
    labels = np.unique(seg_volume.astype(int))
    labels = labels[labels > 0]

    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    label_colors = {int(l): palette[i % len(palette)] for i, l in enumerate(labels)}

    vox, mm3, ml = calculate_segmentation_volume(seg_volume, meta["affine"])
    st.sidebar.subheader("Segmentation Volume")
    st.sidebar.write(f"Voxel count: {vox:,}")
    if mm3:
        st.sidebar.write(f"{mm3:,.2f} mm³ ({ml:,.2f} ml)")

axis = 2
max_slices = min(v.shape[axis] for v in volumes.values())
idx = st.slider("Axial slice", 0, max_slices - 1, max_slices // 2)

cols = st.columns(3)
for col, (name, vol) in zip(cols, volumes.items()):
    with col:
        img = resize_img(get_slice(vol, axis, idx))
        if seg_volume is not None and seg_volume.shape == vol.shape:
            seg = resize_mask(get_slice(seg_volume, axis, idx))
            img = overlay_segmentation(img, seg, label_colors)
        st.image(img, caption=f"{name} — Axial {idx}", use_column_width=True)
