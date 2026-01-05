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

def validate_braTS_file(file, expected_tag=None):
    name = file.name.lower()

    if not name.endswith(".nii.gz"):
        st.error(f"{file.name} is not a .nii.gz file")
        st.stop()

    if expected_tag and expected_tag not in name:
        st.error(
            f"{file.name} does not match expected BraTS modality tag {expected_tag}"
        )
        st.stop()


def load_nifti(file) -> Tuple[np.ndarray, np.ndarray]:
    if nib is None:
        raise ImportError("Please install nibabel")

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


def calculate_segmentation_volume(seg, affine):
    voxel_count = int(np.sum(seg > 0))
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_volume = np.prod(spacing)
    volume_mm3 = voxel_count * voxel_volume
    volume_ml = volume_mm3 / 1000.0
    return voxel_count, volume_mm3, volume_ml


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroINK", layout="wide")
st.title("🧠 NeuroINK — BraTS MRI Viewer")

with st.sidebar:
    st.header("MRI Sequences (BraTS .nii.gz)")

    t1c_file = st.file_uploader(
        "Upload T1C (_0000.nii.gz)",
        type=["gz"],
        key="t1c",
    )

    t2f_file = st.file_uploader(
        "Upload T2F (_0002.nii.gz)",
        type=["gz"],
        key="t2f",
    )

    t2w_file = st.file_uploader(
        "Upload T2W (_0003.nii.gz)",
        type=["gz"],
        key="t2w",
    )

    st.header("Segmentation")
    seg_file = st.file_uploader(
        "Upload Segmentation (.nii.gz)",
        type=["gz"],
        key="seg",
    )


# ==============================
# Validation
# ==============================

if not all([t1c_file, t2f_file, t2w_file]):
    st.info("👈 Upload T1C, T2F, and T2W to begin.")
    st.stop()

validate_braTS_file(t1c_file, "_0000")
validate_braTS_file(t2f_file, "_0002")
validate_braTS_file(t2w_file, "_0003")

if seg_file:
    validate_braTS_file(seg_file)


# ==============================
# Load volumes
# ==============================

t1c, _ = load_nifti(t1c_file)
t2f, _ = load_nifti(t2f_file)
t2w, _ = load_nifti(t2w_file)

volumes = {
    "T1C": t1c,
    "T2F": t2f,
    "T2W": t2w,
}

seg_volume = None
seg_affine = None
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

    voxels, mm3, ml = calculate_segmentation_volume(
        seg_volume, seg_affine
    )

    st.sidebar.subheader("Segmentation Volume")
    st.sidebar.write(f"Voxel count: {voxels:,}")
    st.sidebar.write(f"Volume: {mm3:,.2f} mm³")
    st.sidebar.write(f"Volume: {ml:,.2f} ml")


# ==============================
# Axial Viewer
# ==============================

axis = 2
max_slices = min(v.shape[axis] for v in volumes.values())

slice_idx = st.slider(
    "Axial slice index",
    0,
    max_slices - 1,
    max_slices // 2,
)

cols = st.columns(3)

for col, (name, vol) in zip(cols, volumes.items()):
    with col:
        img = resize_img(get_slice(vol, axis, slice_idx))

        if seg_volume is not None and seg_volume.shape == vol.shape:
            seg_slc = resize_mask(
                get_slice(seg_volume, axis, slice_idx)
            )
            img = overlay_segmentation(img, seg_slc, label_colors)

        st.image(
            img,
            caption=f"{name} — Axial {slice_idx}",
            use_column_width=True,
        )
