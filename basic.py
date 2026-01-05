import io
import os
import tempfile
from typing import Dict, Tuple

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

def load_volume(file) -> Tuple[np.ndarray, str, dict]:
    """
    Load a 3D volume or 2D image from an uploaded file.
    Supports .nii, .nii.gz, .npy, .npz, and image files.
    Returns: (volume, label, metadata)
    """
    filename = file.name.lower()
    label = file.name
    meta = {"affine": None}

    # ---- NIfTI ----
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("Please install nibabel for NIfTI support.")

        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            img = nib.load(tmp_path)
            vol = img.get_fdata().astype(np.float32)
            meta["affine"] = img.affine
        finally:
            os.remove(tmp_path)

    # ---- NumPy ----
    elif filename.endswith(".npy"):
        vol = np.load(io.BytesIO(file.read())).astype(np.float32)

    elif filename.endswith(".npz"):
        data = np.load(io.BytesIO(file.read()))
        key = list(data.keys())[0]
        vol = data[key].astype(np.float32)
        label = f"{label} ({key})"

    # ---- Image ----
    else:
        img = Image.open(io.BytesIO(file.read())).convert("L")
        vol = np.array(img, dtype=np.float32)[..., None]

    # Ensure (X, Y, Z)
    if vol.ndim == 2:
        vol = vol[..., None]
    elif vol.ndim > 3:
        vol = vol[..., 0]

    return vol, label, meta


def get_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    slc = np.take(vol, index, axis=axis)
    slc = np.nan_to_num(slc)
    vmin, vmax = slc.min(), slc.max()
    if vmax > vmin:
        slc = (slc - vmin) / (vmax - vmin)
    else:
        slc = np.zeros_like(slc)
    return np.rot90(slc)


def get_seg_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    slc = np.take(vol, index, axis=axis)
    return np.rot90(np.nan_to_num(slc))


def resize_slice(slc: np.ndarray, size=256) -> np.ndarray:
    img = Image.fromarray((np.clip(slc, 0, 1) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img) / 255.0


def resize_mask(mask: np.ndarray, size=256) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.NEAREST)
    return np.asarray(img).astype(int)


def overlay_segmentation(img, mask, label_colors, alpha=0.4):
    base = (img * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)

    for lab, color in label_colors.items():
        region = mask == lab
        rgb[region] = (1 - alpha) * rgb[region] + alpha * np.array(color)

    return np.clip(rgb / 255.0, 0, 1)


def calculate_segmentation_volume(seg: np.ndarray, affine=None):
    voxel_count = int(np.sum(seg > 0))
    volume_mm3 = None
    volume_ml = None

    if affine is not None:
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        voxel_volume = float(np.prod(spacing))
        volume_mm3 = voxel_count * voxel_volume
        volume_ml = volume_mm3 / 1000.0

    return voxel_count, volume_mm3, volume_ml


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroINK", layout="wide")
st.title("🧠 NeuroINK — MRI Sequence Viewer")

with st.sidebar:
    st.header("MRI Sequences")
    uploaded_files = st.file_uploader(
        "Upload exactly 3 sequences (T1c, T2f, T2w)",
        type=["nii", "nii.gz", "npy", "npz"],
        accept_multiple_files=True,
    )

    st.header("Segmentation (optional)")
    seg_file = st.file_uploader(
        "Upload segmentation mask",
        type=["nii", "nii.gz", "npy", "npz"],
    )

# ==============================
# Load MRI volumes
# ==============================

volumes: Dict[str, np.ndarray] = {}

if uploaded_files:
    for f in uploaded_files:
        vol, label, _ = load_volume(f)
        volumes[label] = vol

if len(volumes) != 3:
    st.warning("Please upload exactly 3 MRI sequences: T1c, T2f, T2w.")
    st.stop()

# ==============================
# Load segmentation
# ==============================

seg_volume = None
seg_affine = None
label_colors = None

if seg_file is not None:
    seg_volume, _, meta = load_volume(seg_file)
    seg_affine = meta.get("affine")

    labels = np.unique(seg_volume.astype(int))
    labels = labels[labels > 0]

    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (255, 165, 0),
    ]

    label_colors = {
        int(l): palette[i % len(palette)]
        for i, l in enumerate(labels)
    }

    voxel_count, vol_mm3, vol_ml = calculate_segmentation_volume(
        seg_volume, seg_affine
    )

    st.sidebar.subheader("Segmentation Volume")
    st.sidebar.write(f"Voxel count: {voxel_count:,}")
    if vol_mm3 is not None:
        st.sidebar.write(f"Volume: {vol_mm3:,.2f} mm³")
        st.sidebar.write(f"Volume: {vol_ml:,.2f} ml")
    else:
        st.sidebar.info("Physical volume unavailable (no voxel spacing).")

# ==============================
# Axial Viewer
# ==============================

axis = 2  # axial
max_slices = min(v.shape[axis] for v in volumes.values())

slice_idx = st.slider(
    "Axial slice index",
    0, max_slices - 1, max_slices // 2
)

cols = st.columns(3)

for col, (name, vol) in zip(cols, volumes.items()):
    with col:
        slc = resize_slice(get_slice(vol, axis, slice_idx))

        if seg_volume is not None and seg_volume.shape == vol.shape:
            seg_slc = resize_mask(get_seg_slice(seg_volume, axis, slice_idx))
            slc = overlay_segmentation(slc, seg_slc, label_colors)

        st.image(
            slc,
            caption=f"{name} — Axial {slice_idx}",
            use_column_width=True,
        )
