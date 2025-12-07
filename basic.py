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


# ------------------------------
# Utility functions
# ------------------------------

def load_volume(file) -> Tuple[np.ndarray, str]:
    """
    Load a 3D volume or 2D image from an uploaded file.
    Supports .nii, .nii.gz, .npy, .npz, and image files.
    Returns: (volume: np.ndarray, label: str)
    """
    filename = file.name.lower()
    label = file.name

    # NIfTI (.nii / .nii.gz)
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
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # NumPy .npy
    elif filename.endswith(".npy"):
        raw = file.read()
        bio = io.BytesIO(raw)
        vol = np.load(bio).astype(np.float32)

    # NumPy .npz  (take first array)
    elif filename.endswith(".npz"):
        raw = file.read()
        bio = io.BytesIO(raw)
        data = np.load(bio)
        key = list(data.keys())[0]
        vol = data[key].astype(np.float32)
        label = f"{filename} ({key})"

    # Image files (PNG/JPG/…)
    else:
        raw = file.read()
        bio = io.BytesIO(raw)
        pil_img = Image.open(bio).convert("L")
        arr = np.array(pil_img, dtype=np.float32)
        vol = arr[..., None]  # (H, W, 1)

    # Ensure (X, Y, Z)
    if vol.ndim == 2:
        vol = vol[..., None]
    elif vol.ndim > 3:
        vol = vol[..., 0]

    return vol, label


def get_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Normalize slice to [0,1] and rotate."""
    if axis >= vol.ndim:
        raise ValueError(f"Axis {axis} out of range for volume shape {vol.shape}")

    slc = np.take(vol, index, axis=axis)
    slc = np.nan_to_num(slc)

    vmin, vmax = float(slc.min()), float(slc.max())
    if vmax > vmin:
        slc = (slc - vmin) / (vmax - vmin)
    else:
        slc = np.zeros_like(slc)

    return np.rot90(slc)


def get_seg_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Rotate segmentation slice."""
    if axis >= vol.ndim:
        raise ValueError(f"Axis {axis} out of range for volume shape {vol.shape}")

    slc = np.take(vol, index, axis=axis)
    slc = np.nan_to_num(slc)
    return np.rot90(slc)


def resize_slice_for_display(slc: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize image slice to (size, size)."""
    slc = np.clip(slc, 0.0, 1.0).astype(np.float32)
    img = Image.fromarray((slc * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_mask_for_display(mask: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize segmentation mask using nearest-neighbor."""
    mask_int = mask.astype(int)
    if mask_int.min() < 0:
        mask_int -= mask_int.min()

    mask_uint8 = np.clip(mask_int, 0, 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    img = img.resize((size, size), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8).astype(int)


def overlay_segmentation_multi(
    img_slice: np.ndarray,
    mask_slice: np.ndarray,
    label_colors: dict,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay multi-label segmentation."""
    img = np.clip(img_slice, 0.0, 1.0).astype(np.float32)
    base = (img * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)

    for lab, color in label_colors.items():
        if lab == 0:
            continue
        region = mask_slice == lab
        if not np.any(region):
            continue
        rgb[region] = (1.0 - alpha) * rgb[region] + alpha * np.array(color)

    return (rgb / 255.0).clip(0.0, 1.0)


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="MRI Sequence Viewer", layout="wide")
st.title("🧠 NeuroINK")

st.markdown(
    """
Upload MRI sequences (T1, T2, FLAIR, etc.) and an optional **segmentation mask**:

- **Single-plane mode**: view all sequences side by side, with mask overlaid  
- **Orthogonal mode**: view one sequence in **axial, coronal, sagittal** with mask overlay  
"""
)

with st.sidebar:
    st.header("Upload MRI Sequences")
    uploaded_files = st.file_uploader(
        "Select one or more files",
        type=["nii", "nii.gz", "npy", "npz", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    st.header("Segmentation (optional)")
    seg_file = st.file_uploader(
        "Upload segmentation mask",
        type=["nii", "nii.gz", "npy", "npz"],
        accept_multiple_files=False,
    )

    st.markdown("---")
    st.header("View Mode")
    view_mode = st.radio(
        "Choose how to view:",
        [
            "Single-plane (all sequences)",
            "Orthogonal (one sequence: axial/coronal/sagittal)",
        ],
        index=1,
    )

    view_plane = st.radio(
        "Single-plane view: choose plane",
        ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
        index=0,
        disabled=(view_mode != "Single-plane (all sequences)"),
    )


# ------------------------------
# Load volumes
# ------------------------------

volumes: Dict[str, np.ndarray] = {}

if uploaded_files:
    for f in uploaded_files:
        try:
            vol, label = load_volume(f)
            volumes[label] = vol
        except Exception as e:
            st.sidebar.error(f"Error loading {f.name}: {e}")

if not volumes:
    st.info("👈 Upload one or more MRI sequences from the sidebar to begin.")
    st.stop()


# ------------------------------
# Load segmentation (optional)
# ------------------------------

seg_volume = None
seg_label = None
label_colors = None

if seg_file is not None:
    try:
        seg_volume, seg_label = load_volume(seg_file)
    except Exception as e:
        st.sidebar.error(f"Error loading segmentation {seg_file.name}: {e}")
        seg_volume = None

# Build label → color mapping
if seg_volume is not None:
    labels = np.unique(seg_volume.astype(int))
    labels = labels[labels > 0]

    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 165, 0), (128, 0, 128), (0, 128, 128),
        (255, 192, 203),
    ]

    label_colors = {
        int(lab): palette[i % len(palette)]
        for i, lab in enumerate(labels)
    }

    st.sidebar.markdown("**Segmentation labels:**")
    for lab, col in label_colors.items():
        r, g, b = col
        st.sidebar.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background-color: rgb({r},{g},{b});margin-right:6px;'></span> Label {lab}",
            unsafe_allow_html=True,
        )


# ------------------------------
# Single-plane View
# ------------------------------

if view_mode == "Single-plane (all sequences)":
    axis_map = {"Axial (Z)": 2, "Coronal (Y)": 1, "Sagittal (X)": 0}
    axis = axis_map[view_plane]

    max_slices = max(vol.shape[axis] for vol in volumes.values())

    slice_index = st.slider(
        "Slice index",
        0,
        max_slices - 1,
        max_slices // 2,
        step=1,
    )

    cols = st.columns(len(volumes))

    for col, (name, vol) in zip(cols, volumes.items()):
        with col:
            slc = resize_slice_for_display(
                get_slice(vol, axis, slice_index)
            )

            if (
                seg_volume is not None
                and seg_volume.shape[:3] == vol.shape[:3]
                and label_colors is not None
            ):
                seg_slc = resize_mask_for_display(
                    get_seg_slice(seg_volume, axis, slice_index)
                )
                slc = overlay_segmentation_multi(
                    slc, seg_slc, label_colors
                )

            st.image(slc, caption=name, use_column_width=True)


# ------------------------------
# Orthogonal View
# ------------------------------

else:
    seq_name = st.selectbox("Select sequence", list(volumes.keys()))
    vol = volumes[seq_name]
    x, y, z = vol.shape[:3]

    scroll_pos = st.slider("Scroll", 0.0, 1.0, 0.5)

    idx_ax = int(scroll_pos * (z - 1))
    idx_cor = int(scroll_pos * (y - 1))
    idx_sag = int(scroll_pos * (x - 1))

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            resize_slice_for_display(get_slice(vol, 2, idx_ax)),
            caption=f"Axial {idx_ax}",
            use_column_width=True,
        )

    with c2:
        st.image(
            resize_slice_for_display(get_slice(vol, 1, idx_cor)),
            caption=f"Coronal {idx_cor}",
            use_column_width=True,
        )

    with c3:
        st.image(
            resize_slice_for_display(get_slice(vol, 0, idx_sag)),
            caption=f"Sagittal {idx_sag}",
            use_column_width=True,
        )
