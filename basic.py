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

        # Save to temporary file so nib.load() gets a real path
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
        # e.g. (X, Y, Z, T) -> first timepoint
        vol = vol[..., 0]

    return vol, label


def get_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Extract a single slice along `axis` at position `index`.
    Normalizes to [0, 1] and rotates for nicer viewing.
    """
    if axis >= vol.ndim:
        raise ValueError(f"Axis {axis} out of range for volume shape {vol.shape}")

    slc = np.take(vol, index, axis=axis)
    slc = np.nan_to_num(slc)

    vmin, vmax = float(slc.min()), float(slc.max())
    if vmax > vmin:
        slc_norm = (slc - vmin) / (vmax - vmin)
    else:
        slc_norm = np.zeros_like(slc)

    slc_norm = np.rot90(slc_norm)
    return slc_norm


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(
    page_title="MRI Sequence Viewer",
    layout="wide",
)

st.title("🧠 MRI Sequences Viewer")

st.markdown(
    """
Upload multiple MRI sequences (T1, T2, FLAIR, etc.) and:

- View **all sequences** in a single plane, or  
- View **one sequence** with **axial, coronal, and sagittal** slices side by side  
  and **scroll through the volume** with a single slider.
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
    st.header("View Mode")
    view_mode = st.radio(
        "Choose how to view:",
        options=[
            "Single-plane (all sequences)",
            "Orthogonal (one sequence: axial/coronal/sagittal)",
        ],
        index=1,
    )

    # Only used in single-plane mode
    view_plane = st.radio(
        "Single-plane view: choose plane",
        options=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
        index=0,
        disabled=(view_mode != "Single-plane (all sequences)"),
    )


# ------------------------------
# Load volumes
# ------------------------------

volumes: Dict[str, np.ndarray] = {}

if uploaded_files:
    for f in uploaded_files:
        f.seek(0)
        try:
            vol, label = load_volume(f)
            volumes[label] = vol
        except Exception as e:
            st.sidebar.error(f"Error loading {f.name}: {e}")

if not volumes:
    st.info("👈 Upload one or more MRI sequences from the sidebar to begin.")
    st.stop()

# ------------------------------
# View mode: Single-plane (all sequences)
# ------------------------------
if view_mode == "Single-plane (all sequences)":
    axis_map = {
        "Axial (Z)": 2,
        "Coronal (Y)": 1,
        "Sagittal (X)": 0,
    }
    axis = axis_map[view_plane]

    # Determine max slices along chosen axis
    max_slices = 0
    for vol in volumes.values():
        if vol.ndim >= 3 and axis < vol.ndim:
            max_slices = max(max_slices, vol.shape[axis])

    if max_slices == 0:
        st.error("Could not determine slice dimension for the chosen plane.")
        st.stop()

    slice_index = st.slider(
        "Slice index",
        min_value=0,
        max_value=max_slices - 1,
        value=max_slices // 2,
        step=1,
    )

    st.caption(
        f"Showing **slice {slice_index}** along axis **{axis}** "
        f"({view_plane.split()[0].lower()}) for all sequences."
    )

    cols = st.columns(len(volumes))

    for col, (name, vol) in zip(cols, volumes.items()):
        with col:
            st.subheader(name, help=str(vol.shape))
            try:
                slc = get_slice(vol, axis=axis, index=slice_index)
                st.image(
                    slc,
                    caption=f"{name} — {view_plane.split()[0]} slice {slice_index}",
                    use_column_width=True,
                    clamp=True,
                )
            except Exception as e:
                st.error(f"Error displaying {name}: {e}")

# ------------------------------
# View mode: Orthogonal (one sequence: axial/coronal/sagittal)
# ------------------------------
else:
    st.subheader("Orthogonal View: Axial • Coronal • Sagittal")

    seq_name = st.selectbox(
        "Select sequence to view",
        options=list(volumes.keys()),
        index=0,
    )

    vol = volumes[seq_name]
    if vol.ndim < 3:
        st.error(f"Selected volume {seq_name} has shape {vol.shape}, need at least 3D.")
        st.stop()

    x, y, z = vol.shape[:3]
    st.caption(f"Volume **{seq_name}** shape: (X={x}, Y={y}, Z={z})")

    # Single scroll slider (0–100%) mapped to indices in each dimension
    scroll_pos = st.slider(
        "Scroll through volume (0–100%)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    idx_axial = int(scroll_pos * (z - 1))     # axis=2
    idx_coronal = int(scroll_pos * (y - 1))   # axis=1
    idx_sagittal = int(scroll_pos * (x - 1))  # axis=0

    st.caption(
        f"Slice indices — Axial (Z): {idx_axial}, Coronal (Y): {idx_coronal}, "
        f"Sagittal (X): {idx_sagittal}"
    )

    col_ax, col_cor, col_sag = st.columns(3)

    # Axial (Z)
    with col_ax:
        st.markdown("**Axial (Z)**")
        try:
            slc_ax = get_slice(vol, axis=2, index=idx_axial)
            st.image(
                slc_ax,
                caption=f"Axial slice {idx_axial}",
                use_column_width=True,
                clamp=True,
            )
        except Exception as e:
            st.error(f"Error axial view: {e}")

    # Coronal (Y)
    with col_cor:
        st.markdown("**Coronal (Y)**")
        try:
            slc_cor = get_slice(vol, axis=1, index=idx_coronal)
            st.image(
                slc_cor,
                caption=f"Coronal slice {idx_coronal}",
                use_column_width=True,
                clamp=True,
            )
        except Exception as e:
            st.error(f"Error coronal view: {e}")

    # Sagittal (X)
    with col_sag:
        st.markdown("**Sagittal (X)**")
        try:
            slc_sag = get_slice(vol, axis=0, index=idx_sagittal)
            st.image(
                slc_sag,
                caption=f"Sagittal slice {idx_sagittal}",
                use_column_width=True,
                clamp=True,
            )
        except Exception as e:
            st.error(f"Error sagittal view: {e}")
