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

    # ------------------------
    # NIfTI (.nii / .nii.gz)
    # ------------------------
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("Please install nibabel for NIfTI support.")

        # Decide suffix for temp file
        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"

        # Write uploaded bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            img = nib.load(tmp_path)
            vol = img.get_fdata().astype(np.float32)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------
    # NumPy .npy
    # ------------------------
    elif filename.endswith(".npy"):
        raw = file.read()
        bio = io.BytesIO(raw)
        vol = np.load(bio).astype(np.float32)

    # ------------------------
    # NumPy .npz  (take first array)
    # ------------------------
    elif filename.endswith(".npz"):
        raw = file.read()
        bio = io.BytesIO(raw)
        data = np.load(bio)
        key = list(data.keys())[0]
        vol = data[key].astype(np.float32)
        label = f"{filename} ({key})"

    # ------------------------
    # Image files (PNG/JPG/…)
    # ------------------------
    else:
        raw = file.read()
        bio = io.BytesIO(raw)
        pil_img = Image.open(bio).convert("L")
        arr = np.array(pil_img, dtype=np.float32)
        vol = arr[..., None]  # (H, W, 1)

    # ------------------------
    # Ensure (X, Y, Z) volume
    # ------------------------
    if vol.ndim == 2:
        vol = vol[..., None]
    elif vol.ndim > 3:
        # e.g. (X, Y, Z, T) -> take first timepoint
        vol = vol[..., 0]

    return vol, label

def get_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Extract a single slice along `axis` at position `index`.
    Normalizes to [0, 1] for display.
    """
    # Take slice along chosen axis
    slc = np.take(vol, index, axis=axis)

    # Handle NaNs/Infs
    slc = np.nan_to_num(slc)

    # Min-max normalize per slice
    vmin, vmax = float(slc.min()), float(slc.max())
    if vmax > vmin:
        slc_norm = (slc - vmin) / (vmax - vmin)
    else:
        slc_norm = np.zeros_like(slc)

    # Rotate for nicer orientation (optional)
    slc_norm = np.rot90(slc_norm)

    return slc_norm


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(
    page_title="MRI Sequence Viewer",
    layout="wide",
)

st.title("🧠 MRI Sequences Side-by-Side Viewer")

st.markdown(
    """
Upload multiple MRI sequences (e.g., T1, T2, FLAIR) as **NIfTI (.nii / .nii.gz)**, 
**NumPy (.npy / .npz)**, or **image files (PNG/JPG)** to visualize them side by side.

- Use the **sidebar** to upload files.
- Choose **view plane** (axial/coronal/sagittal).
- Move the **slice slider** to scroll through the volume.
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
    st.header("View Settings")
    view_plane = st.radio(
        "View plane",
        options=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
        index=0,
    )
    # Map view name to axis index
    axis_map = {
        "Axial (Z)": 2,
        "Coronal (Y)": 1,
        "Sagittal (X)": 0,
    }
    axis = axis_map[view_plane]


# ------------------------------
# Load volumes
# ------------------------------

volumes: Dict[str, np.ndarray] = {}

if uploaded_files:
    for f in uploaded_files:
        try:
            f.seek(0)
            vol, label = load_volume(f)
            volumes[label] = vol
        except Exception as e:
            st.sidebar.error(f"Error loading {f.name}: {e}")

if not volumes:
    st.info("👈 Upload one or more MRI sequences from the sidebar to begin.")
    st.stop()

# Ensure all volumes have at least 3D shape
shapes = {name: vol.shape for name, vol in volumes.items()}
min_dim = min(len(s) for s in shapes.values())
if min_dim < 3:
    st.warning("Some volumes have fewer than 3 dimensions; they will be treated accordingly.")

# ------------------------------
# Slice selection
# ------------------------------

# Get max number of slices along chosen axis
max_slices = 0
for vol in volumes.values():
    if vol.ndim >= 3:
        if axis < vol.ndim:
            max_slices = max(max_slices, vol.shape[axis])

if max_slices == 0:
    st.error("Could not determine slice dimension. Check your volumes.")
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
    f"({view_plane.split()[0].lower()})"
)

# ------------------------------
# Display side-by-side
# ------------------------------

n_vols = len(volumes)
cols = st.columns(n_vols)

for col, (name, vol) in zip(cols, volumes.items()):
    with col:
        st.subheader(name, help=str(vol.shape))

        try:
            if axis >= vol.ndim:
                st.error(f"Volume has shape {vol.shape}, cannot slice along axis {axis}.")
                continue

            slc = get_slice(vol, axis=axis, index=slice_index)
            st.image(slc, caption=f"{name} — slice {slice_index}", use_column_width=True, clamp=True)
        except Exception as e:
            st.error(f"Error displaying {name}: {e}")
