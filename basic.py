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


def resize_slice_for_display(slc: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Resize a slice to (size, size) so all views appear the same size.
    Input/Output are float32 in [0, 1].
    """
    slc = np.clip(slc, 0.0, 1.0).astype(np.float32)
    img = Image.fromarray((slc * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_mask_for_display(mask: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Resize a binary/label mask to (size, size) using nearest-neighbor.
    Returns a bool mask (True where label > 0).
    """
    mask_bin = (mask > 0).astype(np.uint8)
    img = Image.fromarray(mask_bin * 255)
    img = img.resize((size, size), Image.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    return arr > 0


def overlay_segmentation(
    img_slice: np.ndarray,
    mask_slice: np.ndarray,
    color=(255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a segmentation mask on a grayscale image.
    img_slice: (H, W) float [0,1]
    mask_slice: (H, W) bool or 0/1
    Returns RGB float [0,1].
    """
    img = np.clip(img_slice, 0.0, 1.0).astype(np.float32)
    base_gray = (img * 255).astype(np.uint8)
    h, w = base_gray.shape

    rgb = np.stack([base_gray] * 3, axis=-1).astype(np.float32)
    mask = mask_slice.astype(bool)

    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)

    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color_arr
    return (rgb / 255.0).clip(0.0, 1.0)


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(
    page_title="MRI Sequence Viewer",
    layout="wide",
)

st.title("🧠 MRI Sequences Viewer with Segmentation")

st.markdown(
    """
Upload MRI sequences (T1, T2, FLAIR, etc.) and an optional **segmentation mask**:

- **Single-plane mode**: view all sequences side by side, with mask overlaid where shape matches  
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
        "Upload segmentation mask (same space/shape as your MRI)",
        type=["nii", "nii.gz", "npy", "npz"],
        accept_multiple_files=False,
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

# Load segmentation (if provided)
seg_volume = None
seg_label = None
if seg_file is not None:
    seg_file.seek(0)
    try:
        seg_volume, seg_label = load_volume(seg_file)
    except Exception as e:
        st.sidebar.error(f"Error loading segmentation {seg_file.name}: {e}")
        seg_volume = None

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
                slc = resize_slice_for_display(slc, size=256)

                # Overlay segmentation if shapes match
                if seg_volume is not None and seg_volume.shape[:3] == vol.shape[:3]:
                    seg_slc = get_slice(seg_volume, axis=axis, index=slice_index)
                    seg_mask = resize_mask_for_display(seg_slc, size=256)
                    img_disp = overlay_segmentation(slc, seg_mask, color=(255, 0, 0), alpha=0.4)
                    st.image(
                        img_disp,
                        caption=f"{name} + seg — {view_plane.split()[0]} slice {slice_index}",
                        use_column_width=True,
                        clamp=True,
                    )
                else:
                    if seg_volume is not None:
                        st.caption("⚠ Seg shape mismatch; not overlaying.")
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

    # Check seg compatibility
    use_seg_here = False
    if seg_volume is not None:
        if seg_volume.shape[:3] == vol.shape[:3]:
            use_seg_here = True
        else:
            st.warning(
                f"Segmentation shape {seg_volume.shape} does not match {vol.shape}; "
                "overlay disabled in orthogonal view."
            )

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

    # ------- Axial (Z) -------
    with col_ax:
        st.markdown("**Axial (Z)**")
        try:
            slc_ax = get_slice(vol, axis=2, index=idx_axial)
            slc_ax = resize_slice_for_display(slc_ax, size=256)

            if use_seg_here:
                seg_ax = get_slice(seg_volume, axis=2, index=idx_axial)
                seg_ax_mask = resize_mask_for_display(seg_ax, size=256)
                img_ax = overlay_segmentation(slc_ax, seg_ax_mask, color=(255, 0, 0), alpha=0.4)
                st.image(
                    img_ax,
                    caption=f"Axial slice {idx_axial} + seg",
                    use_column_width=True,
                    clamp=True,
                )
            else:
                st.image(
                    slc_ax,
                    caption=f"Axial slice {idx_axial}",
                    use_column_width=True,
                    clamp=True,
                )
        except Exception as e:
            st.error(f"Error axial view: {e}")

    # ------- Coronal (Y) -------
    with col_cor:
        st.markdown("**Coronal (Y)**")
        try:
            slc_cor = get_slice(vol, axis=1, index=idx_coronal)
            slc_cor = resize_slice_for_display(slc_cor, size=256)

            if use_seg_here:
                seg_cor = get_slice(seg_volume, axis=1, index=idx_coronal)
                seg_cor_mask = resize_mask_for_display(seg_cor, size=256)
                img_cor = overlay_segmentation(slc_cor, seg_cor_mask, color=(255, 0, 0), alpha=0.4)
                st.image(
                    img_cor,
                    caption=f"Coronal slice {idx_coronal} + seg",
                    use_column_width=True,
                    clamp=True,
                )
            else:
                st.image(
                    slc_cor,
                    caption=f"Coronal slice {idx_coronal}",
                    use_column_width=True,
                    clamp=True,
                )
        except Exception as e:
            st.error(f"Error coronal view: {e}")

    # ------- Sagittal (X) -------
    with col_sag:
        st.markdown("**Sagittal (X)**")
        try:
            slc_sag = get_slice(vol, axis=0, index=idx_sagittal)
            slc_sag = resize_slice_for_display(slc_sag, size=256)

            if use_seg_here:
                seg_sag = get_slice(seg_volume, axis=0, index=idx_sagittal)
                seg_sag_mask = resize_mask_for_display(seg_sag, size=256)
                img_sag = overlay_segmentation(slc_sag, seg_sag_mask, color=(255, 0, 0), alpha=0.4)
                st.image(
                    img_sag,
                    caption=f"Sagittal slice {idx_sagittal} + seg",
                    use_column_width=True,
                    clamp=True,
                )
            else:
                st.image(
                    slc_sag,
                    caption=f"Sagittal slice {idx_sagittal}",
                    use_column_width=True,
                    clamp=True,
                )
        except Exception as e:
            st.error(f"Error sagittal view: {e}")
