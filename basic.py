import os
import tempfile
from collections import defaultdict

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
SEG_TAG = "-seg"


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
    return np.rot90(slc, k=2)  # 180° rotation


def resize_img(img, size=256):
    im = Image.fromarray((img * 255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im) / 255.0


def resize_mask(mask, size=256):
    im = Image.fromarray(mask.astype(np.uint8), "L")
    im = im.resize((size, size), Image.NEAREST)
    return np.asarray(im).astype(int)


def overlay_segmentation(img, mask, alpha=0.4):
    """
    Overlay segmentation mask on grayscale image.
    """
    base = (img * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)

    # Red overlay for tumor
    tumor = mask > 0
    rgb[tumor] = (1 - alpha) * rgb[tumor] + alpha * np.array([255, 0, 0])

    return np.clip(rgb / 255.0, 0, 1)


def calculate_spacing(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def calculate_total_volume(seg, affine):
    spacing = calculate_spacing(affine)
    voxel_volume = np.prod(spacing)
    voxels = int(np.sum(seg > 0))
    return voxels * voxel_volume


def calculate_per_lesion_volumes(seg, affine):
    spacing = calculate_spacing(affine)
    voxel_volume = np.prod(spacing)

    structure = np.ones((3, 3, 3), dtype=int)
    labeled, num = cc_label(seg > 0, structure=structure)

    lesion_volumes = {}
    for lab in range(1, num + 1):
        voxels = np.sum(labeled == lab)
        lesion_volumes[lab] = voxels * voxel_volume

    return lesion_volumes


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroINK Longitudinal Viewer", layout="wide")
st.title("🧠 NeuroINK — BraTS-GLI Longitudinal Viewer")

files = st.file_uploader(
    "Upload all BraTS-GLI .nii.gz files (MRI + seg)",
    type=["gz"],
    accept_multiple_files=True,
)

if not files:
    st.stop()


# ==============================
# Group files by timepoint
# ==============================

timepoints = defaultdict(lambda: {"modalities": {}, "seg": None})

for f in files:
    name = f.name.lower()

    # SEGMENTATION
    if name.endswith("-seg.nii.gz"):
        tp_id = name.replace("-seg.nii.gz", "")
        timepoints[tp_id]["seg"] = f
        continue

    # MRI MODALITIES
    matched = False
    for tag, mod in MODALITY_MAP.items():
        if tag in name:
            tp_id = name.split(tag)[0]
            timepoints[tp_id]["modalities"][mod] = f
            matched = True
            break

    if not matched:
        st.warning(f"Ignored file (unknown modality): {f.name}")


# ==============================
# Layout: Viewer | Metrics
# ==============================

viewer_col, metrics_col = st.columns([3, 1])
metrics_data = []


# ==============================
# LEFT: Viewer
# ==============================

with viewer_col:
    axis = 2

    for tp_idx, (tp, data) in enumerate(sorted(timepoints.items()), start=1):
        st.markdown("---")
        st.subheader(f"⏱ Timepoint {tp_idx}: {tp}")

        mods = data["modalities"]
        seg_file = data["seg"]

        missing = [m for m in MODALITY_ORDER if m not in mods]
        if missing:
            st.error(f"Missing modalities: {', '.join(missing)}")
            continue

        volumes = {}
        affine_ref = None

        for mod in MODALITY_ORDER:
            vol, affine = load_nifti(mods[mod])
            volumes[mod] = vol
            affine_ref = affine

        seg = None
        seg_affine = None

        if seg_file:
            seg, seg_affine = load_nifti(seg_file)

        max_slices = min(v.shape[axis] for v in volumes.values())
        slice_idx = st.slider(
            f"Axial slice — {tp}",
            0,
            max_slices - 1,
            max_slices // 2,
            key=f"slice_{tp}",
        )

        cols = st.columns(4)
        for col, mod in zip(cols, MODALITY_ORDER):
            with col:
                img = resize_img(get_slice(volumes[mod], axis, slice_idx))

                if seg is not None and seg.shape == volumes[mod].shape:
                    seg_slc = resize_mask(
                        get_slice(seg, axis, slice_idx)
                    )
                    img = overlay_segmentation(img, seg_slc)

                st.image(img, caption=mod, use_column_width=True)

        # ---- Metrics computation ----
        if seg is not None:
            total_vol = calculate_total_volume(seg, seg_affine)
            per_lesion = calculate_per_lesion_volumes(seg, seg_affine)

            metrics_data.append(
                {
                    "timepoint": tp,
                    "total_volume": total_vol,
                    "lesions": len(per_lesion),
                    "per_lesion": per_lesion,
                }
            )
        else:
            metrics_data.append(
                {
                    "timepoint": tp,
                    "total_volume": None,
                    "lesions": None,
                    "per_lesion": None,
                }
            )


# ==============================
# RIGHT: Metrics Pane
# ==============================

with metrics_col:
    st.subheader("📊 Timepoint Metrics")

    for i, m in enumerate(metrics_data, start=1):
        st.markdown(f"### ⏱ Timepoint {i}")
        st.markdown(f"`{m['timepoint']}`")

        if m["total_volume"] is None:
            st.markdown("_No segmentation found_")
            st.markdown("---")
            continue

        st.markdown(f"- **Total volume:** {m['total_volume']:,.2f} mm³")
        st.markdown(f"- **Number of lesions:** {m['lesions']}")

        st.markdown("**Per-lesion volumes (mm³):**")
        for lab, vol in sorted(
            m["per_lesion"].items(), key=lambda x: x[1], reverse=True
        ):
            st.markdown(f"• Lesion {lab}: {vol:,.2f}")

        st.markdown("---")
