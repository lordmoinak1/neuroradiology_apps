import io
import os
import tempfile
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

try:
    import nibabel as nib
except ImportError:
    nib = None

# ==============================
# Utility functions
# ==============================

def load_volume(file) -> Tuple[np.ndarray, str]:
    filename = file.name.lower()
    label = file.name

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
            os.remove(tmp_path)

    elif filename.endswith(".npy"):
        vol = np.load(io.BytesIO(file.read())).astype(np.float32)

    elif filename.endswith(".npz"):
        data = np.load(io.BytesIO(file.read()))
        key = list(data.keys())[0]
        vol = data[key].astype(np.float32)
        label = f"{filename} ({key})"

    else:
        pil_img = Image.open(io.BytesIO(file.read())).convert("L")
        vol = np.array(pil_img, dtype=np.float32)[..., None]

    if vol.ndim == 2:
        vol = vol[..., None]
    elif vol.ndim > 3:
        vol = vol[..., 0]

    return vol, label


def get_slice(vol, axis, index):
    slc = np.take(vol, index, axis=axis)
    slc = np.nan_to_num(slc)
    vmin, vmax = float(slc.min()), float(slc.max())
    if vmax > vmin:
        slc = (slc - vmin) / (vmax - vmin)
    else:
        slc = np.zeros_like(slc)
    return np.rot90(slc)


def resize_display(img, size=256):
    img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def overlay_slice_number(img_arr, text):
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, -1)
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.rectangle([5, 5, 120, 30], fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(img).astype(np.float32) / 255.0


# ==============================
# Mouse + Drag + Autoplay JS
# ==============================

def inject_controls_js(key, min_val, max_val, step=1):
    val = st.session_state.get(key, (min_val + max_val) // 2)
    js = """
    <script>
    (() => {{
        let v = {val};
        document.addEventListener("wheel", e => {{
            v += Math.sign(e.deltaY) * {step};
            v = Math.max({min_val}, Math.min({max_val}, v));
            window.parent.postMessage({{type:"streamlit:setSessionState",key:"{key}",value:v}},"*");
            e.preventDefault();
        }}, {{passive:false}});

        let lastY = null;
        document.addEventListener("pointerdown", e => lastY = e.clientY);
        document.addEventListener("pointermove", e => {{
            if(lastY === null) return;
            let dy = e.clientY - lastY;
            if(Math.abs(dy) > 5){{
                v -= Math.sign(dy) * {step};
                v = Math.max({min_val}, Math.min({max_val}, v));
                window.parent.postMessage({{type:"streamlit:setSessionState",key:"{key}",value:v}},"*");
                lastY = e.clientY;
            }}
        }});
        document.addEventListener("pointerup", e => lastY = null);
    }})();
    </script>
    """
    components.html(js, height=0)


def inject_autoplay_js(key, min_val, max_val, step, interval):
    js = """
    <script>
    let v = {st.session_state[key]};
    setInterval(() => {{
        v = (v + {step}) % ({max_val}+1);
        window.parent.postMessage({{type:"streamlit:setSessionState",key:"{key}",value:v}},"*");
    }}, {interval});
    </script>
    """
    components.html(js, height=0)

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="NeuroINK MRI Viewer", layout="wide")
st.title("🧠 NeuroINK MRI Viewer")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload MRI volumes", type=["nii", "nii.gz", "npy", "npz"], accept_multiple_files=True)
    view_mode = st.radio("View Mode", ["Single-plane", "Orthogonal"])

volumes = {}
if uploaded_files:
    for f in uploaded_files:
        vol, label = load_volume(f)
        volumes[label] = vol

if not volumes:
    st.stop()

# ==============================
# SINGLE-PLANE MODE
# ==============================

if view_mode == "Single-plane":
    axis = 2
    vol0 = list(volumes.values())[0]
    max_slices = vol0.shape[axis]

    if "slice_scroll" not in st.session_state:
        st.session_state["slice_scroll"] = max_slices // 2
    if "playing" not in st.session_state:
        st.session_state["playing"] = False

    inject_controls_js("slice_scroll", 0, max_slices - 1)

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state["playing"]:
            if st.button("⏸ Stop"):
                st.session_state["playing"] = False
        else:
            if st.button("▶ Play"):
                st.session_state["playing"] = True
    with col2:
        speed = st.selectbox("Speed (ms)", [50, 100, 150, 250], index=1)

    if st.session_state["playing"]:
        inject_autoplay_js("slice_scroll", 0, max_slices - 1, 1, speed)

    slice_idx = st.slider("Slice", 0, max_slices - 1, st.session_state["slice_scroll"], key="slice_scroll")

    cols = st.columns(len(volumes))
    for col, (name, vol) in zip(cols, volumes.items()):
        with col:
            slc = get_slice(vol, 2, slice_idx)
            img = resize_display(slc)
            img = overlay_slice_number(img, f"Slice {slice_idx}")
            st.image(img, caption=name, use_column_width=True)

# ==============================
# ORTHOGONAL MODE
# ==============================

else:
    seq_name = st.selectbox("Select volume", list(volumes.keys()))
    vol = volumes[seq_name]
    x, y, z = vol.shape

    if "ortho_scroll" not in st.session_state:
        st.session_state["ortho_scroll"] = 50
    if "ortho_play" not in st.session_state:
        st.session_state["ortho_play"] = False

    inject_controls_js("ortho_scroll", 0, 100)

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state["ortho_play"]:
            if st.button("⏸ Stop Ortho"):
                st.session_state["ortho_play"] = False
        else:
            if st.button("▶ Play Ortho"):
                st.session_state["ortho_play"] = True
    with col2:
        speed = st.selectbox("Speed (ms)", [50, 100, 150, 250], index=1, key="ortho_speed")

    if st.session_state["ortho_play"]:
        inject_autoplay_js("ortho_scroll", 0, 100, 1, speed)

    scroll = st.slider("Scroll Volume", 0, 100, st.session_state["ortho_scroll"], key="ortho_scroll")
    pos = scroll / 100.0

    idx_ax = int(pos * (z - 1))
    idx_cor = int(pos * (y - 1))
    idx_sag = int(pos * (x - 1))

    c1, c2, c3 = st.columns(3)

    with c1:
        img = resize_display(get_slice(vol, 2, idx_ax))
        img = overlay_slice_number(img, f"Axial {idx_ax}")
        st.image(img, use_column_width=True)

    with c2:
        img = resize_display(get_slice(vol, 1, idx_cor))
        img = overlay_slice_number(img, f"Coronal {idx_cor}")
        st.image(img, use_column_width=True)

    with c3:
        img = resize_display(get_slice(vol, 0, idx_sag))
        img = overlay_slice_number(img, f"Sagittal {idx_sag}")
        st.image(img, use_column_width=True)
