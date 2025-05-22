import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def peregangan_kontras(img, r1, s1, r2, s2):
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i, j]
            if r < r1:
                result[i, j] = int(s1 / r1 * r) if r1 > 0 else 0
            elif r <= r2:
                result[i, j] = int(((s2 - s1) / (r2 - r1)) * (r - r1) + s1) if r2 > r1 else s1
            else:
                result[i, j] = int(((255 - s2) / (255 - r2)) * (r - r2) + s2) if r2 < 255 else s2
    return np.clip(result, 0, 255).astype(np.uint8)

def compute_lut(r1, s1, r2, s2):
    lut = np.zeros(256)
    for x in range(256):
        if x < r1:
            lut[x] = (s1 / r1) * x if r1 > 0 else 0
        elif x <= r2:
            lut[x] = ((s2 - s1) / (r2 - r1)) * (x - r1) + s1 if r2 > r1 else s1
        else:
            lut[x] = ((255 - s2) / (255 - r2)) * (x - r2) + s2 if r2 < 255 else s2
    return lut

st.markdown("<h3 style='text-align: center;'>Peregangan Kontras</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="small")
with col1:
    st.markdown("<h6 style='text-align: left; margin-bottom:8px;'>Gambar Asli</h6>", unsafe_allow_html=True)
    img_placeholder = st.empty()

with col2:
    st.markdown("<h6 style='text-align: left; margin-bottom:8px;'>Hasil Proses Peregangan Kontras</h6>", unsafe_allow_html=True)
    result_placeholder = st.empty()

col_kiri, col_kanan = st.columns([1, 1.2], gap="small")
with col_kiri:
    uploaded_file = st.file_uploader("Input File Gambar", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"])
    param_cols = st.columns(4)
    with param_cols[0]:
        st.markdown("<div style='text-align:center;font-size:13px;'>Xmin</div>", unsafe_allow_html=True)
        r1 = st.slider("r1", 0, 255, 50, key="r1_slider", label_visibility="collapsed")
    with param_cols[1]:
        st.markdown("<div style='text-align:center;font-size:13px;'>Ymin</div>", unsafe_allow_html=True)
        s1 = st.slider("s1", 0, 255, 0, key="s1_slider", label_visibility="collapsed")
    with param_cols[2]:
        st.markdown("<div style='text-align:center;font-size:13px;'>Xmax</div>", unsafe_allow_html=True)
        r2 = st.slider("r2", 0, 255, 200, key="r2_slider", label_visibility="collapsed")
    with param_cols[3]:
        st.markdown("<div style='text-align:center;font-size:13px;'>Ymax</div>", unsafe_allow_html=True)
        s2 = st.slider("s2", 0, 255, 255, key="s2_slider", label_visibility="collapsed")

with col_kanan:
    st.markdown("<p style='text-align: center; margin-bottom:8px;'>Grafik Peregangan Kontras</p>", unsafe_allow_html=True)
    graph_placeholder = st.empty()
    st.markdown("<div style='height:180px'></div>", unsafe_allow_html=True)

lut = compute_lut(r1, s1, r2, s2)
fig, ax = plt.subplots(figsize=(4, 2.1))
ax.plot([0, 255], [0, 255], 'r--', linewidth=1, label='y=x')
ax.plot(range(256), lut, 'b-')
ax.grid(True)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_xlabel("Intensitas Input", fontsize=8)
ax.set_ylabel("Intensitas Output", fontsize=8)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7)
graph_placeholder.pyplot(fig, use_container_width=True)

img_placeholder.markdown(
    "<div style='display: flex; justify-content: center; align-items: center; height: 160px;'><p>Upload gambar dulu</p></div>",
    unsafe_allow_html=True
)
result_placeholder.markdown(
    "<div style='display: flex; justify-content: center; align-items: center; height: 160px;'><p>Hasilnya disini</p></div>",
    unsafe_allow_html=True
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_display = img
    img_placeholder.image(img_display, width=220)
    if len(img.shape) == 3:
        result_b = peregangan_kontras(img[:,:,0], r1, s1, r2, s2)
        result_g = peregangan_kontras(img[:,:,1], r1, s1, r2, s2)
        result_r = peregangan_kontras(img[:,:,2], r1, s1, r2, s2)
        transformed_array = cv2.merge([result_b, result_g, result_r])
    else:
        transformed_array = peregangan_kontras(img, r1, s1, r2, s2)
    if len(transformed_array.shape) == 3:
        transformed_img = cv2.cvtColor(transformed_array, cv2.COLOR_BGR2RGB)
    else:
        transformed_img = transformed_array
    result_placeholder.image(transformed_img, width=220)
