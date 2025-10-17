# app.py
import streamlit as st
import numpy as np
import cv2
from typing import Tuple

st.set_page_config(layout="wide", page_title="Image Enhancing demo")

# ---------- Utilities] ----------
def to_rgb(cv_img: np.ndarray) -> np.ndarray:
    if cv_img.ndim == 2:
        return cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

def read_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img
#----Denoise-------
def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    img_f = img.astype(np.float32)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noised = img_f + noise
    noised = np.clip(noised, 0, 255).astype(np.uint8)
    return noised

def mean_denoise(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return img.copy()
    return cv2.blur(img, (ksize, ksize))

def median_denoise(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return img.copy()
    # medianBlur requires single channel or 3-channel with odd k
    k = ksize if ksize % 2 == 1 else ksize + 1
    if k < 3: k = 3
    return cv2.medianBlur(img, k)
#----Sharpening-------
def sharp_with_kernel(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]], dtype=np.float32)
    sharp = cv2.filter2D(img, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp

def unsharp_mask(img: np.ndarray, amount: float, radius: float) -> np.ndarray:
    if amount <= 0:
        return img.copy()
    # Use Gaussian blur for mask
    blurred = cv2.GaussianBlur(img, (0,0), radius)
    sharp = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp
#----Edge Detection-------
def sobel_edges(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.uint8(255 * (mag / (mag.max()+1e-8)))
    return mag

def prewitt_edges(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[ -1, 0, 1],
                   [ -1, 0, 1],
                   [ -1, 0, 1]], dtype=np.float32)
    ky = np.array([[  1,  1,  1],
                   [  0,  0,  0],
                   [ -1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
    gy = cv2.filter2D(gray.astype(np.float32), -1, ky)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.uint8(255 * (mag / (mag.max()+1e-8)))
    return mag

def canny_edges(gray: np.ndarray, t1: int, t2: int) -> np.ndarray:
    edges = cv2.Canny(gray, t1, t2)
    return edges

# ---------- UI ----------
st.title("Image Enhancing demo")

uploaded = st.file_uploader("Upload một ảnh màu", type=["png","jpg","jpeg","bmp","tif","tiff"])
if uploaded is None:
    st.info("Vui lòng upload một ảnh để bắt đầu.")
    st.stop()

img_bgr = read_image(uploaded)
if img_bgr is None:
    st.error("Không đọc được ảnh. Thử file khác.")
    st.stop()

img_rgb = to_rgb(img_bgr)
img_display = img_rgb.copy()

# Sidebar: global settings
st.sidebar.header("Noise & Denoise")
noise_sigma = st.sidebar.slider("Gaussian noise σ (pixel std)", 0.0, 100.0, 20.0, step=1.0)
mean_ks = st.sidebar.slider("Mean filter kernel size (odd)", 1, 31, 5, step=2)
median_ks = st.sidebar.slider("Median filter kernel size (odd)", 1, 31, 5, step=2)

st.sidebar.header("Sharpening")
sharp_choice = st.sidebar.selectbox("Sharpen method", ["Unsharp Mask", "Sharpen with Kernel"])
if sharp_choice == "Unsharp Mask":
    sharp_amount = st.sidebar.slider("Sharpen amount (unsharp mask)", 0.0, 2.0, 0.8, step=0.05)
    sharp_radius = st.sidebar.slider("Sharpen radius (Gaussian σ)", 0.5, 10.0, 1.0, step=0.1)


st.sidebar.header("Edge Detection")
canny_t1 = st.sidebar.slider("Canny threshold1", 0, 255, 50)
canny_t2 = st.sidebar.slider("Canny threshold2", 0, 255, 150)

# ---------- Denoise section ----------
st.subheader("1. Denoising / Smoothing")
col1, col2, col3, col4 = st.columns(4)
noised = add_gaussian_noise(img_bgr, noise_sigma)
mean_dn = mean_denoise(noised, mean_ks)
median_dn = median_denoise(noised, median_ks)

with col1:
    st.image(to_rgb(img_bgr), caption="Original", 
use_container_width
=True)
with col2:
    st.image(to_rgb(noised), caption=f"Noisy (σ={noise_sigma})", 
use_container_width
=True)
with col3:
    st.image(to_rgb(mean_dn), caption=f"Mean Denoised (k={mean_ks})", 
use_container_width
=True)
with col4:
    st.image(to_rgb(median_dn), caption=f"Median Denoised (k={median_ks})", 
use_container_width
=True)

# ---------- Sharpening ----------
st.subheader("2. Sharpening")
col1, col2 = st.columns(2)
if( sharp_choice == "Sharpen with Kernel"):
    sharpened = sharp_with_kernel(img_bgr)
    caption = f"Sharpened with Kernel (size=3)"
else:
    sharpened = unsharp_mask(img_bgr, sharp_amount, sharp_radius)
    caption = f"Sharpened (amount={sharp_amount}, radius={sharp_radius})"

with col1:
    st.image(to_rgb(img_bgr), caption="Original", 
use_container_width
=True)
with col2:
    st.image(to_rgb(sharpened), caption=caption, 
use_container_width
=True)

# ---------- Edge detection ----------
st.subheader("3. Edge Detection Filters")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
sob = sobel_edges(gray)
pre = prewitt_edges(gray)
can = canny_edges(gray, canny_t1, canny_t2)

c1, c2, c3 = st.columns(3)
with c1:
    st.image(sob, caption="Sobel (magnitude)", 
use_container_width
=True)
with c2:
    st.image(pre, caption="Prewitt (magnitude)", 
use_container_width
=True)
with c3:
    st.image(can, caption=f"Canny (t1={canny_t1}, t2={canny_t2})", 
use_container_width
=True)


