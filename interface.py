import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageChops, ImageEnhance
import requests
from io import BytesIO
from torchvision import models
import torch.nn as nn
import pandas as pd
import altair as alt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import torchvision.transforms.functional as TF
from base64 import b64encode

# ----------------------------------
# Streamlit page configuration
st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# ----------------------------------
# Sidebar configuration for parameters
st.sidebar.header("Settings")
fake_threshold = st.sidebar.slider("Fake Detection Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
ela_quality = st.sidebar.slider("ELA Quality", min_value=10, max_value=100, value=90, step=5)

# ----------------------------------
# Device configuration (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Running on: {device}")

# ----------------------------------
# Download link helper for processed images
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# ----------------------------------
# Load Model with caching and device support
@st.cache_resource
def load_model():
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 1)
    )
    # Load the model state. Adjust "model_finetuned.pth" path if needed.
    model.load_state_dict(torch.load("model_finetuned.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


# ----------------------------------
# Define image transformation with proper normalization for 3 channels
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# ----------------------------------
# Donut Chart Function
def make_confidence_donut(score, label):
    score = max(min(score, 1), 0)
    color = {
        "Real": ["#28a745", "#eeeeee"],
        "Fake": ["#dc3545", "#eeeeee"]
    }
    source = pd.DataFrame({"Category": [label, ""], "Value": [score, 1 - score]})
    chart = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="Value",
        color=alt.Color("Category:N", scale=alt.Scale(domain=[label, ""], range=color[label]), legend=None)
    ).properties(width=130, height=130)
    text = alt.Chart(pd.DataFrame({'score': [f"{score*100:.1f}%"]})).mark_text(
        align='center', fontSize=22, fontWeight=600, color="white"
    ).encode(text="score:N")
    return chart + text

# ----------------------------------
# ELA Heatmap Function using in-memory BytesIO
def compute_ela(image, quality=90):
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    ela_image = Image.open(buffer)
    ela_diff = ImageChops.difference(image, ela_image)
    enhancer = ImageEnhance.Contrast(ela_diff)
    ela_result = enhancer.enhance(10)
    return ela_result

# ----------------------------------
# UI Header and information
st.title("🔍 Deepfake Image Detection")
st.markdown("Upload or paste an image to detect whether it's **Real or Fake** using a fine-tuned ResNeXt50 model.")
with st.expander("ℹ️ How does this work?"):
    st.write("""
    The model uses a pre-trained ResNeXt50 architecture, fine-tuned on real vs. deepfake images.
    
    After processing, it provides:
    - **Real Confidence**: Likelihood that the image is real.
    - **Fake Confidence**: Likelihood that the image is generated or tampered.
    
    A prediction is made if the fake confidence exceeds a set threshold.
    """)

# ----------------------------------
# Image Input Options (Upload File or Paste URL)
st.header("Upload / Paste Image")
method = st.radio("Select input method:", ["Upload Image File", "Paste an Image URL"])
image = None

if method == "Upload Image File":
    uploaded = st.file_uploader("Choose an image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Error opening image file: {e}")
elif method == "Paste an Image URL":
    image_url = st.text_input("Paste image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Unable to load image from the URL. Error: {e}")

# ----------------------------------
# Prediction & Results
if image:
    st.divider()

    # Preprocess image and move input tensor to device
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()  # Enable gradient tracking

    # Prepare GradCAM and inference inside a spinner for user feedback
    with st.spinner("Running deepfake detection..."):
        cam_extractor = GradCAM(model, target_layer="layer4")
        output = model(input_tensor)
        # Compute probability using sigmoid
        prob = torch.sigmoid(output).item()

        # Grad-CAM Heatmap
        activation_map = cam_extractor(class_idx=0, scores=output)
        resized_image = TF.to_pil_image(input_tensor.squeeze().detach().cpu())
        cam_result = overlay_mask(resized_image, TF.to_pil_image(activation_map[0]), alpha=0.5)

        # ELA Heatmap computed in-memory
        ela_result = compute_ela(image.resize((224, 224)), quality=ela_quality)

    # Calculate confidence scores
    real_prob = 1 - prob
    fake_prob = prob

    # Split layout columns
    col_img, col_space, col_score = st.columns([1.2, 0.1, 1.0])

    if fake_prob > fake_threshold:
        # For Fake Image detection tabs
        with col_img:
            tabs = st.tabs(["Original Image", "Grad-CAM", "ELA Heatmap"])
            with tabs[0]:
                st.markdown("### Preview and Detect")
                st.image(image, width=300, caption="Selected Image")
                st.error("**Fake Image Detected** 🚨")
                st.caption(f"Image Size: {image.size}")
                st.markdown(get_image_download_link(image, "original.png", "Download Original"), unsafe_allow_html=True)
            with tabs[1]:
                st.markdown("### Grad-CAM Heatmap")
                st.image(cam_result, caption="Model Attention (Grad-CAM)", width=300)
                with st.expander("Understanding Grad-CAM Colors for Deepfake Detection"):
                    st.markdown("""
                    **The meaning of the heatmap colors is explained below:**

                    - **Warm/Brighter Colors (red, orange, yellow):**  
                    It is shown that these areas are given more attention by the model. In deepfake detection, these parts are used more to decide if the image is fake.
                    
                    - **Cool/Darker Colors (blue, purple):**  
                    It is shown that these areas are less important in the decision of the model.

                    With this, you can see which parts of the image are focused on by the model, and a little explanation is given on the colors.
                    """)
                st.error("**Fake Image Detected** 🚨")
                st.caption(f"Image Size: {image.size}")
                st.markdown(get_image_download_link(cam_result, "gradcam.png", "Download Grad-CAM"), unsafe_allow_html=True)
            with tabs[2]:
                st.markdown("### ELA Heatmap")
                st.image(ela_result, caption="ELA Heatmap", width=300)
                st.error("**Fake Image Detected** 🚨")
                st.caption(f"Image Size: {image.size}")
                st.markdown(get_image_download_link(ela_result, "ela_heatmap.png", "Download ELA Heatmap"), unsafe_allow_html=True)
        with col_score:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.altair_chart(make_confidence_donut(real_prob, "Real"))
            st.caption("🟢 Real Confidence")
            st.markdown("<br>", unsafe_allow_html=True)
            st.altair_chart(make_confidence_donut(fake_prob, "Fake"))
            st.caption("🔴 Fake Confidence")
    else:
        # For Real Image detection tabs (without Grad-CAM)
        with col_img:
            tabs = st.tabs(["Original Image", "ELA Heatmap"])
            with tabs[0]:
                st.markdown("### Preview and Detect")
                st.image(image, width=300, caption="Selected Image")
                st.success("**Real Image Detected** ✅")
                st.caption(f"Image Size: {image.size}")
                st.markdown(get_image_download_link(image, "original.png", "Download Original"), unsafe_allow_html=True)
            with tabs[1]:
                st.markdown("### ELA Heatmap")
                st.image(ela_result, caption="ELA Heatmap", width=300)
                st.success("**Real Image Detected** ✅")
                st.caption(f"Image Size: {image.size}")
                st.markdown(get_image_download_link(ela_result, "ela_heatmap.png", "Download ELA Heatmap"), unsafe_allow_html=True)
        with col_score:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.altair_chart(make_confidence_donut(real_prob, "Real"))
            st.caption("🟢 Real Confidence")
            st.markdown("<br>", unsafe_allow_html=True)
            st.altair_chart(make_confidence_donut(fake_prob, "Fake"))
            st.caption("🔴 Fake Confidence")
