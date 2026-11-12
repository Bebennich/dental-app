
# ============================================================
# DENTAL AI - XCEPTION MULTI-TASK STREAMLIT APPLICATION
# Age Estimation + Gender Classification + Grad-CAM++
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import streamlit as st

import torch
import torch.nn as nn

import cv2
import numpy as np
import timm
import albumentations as A

import gc
import io
import hashlib
import traceback
import os

from PIL import Image

from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus

from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dental AI | Xception",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best_model.pth"

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 768

AGE_MAX = 100.0

GENDER_THRESHOLD = 0.5

# CPU resource management
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =================================
   GLOBAL
================================= */

.stApp {
    background-color: #F3F6FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* =================================
   TYPOGRAPHY
================================= */

h1, h2, h3 {
    color: #142D46;
}

p {
    color: #607286;
}

/* =================================
   SIDEBAR
================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #10243B 0%,
        #173D58 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* =================================
   HERO HEADER
================================= */

.hero {
    background: linear-gradient(
        120deg,
        #10243B 0%,
        #176B87 100%
    );

    padding: 35px;
    border-radius: 18px;

    margin-bottom: 28px;

    box-shadow: 0 8px 25px rgba(16,36,59,0.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 32px;
    font-weight: 750;
    margin-bottom: 10px;
}

.hero p {
    color: #D6EAF2 !important;
    font-size: 14px;
    margin-bottom: 0;
}

.hero-label {
    color: #90D9E8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

/* =================================
   SECTION TITLES
================================= */

.section-title {
    color: #142D46;
    font-size: 21px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #7A8C9F;
    font-size: 13px;
    margin-bottom: 18px;
}

/* =================================
   CARDS
================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E4EBF2;
    border-radius: 14px;
}

/* =================================
   METRICS
================================= */

div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 18px;
    border: 1px solid #E5ECF3;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #6C7F92;
}

div[data-testid="stMetricValue"] {
    color: #142D46;
    font-weight: 750;
}

/* =================================
   BUTTONS
================================= */

.stButton > button {
    min-height: 45px;
    border-radius: 10px;
    font-weight: 650;
    transition: 0.2s;
}

.stButton > button[kind="primary"] {
    background: #147D9B;
    color: white;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    background: #105E78;
    color: white;
}

/* =================================
   FILE UPLOADER
================================= */

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #91B9CA;
    border-radius: 14px;
    padding: 15px;
}

/* =================================
   IMAGES
================================= */

div[data-testid="stImage"] img {
    border-radius: 10px;
}

/* =================================
   FOOTER
================================= */

.footer {
    text-align: center;
    color: #8192A3;
    font-size: 12px;
    padding: 20px 0;
}

hr {
    border-color: #E3EAF1;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. XCEPTION MULTI-TASK ARCHITECTURE
# ============================================================

def make_head(in_features):

    return nn.Sequential(

        nn.Linear(
            in_features,
            256
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            p=0.2
        ),

        nn.Linear(
            256,
            1
        )

    )


class MultiTaskNet(nn.Module):

    def __init__(
        self,
        backbone_name="legacy_xception",
        pretrained=False
    ):

        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )

        n_feat = self.backbone.num_features

        self.gender_head = make_head(n_feat)

        self.age_head = make_head(n_feat)

    def forward(self, x):

        features = self.backbone(x)

        gender_logit = self.gender_head(
            features
        ).squeeze(-1)

        age_norm = self.age_head(
            features
        ).squeeze(-1)

        return {

            "gender_logit": gender_logit,

            "age_norm": age_norm

        }


# ============================================================
# 5. GRAD-CAM WRAPPER
# ============================================================

class HeadWrapper(nn.Module):

    def __init__(self, model, key):

        super().__init__()

        self.model = model

        self.key = key

    def forward(self, x):

        output = self.model(x)

        return output[self.key].unsqueeze(1)


# ============================================================
# 6. MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    device = torch.device("cpu")

    try:

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: {MODEL_PATH}"
            )

        # Initialize architecture
        model = MultiTaskNet(
            backbone_name="legacy_xception",
            pretrained=False
        )

        # Load checkpoint
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        # Support multiple checkpoint formats
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            else:

                state_dict = checkpoint

        else:

            raise TypeError(
                "Format checkpoint tidak didukung. "
                "Gunakan state_dict model."
            )

        # Remove DataParallel prefix
        state_dict = {

            key.replace("module.", "", 1): value

            for key, value in state_dict.items()

        }

        # Load trained weights
        model.load_state_dict(
            state_dict,
            strict=True
        )

        model.to(device)

        model.eval()

        return model, device, None

    except Exception as e:

        return None, device, str(e)


# ============================================================
# 7. LETTERBOX PREPROCESSING
# ============================================================

def resize_letterbox(
    img,
    target_hw=(384, 768),
    interpolation=cv2.INTER_CUBIC
):

    target_h, target_w = target_hw

    h, w = img.shape[:2]

    scale = min(
        target_h / h,
        target_w / w
    )

    new_h = max(
        1,
        round(h * scale)
    )

    new_w = max(
        1,
        round(w * scale)
    )

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interpolation
    )

    top = (target_h - new_h) // 2

    bottom = target_h - new_h - top

    left = (target_w - new_w) // 2

    right = target_w - new_w - left

    output = cv2.copyMakeBorder(

        resized,

        top,
        bottom,
        left,
        right,

        cv2.BORDER_CONSTANT,

        value=(0, 0, 0)

    )

    return output


# ============================================================
# 8. EVALUATION TRANSFORMATION
# ============================================================

eval_trans = A.Compose([

    A.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    ),

    ToTensorV2()

])


# ============================================================
# 9. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_pil):

    image_np = np.array(
        image_pil.convert("RGB")
    )

    # Letterbox resizing
    img_resized = resize_letterbox(

        image_np,

        target_hw=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ),

        interpolation=cv2.INTER_CUBIC

    )

    # ImageNet normalization
    tensor = eval_trans(
        image=img_resized
    )["image"]

    tensor = tensor.unsqueeze(0)

    return tensor, img_resized


# ============================================================
# 10. MEMORY CLEANUP
# ============================================================

def cleanup_memory(device):

    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# 11. GENERATE GRAD-CAM++
# ============================================================

def generate_gradcam(
    model,
    tensor,
    base_img,
    task_key,
    device
):

    wrapper = None

    cam_result = None

    try:

        model.zero_grad(set_to_none=True)

        # Task-specific wrapper
        wrapper = HeadWrapper(
            model,
            task_key
        )

        target_layers = [
            model.backbone.act4
        ]

        targets = [
            ClassifierOutputTarget(0)
        ]

        # Grad-CAM++ requires gradient
        with torch.enable_grad():

            with GradCAMPlusPlus(

                model=wrapper,

                target_layers=target_layers

            ) as cam:

                cam_result = cam(

                    input_tensor=tensor,

                    targets=targets

                )[0]

        # Generate heatmap overlay
        visualization = show_cam_on_image(

            base_img,

            cam_result,

            use_rgb=True

        )

        return visualization

    finally:

        model.zero_grad(set_to_none=True)

        if wrapper is not None:
            del wrapper

        if cam_result is not None:
            del cam_result

        cleanup_memory(device)


# ============================================================
# 12. MAIN PREDICTION FUNCTION
# ============================================================

def predict(image_pil, model, device):

    tensor = None

    outputs = None

    try:

        # ------------------------------------------
        # A. PREPROCESSING
        # ------------------------------------------

        tensor, img_resized = preprocess_image(
            image_pil
        )

        tensor = tensor.to(device)

        base_img = (
            img_resized.astype(np.float32) / 255.0
        )

        # ------------------------------------------
        # B. INFERENCE
        # ------------------------------------------

        with torch.no_grad():

            outputs = model(tensor)

            gender_prob = torch.sigmoid(
                outputs["gender_logit"]
            ).item()

            gender_pred = (

                "Laki-laki (Male)"

                if gender_prob >= GENDER_THRESHOLD

                else "Perempuan (Female)"

            )

            age_pred = (
                outputs["age_norm"].item() * AGE_MAX
            )

        del outputs

        outputs = None

        # ------------------------------------------
        # C. GRAD-CAM++ GENDER
        # ------------------------------------------

        vis_gender = generate_gradcam(

            model=model,

            tensor=tensor,

            base_img=base_img,

            task_key="gender_logit",

            device=device

        )

        # ------------------------------------------
        # D. GRAD-CAM++ AGE
        # ------------------------------------------

        vis_age = generate_gradcam(

            model=model,

            tensor=tensor,

            base_img=base_img,

            task_key="age_norm",

            device=device

        )

        # ------------------------------------------
        # E. CONFIDENCE
        # ------------------------------------------

        gender_confidence = (

            gender_prob

            if gender_prob >= GENDER_THRESHOLD

            else 1.0 - gender_prob

        )

        # ------------------------------------------
        # F. RESULT
        # ------------------------------------------

        result = {

            "gender": gender_pred,

            "gender_probability": gender_prob,

            "gender_confidence": gender_confidence,

            "age": round(age_pred, 1),

            "gender_cam": vis_gender,

            "age_cam": vis_age,

            "processed_image": img_resized

        }

        return result

    finally:

        if outputs is not None:
            del outputs

        if tensor is not None:
            del tensor

        model.zero_grad(set_to_none=True)

        cleanup_memory(device)


# ============================================================
# 13. FILE HASH
# ============================================================

def get_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# 14. LOAD MODEL
# ============================================================

with st.spinner(
    "Initializing Dental AI Engine..."
):

    model, device, error = load_model()


if model is None:

    st.error(
        "Unable to load Xception model."
    )

    st.code(error)

    st.info(
        "Pastikan best_model.pth tersedia "
        "di direktori yang sama dengan app.py."
    )

    st.stop()


# ============================================================
# 15. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:20px 5px;">

        <div style="
            background:#1D506A;
            width:68px;
            height:68px;
            border-radius:18px;
            margin:auto;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:34px;">

            🦷

        </div>

        <h2 style="
            color:white;
            margin-top:14px;
            margin-bottom:4px;">

            DENTAL AI

        </h2>

        <p style="
            color:#B7D3E3;
            font-size:12px;">

            Xception Multi-task Intelligence

        </p>

    </div>

    <hr style="border-color:#456078;">
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 Model Information")

    st.markdown("""
    **Architecture**

    Xception

    **Learning Approach**

    Multi-task Learning

    **Backbone**

    Legacy Xception

    **Input Resolution**

    384 × 768

    **Normalization**

    ImageNet

    **Prediction Tasks**

    - Gender Classification
    - Age Regression

    **Explainability**

    Grad-CAM++
    """)

    st.divider()

    st.markdown("### ⚙️ System Status")

    st.success("Model Loaded")

    st.caption(
        f"Device: {device}"
    )

    st.caption(
        "Model: Xception Multi-task"
    )

    st.divider()

    st.markdown("""
    <div style="
        text-align:center;
        color:#B7D3E3;
        font-size:11px;">

        Academic Research Prototype

        <br>

        Dental AI System

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 16. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-label">

        ARTIFICIAL INTELLIGENCE • MEDICAL IMAGING

    </div>

    <h1>

        Dental Age & Gender Detection

    </h1>

    <p>

        Intelligent analysis of panoramic dental X-ray
        images using Xception Multi-task Learning.

    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 17. SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None


if "current_file_hash" not in st.session_state:

    st.session_state.current_file_hash = None


# ============================================================
# 18. IMAGE UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">01. Image Upload</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a panoramic dental X-ray image for AI analysis.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(

    "Choose panoramic dental X-ray image",

    type=["jpg", "jpeg", "png"],

    help="Supported image formats: JPG, JPEG, PNG.",

    key="dental_image_upload"

)


if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = get_file_hash(
        file_bytes
    )

    # Reset previous result when image changes
    if current_hash != st.session_state.current_file_hash:

        st.session_state.current_file_hash = current_hash

        st.session_state.analysis_result = None

    try:

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Cannot read uploaded image: {e}"
        )

        st.stop()

    # ------------------------------------------
    # PREPROCESSING PREVIEW
    # ------------------------------------------

    processed_preview = resize_letterbox(

        np.array(image),

        target_hw=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        )

    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        with st.container(border=True):

            st.markdown("**Original Dental X-Ray**")

            st.image(
                image,
                width="stretch"
            )

            st.caption(
                f"Resolution: {image.width} × {image.height}"
            )

    with col2:

        with st.container(border=True):

            st.markdown("**Preprocessed Dental X-Ray**")

            st.image(
                processed_preview,
                width="stretch"
            )

            st.caption(
                "Letterbox | 384 × 768 pixels"
            )


    # ========================================================
    # 19. ANALYSIS SECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">02. AI Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Execute the Xception model to estimate age and classify gender.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(

        "🔍  ANALYZE DENTAL IMAGE",

        type="primary",

        width="stretch",

        key="analyze_button"

    ):

        try:

            with st.spinner(
                "Xception is analyzing the dental image..."
            ):

                result = predict(

                    image,

                    model,

                    device

                )

                st.session_state.analysis_result = result

            st.success(
                "Dental image analysis completed successfully!"
            )

        except Exception as e:

            st.session_state.analysis_result = None

            st.error(
                f"Analysis failed: {str(e)}"
            )

            with st.expander(
                "View technical error"
            ):

                st.code(
                    traceback.format_exc()
                )


    # ========================================================
    # 20. PREDICTION RESULTS
    # ========================================================

    if st.session_state.analysis_result is not None:

        results = st.session_state.analysis_result

        st.divider()

        st.markdown(
            '<div class="section-title">03. Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Results generated by the trained Xception Multi-task model.'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        # ------------------------------------------
        # GENDER RESULT
        # ------------------------------------------

        with col1:

            with st.container(border=True):

                st.markdown(
                    "### 👤 Gender Classification"
                )

                st.metric(

                    label="Predicted Gender",

                    value=results["gender"]

                )

                confidence = results["gender_confidence"]

                st.progress(
                    float(
                        np.clip(confidence, 0, 1)
                    )
                )

                st.markdown(
                    f"**Prediction Confidence:** "
                    f"`{confidence * 100:.2f}%`"
                )

                st.caption(
                    f"Raw sigmoid output: "
                    f"{results['gender_probability']:.4f}"
                )

        # ------------------------------------------
        # AGE RESULT
        # ------------------------------------------

        with col2:

            with st.container(border=True):

                st.markdown(
                    "### 🎂 Age Estimation"
                )

                st.metric(

                    label="Estimated Age",

                    value=f"{results['age']:.1f} Years"

                )

                st.markdown(
                    "**Age Regression Output**"
                )

                st.caption(
                    "Normalized age prediction × 100."
                )


        # ====================================================
        # 21. GRAD-CAM++ VISUALIZATION
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">04. Model Explainability</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Visual explanation of feature activations using Grad-CAM++.'
            '</div>',
            unsafe_allow_html=True
        )

        cam_col1, cam_col2 = st.columns(
            2,
            gap="large"
        )

        with cam_col1:

            with st.container(border=True):

                st.markdown(
                    "### Gender Grad-CAM++"
                )

                st.image(
                    results["gender_cam"],
                    width="stretch"
                )

                st.caption(
                    "Feature activation for gender classification."
                )

        with cam_col2:

            with st.container(border=True):

                st.markdown(
                    "### Age Grad-CAM++"
                )

                st.image(
                    results["age_cam"],
                    width="stretch"
                )

                st.caption(
                    "Feature activation for age regression."
                )


        # ====================================================
        # 22. ANALYSIS SUMMARY
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">05. Analysis Summary</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            st.markdown(
                "### 📋 Prediction Report"
            )

            report_col1, report_col2, report_col3 = st.columns(3)

            with report_col1:

                st.markdown("**Gender**")

                st.write(
                    results["gender"]
                )

            with report_col2:

                st.markdown("**Estimated Age**")

                st.write(
                    f"{results['age']:.1f} years"
                )

            with report_col3:

                st.markdown("**Confidence**")

                st.write(
                    f"{confidence * 100:.2f}%"
                )

            st.caption(
                "This result is an AI-based estimation, "
                "not a definitive medical or forensic conclusion."
            )


        # ====================================================
        # 23. RESET ANALYSIS
        # ====================================================

        st.divider()

        if st.button(

            "🔄  RESET ANALYSIS",

            width="stretch",

            key="reset_button"

        ):

            st.session_state.analysis_result = None

            gc.collect()

            st.rerun()


else:

    # Clear previous results when upload is removed
    st.session_state.current_file_hash = None

    st.session_state.analysis_result = None

    # ------------------------------------------
    # EMPTY STATE
    # ------------------------------------------

    st.markdown("""
    <div style="
        background:#FFFFFF;
        border:1px dashed #A8C5D5;
        border-radius:16px;
        padding:45px 20px;
        text-align:center;
        margin-top:15px;">

        <div style="font-size:48px;">
            🦷
        </div>

        <h3 style="color:#183B56;">
            Ready for Dental Analysis?
        </h3>

        <p style="
            color:#8293A3;
            font-size:14px;">

            Upload a panoramic dental X-ray image above
            to start the prediction process.

        </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 24. FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <hr>

    <b style="color:#176B87; font-size:14px;">
        DENTAL AI
    </b>

    <br><br>

    Xception Multi-task Learning for Age and Gender Detection

    <br>

    Research Prototype | Academic Research Project

    <br><br>

    © 2026 Dental AI

</div>
""", unsafe_allow_html=True)
