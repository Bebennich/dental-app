
# ============================================================
# DENTAL AI - XCEPTION MULTI-TASK LEARNING
# Age Estimation & Gender Classification
# Streamlit Deployment
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

import os
import gc
import io
import hashlib
import textwrap
import traceback

from pathlib import Path
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

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.pth"

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 768

AGE_MAX = 100.0
GENDER_THRESHOLD = 0.5

DEVICE = torch.device("cpu")

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 3. HTML RENDER HELPER
# ============================================================

def render_html(html):

    st.markdown(
        textwrap.dedent(html).strip(),
        unsafe_allow_html=True
    )


# ============================================================
# 4. CUSTOM CSS
# ============================================================

render_html("""
<style>

.stApp {
    background-color: #F3F6FA;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #142D46;
}

p {
    color: #607286;
}

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

.hero {
    background: linear-gradient(
        120deg,
        #10243B,
        #176B87
    );
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(16,36,59,0.12);
}

.hero h1 {
    color: white !important;
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

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white;
    border: 1px solid #E4EBF2;
    border-radius: 14px;
}

div[data-testid="stMetric"] {
    background: white;
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

.stButton > button {
    min-height: 45px;
    border-radius: 10px;
    font-weight: 650;
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

div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #91B9CA;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stImage"] img {
    border-radius: 10px;
}

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
""")


# ============================================================
# 5. MODEL ARCHITECTURE
# ============================================================

def make_head(in_features):

    return nn.Sequential(

        nn.Linear(in_features, 256),

        nn.ReLU(inplace=True),

        nn.Dropout(p=0.2),

        nn.Linear(256, 1)

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
# 6. GRAD-CAM WRAPPER
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
# 7. CHECKPOINT EXTRACTION
# ============================================================

def extract_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):

        raise TypeError(
            "Checkpoint bukan dictionary. "
            "File harus berisi state_dict atau checkpoint dictionary."
        )

    # Format checkpoint yang umum digunakan
    possible_keys = [
        "model_state_dict",
        "state_dict",
        "model"
    ]

    state_dict = None

    for key in possible_keys:

        if key in checkpoint and isinstance(
            checkpoint[key], dict
        ):

            state_dict = checkpoint[key]

            print(f"Checkpoint format detected: {key}")

            break

    # Jika checkpoint langsung berisi state_dict
    if state_dict is None:

        state_dict = checkpoint

    # Validasi parameter
    if not state_dict:

        raise ValueError(
            "State dictionary kosong."
        )

    if not all(
        isinstance(k, str) for k in state_dict.keys()
    ):

        raise ValueError(
            "Checkpoint memiliki nama parameter yang tidak valid."
        )

    # Remove DataParallel prefix
    cleaned_state = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state[key] = value

    return cleaned_state


# ============================================================
# 8. LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    print("=" * 60)
    print("DENTAL AI MODEL INITIALIZATION")
    print("=" * 60)

    print("Model path:", MODEL_PATH)
    print("File exists:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    file_size = MODEL_PATH.stat().st_size / (1024 ** 2)

    print(f"Checkpoint size: {file_size:.2f} MB")

    if file_size == 0:

        raise ValueError(
            "File best_model.pth kosong."
        )

    # Build architecture
    print("[1/4] Creating Xception architecture...")

    model = MultiTaskNet(
        backbone_name="legacy_xception",
        pretrained=False
    )

    # Load checkpoint
    print("[2/4] Reading checkpoint...")

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:

        # Compatibility for older PyTorch
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

    print("Checkpoint type:", type(checkpoint))

    # Extract state_dict
    print("[3/4] Extracting model parameters...")

    state_dict = extract_state_dict(checkpoint)

    print("Total checkpoint parameters:", len(state_dict))

    # Strict loading
    print("[4/4] Loading model weights...")

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as e:

        raise RuntimeError(
            "Struktur checkpoint tidak cocok dengan "
            "arsitektur Xception Multi-task.\n\n"
            f"Detail: {str(e)}"
        ) from e

    model.to(DEVICE)

    model.eval()

    # Check Grad-CAM layer
    if not hasattr(model.backbone, "act4"):

        raise AttributeError(
            "Layer act4 tidak ditemukan pada backbone. "
            "Periksa versi timm dan arsitektur Xception."
        )

    print("MODEL SUCCESSFULLY LOADED")

    print("=" * 60)

    return model


# ============================================================
# 9. LETTERBOX PREPROCESSING
# ============================================================

def resize_letterbox(
    img,
    target_hw=(384, 768),
    interpolation=cv2.INTER_CUBIC
):

    target_h, target_w = target_hw

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:

        raise ValueError(
            "Ukuran gambar tidak valid."
        )

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
# 10. EVALUATION TRANSFORM
# ============================================================

eval_trans = A.Compose([

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()

])


# ============================================================
# 11. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_pil):

    image_np = np.array(
        image_pil.convert("RGB")
    )

    img_resized = resize_letterbox(
        image_np,
        target_hw=(IMAGE_HEIGHT, IMAGE_WIDTH),
        interpolation=cv2.INTER_CUBIC
    )

    tensor = eval_trans(
        image=img_resized
    )["image"]

    tensor = tensor.unsqueeze(0).to(DEVICE)

    return tensor, img_resized


# ============================================================
# 12. MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 13. GRAD-CAM++
# ============================================================

def generate_gradcam(
    model,
    tensor,
    base_img,
    task_key
):

    wrapper = HeadWrapper(
        model,
        task_key
    )

    cam_result = None

    try:

        target_layers = [
            model.backbone.act4
        ]

        targets = [
            ClassifierOutputTarget(0)
        ]

        model.zero_grad(set_to_none=True)

        # Grad-CAM++ needs gradients
        with torch.enable_grad():

            with GradCAMPlusPlus(
                model=wrapper,
                target_layers=target_layers
            ) as cam:

                cam_result = cam(
                    input_tensor=tensor,
                    targets=targets
                )[0]

        visualization = show_cam_on_image(
            base_img,
            cam_result,
            use_rgb=True
        )

        return visualization

    finally:

        model.zero_grad(set_to_none=True)

        if cam_result is not None:

            del cam_result

        del wrapper

        cleanup_memory()


# ============================================================
# 14. PREDICTION FUNCTION
# ============================================================

def predict(image_pil, model):

    tensor = None

    outputs = None

    gender_cam = None

    age_cam = None

    try:

        # --------------------------------------------
        # PREPROCESSING
        # --------------------------------------------

        tensor, img_resized = preprocess_image(
            image_pil
        )

        base_img = (
            img_resized.astype(np.float32) / 255.0
        )

        # --------------------------------------------
        # INFERENCE
        # --------------------------------------------

        with torch.inference_mode():

            outputs = model(tensor)

            gender_prob = torch.sigmoid(
                outputs["gender_logit"]
            ).item()

            age_pred = (
                outputs["age_norm"].item() * AGE_MAX
            )

        del outputs

        outputs = None

        # --------------------------------------------
        # GENDER PREDICTION
        # --------------------------------------------

        if gender_prob >= GENDER_THRESHOLD:

            gender_pred = "Laki-laki (Male)"

            confidence = gender_prob

        else:

            gender_pred = "Perempuan (Female)"

            confidence = 1.0 - gender_prob

        # --------------------------------------------
        # GRAD-CAM GENDER
        # --------------------------------------------

        gender_cam = generate_gradcam(
            model=model,
            tensor=tensor,
            base_img=base_img,
            task_key="gender_logit"
        )

        # --------------------------------------------
        # GRAD-CAM AGE
        # --------------------------------------------

        age_cam = generate_gradcam(
            model=model,
            tensor=tensor,
            base_img=base_img,
            task_key="age_norm"
        )

        # --------------------------------------------
        # RESULT
        # --------------------------------------------

        result = {

            "gender": gender_pred,

            "gender_probability": gender_prob,

            "gender_confidence": confidence,

            "age": round(age_pred, 1),

            "gender_cam": gender_cam,

            "age_cam": age_cam,

            "processed_image": img_resized

        }

        return result

    finally:

        if outputs is not None:

            del outputs

        if tensor is not None:

            del tensor

        if gender_cam is not None:

            del gender_cam

        if age_cam is not None:

            del age_cam

        model.zero_grad(set_to_none=True)

        cleanup_memory()


# ============================================================
# 15. FILE HASH
# ============================================================

def get_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# 16. MODEL INITIALIZATION
# ============================================================

try:

    with st.spinner(
        "Initializing Dental AI Engine..."
    ):

        model = load_model()

except Exception as e:

    st.error(
        "Xception model gagal dimuat."
    )

    st.error(
        f"{type(e).__name__}: {str(e)}"
    )

    with st.expander(
        "Lihat detail error"
    ):

        st.code(
            traceback.format_exc()
        )

    st.warning(
        "Pastikan best_model.pth tersedia dan "
        "struktur checkpoint sesuai dengan arsitektur training."
    )

    st.stop()


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
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

        <h2 style="color:white; margin-top:14px;">
            DENTAL AI
        </h2>

        <p style="color:#B7D3E3; font-size:12px;">
            Xception Multi-task Intelligence
        </p>

    </div>

    <hr style="border-color:#456078;">
    """)

    st.markdown("### 🧠 Model Information")

    st.markdown("""
    **Architecture**

    Xception

    **Learning Approach**

    Multi-task Learning

    **Backbone**

    Legacy Xception

    **Input Resolution**

    384 × 768 pixels

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

    st.caption(f"Device: {DEVICE}")

    st.caption("Model: Xception Multi-task")

    st.divider()

    render_html("""
    <div style="
        text-align:center;
        color:#B7D3E3;
        font-size:11px;
        padding:15px;">
        Academic Research Prototype
        <br><br>
        Dental AI System
    </div>
    """)


# ============================================================
# 18. HERO HEADER
# ============================================================

render_html("""
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
""")


# ============================================================
# 19. SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None

if "current_file_hash" not in st.session_state:

    st.session_state.current_file_hash = None


# ============================================================
# 20. IMAGE UPLOAD
# ============================================================

render_html("""
<div class="section-title">
    01. Image Upload
</div>

<div class="section-subtitle">
    Upload a panoramic dental X-ray image for AI analysis.
</div>
""")

uploaded_file = st.file_uploader(
    "Choose panoramic dental X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG.",
    key="dental_image_upload"
)


# ============================================================
# 21. IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = get_file_hash(
        file_bytes
    )

    if current_hash != st.session_state.current_file_hash:

        st.session_state.current_file_hash = current_hash

        st.session_state.analysis_result = None

    try:

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Cannot read image: {str(e)}"
        )

        st.stop()

    processed_preview = resize_letterbox(
        np.array(image),
        target_hw=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        with st.container(border=True):

            st.markdown("### Original Dental X-Ray")

            st.image(
                image,
                width="stretch"
            )

            st.caption(
                f"Original resolution: "
                f"{image.width} × {image.height}"
            )

    with col2:

        with st.container(border=True):

            st.markdown("### Preprocessed Dental X-Ray")

            st.image(
                processed_preview,
                width="stretch"
            )

            st.caption(
                "Letterbox | 384 × 768 pixels"
            )


    # ========================================================
    # 22. ANALYSIS BUTTON
    # ========================================================

    st.divider()

    render_html("""
    <div class="section-title">
        02. AI Analysis
    </div>

    <div class="section-subtitle">
        Execute Xception to estimate age and classify gender.
    </div>
    """)

    if st.button(
        "🔍 ANALYZE DENTAL IMAGE",
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
                    model
                )

                st.session_state.analysis_result = result

            st.success(
                "Dental image analysis completed successfully!"
            )

        except Exception as e:

            st.session_state.analysis_result = None

            st.error(
                f"Analysis failed: {type(e).__name__}: {str(e)}"
            )

            with st.expander(
                "Technical error details"
            ):

                st.code(
                    traceback.format_exc()
                )


    # ========================================================
    # 23. PREDICTION RESULTS
    # ========================================================

    if st.session_state.analysis_result is not None:

        results = st.session_state.analysis_result

        st.divider()

        render_html("""
        <div class="section-title">
            03. Prediction Results
        </div>

        <div class="section-subtitle">
            Results generated by the trained Xception Multi-task model.
        </div>
        """)

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

            with st.container(border=True):

                st.markdown("### 👤 Gender Classification")

                st.metric(
                    label="Predicted Gender",
                    value=results["gender"]
                )

                confidence = results["gender_confidence"]

                st.progress(
                    float(np.clip(confidence, 0, 1))
                )

                st.markdown(
                    f"**Prediction Confidence:** "
                    f"`{confidence * 100:.2f}%`"
                )

                st.caption(
                    f"Raw sigmoid output: "
                    f"{results['gender_probability']:.4f}"
                )

        with col2:

            with st.container(border=True):

                st.markdown("### 🎂 Age Estimation")

                st.metric(
                    label="Estimated Age",
                    value=f"{results['age']:.1f} Years"
                )

                st.markdown(
                    "**Age Regression Output**"
                )

                st.caption(
                    "Normalized age prediction multiplied by 100."
                )


        # ====================================================
        # 24. GRAD-CAM++
        # ====================================================

        st.divider()

        render_html("""
        <div class="section-title">
            04. Model Explainability
        </div>

        <div class="section-subtitle">
            Visual explanation of feature activations using Grad-CAM++.
        </div>
        """)

        cam_col1, cam_col2 = st.columns(
            2,
            gap="large"
        )

        with cam_col1:

            with st.container(border=True):

                st.markdown("### Gender Grad-CAM++")

                st.image(
                    results["gender_cam"],
                    width="stretch"
                )

                st.caption(
                    "Feature activation for gender classification."
                )

        with cam_col2:

            with st.container(border=True):

                st.markdown("### Age Grad-CAM++")

                st.image(
                    results["age_cam"],
                    width="stretch"
                )

                st.caption(
                    "Feature activation for age regression."
                )


        # ====================================================
        # 25. ANALYSIS SUMMARY
        # ====================================================

        st.divider()

        render_html("""
        <div class="section-title">
            05. Analysis Summary
        </div>
        """)

        with st.container(border=True):

            st.markdown("### 📋 Prediction Report")

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
                    f"{results['gender_confidence'] * 100:.2f}%"
                )

            st.caption(
                "This result is an AI-based estimation, "
                "not a definitive medical or forensic conclusion."
            )


        # ====================================================
        # 26. RESET RESULTS
        # ====================================================

        st.divider()

        if st.button(
            "🔄 CLEAR ANALYSIS RESULTS",
            width="stretch",
            key="clear_button"
        ):

            st.session_state.analysis_result = None

            st.rerun()


else:

    st.session_state.current_file_hash = None

    st.session_state.analysis_result = None

    render_html("""
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

        <p style="color:#8293A3; font-size:14px;">
            Upload a panoramic dental X-ray image above
            to start the prediction process.
        </p>

    </div>
    """)


# ============================================================
# 27. FOOTER
# ============================================================

render_html("""
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
""")
