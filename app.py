
import gc
import hashlib
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np

from PIL import Image
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dental AI | Xception",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.html("""
<style>

/* MAIN APPLICATION */
.stApp {
    background-color: #F3F6FA;
    color: #18324B;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* GENERAL TEXT */
.stApp p,
.stApp label,
.stApp span {
    color: #42566B;
}

/* HEADINGS */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #142D46 !important;
    opacity: 1 !important;
    font-weight: 750 !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #10243B 0%,
        #173D58 100%
    );
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* HERO */
.hero {
    background: linear-gradient(
        120deg,
        #10243B,
        #176B87
    );
    padding: 38px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(16,36,59,0.12);
}

.hero-label {
    color: #90D9E8 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 32px;
    font-weight: 800;
    margin: 12px 0;
}

.hero p {
    color: #D6EAF2 !important;
    font-size: 14px;
}

/* SECTION */
.section-title {
    color: #142D46 !important;
    font-size: 23px;
    font-weight: 800;
    margin-top: 15px;
}

.section-subtitle {
    color: #607286 !important;
    font-size: 14px;
    margin-bottom: 20px;
}

/* CARDS */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE5ED;
    border-radius: 14px;
}

/* METRICS */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 18px;
    border: 1px solid #E5ECF3;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #607286 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #142D46 !important;
    font-weight: 800 !important;
}

/* FILE UPLOADER */
div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #91B9CA;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stFileUploader"] * {
    color: #263D52 !important;
}

/* BUTTON */
.stButton > button {
    min-height: 48px;
    border-radius: 10px;
    font-weight: 750 !important;
    font-size: 14px !important;
    opacity: 1 !important;
}

/* PRIMARY BUTTON */
.stButton > button[kind="primary"] {
    background-color: #147D9B !important;
    color: #FFFFFF !important;
    border: none !important;
}

.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #105E78 !important;
}

/* SECONDARY BUTTON */
.stButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #18324B !important;
    border: 1px solid #B8C9D8 !important;
}

.stButton > button[kind="secondary"] p,
.stButton > button[kind="secondary"] span {
    color: #18324B !important;
}

/* IMAGE */
div[data-testid="stImage"] img {
    border-radius: 10px;
}

/* DIVIDER */
hr {
    border-color: #D5DEE7 !important;
}

/* FOOTER */
.dental-footer {
    text-align: center;
    padding: 30px 15px;
    margin-top: 35px;
    background: #FFFFFF;
    border: 1px solid #E4EBF2;
    border-radius: 14px;
}

.dental-footer .brand {
    color: #176B87 !important;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 2px;
}

.dental-footer p {
    color: #607286 !important;
    font-size: 12px;
}

</style>
""")


# ============================================================
# 3. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.pth"

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 768

AGE_MAX = 100.0

GENDER_THRESHOLD = 0.5

DEVICE = torch.device("cpu")


# ============================================================
# 4. MODEL ARCHITECTURE
# ============================================================

def make_head(in_features):

    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1)
    )


class XceptionMultiTask(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        n_features = self.backbone.num_features

        self.gender_head = make_head(n_features)

        self.age_head = make_head(n_features)

    def forward(self, x):

        features = self.backbone(x)

        gender_logit = self.gender_head(features).squeeze(-1)

        age_norm = self.age_head(features).squeeze(-1)

        return {
            "gender_logit": gender_logit,
            "age_norm": age_norm
        }


# ============================================================
# 5. LOAD MODEL WITH CACHE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path, model_mtime):

    torch.set_num_threads(2)

    model = XceptionMultiTask()

    try:

        try:
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(
                model_path,
                map_location="cpu"
            )

        # Support several checkpoint formats
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model" in checkpoint and isinstance(
                checkpoint["model"], dict
            ):
                state_dict = checkpoint["model"]

            else:
                state_dict = checkpoint

            # Remove DataParallel prefix if present
            cleaned_state_dict = {}

            for key, value in state_dict.items():

                new_key = key

                if new_key.startswith("module."):
                    new_key = new_key[7:]

                cleaned_state_dict[new_key] = value

            model.load_state_dict(
                cleaned_state_dict,
                strict=True
            )

        elif isinstance(checkpoint, nn.Module):

            model = checkpoint

        else:

            raise ValueError(
                "Format checkpoint tidak dikenali."
            )

        model = model.to(DEVICE)
        model.eval()

        return model, None

    except Exception as error:

        return None, str(error)


# ============================================================
# 6. PREPROCESSING
# ============================================================

def resize_letterbox(
    image,
    target_hw=(384, 768),
    interpolation=cv2.INTER_CUBIC
):

    target_h, target_w = target_hw

    original_h, original_w = image.shape[:2]

    scale = min(
        target_h / original_h,
        target_w / original_w
    )

    new_h = max(1, round(original_h * scale))
    new_w = max(1, round(original_w * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation
    )

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top

    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    result = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return result


eval_transform = Compose([
    Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


# ============================================================
# 7. GRAD-CAM++ HELPER
# ============================================================

class HeadWrapper(nn.Module):

    def __init__(self, model, key):

        super().__init__()

        self.model = model
        self.key = key

    def forward(self, x):

        return self.model(x)[self.key].unsqueeze(1)


def generate_gradcam(model, tensor, base_image, task_key):

    target_layer = model.backbone.act4

    wrapped_model = HeadWrapper(
        model,
        task_key
    )

    targets = [
        ClassifierOutputTarget(0)
    ]

    with GradCAMPlusPlus(
        model=wrapped_model,
        target_layers=[target_layer]
    ) as cam:

        grayscale_cam = cam(
            input_tensor=tensor,
            targets=targets
        )[0, :]

    visualization = show_cam_on_image(
        base_image.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    del wrapped_model

    gc.collect()

    return visualization


# ============================================================
# 8. PREDICTION FUNCTION
# ============================================================

def predict_image(model, image_pil):

    # Convert uploaded image to RGB
    image_rgb = np.array(
        image_pil.convert("RGB")
    )

    # Letterbox 384 x 768
    image_resized = resize_letterbox(
        image_rgb,
        target_hw=(IMAGE_HEIGHT, IMAGE_WIDTH),
        interpolation=cv2.INTER_CUBIC
    )

    # Normalization and tensor conversion
    tensor = eval_transform(
        image=image_resized
    )["image"].unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.inference_mode():

        outputs = model(tensor)

        gender_probability = torch.sigmoid(
            outputs["gender_logit"]
        ).item()

        age_prediction = (
            outputs["age_norm"].item() * AGE_MAX
        )

    # Gender classification
    if gender_probability >= GENDER_THRESHOLD:

        gender_prediction = "Laki-laki (Male)"

        gender_confidence = gender_probability

    else:

        gender_prediction = "Perempuan (Female)"

        gender_confidence = 1.0 - gender_probability

    # Prepare original RGB image for Grad-CAM
    base_image = (
        image_resized.astype(np.float32) / 255.0
    )

    # Gender Grad-CAM++
    gender_cam = generate_gradcam(
        model,
        tensor,
        base_image,
        "gender_logit"
    )

    # Age Grad-CAM++
    age_cam = generate_gradcam(
        model,
        tensor,
        base_image,
        "age_norm"
    )

    # Cleanup
    del tensor
    gc.collect()

    return {

        "gender": gender_prediction,

        "gender_probability": gender_probability,

        "gender_confidence": gender_confidence,

        "age": round(age_prediction, 1),

        "original": image_rgb,

        "preprocessed": image_resized,

        "gender_cam": gender_cam,

        "age_cam": age_cam

    }


# ============================================================
# 9. SESSION STATE MANAGEMENT
# ============================================================

def clear_analysis():

    st.session_state.analysis_result = None

    st.session_state.analysis_error = None

    st.session_state.last_image_hash = None


def reset_application():

    clear_analysis()

    st.session_state.image_uploader = None


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None

if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:20px 5px;">
            <div style="font-size:48px;">🦷</div>
            <h2 style="color:white !important;">
                DENTAL AI
            </h2>
            <p style="color:#BBD6E5;">
                Xception Multi-task Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

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

    if MODEL_PATH.exists():

        st.success("Model checkpoint ditemukan")

    else:

        st.error("Model checkpoint tidak ditemukan")

    st.caption(f"Device: {DEVICE}")

    st.caption("Model: Xception Multi-task")


# ============================================================
# 11. LOAD MODEL
# ============================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


if not MODEL_PATH.exists():

    st.error(
        "Xception model gagal ditemukan."
    )

    st.info(
        "Pastikan file best_model.pth berada "
        "dalam direktori yang sama dengan app.py."
    )

    st.stop()


with st.spinner("Initializing Xception AI Engine..."):

    model, model_error = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )


if model is None:

    st.error("Xception model gagal dimuat.")

    with st.expander("Detail kesalahan model"):

        st.code(model_error)

    st.info(
        "Periksa kesesuaian arsitektur model, "
        "nama layer, dan checkpoint best_model.pth."
    )

    st.stop()


# ============================================================
# 12. IMAGE UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">01. Image Upload</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a panoramic dental X-ray image for analysis.'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose panoramic dental X-ray image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader",
    help="Supported formats: JPG, JPEG, PNG",
    on_change=clear_analysis
)


# ============================================================
# 13. IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    try:

        image_pil = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image_pil)

        original_height, original_width = image_array.shape[:2]

        preprocessed_image = resize_letterbox(
            image_array,
            target_hw=(IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

            with st.container(border=True):

                st.markdown(
                    """
                    <h3 style="color:#142D46 !important;">
                        Original Dental X-Ray
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

                st.image(
                    image_pil,
                    width="stretch"
                )

                st.caption(
                    f"Original resolution: "
                    f"{original_width} × {original_height}"
                )

        with col2:

            with st.container(border=True):

                st.markdown(
                    """
                    <h3 style="color:#142D46 !important;">
                        Preprocessed Dental X-Ray
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

                st.image(
                    preprocessed_image,
                    width="stretch"
                )

                st.caption(
                    "Letterbox | 384 × 768 pixels"
                )

        st.divider()

        # Image fingerprint
        image_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        # If image changes, remove previous analysis
        if (
            st.session_state.last_image_hash is not None
            and st.session_state.last_image_hash != image_hash
        ):

            clear_analysis()

        # ----------------------------------------------------
        # ANALYSIS SECTION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">02. AI Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Execute Xception to estimate age and classify gender.'
            '</div>',
            unsafe_allow_html=True
        )

        analyze_col, reset_col = st.columns(
            [3, 1],
            gap="medium"
        )

        with analyze_col:

            analyze_clicked = st.button(
                "🔍 ANALYZE DENTAL IMAGE",
                type="primary",
                width="stretch",
                key="analyze_button"
            )

        with reset_col:

            reset_clicked = st.button(
                "🗑️ CLEAR",
                type="secondary",
                width="stretch",
                key="reset_button"
            )

        # ----------------------------------------------------
        # RESET ALL
        # ----------------------------------------------------

        if reset_clicked:

            reset_application()

            st.rerun()

        # ----------------------------------------------------
        # RUN ANALYSIS
        # ----------------------------------------------------

        if analyze_clicked:

            clear_analysis()

            try:

                with st.spinner(
                    "Xception is analyzing dental image..."
                ):

                    result = predict_image(
                        model,
                        image_pil
                    )

                st.session_state.analysis_result = result

                st.session_state.last_image_hash = image_hash

                st.session_state.analysis_error = None

            except Exception as error:

                st.session_state.analysis_result = None

                st.session_state.analysis_error = str(error)

                gc.collect()

        # ----------------------------------------------------
        # DISPLAY ERROR
        # ----------------------------------------------------

        if st.session_state.analysis_error:

            st.error(
                "Terjadi kesalahan ketika melakukan analisis."
            )

            with st.expander("Detail error"):

                st.code(
                    st.session_state.analysis_error
                )

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        result = st.session_state.analysis_result

        if result is not None:

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '03. Prediction Results'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">'
                'Xception Multi-task prediction results.'
                '</div>',
                unsafe_allow_html=True
            )

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    label="Estimated Age",
                    value=f"{result['age']:.1f} Years"
                )

            with metric2:

                st.metric(
                    label="Predicted Gender",
                    value=result["gender"]
                )

            with metric3:

                st.metric(
                    label="Gender Confidence",
                    value=f"{result['gender_confidence'] * 100:.2f}%"
                )

            st.info(
                "Probabilitas kelas Male: "
                f"{result['gender_probability'] * 100:.2f}%"
            )

            # ------------------------------------------------
            # GRAD-CAM++
            # ------------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '04. Model Explainability'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">'
                'Grad-CAM++ visualization for gender and age prediction.'
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
                        """
                        <h3 style="color:#142D46 !important;">
                            Gender Grad-CAM++
                        </h3>
                        """,
                        unsafe_allow_html=True
                    )

                    st.image(
                        result["gender_cam"],
                        width="stretch"
                    )

                    st.caption(
                        "Activation visualization for gender classification."
                    )

            with cam_col2:

                with st.container(border=True):

                    st.markdown(
                        """
                        <h3 style="color:#142D46 !important;">
                            Age Grad-CAM++
                        </h3>
                        """,
                        unsafe_allow_html=True
                    )

                    st.image(
                        result["age_cam"],
                        width="stretch"
                    )

                    st.caption(
                        "Activation visualization for age regression."
                    )

            st.success(
                "Analisis berhasil diselesaikan."
            )

            st.caption(
                "Untuk melakukan analisis gambar lain, "
                "hapus gambar sebelumnya melalui tombol X "
                "pada uploader atau gunakan tombol CLEAR."
            )

    except Exception as error:

        clear_analysis()

        st.error(
            "Gagal membaca gambar yang diunggah."
        )

        with st.expander("Detail error"):

            st.code(str(error))


else:

    # --------------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------------

    # Ensure old analysis is not displayed without an image
    if st.session_state.analysis_result is not None:

        clear_analysis()

    st.divider()

    with st.container(border=True):

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:55px 20px;
            ">

                <div style="font-size:55px;">
                    🦷
                </div>

                <h2 style="
                    color:#142D46 !important;
                    font-size:24px;
                ">
                    Ready for Dental Analysis
                </h2>

                <p style="
                    color:#607286 !important;
                    font-size:14px;
                ">
                    Upload a panoramic dental X-ray image above
                    to start the prediction process.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown(
    """
    <div class="dental-footer">

        <div class="brand">
            DENTAL AI
        </div>

        <p>
            Xception Multi-task Learning for Age and Gender Detection
        </p>

        <p>
            Research Prototype | Academic Research Project
        </p>

        <p>
            AI-assisted dental image analysis
        </p>

        <p>
            © 2026 Dental AI
        </p>

    </div>
    """,
    unsafe_allow_html=True
)
