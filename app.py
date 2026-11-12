
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import timm
import albumentations as A
import gc
import hashlib
import io
import traceback

from PIL import Image
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# =========================================================
# 1. KONFIGURASI STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Xception Dental AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best_model.pth"

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 768
AGE_MAX = 100.0

# Membatasi thread CPU agar penggunaan resource lebih terkontrol.
torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# =========================================================
# 2. ARSITEKTUR XCEPTION MULTI-TASK
# =========================================================

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

        feat = self.backbone(x)

        gender_logit = self.gender_head(feat).squeeze(-1)
        age_norm = self.age_head(feat).squeeze(-1)

        return {
            "gender_logit": gender_logit,
            "age_norm": age_norm
        }


# =========================================================
# 3. WRAPPER GRAD-CAM++
# =========================================================

class HeadWrapper(nn.Module):

    def __init__(self, model, key):

        super().__init__()

        self.model = model
        self.key = key

    def forward(self, x):

        output = self.model(x)

        return output[self.key].unsqueeze(1)


# =========================================================
# 4. LOAD CHECKPOINT MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MultiTaskNet(
        backbone_name="legacy_xception",
        pretrained=False
    )

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        # Mendukung beberapa format checkpoint.
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            else:

                state_dict = checkpoint

        else:

            raise TypeError(
                "Checkpoint bukan state_dict yang valid."
            )

        # Menghapus prefix DataParallel jika tersedia.
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

        # Memuat bobot model.
        model.load_state_dict(
            state_dict,
            strict=True
        )

        model.to(device)
        model.eval()

        return model, device, None

    except Exception as e:

        return None, device, str(e)


# =========================================================
# 5. PREPROCESSING LETTERBOX 384 x 768
# =========================================================

def resize_letterbox(
    img,
    target_hw=(384, 768),
    interpolation=cv2.INTER_CUBIC
):

    th, tw = target_hw

    h, w = img.shape[:2]

    scale = min(
        th / h,
        tw / w
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

    top = (th - new_h) // 2
    bottom = th - new_h - top

    left = (tw - new_w) // 2
    right = tw - new_w - left

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


# =========================================================
# 6. TRANSFORMASI EVALUASI
# =========================================================

eval_trans = A.Compose([

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()

])


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

    tensor = tensor.unsqueeze(0)

    return tensor, img_resized


# =========================================================
# 7. FUNGSI MEMBERSIHKAN MEMORI
# =========================================================

def cleanup_memory(device):

    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()


# =========================================================
# 8. FUNGSI GRAD-CAM++
# =========================================================

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

        # Grad-CAM++ membutuhkan gradient.
        with torch.enable_grad():

            with GradCAMPlusPlus(
                model=wrapper,
                target_layers=target_layers
            ) as cam:

                cam_result = cam(
                    input_tensor=tensor,
                    targets=targets
                )[0]

        # Membuat overlay heatmap.
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


# =========================================================
# 9. FUNGSI PREDIKSI MULTI-TASK
# =========================================================

def predict(image_pil, model, device):

    tensor = None
    outputs = None

    try:

        # ---------------------------------------------
        # A. PREPROCESSING
        # ---------------------------------------------

        tensor, img_resized = preprocess_image(
            image_pil
        )

        tensor = tensor.to(device)

        base_img = (
            img_resized.astype(np.float32) / 255.0
        )

        # ---------------------------------------------
        # B. INFERENSI UMUR DAN GENDER
        # ---------------------------------------------

        with torch.inference_mode():

            outputs = model(tensor)

            gender_prob = torch.sigmoid(
                outputs["gender_logit"]
            ).item()

            gender_pred = (
                "Laki-laki (Male)"
                if gender_prob >= 0.5
                else "Perempuan (Female)"
            )

            age_pred = (
                outputs["age_norm"].item() * AGE_MAX
            )

        del outputs
        outputs = None

        # ---------------------------------------------
        # C. GRAD-CAM++ GENDER
        # ---------------------------------------------

        vis_gender = generate_gradcam(
            model=model,
            tensor=tensor,
            base_img=base_img,
            task_key="gender_logit",
            device=device
        )

        # ---------------------------------------------
        # D. GRAD-CAM++ UMUR
        # ---------------------------------------------

        vis_age = generate_gradcam(
            model=model,
            tensor=tensor,
            base_img=base_img,
            task_key="age_norm",
            device=device
        )

        # ---------------------------------------------
        # E. SIMPAN HASIL
        # ---------------------------------------------

        result = {

            "gender": gender_pred,

            "gender_probability": gender_prob,

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


# =========================================================
# 10. IDENTITAS FILE UPLOAD
# =========================================================

def get_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# =========================================================
# 11. MEMUAT MODEL
# =========================================================

st.title(
    "🦷 Deteksi Umur dan Gender Manusia"
)

st.markdown(
    """
    Aplikasi deteksi umur dan gender manusia menggunakan
    arsitektur **Xception Multi-task Learning** berdasarkan
    citra *Panoramic Dental X-Ray*.

    Aplikasi dilengkapi dengan visualisasi **Grad-CAM++**
    untuk melihat area citra yang berkontribusi terhadap
    hasil prediksi model.
    """
)

st.divider()

with st.spinner("Memuat model Xception..."):

    model, device, error = load_model()


if model is None:

    st.error(
        "Model gagal dimuat."
    )

    st.code(error)

    st.info(
        "Pastikan best_model.pth berada di folder yang sama "
        "dengan app.py dan checkpoint sesuai arsitektur model."
    )

    st.stop()

st.success(
    f"Model berhasil dimuat | Device: {device}"
)


# =========================================================
# 12. SIDEBAR INFORMASI
# =========================================================

with st.sidebar:

    st.header("🦷 Informasi Model")

    st.write("**Arsitektur:** Xception")

    st.write("**Pendekatan:** Multi-task Learning")

    st.write("**Backbone:** Legacy Xception")

    st.write("**Input:** 384 × 768 piksel")

    st.write("**Task 1:** Gender Classification")

    st.write("**Task 2:** Age Regression")

    st.write("**Visualisasi:** Grad-CAM++")

    st.divider()

    st.caption(
        "Model ini digunakan sebagai prototipe "
        "penelitian deteksi umur dan gender berdasarkan "
        "citra panoramic dental X-ray."
    )

    st.write(
        f"Perangkat komputasi: {device}"
    )


# =========================================================
# 13. INISIALISASI SESSION STATE
# =========================================================

if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None

if "current_file_hash" not in st.session_state:

    st.session_state.current_file_hash = None


# =========================================================
# 14. UPLOAD CITRA
# =========================================================

st.subheader("1. Unggah Citra Panoramic Dental X-Ray")

uploaded_file = st.file_uploader(
    "Pilih citra panoramic dental X-Ray",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, dan PNG."
)


if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = get_file_hash(file_bytes)

    # Menghapus hasil lama jika gambar berubah.
    if current_hash != st.session_state.current_file_hash:

        st.session_state.current_file_hash = current_hash

        st.session_state.analysis_result = None

    try:

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Gagal membaca gambar: {e}"
        )

        st.stop()

    # ---------------------------------------------
    # TAMPILAN CITRA
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Citra Dental X-Ray Original",
            width="stretch"
        )

    with col2:

        preview = resize_letterbox(
            np.array(image),
            target_hw=(IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        st.image(
            preview,
            caption="Citra Setelah Preprocessing",
            width="stretch"
        )

    st.divider()

    # =================================================
    # 15. TOMBOL ANALISIS
    # =================================================

    st.subheader("2. Analisis Citra")

    if st.button(
        "🔍 Mulai Analisis",
        type="primary",
        width="stretch"
    ):

        try:

            with st.spinner(
                "Model sedang menganalisis citra..."
            ):

                result = predict(
                    image,
                    model,
                    device
                )

                st.session_state.analysis_result = result

            st.success(
                "Analisis berhasil dilakukan!"
            )

        except Exception as e:

            st.session_state.analysis_result = None

            st.error(
                f"Terjadi kesalahan saat analisis: {e}"
            )

            with st.expander("Detail kesalahan"):

                st.code(
                    traceback.format_exc()
                )

    # =================================================
    # 16. MENAMPILKAN HASIL
    # =================================================

    if st.session_state.analysis_result is not None:

        results = st.session_state.analysis_result

        st.divider()

        st.subheader("3. Hasil Prediksi")

        # ---------------------------------------------
        # HASIL GENDER DAN UMUR
        # ---------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Prediksi Gender",
                value=results["gender"]
            )

            st.metric(
                label="Probabilitas Gender",
                value=(
                    f"{results['gender_probability'] * 100:.2f}%"
                )
            )

        with col2:

            st.metric(
                label="Estimasi Umur",
                value=f"{results['age']} Tahun"
            )

            st.metric(
                label="Ukuran Input Model",
                value="384 × 768"
            )

        st.divider()

        # ---------------------------------------------
        # VISUALISASI GRAD-CAM++
        # ---------------------------------------------

        st.subheader(
            "4. Visualisasi Grad-CAM++"
        )

        st.markdown(
            """
            Visualisasi Grad-CAM++ digunakan untuk
            mengidentifikasi area citra yang memberikan
            kontribusi terhadap keluaran model pada
            masing-masing tugas prediksi.
            """
        )

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:

            st.markdown(
                "**Grad-CAM++ Gender**"
            )

            st.image(
                results["gender_cam"],
                width="stretch"
            )

        with cam_col2:

            st.markdown(
                "**Grad-CAM++ Estimasi Umur**"
            )

            st.image(
                results["age_cam"],
                width="stretch"
            )

        st.divider()

        # ---------------------------------------------
        # RESET HASIL
        # ---------------------------------------------

        if st.button(
            "🔄 Hapus Hasil Analisis",
            width="stretch"
        ):

            st.session_state.analysis_result = None

            cleanup_memory(device)

            st.rerun()

else:

    # Reset ketika upload dihapus.
    st.session_state.current_file_hash = None
    st.session_state.analysis_result = None

    st.info(
        "Silakan unggah citra panoramic dental X-Ray "
        "untuk memulai analisis."
    )


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    """
    Sistem ini merupakan prototipe penelitian akademik.
    Hasil prediksi merupakan estimasi berbasis model
    pembelajaran mendalam dan bukan pengganti pemeriksaan
    atau penilaian profesional.
    """
)
