
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import timm
import albumentations as A
import gc

from PIL import Image
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="Xception Dental AI",
    page_icon="🦷",
    layout="wide"
)

MODEL_PATH = "best_model.pth"
IMAGE_SIZE = (384, 768)
AGE_MAX = 100.0


# =========================================================
# 2. ARSITEKTUR MODEL XCEPTION MULTI-TASK
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
            num_classes=0,
            global_pool="avg"
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
# 3. WRAPPER UNTUK GRAD-CAM++
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
# 4. LOAD MODEL
# =========================================================

@st.cache_resource
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
            map_location=device,
            weights_only=True
        )

        # Mendukung checkpoint dengan beberapa format
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            else:
                state_dict = checkpoint

            # Mengatasi prefix DataParallel
            state_dict = {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }

            model.load_state_dict(state_dict)

        else:
            model = checkpoint

        model.to(device)
        model.eval()

        return model, device, None

    except Exception as e:

        return None, device, str(e)


# =========================================================
# 5. PREPROCESSING CITRA LETTERBOX
# =========================================================

def resize_letterbox(
    img,
    target_hw=(384, 768),
    interpolation=cv2.INTER_CUBIC
):

    th, tw = target_hw

    h, w = img.shape[:2]

    scale = min(th / h, tw / w)

    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

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
        target_hw=IMAGE_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    tensor = eval_trans(
        image=img_resized
    )["image"]

    tensor = tensor.unsqueeze(0)

    return tensor, img_resized


# =========================================================
# 7. PREDIKSI UMUR DAN GENDER
# =========================================================

def predict(image_pil, model, device):

    tensor, img_resized = preprocess_image(image_pil)

    tensor = tensor.to(device)

    # -----------------------------------------------------
    # Inferensi
    # -----------------------------------------------------

    with torch.no_grad():

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

    # -----------------------------------------------------
    # Grad-CAM++ Gender
    # -----------------------------------------------------

    target_layers = [
        model.backbone.act4
    ]

    targets = [
        ClassifierOutputTarget(0)
    ]

    base_img = (
        img_resized.astype(np.float32) / 255.0
    )

    with torch.enable_grad():

        gender_wrapper = HeadWrapper(
            model,
            "gender_logit"
        )

        with GradCAMPlusPlus(
            model=gender_wrapper,
            target_layers=target_layers
        ) as cam_gender:

            cam_g = cam_gender(
                input_tensor=tensor,
                targets=targets
            )[0]

        vis_gender = show_cam_on_image(
            base_img,
            cam_g,
            use_rgb=True
        )

        # -------------------------------------------------
        # Grad-CAM++ Age
        # -------------------------------------------------

        age_wrapper = HeadWrapper(
            model,
            "age_norm"
        )

        with GradCAMPlusPlus(
            model=age_wrapper,
            target_layers=target_layers
        ) as cam_age:

            cam_a = cam_age(
                input_tensor=tensor,
                targets=targets
            )[0]

        vis_age = show_cam_on_image(
            base_img,
            cam_a,
            use_rgb=True
        )

    # Membersihkan memori
    del tensor
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "gender": gender_pred,
        "gender_probability": gender_prob,
        "age": round(age_pred, 1),
        "gender_cam": vis_gender,
        "age_cam": vis_age,
        "processed_image": img_resized
    }


# =========================================================
# 8. TAMPILAN STREAMLIT
# =========================================================

st.title("🦷 Deteksi Umur dan Gender Manusia")

st.markdown(
    """
    Aplikasi berbasis **Xception Multi-task Learning**
    untuk melakukan estimasi umur dan klasifikasi gender
    berdasarkan citra *Panoramic Dental X-Ray*.

    Model dilengkapi dengan visualisasi **Grad-CAM++**
    untuk membantu melihat area citra yang berkontribusi
    terhadap hasil prediksi.
    """
)

st.divider()


# =========================================================
# 9. MEMUAT MODEL
# =========================================================

with st.spinner("Memuat model Xception..."):

    model, device, error = load_model()

if model is None:

    st.error(
        "Gagal memuat model Xception."
    )

    st.code(error)

    st.info(
        "Pastikan file best_model.pth berada "
        "dalam folder yang sama dengan app.py."
    )

    st.stop()

else:

    st.success(
        f"Model berhasil dimuat! Perangkat: {device}"
    )


# =========================================================
# 10. UPLOAD CITRA
# =========================================================

st.subheader("1. Unggah Citra Dental X-Ray")

uploaded_file = st.file_uploader(
    "Pilih citra panoramic dental X-ray",
    type=["jpg", "jpeg", "png"],
    help="Format citra yang didukung: JPG, JPEG, dan PNG."
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Citra Dental X-Ray Original",
            use_container_width=True
        )

    with col2:

        st.image(
            resize_letterbox(
                np.array(image),
                target_hw=IMAGE_SIZE
            ),
            caption="Citra Setelah Preprocessing",
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # 11. TOMBOL PREDIKSI
    # =====================================================

    st.subheader("2. Hasil Prediksi Model")

    if st.button(
        "🔍 Mulai Analisis",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Model sedang menganalisis citra..."
            ):

                results = predict(
                    image,
                    model,
                    device
                )

            st.success(
                "Analisis citra berhasil dilakukan!"
            )

            # ---------------------------------------------
            # HASIL PREDIKSI
            # ---------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    label="Prediksi Gender",
                    value=results["gender"]
                )

                st.metric(
                    label="Probabilitas Gender",
                    value=f"{results['gender_probability'] * 100:.2f}%"
                )

            with col2:

                st.metric(
                    label="Estimasi Umur",
                    value=f"{results['age']} Tahun"
                )

                st.metric(
                    label="Ukuran Citra",
                    value="384 × 768"
                )

            st.divider()

            # ---------------------------------------------
            # GRAD-CAM++
            # ---------------------------------------------

            st.subheader(
                "3. Visualisasi Grad-CAM++"
            )

            st.markdown(
                """
                Heatmap menunjukkan area citra yang memiliki
                kontribusi terhadap keluaran masing-masing
                task pada model.
                """
            )

            cam_col1, cam_col2 = st.columns(2)

            with cam_col1:

                st.markdown(
                    "**Grad-CAM++ - Gender**"
                )

                st.image(
                    results["gender_cam"],
                    use_container_width=True
                )

            with cam_col2:

                st.markdown(
                    "**Grad-CAM++ - Estimasi Umur**"
                )

                st.image(
                    results["age_cam"],
                    use_container_width=True
                )

            st.divider()

            st.caption(
                """
                Catatan: Hasil aplikasi merupakan estimasi
                model pembelajaran mendalam dan bukan
                pengganti pemeriksaan atau penilaian profesional.
                """
            )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat analisis: {e}"
            )

else:

    st.info(
        "Silakan unggah citra panoramic dental X-ray "
        "untuk memulai analisis."
    )


# =========================================================
# 12. INFORMASI MODEL
# =========================================================

with st.expander("Informasi Arsitektur Model"):

    st.write(
        """
        **Arsitektur:** Xception

        **Pendekatan:** Multi-task Learning

        **Backbone:** Legacy Xception (timm)

        **Task 1:** Klasifikasi Gender

        **Task 2:** Regresi Umur

        **Ukuran Input:** 384 × 768 piksel

        **Normalisasi:** ImageNet

        **Visualisasi:** Grad-CAM++

        **Output Gender:** Female / Male

        **Output Umur:** Tahun
        """
    )
