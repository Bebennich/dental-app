import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import gc
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dental X-Ray Biometrics",
    page_icon="🦷",
    layout="wide"
)

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI ARSITEKTUR MODEL & HELPER
# ==========================================
class HeadWrapper(nn.Module):
    def __init__(self, model, key):
        super().__init__()
        self.model = model
        self.key = key

    def forward(self, x):
        return self.model(x)[self.key].unsqueeze(1)

@st.cache_resource
def load_model():
    def make_head(in_features):
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    class MultiTaskNet(nn.Module):
        def __init__(self, backbone_name="legacy_xception", task_mode="multi", pretrained=False):
            super().__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            n_feat = self.backbone.num_features
            self.gender_head = make_head(n_feat)
            self.age_head = make_head(n_feat)

        def forward(self, x):
            feat = self.backbone(x)
            out = {}
            if hasattr(self, "gender_head"):
                out["gender_logit"] = self.gender_head(feat).squeeze(-1)
            if hasattr(self, "age_head"):
                out["age_norm"] = self.age_head(feat).squeeze(-1)
            return out

    device = torch.device("cpu")
    model = MultiTaskNet(backbone_name="legacy_xception", task_mode="multi", pretrained=False)
    
    try:
        state_dict = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, device
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. PRA-PEMROSESAN CITRA (Letterbox 384x768)
# ==========================================
def resize_letterbox(img, target_hw=(384, 768), interpolation=cv2.INTER_CUBIC):
    th, tw = target_hw
    h, w = img.shape[:2]
    scale = min(th / h, tw / w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    top = (th - new_h) // 2
    bottom = th - new_h - top
    left = (tw - new_w) // 2
    right = tw - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

eval_trans = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ==========================================
# 4. ANTARMUKA UTAMA
# ==========================================
model, load_status = load_model()

with st.sidebar:
    st.title("Tentang Sistem")
    st.info("""
    **Sistem Biometrik Dental X-Ray**
    
    Aplikasi pendukung demonstrasi penelitian skripsi menggunakan arsitektur **Xception Multi-task**.
    
    * **Input:** Citra Panoramik Gigi
    * **Output:** Estimasi Umur & Prediksi Gender
    * **Explainability:** Grad-CAM++
    """)
    st.markdown("---")
    st.caption("Muhammad Rifqi Rahmanda")

st.title("🦷 Deteksi Umur & Gender Berdasarkan Citra Panoramik")
st.markdown("Unggah citra rontgen panoramik gigi untuk melihat hasil prediksi beserta visualisasi Grad-CAM++.")

if model is None:
    st.error(f"Gagal memuat model `best_model.pth`. Pastikan file berada di direktori yang sama. Detail error: {load_status}")
else:
    uploaded_file = st.file_uploader("Pilih file gambar rontgen (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)
        
        col_img, col_res = st.columns([1, 2])
        
        with col_img:
            st.image(image_pil, caption="Citra Input Asli", width="stretch")
            analyze_btn = st.button("Mulai Analisis", type="primary", width="stretch")
        
        if analyze_btn:
            with st.spinner("Menjalankan inferensi model dan Grad-CAM++..."):
                # 1. Resize Letterbox
                img_resized = resize_letterbox(image_np, target_hw=(384, 768), interpolation=cv2.INTER_CUBIC)
                tensor = eval_trans(image=img_resized)["image"].unsqueeze(0).to(device)
                
                # 2. Inferensi
                with torch.no_grad():
                    outputs = model(tensor)
                    g_prob = torch.sigmoid(outputs["gender_logit"]).item()
                    gender_pred = "Laki-laki" if g_prob >= 0.5 else "Perempuan"
                    age_pred = outputs["age_norm"].item() * 100.0
                
                # 3. Grad-CAM++
                target_layers = [model.backbone.act4]
                targets = [ClassifierOutputTarget(0)]
                base_img = img_resized.astype(np.float32) / 255.0
                
                # Grad-CAM Gender
                with GradCAMPlusPlus(model=HeadWrapper(model, "gender_logit"), target_layers=target_layers) as cam_g:
                    gray_cam_g = cam_g(input_tensor=tensor, targets=targets)[0, :]
                vis_gender = show_cam_on_image(base_img, gray_cam_g, use_rgb=True)
                gc.collect()
                
                # Grad-CAM Umur
                with GradCAMPlusPlus(model=HeadWrapper(model, "age_norm"), target_layers=target_layers) as cam_a:
                    gray_cam_a = cam_a(input_tensor=tensor, targets=targets)[0, :]
                vis_age = show_cam_on_image(base_img, gray_cam_a, use_rgb=True)
                
                del tensor
                gc.collect()
                
                # 4. Tampilkan Hasil
                with col_res:
                    st.success("Analisis berhasil diselesaikan!")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(label="🧑 Prediksi Gender", value=gender_pred, delta=f"Prob: {g_prob:.2f}")
                    with m2:
                        st.metric(label="🎂 Estimasi Umur", value=f"{round(age_pred, 1)} Tahun")
                    
                    st.markdown("### 🔍 Hasil Interpretabilitas (Grad-CAM++)")
                    tab1, tab2, tab3 = st.tabs(["Citra Prapemrosesan", "Heatmap Gender", "Heatmap Umur"])
                    
                    with tab1:
                        st.image(img_resized, caption="Hasil Resize Letterbox (384x768)", width="stretch")
                    with tab2:
                        st.image(vis_gender, caption=f"Fokus model memprediksi {gender_pred}", width="stretch")
                    with tab3:
                        st.image(vis_age, caption=f"Fokus model mengestimasi umur {round(age_pred, 1)} thn", width="stretch")