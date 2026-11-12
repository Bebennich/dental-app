import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. Definisi Arsitektur & Muat Model (Cache)
# ==========================================
# Gunakan st.cache_resource agar bobot model (~80MB) hanya dimuat 1 kali ke memori server
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
            if task_mode in ("multi", "gender"):
                self.gender_head = make_head(n_feat)
            if task_mode in ("multi", "age"):
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
        # Pastikan file best_model.pth ada di folder yang sama
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# ==========================================
# 2. Pra-pemrosesan (Sesuai Subbab 3.2.2.B)
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
# 3. Antarmuka Pengguna (UI) Streamlit
# ==========================================
st.set_page_config(page_title="Dental X-Ray Biometrics", page_icon="🦷", layout="centered")

st.title("🦷 Deteksi Umur & Gender")
st.markdown("Unggah citra rontgen panoramik gigi untuk memprediksi jenis kelamin dan estimasi umur menggunakan arsitektur **Xception Multi-task**.")

# Inisialisasi model
model, device = load_model()

# Komponen Unggah File
uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    # Tampilkan gambar yang diunggah
    st.image(image_pil, caption="Citra Rontgen Panoramik", use_container_width=True)
    
    # Tombol Eksekusi
    if st.button("Mulai Analisis", type="primary") and model is not None:
        with st.spinner("Memproses citra dan menjalankan inferensi..."):
            image_np = np.array(image_pil)
            
            # 1. Resize & Letterbox
            img_resized = resize_letterbox(image_np, target_hw=(384, 768), interpolation=cv2.INTER_CUBIC)
            
            # 2. Normalisasi & Konversi ke Tensor
            tensor = eval_trans(image=img_resized)["image"].unsqueeze(0).to(device)
            
            # 3. Inferensi
            with torch.no_grad():
                outputs = model(tensor)
                
                # Prediksi Gender
                g_prob = torch.sigmoid(outputs["gender_logit"]).item()
                gender_pred = "Laki-laki (Male)" if g_prob >= 0.5 else "Perempuan (Female)"
                
                # Prediksi Umur (Y_norm * 100)
                age_pred = outputs["age_norm"].item() * 100.0
            
            # 4. Tampilkan Hasil
            st.success("Analisis Selesai!")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="🧑 Prediksi Gender", value=gender_pred, delta=f"Prob. Laki-laki: {g_prob:.2f}", delta_color="off")
            with col2:
                st.metric(label="🎂 Estimasi Umur", value=f"{round(age_pred, 1)} Tahun")