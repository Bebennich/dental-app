
# =========================================================
# 11. LOAD MODEL
# =========================================================

with st.spinner("Initializing Dental AI Engine..."):
    model, device, error = load_model()

if model is None:
    st.error("Model gagal dimuat.")
    st.code(error)
    st.stop()


# =========================================================
# 12. CUSTOM CSS - MODERN MEDICAL DASHBOARD
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F7FB;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10243B 0%, #183B56 100%);
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Main typography */
h1, h2, h3 {
    color: #142D46;
}

p {
    color: #53677C;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #10243B, #176B87);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(16, 36, 59, 0.12);
}

.hero h1 {
    color: white !important;
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    color: #D4EAF2 !important;
    font-size: 15px;
    margin-bottom: 0;
}

/* Section header */
.section-title {
    color: #142D46;
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #74869A;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E5ECF3;
    border-radius: 14px;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 18px;
    border: 1px solid #E5ECF3;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #64788D;
}

div[data-testid="stMetricValue"] {
    color: #142D46;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 45px;
    transition: 0.2s;
}

.stButton > button[kind="primary"] {
    background: #147D9B;
    border: none;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: #105E78;
    color: white;
}

/* Upload */
div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #8DB8CB;
    border-radius: 14px;
    padding: 15px;
}

/* Image */
div[data-testid="stImage"] img {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #E2EAF1;
}

/* Footer */
.footer {
    text-align: center;
    color: #8192A3;
    font-size: 12px;
    padding: 20px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 13. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:20px 5px;">
        <div style="
            background:#1D506A;
            width:65px;
            height:65px;
            border-radius:18px;
            margin:auto;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:32px;">
            🦷
        </div>
        <h2 style="color:white; margin-top:12px;">
            DENTAL AI
        </h2>
        <p style="color:#B7D3E3; font-size:12px;">
            Xception Multi-task Intelligence
        </p>
    </div>
    <hr style="border-color:#456078;">
    """, unsafe_allow_html=True)

    st.markdown("### Model Information")

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

    **Output Tasks**

    - Gender Classification
    - Age Regression

    **Explainability**

    Grad-CAM++
    """)

    st.divider()

    st.markdown("### System Information")

    st.markdown(
        f"**Device:** `{device}`"
    )

    st.markdown(
        "**Model Status:** :green[Loaded]"
    )

    st.divider()

    st.caption(
        "Research Prototype | Xception Dental AI"
    )


# =========================================================
# 14. HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div style="
        color:#91D8E8;
        font-size:12px;
        font-weight:700;
        letter-spacing:2px;
        margin-bottom:10px;">
        ARTIFICIAL INTELLIGENCE • MEDICAL IMAGING
    </div>

    <h1>Dental Age & Gender Detection</h1>

    <p>
        Intelligent analysis of panoramic dental X-ray images
        using Xception Multi-task Learning.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 15. SESSION STATE
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None


# =========================================================
# 16. IMAGE UPLOAD
# =========================================================

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
    help="Supported formats: JPG, JPEG, PNG.",
    key="dental_image_upload"
)


if uploaded_file is not None:

    import hashlib
    import io

    file_bytes = uploaded_file.getvalue()

    current_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    # Reset hasil ketika gambar berubah.
    if current_hash != st.session_state.current_file_hash:

        st.session_state.current_file_hash = current_hash

        st.session_state.analysis_result = None

    image = Image.open(
        io.BytesIO(file_bytes)
    ).convert("RGB")

    processed_preview = resize_letterbox(
        np.array(image),
        target_hw=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        with st.container(border=True):

            st.markdown("**Original Image**")

            st.image(
                image,
                width="stretch"
            )

            st.caption(
                f"Original resolution: {image.width} × {image.height}"
            )

    with col2:

        with st.container(border=True):

            st.markdown("**Preprocessed Image**")

            st.image(
                processed_preview,
                width="stretch"
            )

            st.caption(
                "Letterbox resizing | 384 × 768"
            )


    # =====================================================
    # 17. ANALYSIS BUTTON
    # =====================================================

    st.markdown(
        '<div class="section-title">02. AI Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Run the Xception model to estimate age and classify gender.'
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
                "Image analysis completed successfully."
            )

        except Exception as e:

            st.session_state.analysis_result = None

            st.error(
                f"Analysis failed: {str(e)}"
            )

            import traceback

            with st.expander("Technical error details"):
                st.code(traceback.format_exc())


    # =====================================================
    # 18. PREDICTION RESULTS
    # =====================================================

    if st.session_state.analysis_result is not None:

        results = st.session_state.analysis_result

        st.divider()

        st.markdown(
            '<div class="section-title">03. Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Prediction results generated by the Xception Multi-task model.'
            '</div>',
            unsafe_allow_html=True
        )

        # ---------------------------------------------
        # RESULT CARDS
        # ---------------------------------------------

        col1, col2 = st.columns(2, gap="large")

        with col1:

            with st.container(border=True):

                st.markdown(
                    "### 👤 Gender Classification"
                )

                st.metric(
                    label="Predicted Gender",
                    value=results["gender"]
                )

                gender_probability = (
                    results["gender_probability"] * 100
                )

                st.progress(
                    min(
                        max(gender_probability / 100, 0.0),
                        1.0
                    )
                )

                st.markdown(
                    f"**Sigmoid output:** "
                    f"`{results['gender_probability']:.4f}`"
                )

                st.caption(
                    f"Probability of Male class: {gender_probability:.2f}%"
                )

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
                    "**Regression output**"
                )

                st.caption(
                    "Predicted normalized age multiplied by 100."
                )


        # =================================================
        # 19. GRAD-CAM VISUALIZATION
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">04. Model Explainability</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Grad-CAM++ visualization of feature activations '
            'for each prediction task.'
            '</div>',
            unsafe_allow_html=True
        )

        cam_col1, cam_col2 = st.columns(2, gap="large")

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
                    "Feature activation visualization for gender classification."
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
                    "Feature activation visualization for age regression."
                )


        # =================================================
        # 20. ANALYSIS SUMMARY
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">05. Analysis Summary</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            st.markdown("**Analysis Report**")

            report_col1, report_col2, report_col3 = st.columns(3)

            with report_col1:

                st.markdown("**Gender**")

                st.write(results["gender"])

            with report_col2:

                st.markdown("**Estimated Age**")

                st.write(f"{results['age']:.1f} years")

            with report_col3:

                st.markdown("**Gender Probability**")

                st.write(f"{gender_probability:.2f}%")

            st.caption(
                "The results represent model predictions and "
                "should not be interpreted as definitive clinical "
                "or forensic conclusions."
            )


        # =================================================
        # 21. RESET BUTTON
        # =================================================

        if st.button(
            "🔄  RESET ANALYSIS",
            width="stretch",
            key="reset_button"
        ):

            st.session_state.analysis_result = None

            st.rerun()


else:

    st.session_state.current_file_hash = None

    st.session_state.analysis_result = None

    # Empty state
    st.markdown("""
    <div style="
        background:white;
        border:1px dashed #A8C5D5;
        border-radius:15px;
        padding:40px 20px;
        text-align:center;
        margin-top:10px;">

        <div style="font-size:45px;">🦷</div>

        <h3 style="color:#183B56;">
            Ready for Dental Analysis?
        </h3>

        <p style="color:#8293A3; font-size:14px;">
            Upload a panoramic dental X-ray image above
            to start the prediction process.
        </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 22. FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    <hr>

    <b style="color:#176B87;">DENTAL AI</b>

    <br>

    Xception Multi-task Learning for Age and Gender Detection

    <br><br>

    Research Prototype | Academic Research Project

</div>
""", unsafe_allow_html=True)
