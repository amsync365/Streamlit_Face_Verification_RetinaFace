import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
from pathlib import Path


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Face Verification",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 35px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .same-person {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }

    .different-person {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }

    .result-title {
        font-size: 30px;
        font-weight: 700;
    }

    .metric-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #f5f5f5;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="main-title">🔍 Face Verification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare two faces using DeepFace and RetinaFace'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# Upload Section
# =========================================================

col1, col2 = st.columns(2, gap="large")


with col1:

    st.subheader("📷 First Image")

    image1_file = st.file_uploader(
        "Upload the first image",
        type=["jpg", "jpeg", "png"],
        key="image1"
    )

    if image1_file:

        image1 = Image.open(image1_file)

        st.image(
            image1,
            caption="First Image",
            use_container_width=True
        )


with col2:

    st.subheader("📷 Second Image")

    image2_file = st.file_uploader(
        "Upload the second image",
        type=["jpg", "jpeg", "png"],
        key="image2"
    )

    if image2_file:

        image2 = Image.open(image2_file)

        st.image(
            image2,
            caption="Second Image",
            use_container_width=True
        )


# =========================================================
# Compare Button
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

compare_col1, compare_col2, compare_col3 = st.columns(
    [1, 2, 1]
)

with compare_col2:

    compare_button = st.button(
        "🔍 Compare Faces",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# Verification
# =========================================================

if compare_button:

    if image1_file is None or image2_file is None:

        st.warning(
            "⚠️ Please upload both images before comparing."
        )

    else:

        with st.spinner(
            "🧠 DeepFace is analyzing the two faces..."
        ):

            # Temporary directory
            temp_dir = Path(tempfile.mkdtemp())

            image1_path = temp_dir / image1_file.name
            image2_path = temp_dir / image2_file.name

            # Save uploaded images
            with open(image1_path, "wb") as f:
                f.write(image1_file.getbuffer())

            with open(image2_path, "wb") as f:
                f.write(image2_file.getbuffer())

            try:

                # -----------------------------------------
                # DeepFace Verification
                # -----------------------------------------

                result = DeepFace.verify(
                    img1_path=str(image1_path),
                    img2_path=str(image2_path),
                    detector_backend="retinaface"
                )

                verified = result["verified"]
                distance = result["distance"]
                threshold = result["threshold"]

                # -----------------------------------------
                # Result
                # -----------------------------------------

                if verified:

                    st.markdown(
                        """
                        <div class="result-box same-person">
                            <div class="result-title">
                                ✅ Same Person
                            </div>
                            <p>
                                DeepFace found the two faces similar enough
                                to be considered the same person.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="result-box different-person">
                            <div class="result-title">
                                ❌ Different People
                            </div>
                            <p>
                                DeepFace found the two faces different.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # -----------------------------------------
                # Metrics
                # -----------------------------------------

                metric1, metric2, metric3 = st.columns(3)

                with metric1:

                    st.metric(
                        "Verification",
                        "Same Person" if verified
                        else "Different"
                    )

                with metric2:

                    st.metric(
                        "Distance",
                        f"{distance:.4f}"
                    )

                with metric3:

                    st.metric(
                        "Threshold",
                        f"{threshold:.4f}"
                    )

                # -----------------------------------------
                # Explanation
                # -----------------------------------------

                with st.expander("ℹ️ How does this work?"):

                    st.write(
                        """
                        DeepFace converts the detected faces into numerical
                        representations called embeddings and compares them.

                        A smaller distance means the two embeddings are more
                        similar.

                        DeepFace compares the calculated distance with the
                        model's threshold. If the distance satisfies the
                        threshold, the two images are classified as belonging
                        to the same person.
                        """
                    )

            except Exception as e:

                st.error(
                    f"❌ An error occurred while processing the images:\n\n{e}"
                )