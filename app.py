import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageOps
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

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("📷 Upload Images")

    st.write(
        "Upload two images to compare the faces."
    )

    image1_file = st.file_uploader(
        "First Image",
        type=["jpg", "jpeg", "png"],
        key="image1"
    )

    image2_file = st.file_uploader(
        "Second Image",
        type=["jpg", "jpeg", "png"],
        key="image2"
    )

    st.divider()

    st.header("⚙️ Settings")

    threshold_multiplier = st.slider(
        "Threshold Multiplier",
        min_value=0.50,
        max_value=2.00,
        value=1.20,
        step=0.05,
        help=(
            "Higher values make the verification more lenient. "
            "Lower values make it stricter."
        )
    )

    st.caption(
        "💡 Increase this value if two images of the same "
        "person are incorrectly classified as different."
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
# Function: Prepare Image
# =========================================================

def prepare_image(image, size=(450, 450)):

    image = image.convert("RGB")

    # Keep aspect ratio while creating the same canvas size
    image = ImageOps.contain(
        image,
        size
    )

    # Create a fixed-size canvas
    canvas = Image.new(
        "RGB",
        size,
        "white"
    )

    # Center the image
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2

    canvas.paste(
        image,
        (x, y)
    )

    return canvas


# =========================================================
# Display Uploaded Images
# =========================================================

if image1_file is not None and image2_file is not None:

    image1 = Image.open(image1_file)
    image2 = Image.open(image2_file)

    display_image1 = prepare_image(image1)
    display_image2 = prepare_image(image2)

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.subheader("📷 First Image")

        st.image(
            display_image1,
            caption="First Image",
            width=450
        )

    with col2:

        st.subheader("📷 Second Image")

        st.image(
            display_image2,
            caption="Second Image",
            width=450
        )

else:

    st.info(
        "👈 Please upload both images from the sidebar."
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

            temp_dir = Path(
                tempfile.mkdtemp()
            )

            image1_path = temp_dir / "image1.jpg"
            image2_path = temp_dir / "image2.jpg"

            # Save first image
            image1.save(
                image1_path,
                format="JPEG"
            )

            # Save second image
            image2.save(
                image2_path,
                format="JPEG"
            )

            try:

                # =================================================
                # DeepFace Verification
                # =================================================

                result = DeepFace.verify(
                    img1_path=str(image1_path),
                    img2_path=str(image2_path),
                    detector_backend="retinaface"
                )

                # Original DeepFace values
                distance = result["distance"]
                original_threshold = result["threshold"]

                # =================================================
                # Custom Threshold
                # =================================================

                custom_threshold = (
                    original_threshold *
                    threshold_multiplier
                )

                # Our final decision
                verified = (
                    distance <= custom_threshold
                )

                # =================================================
                # Result
                # =================================================

                if verified:

                    st.markdown(
                        f"""
                        <div class="result-box same-person">
                            <div class="result-title">
                                ✅ Same Person
                            </div>
                            <p>
                                The two faces are similar enough
                                according to the selected threshold.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="result-box different-person">
                            <div class="result-title">
                                ❌ Different People
                            </div>
                            <p>
                                The distance between the two faces
                                is greater than the selected threshold.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # =================================================
                # Metrics
                # =================================================

                metric1, metric2, metric3, metric4 = st.columns(4)

                with metric1:

                    st.metric(
                        "Verification",
                        "Same Person"
                        if verified
                        else "Different"
                    )

                with metric2:

                    st.metric(
                        "Distance",
                        f"{distance:.4f}"
                    )

                with metric3:

                    st.metric(
                        "Original Threshold",
                        f"{original_threshold:.4f}"
                    )

                with metric4:

                    st.metric(
                        "Custom Threshold",
                        f"{custom_threshold:.4f}"
                    )

                # =================================================
                # Threshold Information
                # =================================================

                st.info(
                    f"""
                    **Threshold Multiplier:** `{threshold_multiplier:.2f}`

                    Original threshold:
                    `{original_threshold:.4f}`

                    Custom threshold:
                    `{custom_threshold:.4f}`

                    Distance:
                    `{distance:.4f}`
                    """
                )

                # =================================================
                # Explanation
                # =================================================

                with st.expander(
                    "ℹ️ How does Face Verification work?"
                ):

                    st.write(
                        """
                        DeepFace detects the faces using RetinaFace
                        and converts them into numerical embeddings.

                        The embeddings are then compared using a
                        distance metric.

                        A smaller distance means the two faces are
                        more similar.

                        The final decision is based on:

                            Distance <= Threshold

                        If this condition is satisfied, the images
                        are classified as belonging to the same person.

                        The Threshold Multiplier allows you to make
                        the verification more or less strict.

                        A higher multiplier makes the system more
                        lenient, while a lower multiplier makes it
                        stricter.
                        """
                    )

            except Exception as e:

                st.error(
                    f"""
                    ❌ An error occurred while processing the images:

                    {e}
                    """
                )
