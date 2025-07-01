import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import pandas as pd

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Brandi - Visual Brand Intelligence", layout="centered")

# === TITLE ===
st.markdown("""
    <h1 style='text-align: center;
               font-size: 64px;
               font-family: "Helvetica Neue", sans-serif;
               font-weight: 900;
               color: #ff007f;
               letter-spacing: 2px;
               margin-bottom: 0;'>
         BRANDI
    </h1>
""", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:white;'>Visual Brand Intelligence</h3>",
            unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:white;'> Upload an Image or Take a Photo</h4>",
            unsafe_allow_html=True)

# === STATE INIT ===
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

# === FILE UPLOADER & CAMERA BUTTON ===
uploaded_file = st.file_uploader(
    "", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="upload_file")
photo_clicked = st.button(" Take Photo", key="take_photo_btn")

# === CUSTOM STYLE FOR BUTTONS INCLUDING FILE UPLOADER ===
st.markdown("""
    <style>
    .stButton>button {
        background-color: transparent;
        color: #ff007f;
        border: 2px solid #ff007f;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 700;
        font-size: 16px;
        width: 100%;
        transition: 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #ff007f;
        color: white;
    }

    .stDownloadButton button {
        background-color: transparent;
        color: #ff007f;
        border: 2px solid #ff007f;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }

    .stDownloadButton button:hover {
        background-color: #ff007f;
        color: white;
    }

    /* Match upload button to take photo style */
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
        background-color: transparent;
        color: #ff007f;
        border: 2px solid #ff007f;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 700;
        font-size: 16px;
        width: 100%;
        transition: 0.3s ease-in-out;
    }

    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
        background-color: #ff007f;
        color: white;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 0;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# === CAMERA INPUT ===
image = None
if photo_clicked:
    st.session_state.show_camera = True

if st.session_state.show_camera:
    cam = st.camera_input("Take a Photo")
    if cam:
        st.session_state.captured_image = cam
        st.session_state.show_camera = False

# === IMAGE SELECTION ===
if "captured_image" in st.session_state:
    image = Image.open(st.session_state.captured_image)
if uploaded_file:
    image = Image.open(uploaded_file)

# === IMAGE PREVIEW & DETECTION ===
if image:
    st.markdown("<h4 style='color:white;'> Image Preview</h4>",
                unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    if st.button(" Detect Logos", key="detect_button"):
        with st.spinner("Detecting logos..."):
            model = YOLO("models/best.pt")
            results = model(image_path)[0]
            annotated_path = image_path.replace(".jpg", "_annotated.jpg")
            results.save(filename=annotated_path)

        detections = []
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            brand = results.names[int(class_id)]
            detections.append({
                "Brand": brand,
                "Confidence": round(score, 2),
                "Bounding Box": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            })

        df = pd.DataFrame(detections)

        st.markdown("<h4 style='color:white;'> Detection Results</h4>",
                    unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        with open(annotated_path, "rb") as f:
            st.download_button(" Download Annotated Image", f,
                               file_name="annotated_result.jpg", mime="image/jpeg", key="download_button")
else:
    st.info(" Upload or take a photo to begin logo detection.")
# === METRICS & ANALYTICS SECTION ===

if image and "df" in locals() and not df.empty:
    # === Store session stats ===
    if "images_processed" not in st.session_state:
        st.session_state.images_processed = 0
    if "total_logos_detected" not in st.session_state:
        st.session_state.total_logos_detected = 0
    if "detection_log" not in st.session_state:
        st.session_state.detection_log = []

    st.session_state.images_processed += 1
    st.session_state.total_logos_detected += len(df)
    st.session_state.detection_log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "brands": df["Brand"].tolist(),
        "avg_conf": round(df["Confidence"].mean(), 2)
    })

    st.markdown("---")
    st.markdown("<h4 style='color:white;'> Detection Summary</h4>",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric(" Images Processed", st.session_state.images_processed)
    col2.metric(" Total Logos Detected", st.session_state.total_logos_detected)
    col3.metric(" Avg. Confidence", f"{df['Confidence'].mean():.2f}")

    # === Pie Chart: Brand Frequency ===
    brand_counts = df["Brand"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(brand_counts, labels=brand_counts.index,
            autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')
    st.markdown("####  Brand Distribution")
    st.pyplot(fig1)

    # === Bar Chart: Confidence by Brand ===
    avg_conf_by_brand = df.groupby(
        "Brand")["Confidence"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots()
    avg_conf_by_brand.plot(kind='bar', color="#ff007f", ax=ax2)
    ax2.set_ylabel("Avg. Confidence")
    ax2.set_title("Confidence per Brand")
    st.markdown("####  Confidence Levels")
    st.pyplot(fig2)

    # === Detection Log Table ===
    st.markdown("####  Detection Log")
    st.dataframe(pd.DataFrame(st.session_state.detection_log)
                 [::-1], use_container_width=True)
