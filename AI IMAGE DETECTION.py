import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍"
)

st.title("🔍 AI Object Detection")
st.write("Upload an image and let AI identify the object.")

# Load model
model = YOLO("yolo11n.pt")

# Upload image
image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if image is not None:
    img = Image.open(image)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Identify Object"):

        with st.spinner("Identifying object..."):

            results = model(img)

            detected_objects = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    object_name = model.names[class_id]

                    if object_name not in detected_objects:
                        detected_objects.append(object_name)

            if detected_objects:
                st.success(
                    "This image contains: "
                    + ", ".join(detected_objects)
                )
            else:
                st.warning("I couldn't identify any object.")