import streamlit as st
import torch
from diffusers import DiffusionPipeline

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI Image Generator")
st.write("Create an image using a text prompt!")

# Load the model only once
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5"
    )

    pipe = pipe.to(device)

    return pipe


# Text input
prompt = st.text_input(
    "Enter a prompt for the image:",
    "A cute cat sitting in a beautiful garden"
)

# Generate button
if st.button("🎨 Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):

            try:
                pipe = load_model()

                image = pipe(prompt).images[0]

                st.image(
                    image,
                    caption="Generated Image",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error generating image: {e}")