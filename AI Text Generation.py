import streamlit as st
from transformers import pipeline

# Page settings
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖"
)

# Title
st.title("🤖 AI Text Generator")
st.write("Enter a sentence and let AI complete it!")

# Load model
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="MiniMaxAI/MiniMax-Music3"
    )

generator = load_model()

# User input
prompt = st.text_area(
    "Enter your text:",
    placeholder="Artificial Intelligence is..."
)

# Generate button
if st.button("Generate Text"):
    if prompt:
        with st.spinner("Generating..."):
            result = generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1
            )

        generated_text = result[0]["generated_text"]

        st.subheader("Generated Text")
        st.write(generated_text)

    else:
        st.warning("Please enter some text first!")