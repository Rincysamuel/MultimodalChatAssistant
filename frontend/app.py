import streamlit as st
import requests
from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="LLaVA Multimodal Chat", page_icon="🤖")

st.title("🤖 LLaVA Multimodal Chat Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input section
with st.form(key="query_form"):
    query = st.text_input("Enter your question or prompt:")
    image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("Send")

if submit and query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query, "image": image})

    try:
        # Decide endpoint
        if image is not None:
            files = {"image": (image.name, image, image.type)}
            data = {"query": query}
            response = requests.post(f"{BACKEND_URL}/ask/image", data=data, files=files)
        else:
            response = requests.post(f"{BACKEND_URL}/ask/text", data={"query": query})

        if response.status_code == 200:
            resp_json = response.json()
            answer = resp_json.get("response", "No response from model.")
        else:
            answer = f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        answer = f"Request failed: {e}"

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer, "image": None})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            if msg["image"]:
                st.image(Image.open(msg["image"]), caption="User uploaded image")
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
