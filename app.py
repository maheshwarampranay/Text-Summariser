import streamlit as st
from summarizer import (
    preprocess,
    extractive_summary,
    abstractive_summary,
    read_file,
)

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("📝 Text Summarizer")
st.markdown("Summarize long documents or pasted text using extractive or abstractive methods.")

# Input Method
input_method = st.radio("Choose input method:", ["Type or paste text", "Upload a file"])
text_input = ""

if input_method == "Type or paste text":
    text_input = st.text_area("Enter your text below:", height=300)
else:
    uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])
    if uploaded_file:
        text_input = read_file(uploaded_file)

# Summary Type
summary_type = st.radio("Choose summarization type:", ["Extractive", "Abstractive"])

if st.button("Summarize"):
    if text_input.strip():
        clean_text = preprocess(text_input)
        if summary_type == "Extractive":
            summary = extractive_summary(clean_text)
        else:
            summary = abstractive_summary(clean_text)
        st.subheader("📄 Summary:")
        st.write(summary)
    else:
        st.warning("Please provide text or upload a valid file.")
