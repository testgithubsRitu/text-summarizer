"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from src.summarizer import Summarizer

st.set_page_config(page_title="Text Summarizer", page_icon="📝")
st.title("📝 Automated Text Summarizer")
st.caption("Abstractive summaries with DistilBART; long documents are handled by chunked map-reduce.")


@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model():
    return Summarizer()


uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
default = uploaded.read().decode("utf-8") if uploaded else ""
text = st.text_area("Or paste text here", value=default, height=250)
max_len = st.slider("Max summary length (tokens)", 40, 250, 130)

if st.button("Summarize", disabled=not text.strip()):
    with st.spinner("Summarizing..."):
        summary = load_model().summarize(text, max_length=max_len)
    st.subheader("Summary")
    st.write(summary)
    st.caption(f"{len(text.split())} words in -> {len(summary.split())} words out")
