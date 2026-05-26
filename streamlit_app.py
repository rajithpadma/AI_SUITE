import streamlit as st
from PIL import Image

from text_summarizer import TextSummarizer
from web_scraper import WebArticleScraper
from speech_to_text_and_neural_style_transfer import (
    NeuralStyleTransfer,
    NSTConfig,
    SpeechToText,
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Suite",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Suite")
st.caption(
    "Text summarizer, article scraping, speech-to-text, and neural style transfer."
)

# ---------------------------------------------------
# CACHE MODELS / SERVICES
# ---------------------------------------------------
@st.cache_resource
def get_summarizer():
    return TextSummarizer()


@st.cache_resource
def get_scraper():
    return WebArticleScraper()


@st.cache_resource
def get_speech_to_text():
    return SpeechToText()


@st.cache_resource
def get_nst():
    # Keep a single NST instance so VGG19 is loaded only once
    return NeuralStyleTransfer(NSTConfig())


# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab_summarize, tab_scrape, tab_stt_nst = st.tabs(
    [
        "Text Summarizer",
        "Article URL Scraper",
        "Speech to Text + Neural Style Transfer",
    ]
)

# ===================================================
# TEXT SUMMARIZER TAB
# ===================================================
with tab_summarize:

    st.subheader("📝 Text Summarizer")
    st.write("Summarize either pasted text or a provided article URL.")

    url_input = st.text_input("Article URL (optional)")

    text_input = st.text_area(
        "Or paste the article text (optional)",
        height=200
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        max_length = st.number_input(
            "Max length",
            min_value=50,
            max_value=512,
            value=225,
            step=5
        )

    with col2:
        min_length = st.number_input(
            "Min length",
            min_value=10,
            max_value=300,
            value=70,
            step=5
        )

    with col3:
        chunk_size_words = st.number_input(
            "Chunk size (words)",
            min_value=100,
            max_value=2000,
            value=512,
            step=50
        )

    summarize_btn = st.button(
        "Summarize",
        type="primary"
    )

    if summarize_btn:

        summarizer = get_summarizer()
        scraper = get_scraper()

        status = st.empty()

        def progress(msg: str):
            status.write(msg)

        try:

            if url_input.strip():

                status.write("Scraping article text...")

                title, article_text = scraper.fetch(
                    url_input.strip()
                )

                st.write("### Title")
                st.write(title)

                text_to_summarize = article_text

            else:
                text_to_summarize = (
                    text_input or ""
                ).strip()

            if not text_to_summarize:
                st.error(
                    "Please provide either a URL or some text."
                )
                st.stop()

            st.write("### Summary")

            summary = summarizer.summarize(
                text_to_summarize,
                max_length=int(max_length),
                min_length=int(min_length),
                chunk_size_words=int(chunk_size_words),
                progress_callback=progress,
            )

            st.write(summary)

        except Exception as e:
            st.error(f"Summarization failed: {e}")

# ===================================================
# WEB SCRAPER TAB
# ===================================================
with tab_scrape:

    st.subheader("🌐 Article URL Scraper")
    st.write(
        "Enter an article URL; the app extracts and cleans the paragraph text."
    )

    url_input_2 = st.text_input("Article URL")

    scrape_btn = st.button(
        "Scrape",
        type="primary"
    )

    if scrape_btn:

        scraper = get_scraper()

        try:

            title, text = scraper.fetch(
                url_input_2.strip()
            )

            st.write("### Title")
            st.write(title)

            st.write("### Extracted Text")
            st.write(text)

            st.download_button(
                "Download extracted text",
                data=text.encode("utf-8"),
                file_name="extracted_article.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"Scraping failed: {e}")

# ===================================================
# SPEECH TO TEXT + NST TAB
# ===================================================
with tab_stt_nst:

    # -----------------------------------------------
    # SPEECH TO TEXT
    # -----------------------------------------------
    st.subheader("🎤 Speech to Text")

    audio_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"]
    )

    language = st.selectbox(
        "Language",
        options=[
            "en-US",
            "en-GB",
            "hi-IN",
            "es-ES"
        ],
        index=0
    )

    if st.button("Transcribe"):

        if not audio_file:
            st.error("Please upload a WAV file.")

        else:

            stt = get_speech_to_text()

            status = st.empty()

            def progress(msg: str):
                status.write(msg)

            try:

                text = stt.transcribe_wav_bytes(
                    audio_file.getvalue(),
                    language=language,
                    progress_callback=progress,
                )

                st.write("### Transcription")
                st.write(text)

            except Exception as e:
                st.error(f"Transcription failed: {e}")

    # -----------------------------------------------
    # DIVIDER
    # -----------------------------------------------
    st.divider()

    # -----------------------------------------------
    # NEURAL STYLE TRANSFER
    # -----------------------------------------------
    st.subheader("🎨 Neural Style Transfer")

    st.write(
        "Upload a content image and a style image, then stylize."
    )

    c_img_file = st.file_uploader(
        "Content image",
        type=["png", "jpg", "jpeg"],
        key="content"
    )

    s_img_file = st.file_uploader(
        "Style image",
        type=["png", "jpg", "jpeg"],
        key="style"
    )

    # -----------------------------------------------
    # SETTINGS
    # -----------------------------------------------
    colA, colB, colC = st.columns(3)

    with colA:
        max_size = st.slider(
            "Max image size (longest side)",
            min_value=128,
            max_value=512,
            value=256,
            step=16
        )

    with colB:
        steps = st.slider(
            "Optimization steps (higher = slower)",
            min_value=10,
            max_value=120,
            value=40,
            step=10
        )

    with colC:
        lr = st.number_input(
            "LBFGS learning rate",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.3f"
        )

    # -----------------------------------------------
    # WEIGHTS
    # -----------------------------------------------
    weight_col1, weight_col2 = st.columns(2)

    with weight_col1:

        # FIXED ERROR HERE
        content_weight = st.number_input(
            "Content weight",
            min_value=100,
            max_value=50000,
            value=10000,
            step=500
        )

    with weight_col2:

        # FIXED ERROR HERE
        style_weight = st.number_input(
            "Style weight",
            min_value=1,
            max_value=50000,
            value=100,
            step=10
        )

    # -----------------------------------------------
    # STYLIZE BUTTON
    # -----------------------------------------------
    if st.button("Stylize"):

        if not c_img_file or not s_img_file:
            st.error(
                "Please upload both content and style images."
            )

        else:

            nst = get_nst()

            status = st.empty()

            def progress(msg: str):
                status.write(msg)

            try:

                content_img = Image.open(
                    c_img_file
                ).convert("RGB")

                style_img = Image.open(
                    s_img_file
                ).convert("RGB")

                cfg = NSTConfig(
                    steps=int(steps),
                    lr=float(lr),
                    content_weight=float(content_weight),
                    style_weight=float(style_weight),
                    max_size=int(max_size),
                )

                out = nst.stylize(
                    content_img,
                    style_img,
                    config=cfg,
                    progress_callback=progress,
                )

                st.image(
                    out,
                    caption="Neural Style Transfer Output",
                    use_container_width=True
                )

                import io as _io

                out_buf = _io.BytesIO()

                out.save(out_buf, format="PNG")

                st.download_button(
                    "Download Stylized Image (PNG)",
                    data=out_buf.getvalue(),
                    file_name="stylized_output.png",
                    mime="image/png",
                )

            except Exception as e:
                st.error(f"NST failed: {e}")