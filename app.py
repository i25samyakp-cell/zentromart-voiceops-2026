"""A small Streamlit interface for the ZentroMart VoiceOps agent.

Run with: streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.agent import ZentroMartAgent
from src.config import get_settings
from src.voice import GnaniVoice


st.set_page_config(page_title="ZentroMart Voice Support", page_icon="🎧", layout="centered")


def extension_for(uploaded_file: object) -> str:
    """Use the uploaded filename's extension, falling back to WAV."""
    name = str(getattr(uploaded_file, "name", ""))
    extension = Path(name).suffix.casefold()
    return extension if extension in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"} else ".wav"


def new_agent(lexical_only: bool) -> ZentroMartAgent:
    """Start a new, isolated conversation for the browser session."""
    settings = get_settings()
    settings.validate_dataset()
    return ZentroMartAgent(settings, use_embeddings=not lexical_only)


def process_audio(uploaded_file: object, agent: ZentroMartAgent) -> dict[str, object]:
    """Transcribe one browser upload, answer it, and return a Gnani audio reply."""
    settings = get_settings()
    settings.validate_dataset()
    suffix = extension_for(uploaded_file)

    with tempfile.TemporaryDirectory(prefix="zentromart_voice_") as temp_directory:
        temp_dir = Path(temp_directory)
        input_path = temp_dir / f"customer_audio{suffix}"
        reply_path = temp_dir / "zentromart_reply.wav"
        input_path.write_bytes(uploaded_file.getvalue())

        voice = GnaniVoice(settings)
        transcript = voice.transcribe(input_path, language_code="en-IN")
        if not transcript:
            return {
                "transcript": "",
                "reply": "Sorry, I didn't catch that. Could you say it again?",
                "audio": None,
                "sources": [],
            }

        result = agent.respond(transcript)
        voice.synthesize(result.reply, reply_path, language="en-IN", voice="Kaveri")
        return {
            "transcript": transcript,
            "reply": result.reply,
            "audio": reply_path.read_bytes(),
            "sources": result.sources,
        }


def process_text(message: str, agent: ZentroMartAgent) -> dict[str, object]:
    """Answer a typed message and synthesize the same voice response."""
    settings = get_settings()
    result = agent.respond(message)
    with tempfile.TemporaryDirectory(prefix="zentromart_text_") as temp_directory:
        reply_path = Path(temp_directory) / "zentromart_reply.wav"
        GnaniVoice(settings).synthesize(result.reply, reply_path, language="en-IN", voice="Kaveri")
        return {
            "transcript": message,
            "reply": result.reply,
            "audio": reply_path.read_bytes(),
            "sources": result.sources,
        }


def show_result(result: dict[str, object]) -> None:
    st.subheader("ZentroMart's reply")
    st.write(str(result["reply"]))

    audio = result.get("audio")
    if isinstance(audio, bytes):
        st.audio(audio, format="audio/wav")

    sources = result.get("sources", [])
    if sources:
        with st.expander("Sources used"):
            for source in sources:
                if isinstance(source, dict):
                    label = source.get("source_id", "Source")
                    title = source.get("title") or source.get("document") or "ZentroMart document"
                    st.write(f"- **{label}** — {title}")


def conversation_agent(lexical_only: bool) -> ZentroMartAgent:
    """Keep one agent in the browser session so it remembers follow-up turns."""
    if (
        "agent" not in st.session_state
        or st.session_state.get("lexical_only") != lexical_only
    ):
        st.session_state.agent = new_agent(lexical_only)
        st.session_state.lexical_only = lexical_only
        st.session_state.history = []
    return st.session_state.agent


def add_to_history(customer_message: str, result: dict[str, object]) -> None:
    st.session_state.history.append({"role": "user", "content": customer_message})
    st.session_state.history.append({"role": "assistant", "content": str(result["reply"])})


def show_history() -> None:
    for item in st.session_state.get("history", []):
        with st.chat_message(item["role"]):
            st.write(item["content"])


st.title("🎧 ZentroMart Voice Support")
st.caption("Speak or upload a customer question. Gnani handles speech; the grounded agent prepares the reply.")

with st.sidebar:
    st.header("Settings")
    lexical_only = st.checkbox(
        "Fast mode (recommended)",
        value=True,
        help="Uses local retrieval and avoids Gemini embedding quota limits.",
    )
    st.info("Your API keys remain in the local .env file and are never shown in this page.")
    if st.button("Start a new conversation"):
        st.session_state.pop("agent", None)
        st.session_state.pop("lexical_only", None)
        st.session_state.history = []
        st.rerun()

show_history()

voice_tab, text_tab = st.tabs(["Voice question", "Text question"])

with voice_tab:
    st.write("Record directly in the browser or upload a WAV, MP3, M4A, FLAC, OGG, or AAC file.")
    recording = st.audio_input("Record a customer question")
    uploaded = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "flac", "ogg", "aac"])
    selected_audio = recording if recording is not None else uploaded

    if st.button("Process voice question", type="primary", disabled=selected_audio is None):
        try:
            with st.spinner("Transcribing, finding policy evidence, and creating a voice reply..."):
                agent = conversation_agent(lexical_only)
                result = process_audio(selected_audio, agent)
            if result["transcript"]:
                st.subheader("Transcript")
                st.write(str(result["transcript"]))
                add_to_history(str(result["transcript"]), result)
            show_result(result)
        except Exception as error:
            st.error(f"Could not process the audio: {error}")

with text_tab:
    message = st.text_area(
        "Customer message",
        placeholder="For example: How fast are refunds to the ZentroMart Wallet?",
    )
    if st.button("Process text question", disabled=not message.strip()):
        try:
            with st.spinner("Finding policy evidence and creating a voice reply..."):
                agent = conversation_agent(lexical_only)
                result = process_text(message.strip(), agent)
            add_to_history(message.strip(), result)
            show_result(result)
        except Exception as error:
            st.error(f"Could not process the message: {error}")

