import streamlit as st
from openai import OpenAI
from huggingface_hub import InferenceClient
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from datetime import datetime
import io

st.set_page_config(
    page_title="Temperature Exploration Studio",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌡️ Temperature Exploration Studio")
st.markdown("Compare **Predictable (0.1)**, **Balanced (0.5)**, and **Creative (0.9)** AI responses side by side.")

GROQ_URL = "https://api.groq.com/openai/v1"

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

HF_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3-8B-Instruct",
    "openai/gpt-oss-120b",
    "MiniMaxAI/MiniMax-M1-80k",
    "deepseek-ai/DeepSeek-V3",
]

st.sidebar.header("⚙️ Model Configuration")
provider = st.sidebar.selectbox("AI Provider", ["Groq", "Hugging Face"])

default_key = ""
try:
    if provider == "Groq":
        default_key = st.secrets.get("GROQ_API_KEY", "")
    else:
        default_key = st.secrets.get("HF_API_KEY", "")
except Exception:
    default_key = ""

api_key = st.sidebar.text_input(
    f"{provider} API Key",
    value=default_key,
    type="password",
    help="Enter your API key or configure it in Streamlit Secrets."
)

max_tokens = st.sidebar.slider("Max Response Length (Tokens)", 100, 3000, 512, 64)

if st.sidebar.button("🗑️ Reset Experiment Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def transcribe_voice(audio_bytes) -> str:
    recognizer = sr.Recognizer()
    audio_file = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language="en-US")
    except sr.UnknownValueError:
        st.sidebar.warning("Could not understand audio. Please try speaking again.")
        return ""
    except Exception as e:
        st.sidebar.error(f"Audio error: {e}")
        return ""

def generate_response(prompt: str, temperature: float = 0.3, tokens: int = 512) -> tuple[str, str]:
    if not prompt or not prompt.strip():
        return "⚠️ Please enter a non-empty prompt.", "None"
    if not api_key:
        return f"⚠️ {provider} API key is missing. Add it in the left sidebar.", "None"

    models = GROQ_MODELS if provider == "Groq" else HF_MODELS
    last_error = None

    for model in models:
        try:
            if provider == "Groq":
                client = OpenAI(api_key=api_key, base_url=GROQ_URL)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=tokens
                )
                content = response.choices[0].message.content or "(The model returned an empty response.)"
                return content, model
            else:
                client = InferenceClient(token=api_key)
                response = client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=tokens
                )
                content = response.choices[0].message.content or "(The model returned an empty response.)"
                return content, model
        except Exception as error:
            last_error = error

    error_msg = (
        f"❌ All {provider} model attempts failed.\n"
        f"Last error: {type(last_error).__name__}: {last_error}"
        if last_error
        else "No model was configured."
    )
    return error_msg, "Failed"

tab1, tab2 = st.tabs(["🌡️ Temperature Experiment", "🔌 Connection Test"])

with tab1:
    st.subheader("Part 1: Temperature Exploration")
    
    voice_data = audio_recorder(text="Speak Creative Prompt", key="temp_mic")
    if voice_data:
        spoken_text = transcribe_voice(voice_data)
        if spoken_text:
            st.session_state["creative_prompt_input"] = spoken_text

    prompt = st.text_area(
        "Enter a creative prompt:",
        value=st.session_state.get("creative_prompt_input", ""),
        placeholder="e.g., Write a short story opening about a clockmaker who discovers a clock running backward.",
        height=100
    )

    if st.button("🚀 Run Temperature Experiment (0.1 vs 0.5 vs 0.9)", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            st.session_state["saved_prompt"] = prompt
            settings = [
                (0.1, "low", "LOW - More Predictable (0.1)"),
                (0.5, "med", "MEDIUM - Balanced (0.5)"),
                (0.9, "high", "HIGH - More Variation (0.9)"),
            ]
            cols = st.columns(3)
            for idx, (temp_val, key_prefix, label) in enumerate(settings):
                with cols[idx]:
                    with st.spinner(f"Running {label}..."):
                        res, used_model = generate_response(prompt, temperature=temp_val, tokens=max_tokens)
                        st.session_state[f"res_{key_prefix}"] = res
                        st.session_state[f"model_{key_prefix}"] = used_model

    if "res_low" in st.session_state:
        st.divider()
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("### ❄️ LOW (0.1)")
            st.caption(f"More predictable | Model: `{st.session_state.get('model_low', '')}`")
            st.info(st.session_state["res_low"])
            
        with c2:
            st.markdown("### ⚖️ MEDIUM (0.5)")
            st.caption(f"Balanced | Model: `{st.session_state.get('model_med', '')}`")
            st.success(st.session_state["res_med"])
            
        with c3:
            st.markdown("### 🔥 HIGH (0.9)")
            st.caption(f"More variation | Model: `{st.session_state.get('model_high', '')}`")
            st.warning(st.session_state["res_high"])

    st.divider()
    st.subheader("📝 Reflection")
    r1 = st.text_area("1. Which response was the most predictable?", height=80)
    r2 = st.text_area("2. Which response had the most creative variation?", height=80)
    r3 = st.text_area("3. Did a higher temperature always improve the answer?", height=80)

    report_text = f"""TEMPERATURE EXPLORATION EXPERIMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Provider: {provider} | Max Tokens: {max_tokens}
====================================================================

CREATIVE PROMPT:
{st.session_state.get('saved_prompt', prompt)}

--------------------------------------------------------------------
[1] LOW TEMPERATURE (0.1 - More Predictable)
Model Used: {st.session_state.get('model_low', 'N/A')}
Response:
{st.session_state.get('res_low', 'Not run yet.')}

--------------------------------------------------------------------
[2] MEDIUM TEMPERATURE (0.5 - Balanced)
Model Used: {st.session_state.get('model_med', 'N/A')}
Response:
{st.session_state.get('res_med', 'Not run yet.')}

--------------------------------------------------------------------
[3] HIGH TEMPERATURE (0.9 - More Variation)
Model Used: {st.session_state.get('model_high', 'N/A')}
Response:
{st.session_state.get('res_high', 'Not run yet.')}

====================================================================
REFLECTION ANSWERS
1. Most predictable response: {r1}
2. Most creative variation: {r2}
3. Did higher temperature always improve the answer?: {r3}
"""

    st.download_button(
        label="💾 Download Experiment Report (.txt)",
        data=report_text,
        file_name=f"temperature_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

with tab2:
    st.subheader("🔌 Test AI Connection")
    st.markdown("Verify that your selected provider and fallback models are responding properly.")
    test_prompt = st.text_input("Test Prompt:", value="Explain prompt engineering in two simple sentences.")
    if st.button("Run Connection Test"):
        with st.spinner("Testing AI connection..."):
            test_res, used_model = generate_response(test_prompt, temperature=0.3, tokens=100)
        st.caption(f"Model responded: `{used_model}`")
        st.write(test_res)
