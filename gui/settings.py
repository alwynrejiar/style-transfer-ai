
import streamlit as st
import os

# We would ideally load/save these to a config file. 
# For now, we use session state or environment variables.

def show():
    st.title("⚙️ Settings")
    
    st.subheader("🤖 AI Model Configuration")
    
    # Local Model Settings
    st.markdown("#### Local Engine (Ollama)")
    ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
    local_model = st.selectbox(
        "Default Local Model", 
        ["gemma3:1b", "gpt-oss:20b", "llama3", "mistral"],
        index=0
    )
    
    st.markdown("---")
    
    # Cloud API Settings
    st.markdown("#### Cloud APIs (Optional)")
    
    with st.expander("OpenAI Configuration"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("Key temporarily set for this session")
            
    with st.expander("Google Gemini Configuration"):
        gemini_key = st.text_input("Gemini API Key", type="password")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            st.success("Key temporarily set for this session")
            
    st.markdown("---")
    
    if st.button("Save Configuration"):
        # Here we would save to a config.json or .env
        st.success("Settings saved! (Mock)")

