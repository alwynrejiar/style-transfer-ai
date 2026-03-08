
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
        ["gemma3:1b", "llama3", "mistral"],
        index=0
    )
    
    st.markdown("---")
    
    if st.button("Save Configuration"):
        # Here we would save to a config.json or .env
        st.success("Settings saved! (Mock)")

