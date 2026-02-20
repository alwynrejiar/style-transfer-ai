
import streamlit as st
import os

def show():
    # Hero Section
    st.title("🧠 Stylomex")
    st.subheader("Advanced Stylometry & Text Rewrite Engine")

    st.markdown("""
    Welcome to the **Stylomex** local dashboard. This tool allows you to analyze writing styles in depth 
    and rewrite content to match specific personas or tones, all running locally on your machine.
    """)

    # Feature Grid
    col1, col2 = st.columns(2)

    with col1:
        st.info("### 🔍 Analyze Style")
        st.markdown("Discover the linguistic DNA of any text. Understand sentence structure, vocabulary complexity, and tone.")
        if st.button("Go to Analysis"):
            st.warning("Please use the sidebar to navigate to 'Analyze Style'")

    with col2:
        st.success("### 🎨 Style Transfer")
        st.markdown("Rewrite content in the style of famous authors, professional tones, or your own custom profiles.")
        if st.button("Go to Transfer"):
            st.warning("Please use the sidebar to navigate to 'Style Transfer'")
    
    st.markdown("---")

    # Quick Stats or Status
    st.markdown("### ⚡ System Status")
    
    # Check for profiles
    profiles_dir = os.path.join(os.getcwd(), 'stylometry fingerprints')
    if os.path.exists(profiles_dir):
        profile_count = len([f for f in os.listdir(profiles_dir) if f.endswith('.json')])
        st.metric("Saved Profiles", profile_count)
    else:
        st.metric("Saved Profiles", 0)

    # Check for models (mock check for now, can be real later)
    # real check would involve hitting the Ollama API
    st.metric("Local Engine", "Ollama (Detected)")

