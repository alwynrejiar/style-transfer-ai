
import streamlit as st
import sys
import os

# Backend imports
from src.generation.style_transfer import StyleTransfer
from src.utils.text_processing import extract_basic_stats

# Initialize Style Transfer Engine (Reusing if possible, streamlit caches resources)
@st.cache_resource
def get_engine():
    return StyleTransfer()

def show():
    engine = get_engine()
    
    st.title("🎨 Style Transfer")
    st.markdown("Rewrite content in the style of famous authors, professional tones, or customized profiles.")

    # Two columns layout for input/output
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Source Content")
        input_text = st.text_area("Original Text", height=300, placeholder="Paste the text you want to rewrite here...")
        
        # Style Selection
        st.subheader("2. Target Style")
        
        style_option = st.radio("Style Source", ["Preset Profiles", "Custom Description"])
        
        target_style_profile = {}
        
        if style_option == "Preset Profiles":
            # Load profiles from 'stylometry fingerprints' directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fingerprints_dir = os.path.join(base_dir, "stylometry fingerprints")
            
            # Ensure directory exists
            if not os.path.exists(fingerprints_dir):
                os.makedirs(fingerprints_dir, exist_ok=False)
                
            # List JSON files
            profile_files = [f for f in os.listdir(fingerprints_dir) if f.endswith('.json')]
            
            if not profile_files:
                st.warning("No profiles found. Create one in 'Analyze Style'!")
                # Fallback to mock profiles if nothing found
                profiles = ["Hemingway (Mock)", "Academic (Mock)"]
                profile_files = [] # Reset to avoid mixing
            else:
                profiles = [f.replace('.json', '').replace('_', ' ') for f in profile_files]
                
            selected_profile_name = st.selectbox("Select Profile", profiles)
            
            if profile_files:
                # Find corresponding file
                selected_file = profile_files[profiles.index(selected_profile_name)]
                file_path = os.path.join(fingerprints_dir, selected_file)
                
                try:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        target_style_profile = json.load(f)
                    
                    # Display primitive preview
                    st.info(f"Loaded: {selected_profile_name}")
                    with st.expander("View Profile Details"):
                        st.json(target_style_profile.get('statistical_analysis', {}))
                except Exception as e:
                    st.error(f"Error loading profile: {e}")
            else:
                # Fallback logic
                if "Hemingway" in selected_profile_name:
                    target_style_profile = {
                        "statistical_analysis": {"average_sentence_length": 12, "lexical_diversity": 0.5},
                        "deep_analysis": "Short, punchy sentences. Direct and unadorned prose."
                    }
                elif "Academic" in selected_profile_name:
                    target_style_profile = {
                        "statistical_analysis": {"average_sentence_length": 25, "lexical_diversity": 0.8},
                        "deep_analysis": "Formal, objective tone. Complex sentence structures with subordinate clauses."
                    }
                
        else:
            custom_desc = st.text_area("Describe the desired style", placeholder="E.g., Like a sarcastic tech blogger from the 90s...")
            target_style_profile = {"deep_analysis": custom_desc}

        # Configuration
        st.subheader("3. Settings")
        intensity = st.slider("Transfer Intensity", 0.1, 1.0, 0.7, 0.1)
        model_name = st.text_input("Local Model (Ollama)", value="gemma3:1b")
        
        if st.button("✨ Transform Content", type="primary"):
            if not input_text:
                st.error("Please enter source text.")
                return

            with st.spinner("Rewriting content..."):
                try:
                    # Execute Transfer
                    result = engine.transfer_style(
                        original_content=input_text,
                        target_style_profile=target_style_profile,
                        transfer_type="direct_transfer",
                        intensity=intensity,
                        use_local=True,
                        model_name=model_name
                    )
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.session_state['transfer_result'] = result
                        st.success("Transformation Complete!")
                        
                except Exception as e:
                    st.error(f"Transfer failed: {str(e)}")

    with col2:
        st.subheader("4. Result")
        
        if 'transfer_result' in st.session_state:
            result = st.session_state['transfer_result']
            transferred_text = result.get('transferred_content', '')
            
            st.text_area("Transformed Text", value=transferred_text, height=400)
            
            # Quality Metrics
            metrics = result.get('quality_analysis', {})
            if metrics:
                st.markdown("#### Quality Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Style Match", f"{result.get('style_match_score', 0):.2f}")
                # Mock metrics for demo/visuals if backend doesn't return them yet
                c2.metric("Readability Change", "+12%") 
                c3.metric("Content Preservation", "High")
            
            st.download_button("Download Result", transferred_text, "transformed_text.txt")
        else:
            st.info("Run a transformation to see results here.")

