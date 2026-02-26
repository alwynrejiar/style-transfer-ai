
import streamlit as st
import pandas as pd
import sys
import os

# Import backend logic
# Assuming src is in path from app.py
from src.analysis.analyzer import analyze_style
from src.analysis.analogy_engine import detect_conceptual_density, AnalogyInjector
from src.utils.text_processing import extract_basic_stats
from src.config.settings import ANALOGY_DOMAINS, DEFAULT_ANALOGY_DOMAIN

def show():
    st.title("🔍 Analyze Style")
    st.markdown("Deep dive into the stylometric fingerprint of your text.")

    # Input Section
    st.subheader("1. Input Text")
    
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
    
    text_to_analyze = ""
    
    if input_method == "Paste Text":
        text_to_analyze = st.text_area("Enter text to analyze:", height=200, placeholder="Paste your text here...")
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt', 'md'])
        if uploaded_file is not None:
            text_to_analyze = uploaded_file.read().decode("utf-8")
            st.info(f"Loaded {len(text_to_analyze)} characters from {uploaded_file.name}")

    # Analysis Configuration
    st.subheader("2. Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        # For now hardcoding to local Ollama as per preference
        model_type = st.selectbox("Model Type", ["Local (Ollama)", "Cloud (OpenAI/Gemini)"])
    
    with col2:
        if "Local" in model_type:
            model_name = st.text_input("Model Name", value="gemma3:1b")
        else:
            model_name = st.text_input("Model Name", value="gpt-4", disabled=True)

    # --- Cognitive Load Optimization ---
    st.subheader("3. Cognitive Load Optimization")
    analogy_enabled = st.toggle("Auto-Inject Contextual Analogies", value=False)
    domain_labels = {k: v["label"] for k, v in ANALOGY_DOMAINS.items()}
    analogy_domain = DEFAULT_ANALOGY_DOMAIN
    if analogy_enabled:
        selected_label = st.selectbox(
            "Base Domain",
            list(domain_labels.values()),
            index=list(domain_labels.keys()).index(DEFAULT_ANALOGY_DOMAIN),
        )
        # Reverse-lookup key from label
        analogy_domain = next(
            (k for k, v in domain_labels.items() if v == selected_label),
            DEFAULT_ANALOGY_DOMAIN,
        )

    # Analyze Button
    if st.button("🚀 Analyze Text", type="primary"):
        if not text_to_analyze.strip():
            st.error("Please enter some text to analyze.")
            return

        with st.spinner("Analyzing stylometric patterns... this may take a moment"):
            try:
                # 1. Basic Stats (Instant)
                stats = extract_basic_stats(text_to_analyze)
                
                # 2. Deep Analysis (AI)
                # We need to handle the fact that analyze_style expects specific args
                # Passing a mock user profile for now or fetching if available
                user_profile = {"name": "User", "culture": "General"}
                
                result = analyze_style(
                    text_to_analyze=text_to_analyze,
                    use_local=("Local" in model_type),
                    model_name=model_name if "Local" in model_type else None,
                    # API handling would go here
                    user_profile=user_profile,
                    processing_mode="enhanced" # or "statistical"
                )
                
                st.success("Analysis Complete!")
                
                # Store in session state for persistence
                st.session_state['analysis_result'] = result
                st.session_state['analysis_stats'] = stats
                
                # --- RESULTS DISPLAY ---
                
                # Basic Metrics Row
                st.markdown("### 📊 Key Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Word Count", stats.get('word_count', 0))
                m2.metric("Avg Sentence Length", f"{stats.get('avg_sentence_length', 0):.1f}")
                m3.metric("Paragraphs", stats.get('paragraph_count', 0))
                # m4.metric("Lexical Diversity", result.get('lexical_diversity', 'N/A')) # If available in result

                # Deep Analysis Output
                st.markdown("### 🧬 Stylometric Profile")
                
                # If result is a string (markdown), display it directly
                if isinstance(result, str):
                    st.markdown(result)
                elif isinstance(result, dict):
                    # If it returns a dict, we can visualize it better
                    st.json(result)

                # --- Cognitive Load / Analogy Engine ---
                density = detect_conceptual_density(text_to_analyze)
                st.markdown("### 🧠 Cognitive Load Optimization")
                col_d1, col_d2 = st.columns(2)
                col_d1.metric("Overall Density", f"{density['overall_density']:.3f}")
                col_d2.metric("Dense Passages", density['high_density_count'])

                if analogy_enabled and density['high_density_count'] > 0:
                    with st.spinner("Generating contextual analogies..."):
                        injector = AnalogyInjector(domain=analogy_domain)
                        analogy_result = injector.augment_text(
                            text_to_analyze,
                            use_local=("Local" in model_type),
                            model_name=model_name if "Local" in model_type else None,
                        )
                    if analogy_result['analogy_count'] > 0:
                        st.markdown("#### 🌉 Cognitive Bridging Notes")
                        for i, item in enumerate(analogy_result.get('analogies', []), 1):
                            with st.expander(
                                f"Note {i} — density {item['density_score']:.2f}"
                            ):
                                st.caption(f"**Dense passage:** {item['source_sentence']}")
                                st.info(f"💡 {item['analogy']}")
                    else:
                        st.info("No analogies generated — text is already accessible.")
                elif analogy_enabled:
                    st.info("No passages exceed the density threshold — no analogies needed.")
                
                # Download Report
                st.download_button(
                    label="Download Analysis Report",
                    data=str(result),
                    file_name="style_analysis_report.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)
        
    # --- Save Profile Section ---
    st.markdown("---")
    st.subheader("💾 Save as Style Profile")
    
    if 'analysis_result' not in st.session_state:
        st.info("ℹ️ Run an analysis above to enable saving.")
    else:
        result = st.session_state['analysis_result']
        stats = st.session_state.get('analysis_stats', {})
        
        # Input with session state key
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("Profile Name", placeholder="e.g., My Personal Style", key="profile_name_input")
        with col2:
            st.write("") # Spacer
            st.write("") # Spacer
            save_clicked = st.button("Save Profile")
        
        # Status message container
        status_container = st.empty()
        
        if save_clicked:
            profile_name = st.session_state.get("profile_name_input", "").strip()
            
            if not profile_name:
                status_container.error("⚠️ Please enter a profile name first!")
            else:
                try:
                    # Path logic
                    current_file = os.path.abspath(__file__)
                    gui_dir = os.path.dirname(current_file)
                    project_root = os.path.dirname(gui_dir)
                    save_dir = os.path.join(project_root, "stylometry fingerprints")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Sanitize
                    safe_name = "".join([c for c in profile_name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip()
                    safe_name = safe_name.replace(" ", "_")
                    
                    file_path = os.path.join(save_dir, f"{safe_name}.json")
                    
                    # Data prep
                    from datetime import datetime
                    import json
                    
                    profile_data = {}
                    if isinstance(result, dict):
                        profile_data = result
                    else:
                        profile_data = {
                            "metadata": {
                                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "analysis_method": "Enhanced Deep Stylometry (Ollama)",
                                "source_filename": "Manual Input"
                            },
                            "text_statistics": stats if stats else {},
                            "consolidated_analysis": result
                        }
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(profile_data, f, indent=4)
                        
                    status_container.success(f"✅ Profile saved successfully: {safe_name}")
                    st.session_state['last_saved_profile'] = safe_name
                    
                except Exception as e:
                    status_container.error(f"Failed to save: {str(e)}")

