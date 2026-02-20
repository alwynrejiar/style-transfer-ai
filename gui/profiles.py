
import streamlit as st
import os
import json
import glob

def show():
    st.title("👤 Style Profiles")
    st.markdown("Manage your personal style fingerprints and saved profiles.")

    # List existing profiles
    # Go up one level from gui/ to style-transfer-ai/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profiles_dir = os.path.join(base_dir, 'stylometry fingerprints')
    
    if not os.path.exists(profiles_dir):
        os.makedirs(profiles_dir, exist_ok=True)
        
    profile_files = glob.glob(os.path.join(profiles_dir, "*.json"))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Saved Profiles")
        if not profile_files:
            st.info("No profiles found. Create one by analyzing a text!")
        
        selected_file = st.radio(
            "Select a profile to view:", 
            [os.path.basename(f) for f in profile_files],
            index=0 if profile_files else None
        )
        
        if st.button("➕ Create New Profile"):
            st.info("Go to 'Analyze Style' and run an analysis to save a new profile.")

    with col2:
        st.subheader("Profile Details")
        if selected_file:
            file_path = os.path.join(profiles_dir, selected_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                st.markdown(f"### {selected_file.replace('.json', '')}")
                
                # Display metadata
                meta = data.get('metadata', {})
                st.text(f"Created: {meta.get('analysis_date', 'Unknown')}")
                st.text(f"Method: {meta.get('analysis_method', 'Unknown')}")
                
                # Display stats
                stats = data.get('text_statistics', {})
                st.markdown("#### Statistics")
                c1, c2 = st.columns(2)
                c1.metric("Word Count", stats.get('word_count', 0))
                c2.metric("Lexical Diversity", f"{stats.get('lexical_diversity', 0):.2f}")
                
                # Display deep analysis
                st.markdown("#### Deep Analysis Summary")
                analysis = data.get('consolidated_analysis', '')
                if isinstance(analysis, str):
                    st.markdown(analysis[:500] + "..." if len(analysis) > 500 else analysis)
                else:
                    st.json(analysis)
            
            except Exception as e:
                st.error(f"Error loading profile: {str(e)}")
            
            # Move delete logic outside the loading try-except block to avoid syntax errors
            st.markdown("---")
            if st.button("🗑️ Delete Profile", type="primary"):
                try:
                    os.remove(file_path)
                    st.success(f"Deleted {selected_file}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
        else:
            st.write("Select a profile from the left to view details.")

