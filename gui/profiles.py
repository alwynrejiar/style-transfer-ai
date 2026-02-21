
import streamlit as st
import os
import json
import glob
import math

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _radar_chart(categories, values, title="Style Fingerprint"):
    """Create a radar / spider chart for a set of named feature values."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name=title,
        line=dict(color='#636EFA'),
        fillcolor='rgba(99,110,250,0.25)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.15 if values else 1])),
        title=dict(text=title, x=0.5),
        margin=dict(l=60, r=60, t=60, b=40),
        height=420,
    )
    return fig


def _comparison_radar(categories, values_a, values_b, name_a, name_b, title="Profile Comparison"):
    """Radar chart overlaying two profiles."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_a + [values_a[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=name_a,
        line=dict(color='#636EFA'),
        fillcolor='rgba(99,110,250,0.15)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=values_b + [values_b[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=name_b,
        line=dict(color='#EF553B'),
        fillcolor='rgba(239,85,59,0.15)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(values_a), max(values_b)) * 1.15 if values_a else 1])),
        title=dict(text=title, x=0.5),
        margin=dict(l=60, r=60, t=60, b=40),
        height=460,
        legend=dict(orientation='h', y=-0.05),
    )
    return fig


def _heatmap_chart(labels, values, title="Function Word Frequency Heatmap"):
    """Create a 1-row heatmap for a vector of values (e.g. function-word freqs)."""
    # Reshape into a grid for readability (10 columns)
    cols = 10
    rows = math.ceil(len(labels) / cols)
    # Pad to fill grid
    padded_labels = labels + [''] * (rows * cols - len(labels))
    padded_values = values + [0.0] * (rows * cols - len(values))

    z = [padded_values[i * cols:(i + 1) * cols] for i in range(rows)]
    text = [padded_labels[i * cols:(i + 1) * cols] for i in range(rows)]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate='%{text}<br>%{z:.4f}',
        textfont=dict(size=10),
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='Freq'),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
        height=40 + rows * 55,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange='reversed'),
    )
    return fig


def _vocab_richness_bar(vr, title="Vocabulary Richness Metrics"):
    """Horizontal bar chart for vocabulary richness metrics."""
    # Normalize values to 0-1 range for display purposes
    display = {
        'Hapax Legomena': vr.get('hapax_legomena_ratio', 0),
        'Dis Legomena': vr.get('dis_legomena_ratio', 0),
        "Simpson's D": vr.get('simpsons_diversity', 0),
    }
    # These have different scales; show as-is with separate axis
    raw_metrics = {
        "Yule's K": vr.get('yules_k', 0),
        "Brunet's W": vr.get('brunet_w', 0),
        "Honore's R": vr.get('honore_r', 0),
    }

    fig = go.Figure()
    # Ratio metrics (0-1 scale)
    names = list(display.keys())
    vals = list(display.values())
    fig.add_trace(go.Bar(y=names, x=vals, orientation='h', name='Ratio (0-1)',
                         marker_color='#636EFA', text=[f'{v:.4f}' for v in vals], textposition='outside'))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='Value',
        margin=dict(l=120, r=20, t=50, b=30),
        height=250,
        showlegend=False,
    )
    return fig, raw_metrics


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def show():
    st.title("\U0001f464 Style Profiles")
    st.markdown("Manage your personal style fingerprints and saved profiles.")

    # List existing profiles
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profiles_dir = os.path.join(base_dir, 'stylometry fingerprints')

    if not os.path.exists(profiles_dir):
        os.makedirs(profiles_dir, exist_ok=True)

    profile_files = sorted(glob.glob(os.path.join(profiles_dir, "*.json")))

    # ---- Sidebar column ----
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Saved Profiles")
        if not profile_files:
            st.info("No profiles found. Create one by analyzing a text!")

        profile_basenames = [os.path.basename(f) for f in profile_files]
        selected_file = st.radio(
            "Select a profile to view:",
            profile_basenames,
            index=0 if profile_basenames else None,
        )

        if st.button("\u2795 Create New Profile"):
            st.info("Go to 'Analyze Style' and run an analysis to save a new profile.")

        # --- Compare two profiles ---
        if len(profile_basenames) >= 2:
            st.markdown("---")
            st.subheader("Compare Profiles")
            compare_a = st.selectbox("Profile A", profile_basenames, index=0, key="cmp_a")
            compare_b = st.selectbox("Profile B", profile_basenames,
                                     index=min(1, len(profile_basenames) - 1), key="cmp_b")
            compare_btn = st.button("Compare")
        else:
            compare_btn = False

    # ---- Detail column ----
    with col2:
        # --- Single profile view ---
        if selected_file and not compare_btn:
            file_path = os.path.join(profiles_dir, selected_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                st.markdown(f"### {selected_file.replace('.json', '')}")

                # Metadata
                meta = data.get('metadata', {})
                st.text(f"Created: {meta.get('analysis_date', 'Unknown')}")
                st.text(f"Method:  {meta.get('analysis_method', meta.get('processing_mode', 'Unknown'))}")

                # Basic stats
                stats = data.get('text_statistics', {})
                st.markdown("#### Statistics")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Word Count", stats.get('word_count', 0))
                mc2.metric("Lexical Diversity", f"{stats.get('lexical_diversity', 0):.2f}")
                mc3.metric("Unique Words", stats.get('unique_words', stats.get('vocabulary_size', '-')))

                # ---- Deep stylometry visualisations ----
                ds = data.get('deep_stylometry', {})
                if ds and PLOTLY_AVAILABLE:
                    st.markdown("---")
                    st.markdown("#### \U0001f4ca Deep Stylometry Visualisations")

                    tab1, tab2, tab3, tab4 = st.tabs([
                        "POS Radar", "Function Words Heatmap",
                        "Vocabulary Richness", "Scalar Features"
                    ])

                    # --- TAB 1: POS radar chart ---
                    with tab1:
                        pos = ds.get('pos_ratios', {})
                        if pos:
                            tags = sorted(pos.keys())
                            vals = [pos[t] for t in tags]
                            st.plotly_chart(_radar_chart(tags, vals, "POS Tag Distribution"),
                                            use_container_width=True)
                        else:
                            st.info("No POS ratio data available.")

                    # --- TAB 2: Function word heatmap ---
                    with tab2:
                        fw = ds.get('function_word_freq', {})
                        if fw:
                            # Sort by frequency descending
                            sorted_fw = sorted(fw.items(), key=lambda x: x[1], reverse=True)
                            labels = [k for k, _ in sorted_fw]
                            vals = [v for _, v in sorted_fw]
                            st.plotly_chart(_heatmap_chart(labels, vals),
                                            use_container_width=True)
                        else:
                            st.info("No function word data available.")

                    # --- TAB 3: Vocabulary richness ---
                    with tab3:
                        vr = ds.get('vocabulary_richness', {})
                        if vr:
                            bar_fig, raw = _vocab_richness_bar(vr)
                            st.plotly_chart(bar_fig, use_container_width=True)
                            st.markdown("**Scale-independent metrics:**")
                            rc1, rc2, rc3 = st.columns(3)
                            yk = raw.get("Yule's K", 0)
                            bw = raw.get("Brunet's W", 0)
                            hr = raw.get("Honore's R", 0)
                            rc1.metric("Yule's K", f"{yk:.2f}")
                            rc2.metric("Brunet's W", f"{bw:.2f}")
                            rc3.metric("Honore's R", f"{hr:.2f}")
                        else:
                            st.info("No vocabulary richness data available.")

                    # --- TAB 4: Scalar features ---
                    with tab4:
                        scalar_keys = [
                            ('avg_word_length', 'Avg Word Length'),
                            ('contraction_rate', 'Contraction Rate'),
                            ('passive_voice_ratio', 'Passive Voice Ratio'),
                            ('punctuation_density', 'Punctuation Density'),
                            ('quotation_density', 'Quotation Density'),
                            ('question_ratio', 'Question Ratio'),
                            ('exclamation_ratio', 'Exclamation Ratio'),
                            ('avg_dependency_depth', 'Avg Dependency Depth'),
                        ]
                        names = [label for _, label in scalar_keys]
                        vals = [ds.get(key, 0) for key, _ in scalar_keys]
                        st.plotly_chart(_radar_chart(names, vals, "Scalar Style Features"),
                                        use_container_width=True)

                        # Sentence length distribution
                        sl = ds.get('sentence_length_distribution', {})
                        if sl:
                            st.markdown("**Sentence Length Distribution**")
                            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                            sc1.metric("Mean", f"{sl.get('mean', 0):.1f}")
                            sc2.metric("Median", f"{sl.get('median', 0):.1f}")
                            sc3.metric("Std Dev", f"{sl.get('std_dev', 0):.1f}")
                            sc4.metric("Min", sl.get('min', 0))
                            sc5.metric("Max", sl.get('max', 0))

                elif ds and not PLOTLY_AVAILABLE:
                    st.warning("Install `plotly` for interactive visualisations: `pip install plotly`")

                # Deep analysis summary
                st.markdown("#### Deep Analysis Summary")
                analysis = data.get('consolidated_analysis', '')
                if isinstance(analysis, str):
                    st.markdown(analysis[:500] + "..." if len(analysis) > 500 else analysis)
                else:
                    st.json(analysis)

            except Exception as e:
                st.error(f"Error loading profile: {str(e)}")

            # Delete button
            st.markdown("---")
            if st.button("\U0001f5d1\ufe0f Delete Profile", type="primary"):
                try:
                    os.remove(file_path)
                    st.success(f"Deleted {selected_file}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

        # --- Compare two profiles view ---
        elif compare_btn:
            try:
                path_a = os.path.join(profiles_dir, compare_a)
                path_b = os.path.join(profiles_dir, compare_b)
                with open(path_a, 'r', encoding='utf-8') as f:
                    data_a = json.load(f)
                with open(path_b, 'r', encoding='utf-8') as f:
                    data_b = json.load(f)

                ds_a = data_a.get('deep_stylometry', {})
                ds_b = data_b.get('deep_stylometry', {})

                if not ds_a or not ds_b:
                    st.warning("One or both profiles lack deep stylometry data (requires v1.3.0+).")
                else:
                    st.markdown(f"### Comparing: **{compare_a}** vs **{compare_b}**")

                    # Compute similarity (import from src)
                    try:
                        import sys as _sys
                        _sys.path.insert(0, base_dir)
                        from src.analysis.metrics import calculate_style_similarity
                        similarity = calculate_style_similarity(ds_a, ds_b)
                    except Exception:
                        similarity = None

                    if similarity:
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Combined", f"{similarity['combined_score']:.1%}")
                        s2.metric("Cosine", f"{similarity['cosine_similarity']:.4f}")
                        s3.metric("Burrows' \u0394", f"{similarity['burrows_delta']:.4f}")
                        s4.metric("N-gram", f"{similarity['ngram_overlap']:.4f}")

                    if PLOTLY_AVAILABLE:
                        # Overlaid POS radar
                        pos_a = ds_a.get('pos_ratios', {})
                        pos_b = ds_b.get('pos_ratios', {})
                        if pos_a and pos_b:
                            tags = sorted(set(pos_a.keys()) | set(pos_b.keys()))
                            va = [pos_a.get(t, 0) for t in tags]
                            vb = [pos_b.get(t, 0) for t in tags]
                            st.plotly_chart(
                                _comparison_radar(tags, va, vb, compare_a, compare_b, "POS Distribution Comparison"),
                                use_container_width=True
                            )

                        # Scalar features overlay
                        scalar_keys = [
                            ('avg_word_length', 'Avg Word Length'),
                            ('contraction_rate', 'Contraction Rate'),
                            ('passive_voice_ratio', 'Passive Voice'),
                            ('punctuation_density', 'Punct Density'),
                            ('question_ratio', 'Question Ratio'),
                            ('exclamation_ratio', 'Exclamation Ratio'),
                        ]
                        names = [label for _, label in scalar_keys]
                        va = [ds_a.get(k, 0) for k, _ in scalar_keys]
                        vb = [ds_b.get(k, 0) for k, _ in scalar_keys]
                        st.plotly_chart(
                            _comparison_radar(names, va, vb, compare_a, compare_b, "Scalar Features Comparison"),
                            use_container_width=True
                        )
                    else:
                        st.warning("Install `plotly` for comparison charts.")

            except Exception as e:
                st.error(f"Comparison error: {e}")

        else:
            st.write("Select a profile from the left to view details.")
