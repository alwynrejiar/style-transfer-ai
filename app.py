
import streamlit as st
import sys
import os

# Add src to path
# Add project root to path (one level up from gui/src)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import GUI modules
# using local imports inside functions or after path setup to avoid import errors
# if files don't exist yet, we will create them in next steps.

# Page Configuration
st.set_page_config(
    page_title="Stylomex",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("🧠 Stylomex")

pages = {
    "Home": "gui.home",
    "Analyze Style": "gui.analyze",
    "Style Transfer": "gui.transfer",
    "Profiles": "gui.profiles",
    "Settings": "gui.settings"
}

selection = st.sidebar.radio("Navigation", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.info(
    "**Version 1.3.0**\n"
    "Local-First Stylometry & Style Transfer"
)


# Routing
try:
    if selection == "Home":
        from gui import home
        home.show()
    elif selection == "Analyze Style":
        from gui import analyze
        analyze.show()
    elif selection == "Style Transfer":
        from gui import transfer
        transfer.show()
    elif selection == "Profiles":
        from gui import profiles
        profiles.show()
    elif selection == "Settings":
        from gui import settings
        settings.show()
except ImportError as e:
    st.error(f"Module for '{selection}' not found yet. Please implement `gui/{pages[selection].split('.')[-1]}.py`")
    st.error(f"Error: {e}")

