# =========================================================
# This file is the main entrance point for the streamlit UI
# =========================================================

import streamlit as st
import pandas as pd

def Main():
    st.set_page_config(
        page_title="Nascar Race Predictions",
        layout="wide"
    )

    st.title("Nascar Race Predictions")



if __name__ == "__main__":
    Main()