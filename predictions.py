import streamlit as st
import pandas as pd
from os import path


DATA_DIR = "data_files/"

st.set_page_config(page_title="Betting Cleanup - Baseball Predictions", page_icon="⚾", layout="wide")

st.image(path.join(DATA_DIR, 'logo.png'), width=250)
st.title("Betting Cleanup - Baseball Predictions")

# Overview description for users
st.markdown(
    """
    **Betting Cleanup** is a lightweight Python-based machine learning
    system that generates daily MLB betting picks. On this site you can:

    - View model-generated **edges** compared to sportsbook odds
    - See confidence tiers and suggested bet sizing
    - Track historical picks and bankroll performance
    - Explore underlying data, features, and model evaluations

    """
)

