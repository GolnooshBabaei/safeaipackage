import streamlit as st

pg = st.navigation([
    st.Page("Home", title="Home", icon="🏠"),
    st.Page("Data Generator Agent", title="Data Generator", icon="🧪"),
    st.Page("Prediction Agent", title="Prediction", icon="🔮"),
])