# run: BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py

import streamlit as st

st.set_page_config(page_title="MalariaScope", page_icon="🔬", layout="wide")

pages = [
    st.Page("predict.py", title="Check a cell", icon="🔬", default=True),
    st.Page("pipeline.py", title="Pipeline", icon="⚙️"),
]
st.navigation(pages).run()
