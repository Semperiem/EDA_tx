import streamlit as st
import pandas as pd
from pages import page1, page2

st.set_page_config(page_title="Multi-Page App", layout="wide")

# File uploader placed in the sidebar or main page
st.sidebar.title('Navigation')
uploaded_file = st.sidebar.file_uploader("Upload another dataset", type=["csv"])

# If you want to process the file before passing it (like loading it into DataFrame)
if uploaded_file is not None:
    import pandas as pd
    data = pd.read_csv(uploaded_file)
if uploaded_file is None:
    data = pd.read_csv('txn_history_dummysample.csv')  # or set a default

# Define pages with modified page functions to accept data
pages = {
    "Exploratory Data Analysis": page1.app,
    "Customer Segmentation Analysis": page2.app
}

selection = st.sidebar.radio("Go to", list(pages.keys()))

# Call the selected page function and pass the uploaded data
page = pages[selection]
page(data)  # Assuming each page function is modified to accept a data parameter
