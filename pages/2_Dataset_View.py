import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import numpy as np
import plotly.express as px

from scipy.stats import norm, randint, skewnorm, expon


datasets = {
    'Dataset 1': {'url': 'https://raw.githubusercontent.com/lpanavas/DPEducationDatasets/master/PUMS_california_demographics_1000.csv', 'description': 'This is dataset of california demographics.'},
    'Dataset 2': {'url': 'https://raw.githubusercontent.com/ryanschaub/US-Census-Demographic-Data/master/acs2015_county_data.csv', 'description': 'In depth demographic dataset'},
    'Dataset 3': {'url': 'https://github.com/user/dataset3', 'description': 'This is dataset 3.'},
    'Dataset 4': {'url': 'https://github.com/user/dataset4', 'description': 'This is dataset 4.'},
    'Dataset 5': {'url': 'https://github.com/user/dataset5', 'description': 'This is dataset 5.'},
    'Dataset 6': {'url': 'https://github.com/user/dataset6', 'description': 'This is dataset 6.'}
}

st.set_page_config(layout="wide")
# Create the main app and tabs
st.title('Dataset Selection')

# Use the `tabs` function to create tabs
tab1, tab2, tab3 = st.tabs(["PreLoaded Datasets", "Synthetic Dataset", "Upload Dataset"])

# Fill content for Tab 1
with tab1:
    st.header('PreLoaded Datasets')
    st.write('Please select one of the uploaded datasets to proceed.')
    keys = list(datasets.keys())
    for i in range(0, len(datasets), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(datasets):
                dataset_name = keys[i + j]
                with col:
                    with st.container(border= True):
                        st.subheader(dataset_name)
                        st.write(datasets[dataset_name]['description'])
                        st.markdown(f"[Dataset URL]({datasets[dataset_name]['url']})")
                        # Add a select button to each container
                        select_button = st.button(f"Select {dataset_name}")
                        if select_button:
                            st.session_state['pages']['dataset'] = True
                            st.session_state.selected_dataset = pd.read_csv(datasets[dataset_name]['url'])
                            st.session_state['dataset_url'] = datasets[dataset_name]['url']
                            switch_page("Query Selection")

# Fill content for Tab 2
from scipy.stats import norm, randint, expon




def generate_categorical_column(n_records, n_categories, category_weights=None):
    if not category_weights:
        category_weights = [1/n_categories] * n_categories
    categories = np.arange(n_categories)
    return np.random.choice(categories, size=n_records, p=category_weights)

with tab2:
    st.header("Synthetic Dataset Generator")

    n_records = st.number_input("Number of Records:", min_value=1, value=100)

    if 'synthetic_data' not in st.session_state:
        st.session_state['synthetic_data'] = {}

    col1, col2 = st.columns(2)

    with col1:
        col_name = st.text_input("Column Name:")
        data_type = st.selectbox("Data Type:", ["Numeric", "Categorical"])

        if data_type == "Numeric":
            data_format =  st.selectbox("Data Format:", ["Integer", "Float"])
            distribution = st.selectbox("Distribution:", ["Normal", "Uniform", "Skewed"])

            if distribution == "Normal":
                mean = st.slider("Mean:", -100, 100, 0)
                std = st.slider("Standard Deviation:", 0.1, 50.0, 1.0)
                low = st.number_input("Low:", value=-50)
                high = st.number_input("High:", value=50)
            elif distribution == "Uniform":
                low = st.number_input("Low:", value=-50)
                high = st.number_input("High:", value=50)
            elif distribution == "Skewed":
                a = st.slider("Skew:", -10.0, 10.0, 0.0)
                low = st.number_input("Low:", value=-50)
                high = st.number_input("High:", value=50)

        elif data_type == "Categorical":
            n_categories = st.slider("Number of Categories:", 2, 10, 3)
            category_weights = [st.slider(f"Weight for Category {i+1}:", 1, 100, 25) for i in range(n_categories)]

        if st.button("Add Column") and col_name:
            if data_type == "Numeric":
                if distribution == "Normal":
                    col_data = []
                    while len(col_data) < n_records:
                        sample = norm.rvs(loc=mean, scale=std)
                        if low <= sample <= high:
                            if data_format == "Integer":
                                sample = round(sample)  # Convert to integer
                            col_data.append(sample)
                elif distribution == "Uniform":
                    col_data = randint.rvs(low, high+1, size=n_records)
                elif distribution == "Skewed":
                    col_data = skewnorm.rvs(a,  scale=a, size=n_records)
                # clip data to specified range
                # col_data = np.clip(col_data, low, high) 
            elif data_type == "Categorical":
                # normalize category weights
                normalized_weights = [float(i)/sum(category_weights) for i in category_weights]
                col_data = np.random.choice(range(n_categories), size=n_records, p=normalized_weights)
            st.session_state['synthetic_data'][col_name] = col_data

    with col2:
        st.header("Distribution Preview")
        if st.session_state['synthetic_data']:
            df = pd.DataFrame(st.session_state['synthetic_data'])
            col_to_preview = st.selectbox('Select column for preview', df.columns)
            fig = px.histogram(df, x=col_to_preview)
            st.plotly_chart(fig)       
    
with tab3:
    st.header('Upload Dataset')
    st.write('Please upload a dataset with column names as the first row. DO NOT UPLOAD SENSITIVE DATA.')
    st.file_uploader("Upload a CSV file", type=["csv"])