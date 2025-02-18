import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import numpy as np
import plotly.express as px

from scipy.stats import norm, randint, skewnorm, expon


import streamlit as st
import pandas as pd
if 'pages' not in st.session_state:
    st.session_state['pages'] = {'dataset': False, 'queries': False}
datasets = {
    'Adult Income': {
        'url': 'https://raw.githubusercontent.com/lpanavas/DPEducationDatasets/master/adult_income.csv',
        'description': 'This dataset contains information about individuals from the 1994 United States Census, including demographic features such as age, education level, marital status, and occupation, as well as their income level.',
        'count': 48842,
        'multiple_user_contributions': False,
        'domain': 'Census Data'
    },

    'Califorgnia Demographics': {
        'url': 'https://raw.githubusercontent.com/lpanavas/DPEducationDatasets/master/PUMS_california_demographics_1000.csv',
        'description': 'This dataset contains demographic information about California residents, including characteristics such as age, gender, race, education level, and income level.',
        'count': 1000,
        'multiple_user_contributions': False,
        'domain': 'General Demographics'
    },

    # 'Student Performance': {
    #     'url': 'https://raw.githubusercontent.com/lpanavas/DPEducationDatasets/master/student_performance.csv',
    #     'description': 'This dataset contains student grades and demographic information.',
    #     'count': 395,
    #     'multiple_user_contributions': False,
    #     'domain': 'Education'
    # },

    # 'Online Retail': {
    #     'url': 'https://raw.githubusercontent.com/lpanavas/DPEducationDatasets/master/online_retail.csv',
    #     'description': 'This dataset contains transactions from an online retailer in the UK.',
    #     'count': 541909,
    #     'multiple_user_contributions': False,
    #     'domain': 'Retail'
    # }
}
st.set_page_config(layout="wide")

# Create the main app and tabs
st.title('Dataset Selection')

# Use the `tabs` function to create tabs
tab1, tab2, tab3 = st.tabs(["PreLoaded Datasets", "Synthetic Dataset", "Upload Dataset"])

if 'selected_dataset' in st.session_state:
    del st.session_state['selected_dataset']

if 'queries' in st.session_state:
    del st.session_state['queries']


@st.cache_data()
def load_datasets(datasets):
    loaded_datasets = {}
    for dataset_name, dataset_info in datasets.items():
        with st.spinner(f"Loading {dataset_name}..."):
            loaded_datasets[dataset_name] = pd.read_csv(dataset_info['url'])
    return loaded_datasets

loaded_datasets = load_datasets(datasets)

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
                    with st.container(border=True):
                        st.subheader(dataset_name)
                        st.write(datasets[dataset_name]['description'])
                        st.markdown(f"**Count:** {datasets[dataset_name]['count']}")
                        st.markdown(f"**Multiple User Contributions:** {datasets[dataset_name]['multiple_user_contributions']}")
                        st.markdown(f"**Domain:** {datasets[dataset_name]['domain']}")
                        st.markdown(f"[Dataset URL]({datasets[dataset_name]['url']})")
                        # Display the first few rows of the dataset
                        st.dataframe(loaded_datasets[dataset_name].head(5))
                        # Add a select button to each container
                        select_button = st.button(f"Select {dataset_name}")
                        if select_button:
                            st.session_state['pages']['dataset'] = True
                            st.session_state.selected_dataset = loaded_datasets[dataset_name]
                            st.session_state['dataset_url'] = datasets[dataset_name]['url']
                            switch_page("Visualizations")

# Fill content for Tab 2
from scipy.stats import norm, randint, expon




def generate_categorical_column(n_records, n_categories, category_weights=None):
    if not category_weights:
        category_weights = [1/n_categories] * n_categories
    categories = np.arange(n_categories)
    return np.random.choice(categories, size=n_records, p=category_weights)

with tab2:
    st.header("Synthetic Dataset Generator")
    n_records_input = st.number_input("Number of Records:", min_value=1, value=100, key='n_records')
    if 'n_records' not in st.session_state:
        st.session_state['n_records'] = n_records_input

    if 'synthetic_data' not in st.session_state:
        st.session_state['synthetic_data'] = {}

    col1, col2 = st.columns(2)

    with col1:
        col_name = st.text_input("Column Name:")

        if col_name:
            if col_name in st.session_state['synthetic_data']:
                st.error("Column name already exists. Please choose a different name.")
            else:
                try:
                    data_type = st.selectbox("Data Type:", ["Numeric", "Categorical"])

                    if data_type == "Numeric":
                        data_format = st.selectbox("Data Format:", ["Integer", "Float"])
                        distribution = st.selectbox("Distribution:", ["Normal", "Uniform", "Skewed"])

                        low = st.number_input("Low:", value=0)
                        high = st.number_input("High:", value=50)

                        if distribution == "Normal":
                            mean = st.slider("Mean:", low, high, (low + high) // 2)
                            std = st.slider("Standard Deviation:", 0.1, (high - low) / 2, 1.0)
                        elif distribution == "Uniform":
                            pass  # No additional parameters needed for uniform distribution
                        elif distribution == "Skewed":
                            a = st.slider("Skew:", -10.0, 10.0, 0.0)

                    elif data_type == "Categorical":
                        n_categories = st.slider("Number of Categories:", 2, 10, 3)
                        category_weights = [st.slider(f"Weight for Category {i+1}:", 1, 100, 25) for i in range(n_categories)]

                    if data_type == "Numeric":
                        if distribution == "Normal":
                            col_data = []
                            while len(col_data) < st.session_state['n_records']:
                                sample = norm.rvs(loc=mean, scale=std)
                                if low <= sample <= high:
                                    if data_format == "Integer":
                                        sample = round(sample)  # Convert to integer
                                    col_data.append(sample)
                        elif distribution == "Uniform":
                            col_data = randint.rvs(low, high+1, size=st.session_state['n_records'])
                        elif distribution == "Skewed":
                            col_data = skewnorm.rvs(a, scale=a, size=st.session_state['n_records'])
                        # clip data to specified range
                        # col_data = np.clip(col_data, low, high)
                    elif data_type == "Categorical":
                        # normalize category weights
                        normalized_weights = [float(i)/sum(category_weights) for i in category_weights]
                        col_data = np.random.choice(range(n_categories), size=st.session_state['n_records'], p=normalized_weights)
                    st.session_state['synthetic_data'][col_name] = col_data
                except ValueError:
                    st.error("Invalid input. Please enter a valid number.")

    with col2:
        st.header("Distribution Preview")
        if 'synthetic_data' in st.session_state:
            df = pd.DataFrame(st.session_state['synthetic_data'])
            col_to_preview = st.selectbox('Select column for preview', df.columns)
            fig = px.histogram(df, x=col_to_preview)
            st.plotly_chart(fig)

            # Add a button to add the synthetic dataframe to the selected dataframe of session state
            if st.button("Use Synthetic Data"):
                st.session_state.selected_dataset = df
                switch_page("Visualizations")
        
# with tab3:
#     st.header('Upload Dataset')
#     st.write('Please upload a dataset with column names as the first row. DO NOT UPLOAD SENSITIVE DATA.')
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             if df.columns.str.contains('Unnamed').any():
#                 st.error("The uploaded file does not have headers as the top row.")
#             elif df.isnull().values.any():
#                 st.error("The uploaded file contains empty values.")
#             elif not all(dtype.name in ['int64', 'float64', 'object'] for dtype in df.dtypes):
#                 st.error("The uploaded file contains columns with unsupported data types.")
#             else:
#                 st.success("The uploaded file is valid.")
#                 if st.button("Use Uploaded Data"):
#                     st.session_state['selected_dataset'] = df
#                     switch_page("Query Selection")
#         except pd.errors.EmptyDataError:
#             st.error("The uploaded file is empty.")
#         except pd.errors.ParserError as e:
#             st.error(f"Error parsing the uploaded file: {e}")
with tab3:
    st.header('Upload Dataset')
    st.write('Please upload a dataset with column names as the first row. DO NOT UPLOAD SENSITIVE DATA.')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.columns.str.contains('Unnamed').any():
                st.error("The uploaded file does not have headers as the top row.")
            elif df.isnull().values.any():
                null_columns = df.columns[df.isnull().any()]
                st.error(f"The uploaded file contains empty values in the following columns: {', '.join(null_columns)}")
            elif not all(dtype.name in ['int64', 'float64', 'object'] for dtype in df.dtypes):
                unsupported_columns = df.columns[~df.dtypes.isin(['int64', 'float64', 'object'])]
                st.error(f"The uploaded file contains columns with unsupported data types: {', '.join(unsupported_columns)}")
            else:
                st.success("The uploaded file is valid.")
                if st.button("Go to Query Selection Page"):
                    st.session_state['selected_dataset'] = df
                    switch_page("Query Selection")
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
        except pd.errors.ParserError as e:
            st.error(f"Error parsing the uploaded file: {e}")
st.session_state['active_page'] = 'Dataset_View'