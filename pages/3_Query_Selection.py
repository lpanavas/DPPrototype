import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
import charts
from typing import Dict, List

# Cache the data loading function


# Function to create and cache chart
def get_chart(data, column):
    if 'charts' not in st.session_state:
        st.session_state['charts'] = {}
    if column not in st.session_state['charts']:
        st.session_state['charts'][column] = charts.column_selection_charts(data, column)
    return st.session_state['charts'][column]

st.title("Query Selection")

# Function to update queries in session state
def update_queries(column, query_type, metadata=None):
    query_key = f"{column}_{query_type}"
    if metadata:
        if query_type in ['average', 'histogram']:
            other_query_type = 'histogram' if query_type == 'average' else 'average'
            other_query_key = f"{column}_{other_query_type}"
            if other_query_key in st.session_state['queries']:
                other_metadata = st.session_state['queries'][other_query_key]
                other_metadata['lower_bound'] = metadata['lower_bound']
                other_metadata['upper_bound'] = metadata['upper_bound']
                other_metadata['column'] = column
                other_metadata['epsilon'] = None
                other_metadata['url'] = st.session_state['dataset_url']
                st.session_state['queries'][other_query_key] = other_metadata
        st.session_state['queries'][query_key] = metadata
    else:
        if query_key in st.session_state['queries']:
            del st.session_state['queries'][query_key]

# Read your dataset
data = st.session_state.selected_dataset

# Initialize session state variables if not present
if 'queries' not in st.session_state:
    st.session_state['queries'] = {}
if 'selected_columns' not in st.session_state:
    st.session_state['selected_columns'] = []

# Ensure active page is set to 'Query_Selection'
st.session_state['active_page'] = 'Query_Selection'

# Instructions for using the tool
st.markdown("""
Welcome to the Query Selection Tool. Please follow these steps to specify your data analysis queries:

1. **Select Columns**: Use the checkboxes to select the columns you wish to analyze.
2. **Choose Analysis Type**: For each selected column, choose the type of analysis you are interested in - Count, Average, or Histogram.
3. **Input Metadata (if applicable)**: 
- For Average and Histogram analyses, specify the lower and upper bounds of the data.
- For Histogram analysis, also specify the number of bins.
4. **Visualize your data**: When you're ready, click the "Visualize the Data" button to see the visualizations of your selected queries.
""")

# Display the dataframe in a collapsible expander
with st.expander("Dataframe"):
    st.write(data)

# Header above the column checkboxes
st.subheader("Select Columns")

# Display columns with checkboxes and options
columns = data.columns.tolist()
num_cols = 5  # number of columns in the row structure
col_containers = st.columns(num_cols)

for i, col in enumerate(columns):
    with col_containers[i % num_cols]:
        if st.checkbox(f"Select {col}", key=f"checkbox_{col}"):
            if col not in st.session_state['selected_columns']:
                st.session_state['selected_columns'].append(col)
                st.session_state[f"dialog_open_{col}"] = True  # Initialize dialog state when column is selected
        else:
            if col in st.session_state['selected_columns']:
                st.session_state['selected_columns'].remove(col)
                for query_type in ['count', 'average', 'histogram']:
                    query_key = f"{col}_{query_type}"
                    if query_key in st.session_state['queries']:
                        del st.session_state['queries'][query_key]
                st.session_state[f"dialog_open_{col}"] = False  # Reset dialog state when column is deselected

# Dialog function to add queries
@st.experimental_dialog("Add Query", width="small")
def add_query(column):
    if f"metadata_{column}" not in st.session_state:
        st.session_state[f"metadata_{column}"] = {}

    metadata = st.session_state[f"metadata_{column}"]
    num_unique_values = data[column].nunique()
    options = ["Count", "Histogram"] if num_unique_values <= 50 else ["Count", "Average", "Histogram"]
    option_states = {option: st.checkbox(f"{option} for {column}") for option in options}

    if option_states.get("Average") or option_states.get("Histogram"):
        if num_unique_values > 50:
            columnType = data[column].dtype
            step = None if columnType == 'float64' else 1

            if st.button("Add default values"):
                metadata['lower_bound'] = data[column].min()
                metadata['upper_bound'] = data[column].max()
                if option_states.get("Histogram"):
                    metadata['bins'] = 10
                st.session_state[f"metadata_{column}"] = metadata

            metadata['lower_bound'] = st.number_input(f"Lower Bound of Column {column}", value=metadata.get('lower_bound', None), step=step)
            metadata['upper_bound'] = st.number_input(f"Upper Bound of Column {column}", value=metadata.get('upper_bound', None), step=step)
            if option_states.get("Histogram"):
                metadata['bins'] = st.number_input(f"Number of Histogram Bins for {column}", value=metadata.get('bins', None), min_value=1, step=1)
            metadata['data_type'] = 'continuous'
        else:
            metadata['data_type'] = 'categorical'
            st.write("In practice you would have to specify your bins but for the sake of simplicity we will automatically gather them from the data.")
    elif option_states.get("Count"):
        st.write("You're all set; no additional metadata to provide.")

    for option, selected in option_states.items():
        if selected:
            if option == "Count":
                update_queries(column, option.lower(), {'metadata': None})
            else:
                update_queries(column, option.lower(), metadata)
        else:
            update_queries(column, option.lower())

    if st.button("Submit"):
        st.session_state[f"dialog_open_{column}"] = False
        st.rerun()

# Update queries based on selected columns
for column in st.session_state['selected_columns']:
    if st.session_state.get(f"dialog_open_{column}", True):
        add_query(column)

# st.write(st.session_state)

# Button to visualize the data
dataset_view = st.button("Go to Visualization Page")
if dataset_view:
    if not st.session_state['queries']:
        st.error("Error: Please select at least one query.")
    else:
        bounds_issue_detected = False
        for query_key, query_info in st.session_state['queries'].items():
            if query_info.get('lower_bound') is not None and query_info.get('upper_bound') is not None:
                if query_info['lower_bound'] == query_info['upper_bound']:
                    st.error(f"Error in {query_key}: Lower and upper bounds cannot be the same.")
                    bounds_issue_detected = True
                    break
                if query_info['lower_bound'] > query_info['upper_bound']:
                    st.error(f"Error in {query_key}: Lower bound cannot be greater than upper bound.")
                    bounds_issue_detected = True
                    break
        if not bounds_issue_detected:
            switch_page("Visualizations")
