import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
import charts

st.title("Query Selection")

data = st.session_state.selected_dataset


from typing import Dict, List

# Read your dataset (replace with your actual dataset)


# # Function to detect categorical variables
# def detect_categorical_variables(df):
#     categorical_vars = [col for col in df.columns if df[col].dtype == 'object']
#     return categorical_vars

# # Store the list of detected categorical variables
# categorical_vars = detect_categorical_variables(data)

# st.title("SQL-like Selection on Categorical Variables")

# # Select a variable for aggregation
# selected_var = st.selectbox("Select a categorical variable", categorical_vars)

# if selected_var:
#     st.header("User's Query")

#     # Function to get the user's query
#     def get_user_query():
#         query = st.text_area("Enter your SQL-like query (e.g., count of men grouped by state)", "")
#         return query

#     user_query = get_user_query()

#     # Function to execute the user's query and return the results as a JSON
#     def execute_query(query):
#         # Parse the query to extract the selected variable and aggregation function
#         selected_agg, group_by = query.split()
#         group_by = group_by.replace("'", "")  # Remove single quotes if present

#         # Perform the aggregation
#         result = data.groupby(group_by)[selected_var].agg(selected_agg)

#         # Convert the result to a JSON dictionary
#         result_dict = result.to_dict()
#         return result_dict

#     # Execute the query and show the results
#     query_results = execute_query(user_query)
#     st.json(query_results)








# Initialize 'queries' in session_state if not present
if 'queries' not in st.session_state:
    st.session_state['queries'] = {}

if st.session_state['active_page'] != 'Query_Selection':
    st.session_state['queries'] = {}

st.session_state['active_page'] = 'Query_Selection'









df = st.session_state.selected_dataset



st.markdown("""

Welcome to the Query Selection Tool. Please follow these steps to specify your data analysis queries:

1. **Select Columns**: Use the checkboxes to select the columns you wish to analyze.
2. **Choose Analysis Type**: For each selected column, choose the type of analysis you are interested in - Count, Average, or Histogram.
3. **Input Metadata (if applicable)**: 
   - For Average and Histogram analyses, specify the lower and upper bounds of the data.
   - For Histogram analysis, also specify the number of bins.
4. **Visualize your data**: When you're ready, click the "Visualize the Data" button to see the visualizations of your selected queries.
""")

dataset_view = st.button("Visualize the Data")
if dataset_view:
# Check 1: Ensure there is at least one query selected
    if not st.session_state['queries']:
        st.error("Error: Please select at least one query.")
    else:
        # Flags to track if any checks fail
        bounds_issue_detected = False

        # Iterate through queries to check bounds conditions
        for query_key, query_info in st.session_state['queries'].items():
            # Check 2 & 3: Bounds conditions, applicable for 'average' and 'histogram'
            if query_info.get('lower_bound') is not None and query_info.get('upper_bound') is not None:
                if query_info['lower_bound'] == query_info['upper_bound']:
                    st.error(f"Error in {query_key}: Lower and upper bounds cannot be the same.")
                    bounds_issue_detected = True
                    break  # Stop checking further as an issue is already found

                if query_info['lower_bound'] > query_info['upper_bound']:
                    st.error(f"Error in {query_key}: Lower bound cannot be greater than upper bound.")
                    bounds_issue_detected = True
                    break  # Stop checking further as an issue is already found
                
        # If no bounds issues detected, proceed to switch page
        if not bounds_issue_detected:
            switch_page("Visualizations")




def update_queries(column, query_type, metadata=None):
    query_key = f"{column}_{query_type}"

    if metadata:
        
        num_unique_values = df[column].nunique()

        if num_unique_values > 30:
        # Before updating, ensure consistency between average and histogram if both exist
            if query_type in ['average', 'histogram']:
                
                # Check if the other query type already exists and update it to share bounds
                other_query_type = 'histogram' if query_type == 'average' else 'average'
                other_query_key = f"{column}_{other_query_type}"
                if other_query_key in st.session_state['queries']:
                    # Update the other query to share bounds
                    other_metadata = st.session_state['queries'][other_query_key]
                    other_metadata['lower_bound'] = metadata['lower_bound']
                    other_metadata['upper_bound'] = metadata['upper_bound']
                    other_metadata['column'] = column
                    other_metadata['epsilon'] = None
                    other_metadata['url'] = st.session_state['dataset_url']
                    st.session_state['queries'][other_query_key] = other_metadata
        # Update the current query in session_state

        st.session_state['queries'][query_key] = metadata
       
    else:
        # Remove the query if it exists in session_state
        if query_key in st.session_state['queries']:
            del st.session_state['queries'][query_key]






for column in df.columns:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    is_selected = col1.checkbox(f"Select {column}", key=f"checkbox_{column}")
    chart = charts.column_selection_charts(df, column)
    col1.altair_chart(chart, use_container_width=True)
    if is_selected:
        # Initialize metadata
        metadata = {}
        
        # Check the number of unique values in the column
        num_unique_values = df[column].nunique()
        
        # Directly track selected options
        options = []
        if num_unique_values > 30:
            options = ["Count", "Average", "Histogram"]
        else:
            options = ["Count", "Histogram"]
        
        option_states = {option: col2.checkbox(f"{option} for {column}", key=f"{option.lower()}_{column}") for option in options}
        
        # Metadata for Average and Histogram, collected once due to shared bounds requirement
        if option_states.get("Average") or option_states.get("Histogram"):
            if num_unique_values > 30:
                columnType = df[column].dtype
                step = None if columnType == 'float64' else 1
                metadata['lower_bound'] = col3.number_input(f"Lower Bound of Column {column}", step=step, key=f"lb_{column}")
                metadata['upper_bound'] = col3.number_input(f"Upper Bound of Column {column}", step=step, key=f"ub_{column}")
                if option_states.get("Histogram"):
                    metadata['bins'] = col3.number_input(f"Number of Histogram Bins for {column}", min_value=1, step=1, key=f"bins_{column}")
                metadata['data_type'] = 'continuous'
            else:
                metadata['data_type'] = 'categorical'
                col3.write("In practice you would have to specify your bins but for the sake of simplicity we will automatically gather them from the data.")
        elif option_states.get("Count"):
            col3.write("You're all set; no additional metadata to provide.")
  
        
        # Handle selections and update session state
        for option, selected in option_states.items():
        
            if selected:
                if option == "Count":
                    update_queries(column, option.lower(), {'metadata': None})
                else:

                    update_queries(column, option.lower(), metadata)
            else:
                # Remove the query if it exists in session_state
                update_queries(column, option.lower())

st.markdown("<br><br>", unsafe_allow_html=True)
