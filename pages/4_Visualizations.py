import streamlit as st
import charts
from NotebookCreator import create_notebook
from nbconvert import PythonExporter
import nbformat as nbf
from CompositionFunctions import *
import pandas as pd

st.session_state['active_page'] = 'Visualization_page'


st.title('Visualization Playground')
# Creating tabs
if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = None
df = st.session_state['selected_dataset']
tab1, tab2, tab3, tab4 = st.tabs(["Simulations", "One Query", "Multiple Queries", "Data Export"])

# Initialize an empty list to store the numbers
if 'simulations_parameter_selection' not in st.session_state:
    st.session_state['simulations_parameter_selection'] = None

if 'simulations_parameter_selection'  in st.session_state:
    st.session_state['simulations_parameter_selection'] = None

with tab1:
    
    st.header("Simulations")
    st.write("Here we will be able to see simulated private outputs on each of your queries. Choose the query you want and select variations of different paramters to explore.")
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.header("Experimental Parameters")  # Add a header for the first column
        st.write("Please select parameters.")  # Placeholder text
    # Add your simulations content or functionality here
        if 'queries' in st.session_state and st.session_state['queries']:
            # Extracting query keys or descriptions to list in the dropdown
            query_keys = list(st.session_state['queries'].keys())

            # Create a select box (dropdown) with the query keys
            # selected_query = st.selectbox("Select a query:", options=query_keys)
            selected_query = st.selectbox( "Which query would you like to visualize?",
                        (query_keys),
                        index=None,
                        placeholder="Select Query ...",
           
                        )
            
            if selected_query:
                st.session_state['visualize_clicked'] = False
                options = ['Epsilon', 'Mechanism']

                if 'average' in selected_query:
                    options.append('Bounds')
                elif 'data_type' in st.session_state['queries'][selected_query] and st.session_state['queries'][selected_query]['data_type'] == 'continuous':
                    options.append('Bounds')
                    options.append('Bins')

                st.session_state['simulations_parameter_selection'] = st.selectbox( "Which implementation parameter are you interested in?",
                                                        options,
                                                        index=None,
                                                        placeholder="Select Parameter ...",
                                                        )
          
                if st.session_state['simulations_parameter_selection']:
                    if st.session_state['simulations_parameter_selection'] == 'Epsilon':
                        if 'epsilon_inputs' not in st.session_state:
                            st.session_state['epsilon_inputs'] = []
                        col3, col4 = st.columns([1, 1])
                        with col3:
                            with st.form('epsilon_form'):
                                st.write('Enter \u03B5 Values (max 4):')
                                new_epsilon = st.number_input('Privacy (\u03B5)', min_value=0.001, step=0.001, value=1.0000, format="%.3f", key='epsilon_input')

                                submitted = st.form_submit_button('Add \u03B5 value')
                                if submitted:
                                    if len(st.session_state['epsilon_inputs']) < 4:
                                        if new_epsilon not in st.session_state['epsilon_inputs']:
                                            st.session_state['epsilon_inputs'].append(new_epsilon)
                                        else:
                                            st.warning('This \u03B5 value has already been added.')
                                    else: 
                                        st.warning('Maximum of 4 \u03B5 values reached.')
                        with col4:
                            for index, epsilon in enumerate(st.session_state['epsilon_inputs']):
                        
                            
                                if st.button(f"Delete {epsilon}", key=f"delete_{index}"):
                                    # Remove epsilon from the list
                                    st.session_state['epsilon_inputs'].remove(epsilon)
                                    # Update the page to reflect the deletion
                                    st.rerun()
                   
                if st.session_state['simulations_parameter_selection']:
                    if st.session_state['simulations_parameter_selection'] == 'Bounds':
                        epsilon = st.number_input('Epsilon', min_value=0.001, step=0.001, value=1.0000, format="%.3f", key='epsilon_input')
                        
                        col3, col4 = st.columns([1, 1])
                        with col3:
                            st.write('Enter Lower and Upper Bounds:')
                            lower_bound = st.number_input('Lower Bound', min_value=-100000000.0, step=0.1, value=float(st.session_state['queries'][selected_query]['lower_bound']), format="%.1f", key='lower_bound_input')
                            upper_bound = st.number_input('Upper Bound', min_value=lower_bound, step=0.1, value=float(st.session_state['queries'][selected_query]['upper_bound']), format="%.1f", key='upper_bound_input')

                            
                            if 'bounds_inputs' not in st.session_state:
                                st.session_state['bounds_inputs'] = []
                            
                            submitted = st.button('Add Bounds')
                            if submitted:
                                new_bounds = (lower_bound, upper_bound)
                                if len(st.session_state['bounds_inputs']) < 4:
                                    if new_bounds not in st.session_state['bounds_inputs']:
                                        st.session_state['bounds_inputs'].append(new_bounds)
                                    else:
                                        st.warning('This Bounds value has already been added.')
                                else: 
                                    st.warning('Maximum of 4 Bounds values reached.')
                                    
                        with col4:
                            for index, bounds in enumerate(st.session_state['bounds_inputs']):
                                if st.button(f"Delete {bounds}", key=f"delete_{index}"):
                                    # Remove bounds from the list
                                    st.session_state['bounds_inputs'].remove(bounds)
                                    # Update the page to reflect the deletion
                                    st.rerun()

                    if st.session_state['simulations_parameter_selection']:
                        if st.session_state['simulations_parameter_selection'] == 'Bins':
                            epsilon = st.number_input('Epsilon', min_value=0.001, step=0.001, value=1.0000, format="%.3f", key='epsilon_input')

                            col3, col4 = st.columns([1, 1])
                            
                            with col3:
                                
                                
                               
                           
                                lower_bound = st.number_input('Lower Bound', min_value=-100000000.0, step=0.1, value=float(st.session_state['queries'][selected_query]['lower_bound']), format="%.1f", key='lower_bound_input')
                                num_bins = st.number_input('Number of Bins', min_value=2, max_value=50, value=10, step=1, key='num_bins_input')
                                
                                if 'bins_inputs' not in st.session_state:
                                    st.session_state['bins_inputs'] = []
                            
                            with col3:
                                
                                if 'bins_inputs' not in st.session_state:
                                    st.session_state['bins_inputs'] = []
                                
                                submitted = st.button('Add Bins')
                                if submitted:
                                    new_bins = (num_bins)
                                    if len(st.session_state['bins_inputs']) < 4:
                                        if new_bins not in st.session_state['bins_inputs']:
                                            st.session_state['bins_inputs'].append(new_bins)
                                        else:
                                            st.warning('This Bins value has already been added.')
                                    else: 
                                        st.warning('Maximum of 4 Bins values reached.')
                                with col4:       
                                    upper_bound = st.number_input('Upper Bound', min_value=lower_bound, step=0.1, value=float(st.session_state['queries'][selected_query]['upper_bound']), format="%.1f", key='upper_bound_input')
                                    st.write('Selected Bins')
                                    for index, bins in enumerate(st.session_state['bins_inputs']):
                                        if st.button(f"Delete {bins}", key=f"delete_{index}"):
                                            # Remove bins from the list
                                            st.session_state['bins_inputs'].remove(bins)
                                            # Update the page to reflect the deletion
                                            st.rerun()
                    elif st.session_state['simulations_parameter_selection'] == 'Mechanism':
                        mechanisms = st.multiselect(
                            'Which mechanism do you want displayed',
                            ['Gaussian', 'Laplace'],
                            ['Gaussian', 'Laplace'],
                            key='simulations_selected_mechanism'

                        )



    
    
    with col2:
        st.header("Visualization")  # Add a header for the second column
        st.write("When you have selected your parameters please click the visualize button to see them.")  # Placeholder text

        # Check if 'visualize_clicked' is not in session state or if the 'Visualize' button is clicked
        if 'visualize_clicked' not in st.session_state or st.button("Visualize"):
            st.session_state['visualize_clicked'] = True  # Update session state to indicate button click
            # Assuming 'parameter_selection' is defined and accessible here
            if st.session_state['simulations_parameter_selection'] == 'Epsilon':
                
                if 'average'  in selected_query or 'count' in selected_query:
                    
                    chart_config = charts.preset_parameters(df, selected_query, st.session_state['simulations_parameter_selection'], st.session_state['epsilon_inputs'])
                    # Store the figure in session state to reuse
                    st.session_state['fig'] = chart_config
                else:
                    
                    chart_config = charts.noisy_histogram_creation(df, selected_query, st.session_state['queries'][selected_query]['data_type'], st.session_state['simulations_parameter_selection'], st.session_state['epsilon_inputs'])
                    st.session_state['fig'] = chart_config
            if st.session_state['simulations_parameter_selection'] == 'Bounds':
                
                chart_config = charts.preset_parameters(df, selected_query, st.session_state['simulations_parameter_selection'], st.session_state['bounds_inputs'])
                # Store the figure in session state to reuse
                st.session_state['fig'] = chart_config


        # Display the chart from session state if it exists and 'visualize_clicked' is True
        if 'visualize_clicked' in st.session_state and st.session_state['visualize_clicked']:
            fig = st.session_state.get('fig', None)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                        







               

























with tab2:
 
    # Add your one query content or functionality here
    col1, col2 = st.columns([1, 2], gap="large")
    if 'one_query_selected_mechanism' not in st.session_state:
        st.session_state['one_query_selected_mechanism'] = None
    if 'alpha' not in st.session_state:
        st.session_state['alpha'] = None
    if 'error_type' not in st.session_state:
        st.session_state['error_type'] = None
    with col1:  
        st.header('Inputs')
        query_keys = list(st.session_state['queries'].keys())
        
        selected_query = st.selectbox(
            "Which query would you like to visualize?",
            options=query_keys,
            index=None,

            placeholder="Select Query ...",
            key='one_query_selected_parameter'  # This links the widget to `st.session_state['one_query_selected_parameter']`
        )
        if selected_query:
            st.session_state['one_query_selected_mechanism'] = st.multiselect(
                                'Which mechanism do you want displayed',
                                ['Gaussian', 'Laplace'],
                                ['Gaussian', 'Laplace' ],
                            )

            epsilon = st.slider('Privacy Parameter (\u03B5)', .01, 1.0, .25, key=f"one_query_epsilon_slider")
            st.session_state['error_type'] = st.selectbox(
                "Which error type would you like to visualize?",
                options=['Absolute Additive Error', 'Relative Additive Error'],
                index=0,
                key='one_query_selected_error',  # This links the widget to `st.session_state['one_query_selected_parameter']`
                help=(
                        "- **Error Bounds**: Indicates predicted values' deviation from true values within 1-beta (\u03B2) confidence. \n"
                        "- **Absolute Error**: Direct difference between predicted and true values. Use for unnormalized error measurement. \n"
                        "- **Relative Error**: Absolute Error normalized by true value, useful for error comparison across scales. \n"
                        "- **Choosing Metric**: \n"
                        "   - Use **Absolute Error** for consistency in measurement units. \n"
                        "   - Choose **Relative Error** for comparative analysis across different magnitudes."
                    )
            )
            st.session_state['alpha'] = st.slider('beta (\u03B2) - High probability bound on accuracy', 0.01, .50, 0.05,     help='Beta represents the confidence level. A beta of 0.05 means that 95% of the hypothetical outputs will fall inside the error bars. You can see this on the chart on the left. As beta increases, more points will fall outside of the red error bounds.'
)
            

          
    





    with col2:
        st.header('Visualization')
        if st.session_state.one_query_selected_mechanism and  st.session_state['alpha'] and  st.session_state['error_type'] is not None:
            one_query_charts = charts.one_query_privacy_accuracy_lines(df, selected_query, st.session_state['one_query_selected_mechanism'], st.session_state['alpha'] , epsilon,  st.session_state['error_type'])
            st.plotly_chart(one_query_charts, use_container_width=True)

            with st.expander("Chart Explanations"):
                mechanism_names = ' and '.join(st.session_state['one_query_selected_mechanism'])  # e.g., "Laplace and Gaussian"
                chart_explanations = f"""
                #### Hypothetical Outputs (Left)

                This chart displays individual hypothetical outputs for the selected privacy-preserving mechanism(s): {mechanism_names}. 
                The true mean is indicated by a dashed red line, serving as a benchmark to assess the accuracy of each mechanism's output. 
                The points represent hypothetical outputs. Since differential privacy adds random noise we cannot say for certain which value will be released. 
                The distribution of points gives a sense of where the noisy value is likely to fall.
                Use this chart to better understand how the implementation choices will influence your noisy output.
                This chart also gives an indication of which mechanism may be better suited for your needs. The closer the distribution to the true mean, the better the accuracy.

                ####  Accuracy vs. Privacy Parameter (ε) (Right)

                This chart illustrates the { st.session_state['error_type']} of the selected mechanism(s): {mechanism_names}, as a function of the privacy parameter ε.
                As ε increases, the error bound  decreases, signifying that less stringent privacy (higher ε) correlates with higher accuracy.
                The lines represent the error bound as specified by beta (\u03B2). The X axis can be interpreted as the true value of the query. 
                The noisy values will fall within within the error bounds with 1-\u03B2 confidence.
                The red dots represent the privacy/accuracy point visualized in the left chart. 
                This chart can help decide the correct level of privacy for your data release. 
                The error often increases exponetially as we decrease the privacy parameter. If possible, increase the epsilon beyond the 'elbow' of the curve to maximize accuracy while maintaining privacy.
                """
                
                st.markdown(chart_explanations)






















import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.express as px

with tab3:
    st.header("Multiple Queries")
    col1, col2 = st.columns([1, 2])  # Create two columns with a ratio of 1:2

    with col1:  # Put the sliders in the first column
        k = st.slider('Number of queries', 1, 100)
        del_g = st.slider('Global Delta (log scale)', -6, -2)
        epsilon_g = st.slider('Global Epsilon', 0.01, 1.0)

    with col2:  # Put the visualization in the second column
        compositors = compare(k, pow(10,del_g), .5, epsilon_g)
        df = pd.DataFrame.from_dict(compositors, orient='index', columns=['Epsilon_0', 'Delta_0'])
        df['Compositor'] = df.index  # Add a new column with the index values
        fig = charts.compare_compositors(df)
        st.plotly_chart(fig)












with tab4:
    st.header("Code Export")
    # Add your data export content or functionality here
    st.write("This is the Data Export tab.")
    
    with st.form(key='data_export_form'):
        query_keys = list(st.session_state['queries'].keys())
        
        selected_query = st.selectbox(
            "Which query would you like to export?",
            options=query_keys,
            index=0,

            placeholder="Select Query ...",
            key='data_export_selected_parameter'  # This links the widget to `st.session_state['one_query_selected_parameter']`
        )
        epsilon = st.slider('Privacy Parameter (\u03B5)', .01, 1.0, .25, key=f"data_export_epsilon_slider")

        submitted = st.form_submit_button('Export')

        if submitted:
            st.session_state['queries'][selected_query]['epsilon'] = epsilon
            st.session_state['queries'][selected_query]['url'] = st.session_state['dataset_url']
       
            nb = create_notebook({selected_query: st.session_state['queries'][selected_query]})
            with open('notebook.ipynb', 'w') as f:
                nbf.write(nb, f)
            with open('notebook.ipynb', 'rb') as f:
                data = f.read()
            st.download_button(label='Download IPython Notebook',
                data=data,
                file_name='notebook.ipynb',
                mime='application/x-ipynb+json')


