
import altair as alt
from typing import List
import streamlit as st
import opendp.prelude as dp
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

dp.enable_features("contrib")



def column_selection_charts(df, column):
    if df[column].dtype == 'object': # Adjust condition for non-numeric columns
        # Create a bar chart for non-numeric data
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{column}:N', sort='-y'),  # N for nominal data
            y=alt.Y('count()', title='Count'),
        ).properties(
            title=column,
            width=150,
            height=200
        )
    else:
        # Create a histogram for numeric data
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{column}:Q', bin=True),  # Q for quantitative data
            y=alt.Y('count()', title='Count'),
        ).properties(
            title=column,
            width=150,
            height=200
        )
    
def preset_parameters(df, column, query,  parameter_list):
    if query == 'Epsilon':
        columnName, queryType = column.split('_')
        epsilon_variations = sorted(parameter_list)
        if queryType == 'average':

            data = []
            columnType = df[columnName].dtype
            data_min = st.session_state['queries'][column]['lower_bound']
            data_max = st.session_state['queries'][column]['upper_bound']
            
        

            domainType = dp.domain_of(List[float])
            if columnType == 'float64':
                domainType = dp.domain_of(List[float])
            elif columnType == 'int64':
                domainType = dp.domain_of(List[int])
            private_releases = {}
            
              # Will store releases for each epsilon
            for epsilon in epsilon_variations:
                releases = []
                for i in range(20): 
                    context = dp.Context.compositor(
                            data=list(df[columnName]),
                            privacy_unit=dp.unit_of(contributions=1),
                            privacy_loss=dp.loss_of(epsilon=epsilon),
                            domain=domainType,
                            split_evenly_over=1
                        )
                    dp_sum = context.query().clamp((data_min, data_max)).sum().laplace()
                    releases.append(dp_sum.release()/df.shape[0])
                private_releases[epsilon] = releases
        
            true_mean = df[columnName].mean()
        
        # Add data points for each epsilon
            for epsilon_str, releases in private_releases.items():
                epsilon = float(epsilon_str)  # Convert epsilon to float for consistent datatype
                for value in releases:
                    data.append({
                        'Epsilon': epsilon,
                        'Value': value,
                        'Type': 'Private Release'
                    })
                
            

            chart_df = pd.DataFrame(data)
            
            # Convert 'Epsilon' back to string for categorical representation, except for true mean
            chart_df['Epsilon'] = chart_df['Epsilon'].apply(lambda x: 'True Mean' if x == -1 else str(x))
            
            min_private_release = min(min(private_releases[epsilon]) for epsilon in epsilon_variations)
            y_axis_min = min(0, min_private_release) 

            
            fig = go.Figure(data=[
                go.Scatter(
                    x=chart_df['Epsilon'],  # Epsilon on x-axis
                    y=chart_df['Value'],    # Value on y-axis
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.1,  # Lower opacity for dots
                        line=dict(
                            width=2,
                            color='blue'  # Outline color
                        )
                    ),
                    name='Possible Private Outputs',  # Updated name for legend entry
                    showlegend=True 
                )
            ])
            fig.add_trace(go.Scatter(
                x=[None],  # No actual data points
                y=[None],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='True Mean'  # Name for the legend
            ))

            fig.update_layout(
                title='Privacy Parameter (\u03B5) vs Hypothetical Private Values',
        
                xaxis=dict(
                title='Privacy Parameter (\u03B5)',
                type='category'  # Explicitly set x-axis type to 'category'
            ),
                yaxis=dict(
                title='Hypothetical Private Values',
                range=[y_axis_min, max(chart_df['Value']) * 1.25]  # Add 5% padding

            ),

            )

            
            fig.add_shape(type="line",
                    x0=0,  # Starting from the first x-axis item
                    y0=true_mean,  # True mean value for y
                    x1=1,  # Ending at the last x-axis item
                    y1=true_mean,  # Same true mean value for y to keep it horizontal
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    ),
                    xref="paper",  # Reference the whole x-axis range
                    yref="y"  # Reference the y-axis values
                    )



            return fig
        elif queryType == 'count':
            data = []
            epsilon_variations = parameter_list
            columnType = df[columnName].dtype
        
            
        

            domainType = dp.domain_of(List[float])
            if columnType == 'float64':
                domainType = dp.domain_of(List[float])
            elif columnType == 'int64':
                domainType = dp.domain_of(List[int])
            private_releases = {}  # Will store releases for each epsilon
            for epsilon in epsilon_variations:
                releases = []
                
                for i in range(20): 
                    context = dp.Context.compositor(
                            data=list(df[columnName]),
                            privacy_unit=dp.unit_of(contributions=1),
                            privacy_loss=dp.loss_of(epsilon=epsilon),
                            domain=domainType,
                            split_evenly_over=1
                        )
                    dp_count = context.query().count().laplace()
                    releases.append(dp_count.release())
                private_releases[epsilon] = releases
        
            true_mean = df[columnName].count()
        
        # Add data points for each epsilon
            for epsilon_str, releases in private_releases.items():
                epsilon = float(epsilon_str)  # Convert epsilon to float for consistent datatype
                for value in releases:
                    data.append({
                        'Epsilon': epsilon,
                        'Value': value,
                        'Type': 'Private Release'
                    })
                
            

            chart_df = pd.DataFrame(data)
            
            # Convert 'Epsilon' back to string for categorical representation, except for true mean
            chart_df['Epsilon'] = chart_df['Epsilon'].apply(lambda x: 'True Mean' if x == -1 else str(x))
            
            
            min_private_release = min(min(private_releases[epsilon]) for epsilon in epsilon_variations)
            y_axis_min = min(0, min_private_release)
            
            fig = go.Figure(data=[go.Scatter(
                x=chart_df['Epsilon'],  # Epsilon on x-axis
                y=chart_df['Value'],    # Value on y-axis
                mode='markers',
                marker=dict(
                    size=10,
                    opacity=0.5  # Lower opacity for dots
                ),
                name='Possible Private Outputs',  # Updated name for legend entry
                showlegend=True 
            )])
            fig.add_trace(go.Scatter(
                x=[None],  # No actual data points
                y=[None],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='True Mean'  # Name for the legend
            ))

            fig.update_layout(
                title='Privacy Parameter (\u03B5) vs Hypothetical Private Values',
        
                xaxis=dict(
                title='Privacy Parameter (\u03B5)',
                type='category'  # Explicitly set x-axis type to 'category'
            ),
                yaxis=dict(
                title='Hypothetical Private Values',
                range=[y_axis_min, max(chart_df['Value']) * 1.25]  # Add 5% padding

            ),

            )

            
            fig.add_shape(type="line",
                    x0=0,  # Starting from the first x-axis item
                    y0=true_mean,  # True mean value for y
                    x1=1,  # Ending at the last x-axis item
                    y1=true_mean,  # Same true mean value for y to keep it horizontal
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    ),
                    xref="paper",  # Reference the whole x-axis range
                    yref="y"  # Reference the y-axis values
                    )
            return fig
    elif query == 'Mechanism':
        return

def generate_gnd_samples(n, beta, alpha, mu=0):
    # Direct generation for special cases: Laplace (beta=1) and Gaussian (beta=2)
    if beta == 1:
        # Laplace distribution
        return np.random.laplace(loc=mu, scale=alpha, size=n)
    elif beta == 2:
        # Gaussian distribution
        return np.random.normal(loc=mu, scale=alpha, size=n)
    else:
        # For arbitrary beta, a more complex method would be required.
        # Placeholder for approximation or numerical method
        print(f"Direct sampling for beta={beta} not implemented.")
        return None
    
def calculate_gaussian_scale(sensitivity, epsilon, delta):
    """Calculate the scale of Gaussian noise for differential privacy.
    
    Parameters:
    - sensitivity (float): The sensitivity of the query.
    - epsilon (float): The epsilon value of differential privacy.
    - delta (float): The delta value of differential privacy.
    
    Returns:
    - float: The scale of the Gaussian noise.
    """
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("Epsilon and delta must be within (0, 1).")
    if sensitivity <= 0:
        raise ValueError("Sensitivity must be positive.")
    
    scale = sensitivity / epsilon * np.sqrt(2 * np.log(1.25 / delta))
    return scale

def one_query_privacy_accuracy_lines(df, selected_query, mechanisms, alpha, epsilon, error_type):

    epsilons = np.linspace(0.01, 1.0, 100) 
    delta = 1e-6
    columnName, queryType = selected_query.split('_')
    
  
    if queryType == 'average':
        columnType = df[columnName].dtype
        data_min = st.session_state['queries'][selected_query]['lower_bound']
        data_max = st.session_state['queries'][selected_query]['upper_bound']
        sensitivity = (data_max - data_min)/df.shape[0]
        summary_df =  summary_df = pd.DataFrame({
            'mechanism': ['Laplace', 'Gaussian'],
            'true_value': [df[columnName].mean(), df[columnName].mean()],  # Example true mean values
            # 'error_margin': [point_df['error'][0] ,point_df['error'][1] ]  # Example error margins
        })

    elif queryType == 'count':
        sensitivity = 1  # Set the sensitivity to 1 for 'count' query
        summary_df =  summary_df = pd.DataFrame({
            'mechanism': ['Laplace', 'Gaussian'],
            'true_value': [df[columnName].count(), df[columnName].count()],  # Example true mean values
            # 'error_margin': [point_df['error'][0] ,point_df['error'][1] ]  # Example error margins
        })

    else:
        # Handle other query types here
        pass

    line_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

    for eps in epsilons:
        if 'Laplace' in mechanisms:
            laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / eps, alpha)  # Ensure you use 'eps' here, not 'epsilon'
            new_row = pd.DataFrame({'epsilon': [eps], 'error': [laplace_error], 'mechanism': 'Laplace'})
            line_df = pd.concat([line_df, new_row], ignore_index=True)

            
        if 'Gaussian' in mechanisms:
            gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, eps, delta), alpha)  # Again, ensure 'eps' is used
            new_row = pd.DataFrame({'epsilon': [eps], 'error': [gaussian_error], 'mechanism': 'Gaussian'})
            line_df = pd.concat([line_df, new_row], ignore_index=True)

    point_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

# Calculate Laplace error for the specified epsilon value and append to the DataFrame
    if 'Laplace' in mechanisms:
        laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / epsilon, alpha)  # Ensure you use 'eps' here, not 'epsilon'
        new_row = pd.DataFrame({'epsilon': [epsilon], 'error': [laplace_error], 'mechanism': 'Laplace'})
        point_df = pd.concat([point_df, new_row], ignore_index=True)

    # Calculate Gaussian error for the specified epsilon value and append to the DataFrame    
    if 'Gaussian' in mechanisms:
        gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, epsilon, delta), alpha)  # Again, ensure 'eps' is used
        new_row = pd.DataFrame({'epsilon': [epsilon], 'error': [gaussian_error], 'mechanism': 'Gaussian'})
        point_df = pd.concat([point_df, new_row], ignore_index=True)
    
    # Create subplots: 1 row, 2 cols
    point_colors = ['#0072B2', '#80b1d3'] 

    error_bar_color = 'red'  # Red for error bars

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hypothetical outputs", "Accuracy vs. Privacy Parameter (\u03B5)"))
    summary_df['error_margin'] = [point_df['error'][0], point_df['error'][1]]

    fig.update_yaxes(title_standoff=10, row=1, col=2) 
    if error_type == 'Absolute Error':
        fig.update_yaxes(title_text="Hypothetical Outputs", row=1, col=1)
        fig.update_yaxes(title_text="Absolute Error Bound",  row=1, col=2)

        pass
    elif error_type == 'Relative Error':
        line_df['error'] = line_df['error'] / summary_df['true_value'][0]
        point_df['error'] = point_df['error'] / summary_df['true_value'][0]
        # summary_df['error_margin'] = summary_df['error_margin'] / summary_df['true_value'][0]
        fig.update_yaxes(title_text="Hypothetical Outputs", row=1, col=1)
        fig.update_yaxes(title_text="Relative Error Bound", tickformat=',.0%', row=1, col=2)
    else:
        raise ValueError("Invalid error_type. Please choose 'Absolute Error' or 'Relative Error'.")
    # Plot 1: Accuracy vs. Privacy for Selected Mechanisms
    for index, mechanism in enumerate(line_df['mechanism'].unique()):
        mechanism_df = line_df[line_df['mechanism'] == mechanism]
        fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines', name=mechanism, line=dict(color=point_colors[index])), row=1, col=2)
    for index, row in point_df.iterrows():
        fig.add_trace(
            go.Scatter(x=[row['epsilon']], y=[row['error']], mode='markers', name=f"{row['mechanism']} (Specific \u03B5)",
                    marker=dict(color='red', size=4), showlegend=False),  # You can adjust the marker size here
            row=1, col=2  # Assuming you want to add these points to the second subplot
        )
    
    
    max_value =point_df['error'].max()

  
    
    laplace_scale =  sensitivity/ epsilon
    # Generate 50 points from Laplace distribution
    vals_laplace = np.random.laplace(loc=summary_df['true_value'][0], scale=laplace_scale, size=50)

        # Given delta value for Gaussian distribution calculation
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    vals_gauss = np.random.normal(loc=summary_df['true_value'][0], scale=sigma, size=50)

    for index, row in summary_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['mechanism']],
                y=[row['true_value']],
                error_y=dict(type='data', array=[row['error_margin']], visible=True, color=error_bar_color),

                mode='markers',
                name=row['mechanism'],
                marker=dict(size=10, opacity=00),  # Adjust opacity for visual emphasis
                showlegend=False,
            ),
            

            row=1, col=1
        )
    
    # Add Laplace points to the plot
    fig.add_trace(
        go.Scatter(
            x=['Laplace'] * 50,  # X-values are categorical labels
            y=vals_laplace,
            mode='markers',
            name='Laplace',
            marker=dict(color=point_colors[0], opacity=.3),
            showlegend=False,
        ),
            

        row=1, col=1
    )

    # Add Gaussian points to the plot
    fig.add_trace(
        go.Scatter(
            x=['Gaussian'] * 50,  # X-values are categorical labels
            y=vals_gauss,
            mode='markers',
            name='Gaussian',
            marker=dict(color=point_colors[1], opacity=.3),
            showlegend=False,
        ),
        

        row=1, col=1
    )

    fig.add_shape(type="line",
                x0=-.1,  # Starting from the first x-axis item
                y0=row['true_value'],  # True mean value for y
                x1=1.1,  # Ending at the last x-axis item
                y1=row['true_value'],  # Same true mean value for y to keep it horizontal
                line=dict(
                    color="red",
                    width=1,
                    dash="dash",
                ),
                xref="paper",  # Reference the whole x-axis range
                yref="y",
                row=1, col=1, # Reference the y-axis values
                showlegend=True,
                name='True Mean'
                
                )
            

    fig.update_yaxes(title_text="Hypothetical outputs", range=[0, (summary_df['true_value'][0]+ max_value)*1.25], row=1, col=1)
    # Update xaxis properties
    fig.update_xaxes(title_text="Privacy Parameter (\u03B5)", row=1, col=1)
    fig.update_xaxes(title_text="Privacy Parameter (\u03B5) (Specific)", row=1, col=2)

    # Update yaxis properties
    
    return fig