
import altair as alt
from typing import List
import streamlit as st
import opendp.prelude as dp
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

dp.enable_features("honest-but-curious")
dp.enable_features('contrib')



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
    

def create_figure(chart_df, true_mean, y_axis_min):

    fig = go.Figure(data=[
        go.Scatter(
            x=chart_df.iloc[:, 0],  # Epsilon on x-axis
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
        line=dict(color='black', width=2, dash='dash'),
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
                color="black",
                width=2,
                dash="dash",
            ),
            xref="paper",  # Reference the whole x-axis range
            yref="y"  # Reference the y-axis values
            )

    return fig



def preset_parameters(df, column, query,  parameter_list):
    data_min = None
    data_max = None

    if query == 'Epsilon':
        columnName, queryType = column.split('_')
        epsilon_variations = sorted(parameter_list)
        if query == 'Epsilon':
            columnName, queryType = column.split('_')
            epsilon_variations = sorted(parameter_list)
            if queryType == 'count':

                data = []
                columnType = df[columnName].dtype
     
                
            

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
                        releases.append(get_query_private_outputs(df, queryType, 'laplace', columnName, epsilon,(data_min, data_max),  .0000001))
                
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

                return create_figure(chart_df, true_mean, y_axis_min)
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
                    releases.append(get_query_private_outputs(df, queryType, 'laplace', columnName, epsilon,(data_min, data_max),  .0000001))
               
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

            return create_figure(chart_df, true_mean, y_axis_min)
    if query == 'Bounds':
       
        columnName, queryType = column.split('_')
        if queryType == 'average':

            data = []
            columnType = df[columnName].dtype
            

            private_releases = {}

            # Iterate over the list of bounds
            for bounds in parameter_list:
                releases = []
                lower_bound, upper_bound = bounds
                for i in range(20):
                    releases.append(get_query_private_outputs(df, queryType, 'laplace', columnName, st.session_state['epsilon_input'], (lower_bound, upper_bound), .0000001))
                private_releases[bounds] = releases

            true_mean = df[columnName].mean()

            # Add data points for each bounds
            for bounds_str, releases in private_releases.items():
                lower_bound, upper_bound = bounds_str
                for value in releases:
                    data.append({
                        'Bounds': f"[{lower_bound}, {upper_bound}]",
                        'Value': value,
                        'Type': 'Private Release'
                    })

            chart_df = pd.DataFrame(data)

            min_private_release = min(min(private_releases[bounds]) for bounds in parameter_list)
            y_axis_min = min(0, min_private_release)

            return create_figure(chart_df, true_mean, y_axis_min)
        




def dataframe_domain():
    return dp.user_domain("DataFrameDomain", lambda x: isinstance(x, pd.DataFrame))

def make_select_column(col_name, T):
    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=T)),
        output_metric=dp.symmetric_distance(),
        function=lambda data: data[col_name].to_numpy(),
        stability_map=lambda d_in: d_in)



def noisy_histogram_creation(df, selected_query, data_type, parameter, parameter_list):
    columnName, queryType = selected_query.split('_')
    epsilon_variations = sorted(parameter_list)
    epsilon_variations.reverse()
    categories = sorted(df[columnName].unique())

    column_type = df[columnName].dtype
    print(column_type==int)
    if np.issubdtype(column_type, np.integer):
        print(column_type)
        df_copy = df.copy()
        df_copy[columnName] = df_copy[columnName].astype(column_type)
        categories_type = np.array(categories, dtype=np.int32)       
        histogram = (
            make_select_column(columnName, int) >>
            dp.t.then_count_by_categories(categories=categories_type)
        )

    if column_type == float:
        df_copy = df.copy()
        df_copy[columnName] = df_copy[columnName].astype('float')
        histogram = (  
          make_select_column(columnName, float) >>
              
          dp.t.then_count_by_categories(categories=categories) 
                  
          )
    if column_type == str:
        df_copy = df.copy()
        df_copy[columnName] = df_copy[columnName].astype('str')
        histogram = (  
          make_select_column(columnName, str) >>
              
          dp.t.then_count_by_categories(categories=categories) 
                  
          )

    data = {}
    true_counts = df[columnName].value_counts().reindex(categories).values
   
    data['True Counts'] = [true_counts]
    for eps in epsilon_variations:
        data[eps] = []
        for i in range(1):
            noisy_histogram = dp.binary_search_chain(
                lambda s: histogram >> dp.m.then_laplace(scale=s),
                d_in=1, d_out=eps)

            sensitive_counts = noisy_histogram(df_copy)
            data[eps].append(sensitive_counts)

    # Add true counts to the data dictionary
    

    return visualize_data(data, categories)


def visualize_data(data, categories):
    fig = go.Figure()

    # Create a list of blue colors for the histograms
    blue_colors = ['rgba(0, 0, 255, 1)', 'rgba(0, 128, 255, 1)', 'rgba(0, 192, 255, 1)', 'rgba(0, 224, 255, 1)']

    for i, (eps, counts) in enumerate(data.items()):
        for j, count in enumerate(counts):
            if eps == 'True Counts':
                color = 'black'  # Red outline for true counts
            else:
                color = blue_colors[i % len(blue_colors)]  # Cycle through blue colors for each epsilon

            fig.add_trace(go.Bar(
                x=categories,
                y=count,
                marker_color=color,  # Transparent fill
                marker_line_color=color,  # Colored outline
                marker_line_width=0.5,  # Thinner outline
                name=f'Epsilon {eps}' if eps != 'True Counts' else 'True Counts',
                showlegend=True if j == 0 else False,  # Only show the first histogram's legend
                offsetgroup=str(eps),  # Offset group for each epsilon
                 # Shared bin group for alignment
            ))

    fig.update_layout(
        barmode='group',
        title='Grouped Outlined Histograms for Different Epsilons',
        xaxis_title='Category',
        yaxis_title='Frequency',
        legend_title='Epsilon',
        legend=dict(
            traceorder="normal",
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor="Black",
            borderwidth=2
        )
    )

    return fig


























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


import math
def calculate_sensitivity(query_type, bounds, df):
    if query_type == 'count':
        return 1
    elif query_type == 'average':
        return (bounds[1] - bounds[0])/df.shape[0]

def calculate_scale(mechanism, sensitivity, epsilon, delta):
    if mechanism == 'laplace':
        return sensitivity / epsilon
    elif mechanism == 'gaussian':
        return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

def create_df_meas(df, column_name, T, mechanism, scale, query_type, bounds):
    df[column_name] = df[column_name].astype(T)  # Explicitly casting to float if T is float

    if query_type == 'count':

        df_meas = (
            make_select_column(column_name, T) >>
            dp.t.then_count() >>
            getattr(dp.m, f'then_{mechanism}')(scale)
        )
    else:
        bounds = (float(bounds[0]), float(bounds[1]))

        df_meas = (
            make_select_column(column_name, T) >>
            dp.t.then_resize(size=len(df), constant=20.) >>
            dp.t.then_clamp(bounds) >>
            dp.t.then_mean() >>
            getattr(dp.m, f'then_{mechanism}')(scale)
        )
    return df_meas(df)

def get_query_private_outputs(df, query_type, mechanism, column_name,  epsilon,  bounds=(None, None), delta=1e-6):
    
    
    sensitivity = calculate_sensitivity(query_type, bounds, df)
    scale = calculate_scale(mechanism, sensitivity, epsilon, delta)
    
    if df[column_name].dtype == int:
        df_float = df.copy()
        df_float[column_name] = df_float[column_name].astype('float')
        return create_df_meas(df_float, column_name, float, mechanism, scale, query_type, bounds)
    else:
        return create_df_meas(df, column_name, float, mechanism, scale, query_type, bounds)
    

        
    

def one_query_privacy_accuracy_lines(df, selected_query, mechanisms, alpha, epsilon, error_type):
    

    epsilons = np.linspace(0.01, 1.0, 100) 
    delta = 1e-6
    columnName, queryType = selected_query.split('_')
    data_min = None
    data_max = None
    
  
    if queryType == 'average':
        columnType = df[columnName].dtype
        data_min = st.session_state['queries'][selected_query]['lower_bound']
        data_max = st.session_state['queries'][selected_query]['upper_bound']
        sensitivity = (data_max - data_min)/df.shape[0]
        summary_df = pd.DataFrame({
            'mechanism': mechanisms,
            'true_value': [df[columnName].mean()] * len(mechanisms),  # Example true mean values
            # 'error_margin': [point_df['error'][0] ,point_df['error'][1] ]  # Example error margins
        })

    elif queryType == 'count':
        sensitivity = 1  # Set the sensitivity to 1 for 'count' query
        summary_df = pd.DataFrame({
            'mechanism': mechanisms,
            'true_value': [df[columnName].count()] * len(mechanisms),  # Example true mean values
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

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hypothetical outputs", "Accuracy vs. Privacy Parameter (\u03B5)"), horizontal_spacing=0.15)
    error_margin = []
    summary_df['error_margin'] = point_df['error'].tolist()

    fig.update_yaxes(title_standoff=10, row=1, col=2) 
    if error_type == 'Absolute Additive Error':
        fig.update_yaxes(title_text="Hypothetical Outputs", row=1, col=1)
        fig.update_yaxes(title_text="Absolute Error Upper Bound",  row=1, col=2)
        pass
    elif error_type == 'Relative Additive Error':
        line_df['error'] = line_df['error'] / summary_df['true_value'][0]
        
        # summary_df['error_margin'] = summary_df['error_margin'] / summary_df['true_value'][0]
        fig.update_yaxes(title_text="Hypothetical Outputs", row=1, col=1)
        fig.update_yaxes(title_text="Relative Error Upper Bound", tickformat=',.0%', row=1, col=2)
    else:
        raise ValueError("Invalid error_type. Please choose 'Absolute Error' or 'Relative Error'.")
    # Plot 1: Accuracy vs. Privacy for Selected Mechanisms
    for index, mechanism in enumerate(line_df['mechanism'].unique()):
        mechanism_df = line_df[line_df['mechanism'] == mechanism]
        fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines', 
                                name=f"Additive Error Upper Bound for {mechanism}", 
                                line=dict(color=point_colors[index])), row=1, col=2)
    for index, row in point_df.iterrows():
        if index == 0:  # Only add one dot to the legend
            showlegend = True
        else:
            showlegend = False
        if error_type == 'Relative Additive Error':
            row['error'] = row['error'] / summary_df['true_value'][0]
        fig.add_trace(
            go.Scatter(x=[row['epsilon']], y=[row['error']], mode='markers', 
                    name="Selected Epsilon",
                    # marker=dict(color=point_colors[point_df['mechanism'].unique().tolist().index(row['mechanism'])], size=10), 
                    marker=dict(color='red', size=6),
                    showlegend=showlegend),  
            row=1, col=2  
        )

    fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
        
    max_value =point_df['error'].max()

    laplace_scale =  sensitivity/ epsilon
    vals_laplace = []
    vals_gauss =[]
    
    if 'Laplace' in mechanisms: 
        laplace_row = point_df.loc[point_df['mechanism'] == 'Laplace']
        for i in range(20):
            vals_laplace.append(get_query_private_outputs(df, queryType, 'laplace', columnName, epsilon,(data_min, data_max),  .0000001))
        fig.add_trace(
            go.Scatter(
                x=['Laplace'],  # Use 0 for Laplace
                y=[np.mean(vals_laplace)],
                error_y=dict(type='data', array=[laplace_row['error'].values[0]], visible=True, color=error_bar_color),

                mode='markers',
                name='Laplace',
                marker=dict(size=10, opacity=00),  # Adjust opacity for visual emphasis
                showlegend=False,
            ),
            

            row=1, col=1
        )    
        fig.add_trace(
            go.Scatter(
                x=['Laplace'] * 20,  # X-values are categorical labels
                y=vals_laplace,
                mode='markers',
                name='Laplace',
                marker=dict(color=point_colors[0], opacity=.3),
                showlegend=False,
            ),
            

            row=1, col=1
        )
        
      

    if 'Gaussian' in mechanisms:
        gaussian_row = point_df.loc[point_df['mechanism'] == 'Gaussian']
        for i in range(20):
            
            vals_gauss.append(get_query_private_outputs(df, queryType, 'gaussian', columnName, epsilon, (data_min, data_max),  .0000001))       
        fig.add_trace(
            go.Scatter(
                x=['Gaussian'],  # Use 1 for Gaussian
                y=[np.mean(vals_gauss)],
                error_y=dict(type='data', array=[gaussian_row['error'].values[0]], visible=True, color=error_bar_color),

                mode='markers',
                name='Gaussian',
                marker=dict(size=10, opacity=00),  # Adjust opacity for visual emphasis
                showlegend=False,
            ),
            

            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=['Gaussian'] * 20,  # X-values are categorical labels
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
                y0=summary_df.iloc[0]['true_value'],  # True mean value for y
                x1=1.1,  # Ending at the last x-axis item
                y1=summary_df.iloc[0]['true_value'],  # Same true mean value for y to keep it horizontal
                line=dict(
                    color="black",
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

def compare_compositors(df):
    # Create a figure with two subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2, subplot_titles=("Epsilon", "Delta"))

    # Get the unique types of compositors
    compositors = df['Compositor'].unique()

    # Loop over each type of compositor and add it to the plot
    for i, comp in enumerate(compositors):
        # Filter the DataFrame for this compositor
        filtered_df = df[df['Compositor'] == comp]

        # Add a trace to the first subplot (epsilon)
        fig.add_trace(go.Scatter(x=filtered_df.index,
                                 y=filtered_df['Epsilon_0'],
                                 mode='markers',
                                 marker=dict(color=f'rgb({i*50}, 100, {255-i*50})')), row=1, col=1)

        # Add a trace to the second subplot (delta)
        fig.add_trace(go.Scatter(x=filtered_df.index,
                                 y=filtered_df['Delta_0'],
                                 mode='markers',
                                 marker=dict(color=f'rgb({i*50}, 100, {255-i*50})')), row=1, col=2)

    # Update layout to show both plots side by side
    fig.update_layout(height=400, width=400,
                    #   title_text="Multiple Queries",
                      xaxis_title='Compositors',
                      yaxis_title='Per query epsilon',
                      xaxis2_title='Compositors',
                      showlegend=False)

    # Start epsilon chart at 0
    fig.update_yaxes(range=[0, None], row=1, col=1)

    fig.update_layout(yaxis2=dict(title='Per query delta', title_standoff=10))
    

    # Add y-axis label to delta chart
    fig.update_yaxes(title_text='Per query delta', row=1, col=2)

    # Rotate x-axis labels to 45 degrees

    fig.update_xaxes(tickangle=45)
    


    return fig

