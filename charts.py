import altair as alt
from typing import List
import streamlit as st
import opendp.prelude as dp
import plotly.graph_objects as go
import plotly.express as px
from opendp.domains import atom_domain, vector_domain
from opendp.metrics import absolute_distance, symmetric_distance


import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

dp.enable_features("honest-but-curious")
dp.enable_features('contrib')


    

def create_figure(chart_df, true_mean, y_axis_min, query):

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
        title=f'{query} vs Hypothetical Private Values',
        
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

import math
# def dataframe_domain():
#     return dp.user_domain("DataFrameDomain", lambda x: isinstance(x, pd.DataFrame))

# def make_select_column(col_name, T):
#     return dp.t.make_user_transformation(
#         input_domain=dataframe_domain(),
#         input_metric=dp.symmetric_distance(),
#         output_domain=dp.vector_domain(dp.atom_domain(T=T)),
#         output_metric=dp.symmetric_distance(),
#         function=lambda data: data[col_name].to_numpy(),
#         stability_map=lambda d_in: d_in)

# def calculate_sensitivity(query_type, bounds, df):
#     if query_type == 'count':
#         return 1
#     elif query_type == 'average':
#         return (bounds[1] - bounds[0])/df.shape[0]

# def calculate_scale(mechanism, sensitivity, epsilon, delta):
#     if mechanism == 'laplace':
#         return sensitivity / epsilon
#     elif mechanism == 'gaussian':
    
#         return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

# def create_df_meas(df, column_name, T, mechanism, scale, query_type, bounds):

#     if query_type == 'count':
#         df_meas = (
#             make_select_column(column_name, T) >>
#             dp.t.then_count() >>
#             getattr(dp.m, f'then_{mechanism}')(scale)
#         )
#     else:
#         bounds = ((float(bounds[0])), (float(bounds[1])))

#         df_meas = (
#             make_select_column(column_name, T) >>
#             dp.t.then_resize(size=len(df), constant=float(bounds[1])) >>
#             dp.t.then_clamp(bounds) >>
#             dp.t.then_mean() >>
#             getattr(dp.m, f'then_{mechanism}')(scale)
#         )
#     return df_meas(df)

# def get_query_private_outputs(df, query_type, mechanism, column_name, epsilon, bounds=(None, None), delta=1e-6):
#     sensitivity = calculate_sensitivity(query_type, bounds, df)
#     scale = calculate_scale(mechanism, sensitivity, epsilon, delta)
#     column_type = df[column_name].dtype
#     if np.issubdtype(column_type, np.integer):
#         df_copy = df.copy(deep=True)
#         df_copy[column_name] = df_copy[column_name].astype(float)
#         return create_df_meas(df_copy, column_name, float, mechanism, scale, query_type, bounds)
#     elif np.issubdtype(column_type, np.floating):
#         return create_df_meas(df, column_name, float, mechanism, scale, query_type, bounds)
#     else:
#         return create_df_meas(df, column_name, str, mechanism, scale, query_type, bounds)

def preset_parameters(df, column, query, parameter_list, hide_non_feasible_values=False):
    column_name, query_type = column.split('_')
    data = []
    
    if query == 'Epsilon':
        

        epsilon_variations = sorted(parameter_list)
        private_releases = {}
        
        for epsilon in epsilon_variations:
            releases = []
            sensitivity = 1 if query_type == 'count' else (st.session_state['query_info']['average_bounds']['upper_bound'] - st.session_state['query_info']['average_bounds']['lower_bound']) / len(df)
            scale = sensitivity / epsilon
            
            for _ in range(20):
                if query_type == 'count':
                    true_value = len(df[column_name])
                    noise = np.random.laplace(0, scale)
                else:
                    bounds = (st.session_state['query_info']['average_bounds']['lower_bound'], st.session_state['query_info']['average_bounds']['upper_bound'])
                    data = np.clip(df[column_name].to_numpy(), bounds[0], bounds[1])
                    true_value = np.mean(data)
                    noise = np.random.laplace(0, scale)
                releases.append(true_value + noise)
            private_releases[epsilon] = releases
            
        true_mean = len(df[column_name]) if query_type == 'count' else df[column_name].mean()
        data = []   
        for epsilon, releases in private_releases.items():
            for value in releases:
                data.append({'Epsilon': float(epsilon), 'Value': value, 'Type': 'Private Release'})
                
        chart_df = pd.DataFrame(data)
        if hide_non_feasible_values:
            chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
        chart_df['Epsilon'] = chart_df['Epsilon'].apply(lambda x: 'True Mean' if x == -1 else str(x))
        
        y_axis_min = min(0, min(min(releases) for releases in private_releases.values()))
        return create_figure(chart_df, true_mean, y_axis_min, query)

    if query == 'Bounds':

        private_releases = {}
        epsilon = st.session_state['epsilon_input']
        
        for bounds in parameter_list:
            releases = []
            lower_bound, upper_bound = bounds
            sensitivity = (upper_bound - lower_bound) / len(df)
            scale = sensitivity / epsilon
            
            for _ in range(20):
                data = np.clip(df[column_name].to_numpy(), lower_bound, upper_bound)
                true_value = np.mean(data)
                noise = np.random.laplace(0, scale)
                releases.append(true_value + noise)
            private_releases[bounds] = releases
            
        true_mean = df[column_name].mean()
        data = []

        for bounds, releases in private_releases.items():
            lower_bound, upper_bound = bounds
            for value in releases:

                data.append({'Bounds': f"[{lower_bound}, {upper_bound}]", 'Value': value, 'Type': 'Private Release'})
                
        chart_df = pd.DataFrame(data)
        if hide_non_feasible_values:
            chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
            
        y_axis_min = min(0, min(min(releases) for releases in private_releases.values()))
        return create_figure(chart_df, true_mean, y_axis_min, query)

    if query == 'Mechanism':
        mechanisms = sorted(parameter_list)
        epsilon = st.session_state['epsilon_input']
        private_releases = {}
        
        bounds = (None, None) if query_type == 'count' else (st.session_state['query_info']['average_bounds']['lower_bound'], st.session_state['query_info']['average_bounds']['upper_bound'])
        
        for mechanism in mechanisms:
            releases = []
            sensitivity = 1 if query_type == 'count' else (bounds[1] - bounds[0]) / len(df)
            scale = sensitivity / epsilon if mechanism == 'laplace' else sensitivity * np.sqrt(2 * np.log(1.25 / 1e-6)) / epsilon
            
            for _ in range(20):
                if query_type == 'count':
                    true_value = len(df[column_name])
                    noise = np.random.laplace(0, scale) if mechanism == 'laplace' else np.random.normal(0, scale)
                else:
                    data = np.clip(df[column_name].to_numpy(), bounds[0], bounds[1])
                    true_value = np.mean(data)
                    noise = np.random.laplace(0, scale) if mechanism == 'laplace' else np.random.normal(0, scale)
                releases.append(true_value + noise)
            private_releases[mechanism] = releases
            
        true_mean = len(df[column_name]) if query_type == 'count' else df[column_name].mean()
        data = []

        for mechanism, releases in private_releases.items():
            for value in releases:
                data.append({'Mechanism': str(mechanism), 'Value': value, 'Type': 'Private Release'})
                
        chart_df = pd.DataFrame(data)
        if hide_non_feasible_values:
            chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
            
        y_axis_min = min(0, min(min(releases) for releases in private_releases.values()))
        return create_figure(chart_df, true_mean, y_axis_min, query)
    




def noisy_histogram_creation(df, selected_query, data_type, parameter, parameter_list, min_val, max_val, num_bins, epsilon_input=None, hide_non_feasible_values=False):
    column_name, _ = selected_query.split('_')
    column_type = df[column_name].dtype
    data = {}
    
    if data_type == 'categorical':
        categories = sorted(df[column_name].unique().tolist())
        true_counts = df[column_name].value_counts().reindex(categories).values
        data['True Counts'] = [true_counts]
        
        if parameter == 'Epsilon':
            epsilon_variations = sorted(parameter_list, reverse=True)
            for eps in epsilon_variations:
                scale = 1 / eps
                noisy_counts = true_counts + np.random.laplace(0, scale, size=len(categories))
                if hide_non_feasible_values:
                    noisy_counts = np.where(noisy_counts < 0, np.nan, noisy_counts)
                data[eps] = [noisy_counts]
                
        elif parameter == 'Mechanism':
            if 'laplace' in parameter_list:
                scale = 1 / epsilon_input
                noisy_counts = true_counts + np.random.laplace(0, scale, size=len(categories))
                if hide_non_feasible_values:
                    noisy_counts = np.where(noisy_counts < 0, np.nan, noisy_counts)
                data['laplace'] = [noisy_counts]
                    
            if 'gaussian' in parameter_list:
                delta = 1e-6
                scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon_input
                noisy_counts = true_counts + np.random.normal(0, scale, size=len(categories))
                if hide_non_feasible_values:
                    noisy_counts = np.where(noisy_counts < 0, np.nan, noisy_counts)
                data['gaussian'] = [noisy_counts]
                
        return visualize_histogram(data, categories, 'Categorical')
    
    else:
        if parameter == 'Bins':
            bins_variations = sorted(parameter_list, reverse=True)
            for bins in bins_variations:
                bin_edges = np.linspace(min_val, max_val, bins + 1, dtype=column_type)
                true_counts, _ = np.histogram(df[column_name], bins=bin_edges)
                scale = 1 / epsilon_input
                noisy_counts = true_counts + np.random.laplace(0, scale, size=len(true_counts))
                if hide_non_feasible_values:
                    noisy_counts = np.where(noisy_counts < 0, np.nan, noisy_counts)
                data[bins] = [noisy_counts]
            return visualize_histogram_bins(data, column_name, df, min_val, max_val)
        
        bin_edges = np.linspace(min_val, max_val, num_bins + 1, dtype=column_type)
        true_counts, _ = np.histogram(df[column_name], bins=bin_edges)
        data['True Counts'] = [true_counts]
        
        if parameter == 'Epsilon':
            epsilon_variations = sorted(parameter_list, reverse=True)
            for eps in epsilon_variations:
                scale = 1 / eps
                noisy_counts = true_counts + np.random.laplace(0, scale, size=len(true_counts))
                if hide_non_feasible_values:
                    noisy_counts = np.where(noisy_counts < 0, np.nan, noisy_counts)
                data[eps] = [noisy_counts]
                
        elif parameter == 'Mechanism':
            if 'laplace' in parameter_list:
                scale = 1 / epsilon_input
                noisy_counts = true_counts + np.random.laplace(0, scale, size=len(true_counts))
                if hide_non_feasible_values:
                    noisy_counts = np.where(noisy_counts < 0, np.nan, noisy_counts)
                data['laplace'] = [noisy_counts]
                    
            if 'gaussian' in parameter_list:
                delta = 1e-6
                scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon_input
                noisy_counts = true_counts + np.random.normal(0, scale, size=len(true_counts))
                if hide_non_feasible_values:
                    noisy_counts = np.where(noisy_counts < 0, np.nan, noisy_counts)
                data['gaussian'] = [noisy_counts]
                
        return visualize_histogram(data, bin_edges, 'Continuous')

def visualize_histogram(data, categories, data_type):
    fig = go.Figure()
    blue_colors = px.colors.sequential.Blues[2:]

    for i, (key, counts) in enumerate(data.items()):
        for j, count in enumerate(counts):
            if key == 'True Counts':
                color = 'black'
                name = 'True Counts'
            elif key == 'laplace':
                color = blue_colors[i % len(blue_colors)]
                name = 'Laplace Mechanism'
            elif key == 'gaussian':
                color = blue_colors[i % len(blue_colors)]
                name = 'Gaussian Mechanism'
            else:
                color = blue_colors[i % len(blue_colors)]
                name = f'Epsilon {key}'

            fig.add_trace(go.Bar(
                x=categories if data_type == 'Categorical' else [(categories[i] + categories[i+1]) / 2 for i in range(len(categories)-1)],
                y=count,
                marker_color=color,
                marker_line_color=color,
                marker_line_width=[0 if val == 0 else 0.5 for val in count],
                name=name,
                showlegend=True if j == 0 else False,
                offsetgroup=str(key),
                cliponaxis=True
            ))

    if data_type == 'Continuous':
        bin_labels = [f'{categories[i]} - {categories[i+1]}' for i in range(len(categories)-1)]
        fig.update_layout(xaxis=dict(tickvals=[(categories[i] + categories[i+1]) / 2 for i in range(len(categories)-1)], ticktext=bin_labels))

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

    if data_type == 'Categorical':
        fig.update_xaxes(tickmode='array', tickvals=categories, ticktext=categories, tickangle=-45)

    return fig

def visualize_histogram_bins(data, column_name, df, min_val, max_val):
    fig = make_subplots(rows=len(data), cols=1, shared_xaxes=False, vertical_spacing=0.1)
    blue_colors = px.colors.sequential.Blues[2:]
    grey_colors = px.colors.sequential.Greys[5:]

    for i, (bins, counts) in enumerate(data.items()):
        bin_edges = np.linspace(min_val, max_val, bins+1)
        categories = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        true_counts, _ = np.histogram(df[column_name], bins=bin_edges)
        private_counts = counts[0]
        bin_labels = [f'{bin_edges[i]} - {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

        fig.add_trace(go.Bar(
            x=categories,
            y=true_counts,
            marker_color=grey_colors[i % len(grey_colors)],
            marker_line_color=grey_colors[i % len(grey_colors)],
            marker_line_width=[0 if val == 0 else 0.5 for val in true_counts],
            name=f'True Count ({bins} bins)',
            showlegend=True,
            offsetgroup='True Mean',
            cliponaxis=True,
            hovertext=[f'{bin_labels[i]}, Count: {format(true_counts[i], ".2f")}' for i in range(len(bin_labels))],
            hoverinfo="text"
        ), row=i+1, col=1)

        fig.add_trace(go.Bar(
            x=categories,
            y=private_counts,
            marker_color=blue_colors[i % len(blue_colors)],
            marker_line_color=blue_colors[i % len(blue_colors)],
            marker_line_width=[0 if val == 0 else 0.5 for val in private_counts],
            name=f'Private Outputs ({bins} bins)',
            showlegend=True,
            offsetgroup='Sensitive Values',
            cliponaxis=True,
            hovertext=[f'{bin_labels[i]}, Count: {format(private_counts[i], ".2f")}' for i in range(len(bin_labels))],
            hoverinfo="text"
        ), row=i+1, col=1)

        fig.update_xaxes(title_text='Category', row=i+1, col=1)
        fig.update_yaxes(title_text='Frequency', row=i+1, col=1)

    height = max(250, len(data) * 300)
    fig.update_layout(
        barmode='group',
        title='Grouped Outlined Histograms for Different Bin Variations',
        legend_title='Legend',
        legend=dict(
            traceorder="normal",
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor="Black",
            borderwidth=2
        ),
        height=height
    )

    return fig



def one_query_privacy_accuracy_lines(df, selected_query, mechanisms, alpha, epsilon_input, error_type, delta=None):
    epsilons = np.linspace(0.01, 1.0, 100)
    column_name, query_type = selected_query.split('_')
    
    if query_type == 'average':
        

        data_min = st.session_state['one_query_average_lower_bound']
        data_max = st.session_state['one_query_average_upper_bound']
        sensitivity = (data_max - data_min) / len(df)
    elif query_type in ['count', 'histogram']:
        sensitivity = 1

    line_df = pd.DataFrame()
    point_df = pd.DataFrame()

    for eps in epsilons:
        if 'laplace' in mechanisms:
            laplace_error = sensitivity / eps * np.log(1 / alpha)
            line_df = pd.concat([line_df, pd.DataFrame({'epsilon': [eps], 'error': [laplace_error], 'mechanism': ['laplace']})], ignore_index=True)
        if 'gaussian' in mechanisms:
            gaussian_scale = sensitivity / eps * np.sqrt(2 * np.log(1.25 / delta))
            gaussian_error = gaussian_scale * np.sqrt(2) * np.sqrt(-np.log(alpha))
            line_df = pd.concat([line_df, pd.DataFrame({'epsilon': [eps], 'error': [gaussian_error], 'mechanism': ['gaussian']})], ignore_index=True)

    if 'laplace' in mechanisms:
        laplace_error = sensitivity / epsilon_input * np.log(1 / alpha)
        point_df = pd.concat([point_df, pd.DataFrame({'epsilon': [epsilon_input], 'error': [laplace_error], 'mechanism': ['laplace']})], ignore_index=True)
    if 'gaussian' in mechanisms:
        gaussian_scale = sensitivity / epsilon_input * np.sqrt(2 * np.log(1.25 / delta))
        gaussian_error = gaussian_scale * np.sqrt(2) * np.sqrt(-np.log(alpha))
        point_df = pd.concat([point_df, pd.DataFrame({'epsilon': [epsilon_input], 'error': [gaussian_error], 'mechanism': ['gaussian']})], ignore_index=True)

    point_colors = ['#0072B2', '#80b1d3']
    error_bar_color = 'red'
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hypothetical outputs", "Accuracy vs. Privacy Parameter (\u03B5)"), horizontal_spacing=0.15, vertical_spacing=0.3)
    mechanism_colors = {'laplace': '#80b1d3', 'gaussian': '#0072B2'}

    if query_type == 'histogram':
        data_type = st.session_state['one_query_info']['data_type']
        min_val = st.session_state['one_query_info']['histogram_bounds']['lower_bound']
        max_val = st.session_state['one_query_info']['histogram_bounds']['upper_bound']
        num_bins = st.session_state["one_query_hist_bins"] if data_type == 'continuous' else df[column_name].nunique()
        
        # left_fig = noisy_histogram_creation(df, selected_query, data_type, 'Mechanism', mechanisms, epsilon_input, hide_non_feasible_values=True)
        left_fig = noisy_histogram_creation(df, selected_query, st.session_state['one_query_info']['data_type'], 'Mechanism', mechanisms, min_val, max_val,num_bins, epsilon_input=st.session_state['one_query_epsilon_slider'], hide_non_feasible_values=st.session_state['hide_non_feasible_values'])

        for trace in left_fig['data']:
            fig.add_trace(trace, row=1, col=1)

        total_count = df[column_name].count()
        avg_bin_count = total_count / num_bins

        if error_type == 'Relative Additive Error':
            line_df['error'] = np.abs(line_df['error'] / avg_bin_count)
            point_df['error'] = np.abs(point_df['error'] / avg_bin_count)

        for mechanism in line_df['mechanism'].unique():
            mechanism_df = line_df[line_df['mechanism'] == mechanism]
            fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines',
                                   name=f"Additive Error Upper Bound for {mechanism}",
                                   line=dict(color=mechanism_colors[mechanism])), row=1, col=2)

    else:
        if query_type == 'average':
            
            true_value = df[column_name].mean()
            
        else:
            true_value = df[column_name].count()
        # true_value = df[column_name].clip.mean() if query_type == 'average' else df[column_name].count()
        summary_df = pd.DataFrame({'mechanism': mechanisms, 'true_value': [true_value] * len(mechanisms)})
        point_df['original_error'] = point_df['error']

        if error_type == 'Relative Additive Error':
            line_df['error'] = np.abs(line_df['error'] / true_value)
            point_df['error'] = np.abs(point_df['error'] / true_value)

        for index, mechanism in enumerate(line_df['mechanism'].unique()):
            mechanism_df = line_df[line_df['mechanism'] == mechanism]
            fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines',
                                   name=f"Additive Error Upper Bound for {mechanism}",
                                   line=dict(color=point_colors[index])), row=1, col=2)

        max_value = point_df['original_error'].max()
        fig.update_yaxes(title_text="Hypothetical outputs", range=[0, (true_value + max_value) * 1.25], row=1, col=1)

        for mechanism in mechanisms:
            scale = sensitivity / epsilon_input if mechanism == 'laplace' else sensitivity / epsilon_input * np.sqrt(2 * np.log(1.25 / delta))
            vals = []
            for _ in range(20):
                if query_type == 'count':
                    value = len(df[column_name])
                else:
                    data = np.clip(df[column_name].to_numpy(), data_min, data_max)
                    value = np.mean(data)
                noise = np.random.laplace(0, scale) if mechanism == 'laplace' else np.random.normal(0, scale)
                vals.append(value + noise)
            
            error_row = point_df.loc[point_df['mechanism'] == mechanism]
            if query_type == 'average':
            
                clipped_value = np.clip(df[column_name].to_numpy(), data_min, data_max).mean()
            else:
                clipped_value = true_value
            fig.add_trace(go.Scatter(x=[mechanism], y=[clipped_value],
                                   error_y=dict(type='data', array=[error_row['original_error'].values[0]], visible=True, color=error_bar_color),
                                   mode='markers', name=mechanism, marker=dict(size=10, opacity=0), showlegend=False),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=[mechanism] * 20, y=vals, mode='markers', name=mechanism,
                                   marker=dict(color=point_colors[mechanisms.index(mechanism)], opacity=.3), showlegend=True),
                         row=1, col=1)

        fig.add_shape(type="line", x0=-.1, y0=true_value, x1=1.1, y1=true_value,
                     line=dict(color="black", width=1, dash="dash"), xref="paper", yref="y",
                     row=1, col=1, showlegend=True, name='True Mean')

        fig.update_xaxes(title_text="Mechanism", row=1, col=1)

    fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2), margin=dict(l=0, r=0, t=0, b=100))
    fig.update_xaxes(title_text="Privacy Parameter (\u03B5)", row=1, col=2)

    if error_type == 'Absolute Additive Error':
        fig.update_yaxes(title_text="Absolute Error Upper Bound", row=1, col=2)
    elif error_type == 'Relative Additive Error':
        fig.update_yaxes(title_text="Relative Error Upper Bound", tickformat=',.0%', row=1, col=2)

    for index, row in point_df.iterrows():
        fig.add_trace(go.Scatter(x=[row['epsilon']], y=[row['error']], mode='markers',
                               name="Selected Epsilon" if index == 0 else None,
                               marker=dict(color='red', size=6), showlegend=index == 0),
                     row=1, col=2)

    return fig





def compare_compositors(df):
    # Create a figure with two subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2, subplot_titles=("Per query epsilon", "Per query delta"))

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

