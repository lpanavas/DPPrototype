
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

import math
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

    if query_type == 'count':
        df_meas = (
            make_select_column(column_name, T) >>
            dp.t.then_count() >>
            getattr(dp.m, f'then_{mechanism}')(scale)
        )
    else:
        bounds = ((float(bounds[0])), (float(bounds[1])))

        df_meas = (
            make_select_column(column_name, T) >>
            dp.t.then_resize(size=len(df), constant=float(bounds[1])) >>
            dp.t.then_clamp(bounds) >>
            dp.t.then_mean() >>
            getattr(dp.m, f'then_{mechanism}')(scale)
        )
    return df_meas(df)

def get_query_private_outputs(df, query_type, mechanism, column_name, epsilon, bounds=(None, None), delta=1e-6):
    sensitivity = calculate_sensitivity(query_type, bounds, df)
    scale = calculate_scale(mechanism, sensitivity, epsilon, delta)
    column_type = df[column_name].dtype
    if np.issubdtype(column_type, np.integer):
        df[column_name] = df[column_name].astype(float)
        # bounds = (float(bounds[0]), float(bounds[1]))
    return create_df_meas(df, column_name, float, mechanism, scale, query_type, bounds)

def preset_parameters(df, column, query,  parameter_list, hide_non_feasible_values=False):

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
                if hide_non_feasible_values:
                        chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
           
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
            if hide_non_feasible_values:
                        chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
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
            if hide_non_feasible_values:
                        chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
            min_private_release = min(min(private_releases[bounds]) for bounds in parameter_list)
            y_axis_min = min(0, min_private_release)

            return create_figure(chart_df, true_mean, y_axis_min)
    if query == 'Mechanism':
        columnName, queryType = column.split('_')
        mechanisms = sorted(parameter_list)
        
        epsilon = st.session_state['epsilon_input']
       
        bounds = (None, None)
        if queryType == 'count':

            data = []
            columnType = df[columnName].dtype

            private_releases = {}
            
            # Will store releases for each mechanism
            for mechanism in mechanisms:
                releases = []
                for i in range(20):
                    releases.append(get_query_private_outputs(df, queryType, mechanism, columnName, epsilon, bounds, .0000001))
                
                private_releases[mechanism] = releases
            
            true_mean = df[columnName].count()
            
            # Add data points for each mechanism
            for mechanism_str, releases in private_releases.items():
                mechanism = str(mechanism_str)  
                for value in releases:
                    data.append({
                        'Mechanism': mechanism,
                        'Value': value,
                        'Type': 'Private Release'
                    })
                    
                

            chart_df = pd.DataFrame(data)
            if hide_non_feasible_values:
                        chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
            min_private_release = min(min(private_releases[mechanism]) for mechanism in mechanisms)
            y_axis_min = min(0, min_private_release) 

            return create_figure(chart_df, true_mean, y_axis_min)
        if queryType == 'average':

            data = []
            columnType = df[columnName].dtype
            
            bounds = (st.session_state['queries'][column]['lower_bound'], st.session_state['queries'][column]['upper_bound'])

            private_releases = {}
            
              # Will store releases for each mechanism
            for mechanism in mechanisms:
                releases = []
                for i in range(20):
                    releases.append(get_query_private_outputs(df, queryType, mechanism, columnName, epsilon, bounds, .0000001))
               
                private_releases[mechanism] = releases
        
            true_mean = df[columnName].mean()
        
        # Add data points for each mechanism
            for mechanism_str, releases in private_releases.items():
                mechanism = str(mechanism_str)  
                for value in releases:
                    data.append({
                        'Mechanism': mechanism,
                        'Value': value,
                        'Type': 'Private Release'
                    })
                
            

            chart_df = pd.DataFrame(data)
            if hide_non_feasible_values:
                        chart_df.loc[chart_df['Value'] < 0, 'Value'] = np.nan
            min_private_release = min(min(private_releases[mechanism]) for mechanism in mechanisms)
            y_axis_min = min(0, min_private_release) 

            return create_figure(chart_df, true_mean, y_axis_min)
        







def at_delta(meas, delta):
    # convert from ρ to ε(δ)
    meas = dp.c.make_zCDP_to_approxDP(meas)
    # convert from ε(δ) to (ε, δ)
    return dp.c.make_fix_delta(meas, delta)   
def noisy_histogram_creation(df, selected_query, data_type, parameter, parameter_list, epsilon_input, hide_non_feasible_values=False):

    columnName, queryType = selected_query.split('_')
    epsilon_variations = sorted(parameter_list)
    epsilon_variations.reverse()
    categories = sorted(df[columnName].unique())

    column_type = df[columnName].dtype
    histogram = {}
    if np.issubdtype(column_type, np.integer):
        df_copy = df.copy(deep=True)
        df_copy[columnName] = df_copy[columnName].astype(column_type)
        categories_type = np.array(categories, dtype=np.int32)

       
        # histogram = (
        #     make_select_column(columnName, int) >>
        #     dp.t.then_count_by_categories(categories=categories_type)
        # )
        categories = np.array(categories, dtype=np.int32)

    if column_type == float:
        df_copy = df.copy(deep=True)
        df_copy[columnName] = df_copy[columnName].astype('float')
        # histogram = (  
        #   make_select_column(columnName, float) >>
              
        #   dp.t.then_count_by_categories(categories=categories) 
                  
        #   )
        categories = np.array(categories, dtype=float)
    if column_type == str:
        df_copy = df.copy(deep=True)
        df_copy[columnName] = df_copy[columnName].astype('str')
        # histogram = (  
        #   make_select_column(columnName, str) >>
              
        #   dp.t.then_count_by_categories(categories=categories) 
                  
        #   )
        categories = np.array(categories, dtype=str)
    data = {}
    true_counts = df[columnName].value_counts().reindex(categories).values

    data['True Counts'] = [true_counts]
    if parameter == 'Epsilon':
        for eps in epsilon_variations:
            data[eps] = []
            histogram = (
                        dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
                        dp.t.make_select_column(columnName, str) >>
                        dp.t.then_cast_default(int) >>
                        dp.t.then_count_by_categories(categories=categories, MO=dp.L1Distance[int] )
                    )
            # histogram(df.to_csv(index=False, header=False))
            noisy_laplace_histogram = dp.binary_search_chain(
                lambda s: histogram >> dp.m.then_laplace(scale=s),
                d_in=1, d_out=eps)
            sensitive_counts = noisy_laplace_histogram(df.to_csv(index=False, header=False))

            # noisy_histogram = dp.binary_search_chain(
            #     lambda s: histogram >> dp.m.then_laplace(scale=s),
            #     d_in=1, d_out=eps)

            # sensitive_counts = noisy_histogram(df_copy)
            if hide_non_feasible_values:
                    sensitive_counts = [np.nan if x < 0 else x for x in sensitive_counts]
            data[eps].append(sensitive_counts[:-1])
    elif parameter == 'Mechanism':
 
        if 'laplace'in parameter_list:
            data['laplace'] = []
            # categories = np.array(df[columnName].unique(), dtype=np.int32)

            # categories = np.array(df[columnName].unique(), dtype=np.int32)

            histogram = (
                        dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
                        dp.t.make_select_column(columnName, str) >>
                        dp.t.then_cast_default(int) >>
                        dp.t.then_count_by_categories(categories=categories, MO=dp.L1Distance[int] )
                    )
            # histogram(df.to_csv(index=False, header=False))
            noisy_laplace_histogram = dp.binary_search_chain(
                lambda s: histogram >> dp.m.then_laplace(scale=s),
                d_in=1, d_out=epsilon_input)

            sensitive_laplace_counts = noisy_laplace_histogram(df.to_csv(index=False, header=False))
            if hide_non_feasible_values:
                    sensitive_laplace_counts = [np.nan if x < 0 else x for x in sensitive_laplace_counts]
            data['laplace'].append(sensitive_laplace_counts[:-1])
        if  'gaussian' in parameter_list:
            data['gaussian'] = []
            # categories = np.array(df[columnName].unique(), dtype=np.int32)

            delta = 1e-6
            t_hist = (
                dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
                dp.t.make_select_column(columnName, str) >>
                dp.t.then_cast_default(int) >>
                dp.t.then_count_by_categories(categories=categories, MO=dp.L2Distance[float])
            )
            m_hist = dp.binary_search_chain(
                lambda s: at_delta(t_hist >> dp.m.then_gaussian(scale=s), delta), 
                d_in=1, 
                d_out=(epsilon_input, delta))
            sensitive_gaussian_counts = m_hist(df.to_csv(index=False, header=False))
            if hide_non_feasible_values:
                    sensitive_gaussian_counts = [np.nan if x < 0 else x for x in sensitive_gaussian_counts]
            data['gaussian'].append(sensitive_gaussian_counts[:-1])
    

    return visualize_data_histogram(data, categories, 'Categorical')


def visualize_data_histogram(data, categories, data_type):

    fig = go.Figure()

    # Create a list of blue colors for the histograms
    # blue_colors = ['rgba(0, 0, 255, 1)', 'rgba(0, 128, 255, 1)', 'rgba(0, 192, 255, 1)', 'rgba(0, 224, 255, 1)']
    blue_colors = px.colors.sequential.Blues[2:] 

    for i, (eps, counts) in enumerate(data.items()):
        for j, count in enumerate(counts):
            if eps == 'True Counts':
                color = 'black'  # Red outline for true counts
                name = 'True Counts'
            elif eps == 'laplace':
                color = blue_colors[i % len(blue_colors)]  # Cycle through blue colors for each epsilon
                name = 'Laplace Mechanism'
            elif eps == 'gaussian':
                color = blue_colors[i % len(blue_colors)]  # Cycle through blue colors for each epsilon
                name = 'Gaussian Mechanism'
            else:
                color = blue_colors[i % len(blue_colors)]  # Cycle through blue colors for each epsilon
                name = f'Epsilon {eps}'

            fig.add_trace(go.Bar(
                x=categories,
                y=count,
                marker_color=color,  # Transparent fill
                marker_line_color=color,  # Colored outline
                marker_line_width=[0 if val == 0 else 0.5 for val in count],  # Set line width to 0 for zero counts  # Thinner outline

                name=name,
                showlegend=True if j == 0 else False,  # Only show the first histogram's legend
                offsetgroup=str(eps), 
                    cliponaxis=True, # Offset group for each epsilon
                    # Shared bin group for alignment
            ))
    if data_type == 'Continuous':
        # Create bin labels for continuous data
        bin_labels = [f'{categories[i]} - {categories[i+1]}' for i in range(len(categories)-1)]
        fig.update_layout(xaxis=dict(tickvals=categories[:-1], ticktext=bin_labels))

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







def noisy_histogram_creation_continuous(df, selected_query, parameter, parameter_list, epsilon_input, hide_non_feasible_values=False, **kwargs):
   
    columnName, queryType = selected_query.split('_')
    column_type = df[columnName].dtype
    data= {}
    min_val = st.session_state['queries'][selected_query]['lower_bound']
    max_val = st.session_state['queries'][selected_query]['upper_bound']
    num_bins = st.session_state['queries'][selected_query]['bins']
    bin_edges = np.linspace(min_val, max_val, num_bins + 1, dtype = column_type)

    if parameter == 'Epsilon':
        epsilon_variations = sorted(parameter_list)
         
        epsilon_variations.reverse()
        
    # Create bins
    

        histogram = ()

        if np.issubdtype(column_type, np.integer):
            df_copy = df.copy(deep=True)
            df_copy[columnName] = df_copy[columnName].astype(column_type)
            
            binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), edges=bin_edges[1:-1])
        
            histogram = (
                make_select_column(columnName, int) >>
                binner >>
                dp.t.then_count_by_categories(categories=np.arange(num_bins-1))
            )

        if column_type == float:
            df_copy = df.copy(deep=True)
            df_copy[columnName] = df_copy[columnName].astype('float')
            
            binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=float)), dp.symmetric_distance(), edges=bin_edges[1:-1])
            histogram = (  
                make_select_column(columnName, float) >>
                binner >>
                dp.t.then_count_by_categories(categories=np.arange(num_bins-1)) 
                )
        

        data = {}
        true_counts, _ = np.histogram(df[columnName], bins=bin_edges)
        data['True Counts'] = [true_counts]
        for eps in epsilon_variations:
            data[eps] = []
            for i in range(1):
                noisy_histogram = dp.binary_search_chain(
                    lambda s: histogram >> dp.m.then_laplace(scale=s),
                    d_in=1, d_out=eps)

                sensitive_counts = noisy_histogram(df_copy)
                if hide_non_feasible_values:
                    sensitive_counts = [np.nan if x < 0 else x for x in sensitive_counts]
                data[eps].append(sensitive_counts)
        # Add true counts to the data dictionary
    if parameter == 'Mechanism':
        
  
           
        true_counts, _ = np.histogram(df[columnName], bins=bin_edges)
        data['True Counts'] = [true_counts] 
         
                
        if  'laplace' in parameter_list:
            data['laplace'] = []
            if np.issubdtype(column_type, np.integer):
                binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), edges=bin_edges[1:-1])
                histogram = (
                    dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
                    dp.t.make_select_column(columnName, str) >>
                    dp.t.then_cast_default(int) >> 
                    binner>>
                    dp.t.then_count_by_categories(categories=np.arange(num_bins-1), MO=dp.L1Distance[int])
                )
            if column_type == float:
          
          
                binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=float)), dp.symmetric_distance(), edges=bin_edges[1:-1])
                histogram = (
                    dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
                    dp.t.make_select_column(columnName, str) >>
                    dp.t.then_cast_default(float) >> 
                    binner>>
                    dp.t.then_count_by_categories(categories=np.arange(num_bins-1))
                )
            noisy_laplace_histogram = dp.binary_search_chain(
                lambda s: histogram >> dp.m.then_laplace(scale=s),
                d_in=1, d_out=epsilon_input)
            sensitive_laplace_counts = noisy_laplace_histogram(df.to_csv(index=False, header=False))
           

            if hide_non_feasible_values:
                    sensitive_laplace_counts = [np.nan if x < 0 else x for x in sensitive_laplace_counts]
            data['laplace'].append(sensitive_laplace_counts[:-1])
        if  'gaussian' in parameter_list:
            data['gaussian'] = []
            delta = 1e-6
            if np.issubdtype(column_type, np.integer):
                binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), edges=bin_edges[1:-1])
                histogram = (
                    dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
                    dp.t.make_select_column(columnName, str) >>
                    dp.t.then_cast_default(int) >> 
                    binner>>
                    dp.t.then_count_by_categories(categories=np.arange(num_bins-1), MO=dp.L2Distance[int])
                )
            if column_type == float:
                
          
                binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=float)), dp.symmetric_distance(), edges=bin_edges[1:-1])
                histogram = (
                    dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
                    dp.t.make_select_column(columnName, str) >>
                    dp.t.then_cast_default(float) >> 
                    binner>>
                    dp.t.then_count_by_categories(categories=np.arange(num_bins-1), MO=dp.L2Distance[float])
                )
            noisy_gaussian_histogram = dp.binary_search_chain(
                lambda s: at_delta(histogram >> dp.m.then_gaussian(scale=s), delta), 
                d_in=1, 
                d_out=(epsilon_input, delta))

            sensitive_gaussian_counts = noisy_gaussian_histogram(df.to_csv(index=False, header=False))
            if hide_non_feasible_values:
                    sensitive_gaussian_counts = [np.nan if x < 0 else x for x in sensitive_gaussian_counts]
            data['gaussian'].append(sensitive_gaussian_counts[:-1])

    if parameter == 'Bins':
        bins_variations = sorted(parameter_list)
        bins_variations.reverse()
         

        min_val = kwargs['lower_bound']
        max_val = kwargs['upper_bound']
        epsilon = epsilon_input
        data = {}
       
        for bins in bins_variations:
            bin_edges = np.linspace(min_val, max_val, bins + 1, dtype = column_type)
            if np.issubdtype(column_type, np.integer):
                df_copy = df.copy(deep=True)
                df_copy[columnName] = df_copy[columnName].astype(column_type)
                
                binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), edges=bin_edges[1:-1])
                histogram = (
                    make_select_column(columnName, int) >>
                    binner >>
                    dp.t.then_count_by_categories(categories=np.arange(bins-1))
                )
            if column_type == float:
                df_copy = df.copy(deep=True)
                df_copy[columnName] = df_copy[columnName].astype('float')
                
                binner = dp.t.make_find_bin(dp.vector_domain(dp.atom_domain(T=float)), dp.symmetric_distance(), edges=bin_edges[1:-1])
                histogram = (  
                    make_select_column(columnName, float) >>
                    binner >>
                    dp.t.then_count_by_categories(categories=np.arange(bins-1)) 
                )

            noisy_histogram = dp.binary_search_chain(
                lambda s: histogram >> dp.m.then_laplace(scale=s),
                d_in=1, d_out=epsilon)

            sensitive_counts = noisy_histogram(df_copy)
            if hide_non_feasible_values:
                sensitive_counts = [np.nan if x < 0 else x for x in sensitive_counts]
            data[bins] = [sensitive_counts]

        return visualize_data_histogram_bins(data,  columnName, df, min_val, max_val)

    return visualize_data_histogram(data, bin_edges, 'Continuous')

        
def visualize_data_histogram_bins(data, columnName, df, min_val, max_val):
    fig = make_subplots(rows=len(data), cols=1, shared_xaxes=False, vertical_spacing=0.1)

    blue_colors = px.colors.sequential.Blues[2:] 
    grey_colors = px.colors.sequential.Greys[5:] 


    for i, (bins, counts) in enumerate(data.items()):
        bin_edges = np.linspace(min_val, max_val, bins+1)
        categories = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

        true_counts, _ = np.histogram(df[columnName], bins=bin_edges)
        private_counts = counts[0]

        bin_labels = [f'{bin_edges[i]} - {bin_edges[i+1]}' for i in range(len(bin_edges)-1)]

        fig.add_trace(go.Bar(
            x=categories,
            y=true_counts,
            marker_color=grey_colors[i % len(grey_colors)],  
            marker_line_color=grey_colors[i % len(grey_colors)],  
            marker_line_width=[0 if val== 0 else 0.5 for val in true_counts],  
            name=f'True Count ({bins} bins)',
            showlegend=True,  # show True counts for all charts
            offsetgroup='True Mean', 
            cliponaxis=True, 
            hovertext=[f'{bin_labels[i]}, Count: {format(true_counts[i], ".2f")}' for i in range(len(bin_labels))],
            hoverinfo="text",

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

        # add x and y axis labels for each subplot
        fig.update_xaxes(title_text='Category', row=i+1, col=1)
        fig.update_yaxes(title_text='Frequency', row=i+1, col=1)

    height = max(250, len(data) * 300)  # adjust the height based on the number of subplots

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
        height=height # adjust height based on number of subplots
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



def one_query_privacy_accuracy_lines_histogram(df, selected_query, mechanisms, alpha, epsilon_input, error_type, delta = None):
   
    epsilons = np.linspace(0.01, 1.0, 100) 
    columnName, queryType = selected_query.split('_')
    data_type = st.session_state['queries'][selected_query]['data_type']
    sensitivity = 1  # Sensitivity for histogram queries is usually 1

    line_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

    for eps in epsilons:
        if 'laplace' in mechanisms:
            laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / eps, alpha)  
            new_row = pd.DataFrame({'epsilon': [eps], 'error': [laplace_error], 'mechanism': 'laplace'})
            line_df = pd.concat([line_df, new_row], ignore_index=True)

        if 'gaussian' in mechanisms:
            gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, eps, delta), alpha)  
            new_row = pd.DataFrame({'epsilon': [eps], 'error': [gaussian_error], 'mechanism': 'gaussian'})
            line_df = pd.concat([line_df, new_row], ignore_index=True)

    point_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

    if 'laplace' in mechanisms:
        laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / epsilon_input, alpha)  
        new_row = pd.DataFrame({'epsilon': [epsilon_input], 'error': [laplace_error], 'mechanism': 'laplace'})
        point_df = pd.concat([point_df, new_row], ignore_index=True)

    if 'gaussian' in mechanisms:
        gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, epsilon_input, delta), alpha)  
        new_row = pd.DataFrame({'epsilon': [epsilon_input], 'error': [gaussian_error], 'mechanism': 'gaussian'})
        point_df = pd.concat([point_df, new_row], ignore_index=True)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hypothetical outputs", "Accuracy vs. Privacy Parameter (\u03B5)"), horizontal_spacing=0.15)
    point_colors = ['#0072B2', '#80b1d3'] 

    error_bar_color = 'red'  
    
    if data_type == 'continuous':
        left_fig = noisy_histogram_creation_continuous(df, selected_query, 'Mechanism', mechanisms, epsilon_input, hide_non_feasible_values=True)
        
    elif data_type == 'categorical':
        left_fig =  left_fig = noisy_histogram_creation(df, selected_query, 'Categorical', 'Mechanism', mechanisms, epsilon_input, hide_non_feasible_values=True)

    for trace in left_fig['data']:
        fig.add_trace(trace, row=1, col=1)

    for index, mechanism in enumerate(line_df['mechanism'].unique()):
        mechanism_df = line_df[line_df['mechanism'] == mechanism]
        fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines', 
                                name=f"Additive Error Upper Bound for {mechanism}", 
                                line=dict(color=point_colors[index])), row=1, col=2)

    for index, row in point_df.iterrows():
        if index == 0:  
            showlegend = True
        else:
            showlegend = False
        fig.add_trace(
            go.Scatter(x=[row['epsilon']], y=[row['error']], mode='markers', 
                    name="Selected Epsilon",
                    marker=dict(color='red', size=6),
                    showlegend=showlegend),  
            row=1, col=2  
        )

    fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
        
    return fig
    
def one_query_privacy_accuracy_lines(df, selected_query, mechanisms, alpha, epsilon_input, error_type, delta = None):
    
    epsilons = np.linspace(0.01, 1.0, 100) 
    columnName, queryType = selected_query.split('_')
    data_min = None
    data_max = None
 

    if queryType == 'average':
        columnType = df[columnName].dtype
        data_min = st.session_state['queries'][selected_query]['lower_bound']
        data_max = st.session_state['queries'][selected_query]['upper_bound']
        sensitivity = (data_max - data_min)/df.shape[0]
  

    elif queryType == 'count':
        sensitivity = 1  # Set the sensitivity to 1 for 'count' query

    elif queryType == 'histogram':
        sensitivity = 1
        if st.session_state['queries'][selected_query]['data_type'] == 'continuous':
            bins = st.session_state['queries'][selected_query]['bins']  # Use specified number of bins for continuous data
        data_type = st.session_state['queries'][selected_query]['data_type']

    line_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

    for eps in epsilons:
        if 'laplace' in mechanisms:
            laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / eps, alpha)  # Ensure you use 'eps' here, not 'epsilon'
            new_row = pd.DataFrame({'epsilon': [eps], 'error': [laplace_error], 'mechanism': 'laplace'})
            line_df = pd.concat([line_df, new_row], ignore_index=True)

            
        if 'gaussian' in mechanisms:
            gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, eps, delta), alpha)  # Again, ensure 'eps' is used
            new_row = pd.DataFrame({'epsilon': [eps], 'error': [gaussian_error], 'mechanism': 'gaussian'})
            line_df = pd.concat([line_df, new_row], ignore_index=True)

    point_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

# Calculate Laplace error for the specified epsilon value and append to the DataFrame
    if 'laplace' in mechanisms:
        laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / epsilon_input, alpha)  # Ensure you use 'eps' here, not 'epsilon'
        new_row = pd.DataFrame({'epsilon': [epsilon_input], 'error': [laplace_error], 'mechanism': 'laplace'})
        point_df = pd.concat([point_df, new_row], ignore_index=True)

    # Calculate Gaussian error for the specified epsilon value and append to the DataFrame    
    if 'gaussian' in mechanisms:
        gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, epsilon_input, delta), alpha)  # Again, ensure 'eps' is used
        new_row = pd.DataFrame({'epsilon': [epsilon_input], 'error': [gaussian_error], 'mechanism': 'gaussian'})
        point_df = pd.concat([point_df, new_row], ignore_index=True)
    
    point_colors = ['#0072B2', '#80b1d3']   

    error_bar_color = 'red'  # Red for error bars

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hypothetical outputs", "Accuracy vs. Privacy Parameter (\u03B5)"), horizontal_spacing=0.15)
    mechanism_colors = {'laplace': '#80b1d3', 'gaussian': '#0072B2'}

    if queryType == 'histogram':
        if data_type == 'continuous':
            left_fig = noisy_histogram_creation_continuous(df, selected_query, 'Mechanism', mechanisms, epsilon_input, hide_non_feasible_values=True)
            num_bins = st.session_state['queries'][selected_query]['bins']
            
            
        elif data_type == 'categorical':
            left_fig =  left_fig = noisy_histogram_creation(df, selected_query, 'Categorical', 'Mechanism', mechanisms, epsilon_input, hide_non_feasible_values=True)
            num_bins = df[columnName].nunique()


        for trace in left_fig['data']:
            fig.add_trace(trace, row=1, col=1)

        total_count = df[columnName].count()
       
        avg_bin_count = total_count / num_bins

        if error_type == 'Relative Additive Error':
            line_df['error'] = np.abs(line_df['error'] / avg_bin_count)
            point_df['error'] = np.abs(point_df['error'] / avg_bin_count)
   

        for index, mechanism in enumerate(line_df['mechanism'].unique()):
            mechanism_df = line_df[line_df['mechanism'] == mechanism]
            fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines', 
                                    name=f"Additive Error Upper Bound for {mechanism}", 
                                    line=dict(color=mechanism_colors[mechanism])), row=1, col=2)

    else:
        if queryType == 'average':
            summary_df = pd.DataFrame({
                'mechanism': mechanisms,
                'true_value': [df[columnName].mean()] * len(mechanisms),  # Example true mean values
                # 'error_margin': [point_df['error'][0] ,point_df['error'][1] ]  # Example error margins
            })
        else:
            summary_df = pd.DataFrame({
                'mechanism': mechanisms,
                'true_value': [df[columnName].count()] * len(mechanisms),  # Example true mean values
                # 'error_margin': [point_df['error'][0] ,point_df['error'][1] ]  # Example error margins
            })
        point_df['original_error'] = point_df['error']

        if error_type == 'Relative Additive Error':
            line_df['error'] = np.abs(line_df['error'] / summary_df['true_value'][0])
            point_df['error'] = np.abs(point_df['error'] / summary_df['true_value'][0])

        for index, mechanism    in enumerate(line_df['mechanism'].unique()):
            mechanism_df = line_df[line_df['mechanism'] == mechanism]
            fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines', 
                                    name=f"Additive Error Upper Bound for {mechanism}", 
                                    line=dict(color=point_colors[index])), row=1, col=2)
        max_value =point_df['original_error'].max()
        fig.update_yaxes(title_text="Hypothetical outputs", range=[0, (summary_df['true_value'][0]+ max_value)*1.25], row=1, col=1)

        laplace_scale =  sensitivity/ epsilon_input
        vals_laplace = []
        vals_gauss =[]
        
        if 'laplace' in mechanisms: 
            laplace_row = point_df.loc[point_df['mechanism'] == 'laplace']
            for i in range(20):
                vals_laplace.append(get_query_private_outputs(df, queryType, 'laplace', columnName, epsilon_input,(data_min, data_max),  delta))
            fig.add_trace(
                go.Scatter(
                    x=['laplace'],  # Use 0 for Laplace
                    y=[summary_df.iloc[0]['true_value']],
                    error_y=dict(type='data', array=[laplace_row['original_error'].values[0]], visible=True, color=error_bar_color),

                    mode='markers',
                    name='laplace',
                    marker=dict(size=10, opacity=00),  # Adjust opacity for visual emphasis
                    showlegend=False,
                ),
                

                row=1, col=1
            )    
            fig.add_trace(
                go.Scatter(
                    x=['laplace'] * 20,  # X-values are categorical labels
                    y=vals_laplace,
                    mode='markers',
                    name='Laplace',
                    marker=dict(color=point_colors[0], opacity=.3),
                    showlegend=False,
                ),
                

                row=1, col=1
            )
            
        

        if 'gaussian' in mechanisms:
            gaussian_row = point_df.loc[point_df['mechanism'] == 'gaussian']
            for i in range(20):
                
                vals_gauss.append(get_query_private_outputs(df, queryType, 'gaussian', columnName, epsilon_input, (data_min, data_max),  delta))       
            fig.add_trace(
                go.Scatter(
                    x=['gaussian'],  # Use 1 for Gaussian
                    y=[summary_df.iloc[0]['true_value']],
                    error_y=dict(type='data', array=[gaussian_row['original_error'].values[0]], visible=True, color=error_bar_color),

                    mode='markers',
                    name='gaussian',
                    marker=dict(size=10, opacity=00),  # Adjust opacity for visual emphasis
                    showlegend=False,
                ),
                

                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=['gaussian'] * 20,  # X-values are categorical labels
                    y=vals_gauss,
                    mode='markers',
                    name='gaussian',
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
                

        # Update xaxis properties
        fig.update_xaxes(title_text="Mechanism", row=1, col=1)

    fig.update_xaxes(title_text="Privacy Parameter (\u03B5)", row=1, col=2)

    # Update yaxis properties
    if error_type == 'Absolute Additive Error':
        fig.update_yaxes(title_text="Absolute Error Upper Bound", row=1, col=2)
    elif error_type == 'Relative Additive Error':
      
        fig.update_yaxes(title_text="Relative Error Upper Bound", tickformat=',.0%', row=1, col=2)
   
    for index, row in point_df.iterrows():
        if index == 0:  
            showlegend = True
        else:
            showlegend = False
        fig.add_trace(
            go.Scatter(x=[row['epsilon']], y=[row['error']], mode='markers', 
                    name="Selected Epsilon",
                    marker=dict(color='red', size=6),
                    showlegend=showlegend),  
            row=1, col=2  
        )

    return fig
# def one_query_privacy_accuracy_lines(df, selected_query, mechanisms, alpha, epsilon_input, error_type, delta = None):
    
#     print(selected_query, mechanisms, alpha, epsilon_input, error_type, delta)
#     epsilons = np.linspace(0.01, 1.0, 100) 
#     columnName, queryType = selected_query.split('_')
#     data_min = None
#     data_max = None
#     data_type = st.session_state['queries'][selected_query]['data_type']

  
#     if queryType == 'average':
#         columnType = df[columnName].dtype
#         data_min = st.session_state['queries'][selected_query]['lower_bound']
#         data_max = st.session_state['queries'][selected_query]['upper_bound']
#         sensitivity = (data_max - data_min)/df.shape[0]
#         summary_df = pd.DataFrame({
#             'mechanism': mechanisms,
#             'true_value': [df[columnName].mean()] * len(mechanisms),  # Example true mean values
#             # 'error_margin': [point_df['error'][0] ,point_df['error'][1] ]  # Example error margins
#         })

#     elif queryType == 'count':
#         sensitivity = 1  # Set the sensitivity to 1 for 'count' query
#         summary_df = pd.DataFrame({
#             'mechanism': mechanisms,
#             'true_value': [df[columnName].count()] * len(mechanisms),  # Example true mean values
#             # 'error_margin': [point_df['error'][0] ,point_df['error'][1] ]  # Example error margins
#         })
#     elif queryType == 'histogram':
#         sensitivity = 1
#         if st.session_state['queries'][selected_query]['data_type'] == 'continuous':
#             bins = st.session_state['queries'][selected_query]['bins']  # Use specified number of bins for continuous data

#     line_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

#     for eps in epsilons:
#         if 'laplace' in mechanisms:
#             laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / eps, alpha)  # Ensure you use 'eps' here, not 'epsilon'
#             new_row = pd.DataFrame({'epsilon': [eps], 'error': [laplace_error], 'mechanism': 'lapalce'})
#             line_df = pd.concat([line_df, new_row], ignore_index=True)

            
#         if 'gaussian' in mechanisms:
#             gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, eps, delta), alpha)  # Again, ensure 'eps' is used
#             new_row = pd.DataFrame({'epsilon': [eps], 'error': [gaussian_error], 'mechanism': 'gaussian'})
#             line_df = pd.concat([line_df, new_row], ignore_index=True)

#     point_df = pd.DataFrame(columns=['epsilon', 'error', 'mechanism'])

# # Calculate Laplace error for the specified epsilon value and append to the DataFrame
#     if 'laplace' in mechanisms:
#         laplace_error = dp.laplacian_scale_to_accuracy(sensitivity / epsilon_input, alpha)  # Ensure you use 'eps' here, not 'epsilon'
#         new_row = pd.DataFrame({'epsilon': [epsilon_input], 'error': [laplace_error], 'mechanism': 'laplace'})
#         point_df = pd.concat([point_df, new_row], ignore_index=True)

#     # Calculate Gaussian error for the specified epsilon value and append to the DataFrame    
#     if 'gaussian' in mechanisms:
#         gaussian_error = dp.gaussian_scale_to_accuracy(calculate_gaussian_scale(sensitivity, epsilon_input, delta), alpha)  # Again, ensure 'eps' is used
#         new_row = pd.DataFrame({'epsilon': [epsilon_input], 'error': [gaussian_error], 'mechanism': 'gaussian'})
#         point_df = pd.concat([point_df, new_row], ignore_index=True)
    
#     # Create subplots: 1 row, 2 cols
#     point_colors = ['#0072B2', '#80b1d3'] 

#     error_bar_color = 'red'  # Red for error bars

#     fig = make_subplots(rows=1, cols=2, subplot_titles=("Hypothetical outputs", "Accuracy vs. Privacy Parameter (\u03B5)"), horizontal_spacing=0.15)
#     error_margin = []
#     summary_df['error_margin'] = point_df['error'].tolist()

#     fig.update_yaxes(title_standoff=10, row=1, col=2) 
#     if error_type == 'Absolute Additive Error':
#         fig.update_yaxes(title_text="Hypothetical Outputs", row=1, col=1)
#         fig.update_yaxes(title_text="Absolute Error Upper Bound",  row=1, col=2)
#         pass
#     elif error_type == 'Relative Additive Error':
#         line_df['error'] = line_df['error'] / summary_df['true_value'][0]
        
#         # summary_df['error_margin'] = summary_df['error_margin'] / summary_df['true_value'][0]
#         fig.update_yaxes(title_text="Hypothetical Outputs", row=1, col=1)
#         fig.update_yaxes(title_text="Relative Error Upper Bound", tickformat=',.0%', row=1, col=2)
#     else:
#         raise ValueError("Invalid error_type. Please choose 'Absolute Error' or 'Relative Error'.")
#     # Plot 1: Accuracy vs. Privacy for Selected Mechanisms
#     for index, mechanism in enumerate(line_df['mechanism'].unique()):
#         mechanism_df = line_df[line_df['mechanism'] == mechanism]
#         fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines', 
#                                 name=f"Additive Error Upper Bound for {mechanism}", 
#                                 line=dict(color=point_colors[index])), row=1, col=2)
        
    
#     for index, row in point_df.iterrows():
#         if index == 0:  # Only add one dot to the legend
#             showlegend = True
#         else:
#             showlegend = False
#         if error_type == 'Relative Additive Error':
#             row['error'] = row['error'] / summary_df['true_value'][0]
#         fig.add_trace(
#             go.Scatter(x=[row['epsilon']], y=[row['error']], mode='markers', 
#                     name="Selected Epsilon",
#                     # marker=dict(color=point_colors[point_df['mechanism'].unique().tolist().index(row['mechanism'])], size=10), 
#                     marker=dict(color='red', size=6),
#                     showlegend=showlegend),  
#             row=1, col=2  
#         )

#     fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
    
#     if queryType == 'histogram':
#         if data_type == 'continuous':
#             print(selected_query, 'Mechanism', mechanisms, hide_non_feasible_values=True)
#             left_fig = noisy_histogram_creation_continuous(df, selected_query, 'Mechanism', mechanisms, epsilon_input, hide_non_feasible_values=True)
            
#         elif data_type == 'categorical':
#             print(selected_query, 'Categorical', 'Mechanism', mechanisms,)
#             left_fig =  left_fig = noisy_histogram_creation(df, selected_query, 'Categorical', 'Mechanism', mechanisms, epsilon_input, hide_non_feasible_values=True)

#         for trace in left_fig['data']:
#             fig.add_trace(trace, row=1, col=1)

#         for index, mechanism in enumerate(line_df['mechanism'].unique()):
#             mechanism_df = line_df[line_df['mechanism'] == mechanism]
#             fig.add_trace(go.Scatter(x=mechanism_df['epsilon'], y=mechanism_df['error'], mode='lines', 
#                                     name=f"Additive Error Upper Bound for {mechanism}", 
#                                     line=dict(color=point_colors[index])), row=1, col=2)

    
#     else:

#         max_value =point_df['error'].max()

#         laplace_scale =  sensitivity/ epsilon_input
#         vals_laplace = []
#         vals_gauss =[]
        
#         if 'laplace' in mechanisms: 
#             laplace_row = point_df.loc[point_df['mechanism'] == 'laplace']
#             for i in range(20):
#                 vals_laplace.append(get_query_private_outputs(df, queryType, 'laplace', columnName, epsilon,(data_min, data_max),  delta))
#             fig.add_trace(
#                 go.Scatter(
#                     x=['Laplace'],  # Use 0 for Laplace
#                     y=[summary_df.iloc[0]['true_value']],
#                     error_y=dict(type='data', array=[laplace_row['error'].values[0]], visible=True, color=error_bar_color),

#                     mode='markers',
#                     name='laplace',
#                     marker=dict(size=10, opacity=00),  # Adjust opacity for visual emphasis
#                     showlegend=False,
#                 ),
                

#                 row=1, col=1
#             )    
#             fig.add_trace(
#                 go.Scatter(
#                     x=['laplace'] * 20,  # X-values are categorical labels
#                     y=vals_laplace,
#                     mode='markers',
#                     name='Laplace',
#                     marker=dict(color=point_colors[0], opacity=.3),
#                     showlegend=False,
#                 ),
                

#                 row=1, col=1
#             )
            
        

#         if 'gaussian' in mechanisms:
#             gaussian_row = point_df.loc[point_df['mechanism'] == 'gaussian']
#             for i in range(20):
                
#                 vals_gauss.append(get_query_private_outputs(df, queryType, 'gaussian', columnName, epsilon, (data_min, data_max),  delta))       
#             fig.add_trace(
#                 go.Scatter(
#                     x=['Gaussian'],  # Use 1 for Gaussian
#                     y=[summary_df.iloc[0]['true_value']],
#                     error_y=dict(type='data', array=[gaussian_row['error'].values[0]], visible=True, color=error_bar_color),

#                     mode='markers',
#                     name='gaussian',
#                     marker=dict(size=10, opacity=00),  # Adjust opacity for visual emphasis
#                     showlegend=False,
#                 ),
                

#                 row=1, col=1
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=['gaussian'] * 20,  # X-values are categorical labels
#                     y=vals_gauss,
#                     mode='markers',
#                     name='gaussian',
#                     marker=dict(color=point_colors[1], opacity=.3),
#                     showlegend=False,
#                 ),
                

#                 row=1, col=1
#             )
            
            

#         fig.add_shape(type="line",
#                     x0=-.1,  # Starting from the first x-axis item
#                     y0=summary_df.iloc[0]['true_value'],  # True mean value for y
#                     x1=1.1,  # Ending at the last x-axis item
#                     y1=summary_df.iloc[0]['true_value'],  # Same true mean value for y to keep it horizontal
#                     line=dict(
#                         color="black",
#                         width=1,
#                         dash="dash",
#                     ),
#                     xref="paper",  # Reference the whole x-axis range
#                     yref="y",
#                     row=1, col=1, # Reference the y-axis values
#                     showlegend=True,
#                     name='True Mean'
                    
#                     )
                

#         fig.update_yaxes(title_text="Hypothetical outputs", range=[0, (summary_df['true_value'][0]+ max_value)*1.25], row=1, col=1)
#         # Update xaxis properties
#         fig.update_xaxes(title_text="Mechanism", row=1, col=1)
#     fig.update_xaxes(title_text="Privacy Parameter (\u03B5)", row=1, col=2)

#     # Update yaxis properties
    
#     return fig

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

