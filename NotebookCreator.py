# import nbformat as nbf
# from nbconvert import PythonExporter
# import json
# from typing import List
# import pandas as pd
# import numpy as np
# import opendp.prelude as dp 
# dp.enable_features("contrib")

# def create_notebook(query_info):
#     print(query_info)
#     nb = nbf.v4.new_notebook()

#     nb['cells'] = [nbf.v4.new_markdown_cell(source='Here is the code to release the queries with the parameters you have selected using the OpenDP library')]
#     nb['cells'].append(nbf.v4.new_code_cell(source='# notebook imports\nimport opendp.prelude as dp\nimport math\nimport pandas as pd\nimport numpy as np\ndp.enable_features("contrib")\ndp.enable_features("honest-but-curious")\nfrom typing import List'))


#         # Heading for the query
#     nb['cells'].append(nbf.v4.new_markdown_cell(source=f"## Query: {query_info['export_selected_query']}"))
#     nb['cells'].append(nbf.v4.new_code_cell(source=f"df = pd.read_csv('{query_info['url']}')"))

#     # Load Data if URL is provided


#     query_info_json = json.dumps(query_info)

#     # Append the JSON representation of query_info to the notebook
#     nb['cells'].append(nbf.v4.new_code_cell(source=f'query_info = {query_info_json}'))

#     # Check the type of query
# # Check the type of query
# # Check the type of query
#     if 'count' in query_info['export_selected_query']:
#         # Add code for count query
#         nb['cells'].append(nbf.v4.new_code_cell(source='''# define dataframe domain
# def dataframe_domain():
#     return dp.user_domain("DataFrameDomain", lambda x: isinstance(x, pd.DataFrame))'''))

#         nb['cells'].append(nbf.v4.new_code_cell(source='''# define select column transformation
# def make_select_column(col_name, T):
#     return dp.t.make_user_transformation(
#         input_domain=dataframe_domain(),
#         input_metric=dp.symmetric_distance(),
#         output_domain=dp.vector_domain(dp.atom_domain(T=T)),
#         output_metric=dp.symmetric_distance(),
#         function=lambda data: data[col_name].to_numpy(),
#         stability_map=lambda d_in: d_in)'''))


#         nb['cells'].append(nbf.v4.new_code_cell(source='''# load data
# df = pd.read_csv('{url}')'''.format(**query_info)))

#     if query_info['export_mechanism'] == 'laplace':
#         nb['cells'].append(nbf.v4.new_code_cell(source='''# get query parameters
# query_type = '{export_selected_query}'
# mechanism = '{export_mechanism}'
# epsilon = {export_epsilon}
# column_name = query_type.split('_')[0]
# column_type = df[column_name].dtype
# if np.issubdtype(column_type, np.integer):
#     T = int
# else:
#     T = float'''.format(**query_info)))
#     elif query_info['export_mechanism'] == 'gaussian':
#         nb['cells'].append(nbf.v4.new_code_cell(source='''# get query parameters
# query_type = '{export_selected_query}'
# mechanism = '{export_mechanism}'
# epsilon = {export_epsilon}
# delta = {export_delta}
# column_name = query_type.split('_')[0]
# column_type = df[column_name].dtype
# if np.issubdtype(column_type, np.integer):
#     T = int
# else:
#     T = float'''.format(**query_info)))

#     # calculate sensitivity and scale
#     if query_info['export_mechanism'] == 'laplace':
#         nb['cells'].append(nbf.v4.new_code_cell(source='''# define scale calculation
# def calculate_scale(mechanism, sensitivity, epsilon, delta):
#     if mechanism == 'laplace':
#         return sensitivity / epsilon'''))

#         nb['cells'].append(nbf.v4.new_code_cell(source='''# calculate sensitivity and scale
# sensitivity = 1
# scale = calculate_scale(mechanism, sensitivity, epsilon, None)'''))
#     elif query_info['export_mechanism'] == 'gaussian':
#         nb['cells'].append(nbf.v4.new_code_cell(source='''# define scale calculation
# def calculate_scale(mechanism, sensitivity, epsilon, delta):
#     if mechanism == 'laplace':
#         return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon'''))
#         nb['cells'].append(nbf.v4.new_code_cell(source='''# calculate sensitivity and scale
# sensitivity = 1
# scale = calculate_scale(mechanism, sensitivity, epsilon, delta)'''))

#         nb['cells'].append(nbf.v4.new_code_cell(source='''# create measurement
# df_meas = (
#     make_select_column(column_name, T) >>
#     dp.t.then_count() >>
#     getattr(dp.m, f'then_{mechanism}')(scale)
# )

# # get private result
# result = df_meas(df)

# print(result)'''))
#         # Add code for other types of queries
#         pass


#     return nb

import nbformat as nbf
from nbconvert import PythonExporter
import json
from typing import List
import pandas as pd
import numpy as np
import opendp.prelude as dp 
dp.enable_features("contrib")

def create_notebook(query_info):
    print(query_info)
    nb = nbf.v4.new_notebook()

    nb['cells'] = [nbf.v4.new_markdown_cell(source='Here is the code to release the queries with the parameters you have selected using the OpenDP library')]
    nb['cells'].append(nbf.v4.new_code_cell(source='# notebook imports\nimport opendp.prelude as dp\nimport math\nimport pandas as pd\nimport numpy as np\ndp.enable_features("contrib")\ndp.enable_features("honest-but-curious")\nfrom typing import List'))


        # Heading for the query
    nb['cells'].append(nbf.v4.new_markdown_cell(source=f"## Query: {query_info['export_selected_query']}"))
    nb['cells'].append(nbf.v4.new_code_cell(source=f"df = pd.read_csv('{query_info['url']}')"))

    # Load Data if URL is provided


    query_info_json = json.dumps(query_info)

    # Append the JSON representation of query_info to the notebook
    nb['cells'].append(nbf.v4.new_code_cell(source=f'query_info = {query_info_json}'))

    if 'count' in query_info['export_selected_query']:
    # Add code for count query
        nb['cells'].append(nbf.v4.new_code_cell(source='''# define dataframe domain
def dataframe_domain():
    return dp.user_domain("DataFrameDomain", lambda x: isinstance(x, pd.DataFrame))'''))

        nb['cells'].append(nbf.v4.new_code_cell(source='''# define select column transformation
def make_select_column(col_name, T):
    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=T)),
        output_metric=dp.symmetric_distance(),
        function=lambda data: data[col_name].to_numpy(),
        stability_map=lambda d_in: d_in)'''))


        nb['cells'].append(nbf.v4.new_code_cell(source='''# load data
df = pd.read_csv('{url}')'''.format(**query_info)))

        if query_info['export_mechanism'] == 'laplace':
            nb['cells'].append(nbf.v4.new_code_cell(source='''# get query parameters
query_type = '{export_selected_query}'
mechanism = '{export_mechanism}'
epsilon = {export_epsilon}
column_name = query_type.split('_')[0]
column_type = df[column_name].dtype
if np.issubdtype(column_type, np.integer):
T = int
else:
T = float'''.format(**query_info)))
        elif query_info['export_mechanism'] == 'gaussian':
            nb['cells'].append(nbf.v4.new_code_cell(source='''# get query parameters
query_type = '{export_selected_query}'
mechanism = '{export_mechanism}'
epsilon = {export_epsilon}
delta = {export_delta}
column_name = query_type.split('_')[0]
column_type = df[column_name].dtype
if np.issubdtype(column_type, np.integer):
T = int
else:
T = float'''.format(**query_info)))

        # calculate sensitivity and scale
        if query_info['export_mechanism'] == 'laplace':
            nb['cells'].append(nbf.v4.new_code_cell(source='''# define scale calculation
def calculate_scale(mechanism, sensitivity, epsilon, delta):
    if mechanism == 'laplace':
        return sensitivity / epsilon'''))

            nb['cells'].append(nbf.v4.new_code_cell(source='''# calculate sensitivity and scale
sensitivity = 1
scale = calculate_scale(mechanism, sensitivity, epsilon, None)'''))
        elif query_info['export_mechanism'] == 'gaussian':
            nb['cells'].append(nbf.v4.new_code_cell(source='''# define scale calculation
def calculate_scale(mechanism, sensitivity, epsilon, delta):
    if mechanism == 'laplace':
        return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon'''))
            nb['cells'].append(nbf.v4.new_code_cell(source='''# calculate sensitivity and scale
sensitivity = 1
scale = calculate_scale(mechanism, sensitivity, epsilon, delta)'''))

            nb['cells'].append(nbf.v4.new_code_cell(source='''# create measurement
df_meas = (
make_select_column(column_name, T) >>
dp.t.then_count() >>
getattr(dp.m, f'then_{mechanism}')(scale)
)

# get private result
result = df_meas(df)

print(result)'''))
        # Add code for other types of queries




    if 'average' in query_info['export_selected_query']:
    # define dataframe domain
        nb['cells'].append(nbf.v4.new_code_cell(source='''# define dataframe domain
def dataframe_domain():
    return dp.user_domain("DataFrameDomain", lambda x: isinstance(x, pd.DataFrame))'''))

        # define select column transformation
        nb['cells'].append(nbf.v4.new_code_cell(source='''# define select column transformation
def make_select_column(col_name, T):
    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=T)),
        output_metric=dp.symmetric_distance(),
        function=lambda data: data[col_name].to_numpy(),
        stability_map=lambda d_in: d_in)'''))

        # load data
        nb['cells'].append(nbf.v4.new_code_cell(source='''# load data
df = pd.read_csv('{url}')'''.format(**query_info)))

        # get query parameters
        if query_info['export_mechanism'] == 'gaussian':
            nb['cells'].append(nbf.v4.new_code_cell(source='''# get query parameters
query_type = '{export_selected_query}'
mechanism = '{export_mechanism}'
epsilon = {export_epsilon}
delta = {export_delta}
column_name = query_type.split('_')[0]
bounds = (query_info['export_min_value'], query_info['export_max_value'])'''.format(**query_info)))
        else:
            nb['cells'].append(nbf.v4.new_code_cell(source='''# get query parameters
query_type = '{export_selected_query}'
mechanism = '{export_mechanism}'
epsilon = {export_epsilon}
column_name = query_type.split('_')[0]
bounds = (query_info['export_min_value'], query_info['export_max_value'])'''.format(**query_info)))

        # calculate sensitivity and scale
        if query_info['export_mechanism'] == 'laplace':
            nb['cells'].append(nbf.v4.new_code_cell(source='''# calculate sensitivity and scale
sensitivity = (bounds[1] - bounds[0])/df.shape[0]
scale = sensitivity / epsilon'''))
        elif query_info['export_mechanism'] == 'gaussian':
            nb['cells'].append(nbf.v4.new_code_cell(source='''# calculate sensitivity and scale
sensitivity = (bounds[1] - bounds[0])/df.shape[0]
scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon'''))

                    # create measurement
        nb['cells'].append(nbf.v4.new_code_cell(source='''# create measurement
bounds = ((float(bounds[0])), (float(bounds[1])))
df[column_name] = df[column_name].astype(float)

df_meas = (
    make_select_column(column_name, float) >>
    dp.t.then_resize(size=len(df), constant=float(bounds[1])) >>
    dp.t.then_clamp(bounds) >>
    dp.t.then_mean() >>
    getattr(dp.m, f'then_{mechanism}')(scale)
)

# get private result    
result = df_meas(df)
print(result)'''))
    if 'histogram' in query_info['export_selected_query']:
        df = pd.read_csv(query_info['url'])
        column_name, _ = query_info['export_selected_query'].split('_')
        if query_info['export_data_type'] == 'categorical':
            categories = sorted(df[column_name].unique())
            column_type = df[column_name].dtype

            # Cell 3: Define at_delta function
            nb['cells'].append(nbf.v4.new_code_cell(source='''def at_delta(meas, delta):
    meas = dp.c.make_zCDP_to_approxDP(meas)
    return dp.c.make_fix_delta(meas, delta)'''))

            # Cell 4: Load data and get parameters
            column_name, _ = query_info['export_selected_query'].split('_')
            categories = sorted(pd.read_csv(query_info['url'])[column_name].unique())
            nb['cells'].append(nbf.v4.new_code_cell(source=f'''# Load data and get parameters
df = pd.read_csv('{query_info['url']}')
query_type = '{query_info['export_selected_query']}'
mechanism = '{query_info['export_mechanism']}'
epsilon = {query_info['export_epsilon']}
column_name = '{column_name}'
categories = {categories}
column_type = df[column_name].dtype'''))

            if query_info['export_mechanism'] == 'laplace':
                # Cell 5: Define and apply Laplace mechanism
                nb['cells'].append(nbf.v4.new_code_cell(source=f'''histogram = (
    dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
    dp.t.make_select_column(column_name, str) >>
    dp.t.then_cast_default(int) >>
    dp.t.then_count_by_categories(categories=categories, MO=dp.L1Distance[int] )
    )
noisy_laplace_histogram = dp.binary_search_chain(
    lambda s: histogram >> dp.m.then_laplace(scale=s),
    d_in=1, d_out=epsilon)
sensitive_counts = noisy_laplace_histogram(df.to_csv(index=False, header=False))'''))

                # Cell 6: Print results
                nb['cells'].append(nbf.v4.new_code_cell(source='''print("Noisy Laplace Counts:", sensitive_counts[:-1])'''))
            elif query_info['export_mechanism'] == 'gaussian':
                # Cell 5: Define and apply Gaussian mechanism
                nb['cells'].append(nbf.v4.new_code_cell(source=f'''delta = {query_info['export_delta']}
t_hist = (
    dp.t.make_split_dataframe(separator=",", col_names=list(df.columns)) >>
    dp.t.make_select_column(column_name, str) >>
    dp.t.then_cast_default(int) >>
    dp.t.then_count_by_categories(categories=categories, MO=dp.L2Distance[float])
)
m_hist = dp.binary_search_chain(
    lambda s: at_delta(t_hist >> dp.m.then_gaussian(scale=s), delta), 
    d_in=1, 
    d_out=(epsilon, delta))
sensitive_gaussian_counts = m_hist(df.to_csv(index=False, header=False))'''))

                # Cell 6: Print results
                nb['cells'].append(nbf.v4.new_code_cell(source='''print("Noisy Gaussian Counts:", sensitive_gaussian_counts[:-1])'''))

    return nb
