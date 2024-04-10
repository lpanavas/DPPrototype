import nbformat as nbf
from nbconvert import PythonExporter
import json
from typing import List
import pandas as pd
import numpy as np
import opendp.prelude as dp 
dp.enable_features("contrib")

def create_notebook(query_info):
    nb = nbf.v4.new_notebook()

    nb['cells'] = [nbf.v4.new_markdown_cell(source='Here is the code to release the queries with the parameters you have selected using the OpenDP library')]
    nb['cells'].append(nbf.v4.new_code_cell(source='# notebook imports\nimport opendp.prelude as dp\nimport pandas as pd\nimport numpy as np\ndp.enable_features("contrib")\nfrom typing import List'))

    for query, info in query_info.items():
        # Heading for the query
        nb['cells'].append(nbf.v4.new_markdown_cell(source=f'## Query: {query}'))

        # Load Data if URL is provided
        if 'url' in info:
            nb['cells'].append(nbf.v4.new_code_cell(source=f"df = pd.read_csv('{info['url']}')"))

        nb['cells'].append(nbf.v4.new_code_cell(source='query_info = ' + json.dumps({query: info})))  

       # Determine domain type  
        if isinstance(info['upper_bound'], float):
            domain_code = "domainType = dp.domain_of(List[float])"
        else:
            domain_code = "domainType = dp.domain_of(List[int])"
        nb['cells'].append(nbf.v4.new_code_cell(source=domain_code))  

        # Private release calculation (run once per query)
        nb['cells'].append(nbf.v4.new_code_cell(source=f'''context = dp.Context.compositor(
                                data=list(df["{info['column']}"]),  # Actual column name
                                privacy_unit=dp.unit_of(contributions=1),
                                privacy_loss=dp.loss_of(epsilon={info['epsilon']}),  # Actual epsilon
                                domain=domainType, 
                                split_evenly_over=1
                            )
dp_sum = context.query().clamp(({info['lower_bound']}, {info['upper_bound']})).sum().laplace()
private_release = dp_sum.release() / df.shape[0]  # Average calculation
print(f"Private release for query={query}: {{private_release}}")''')) 

    return nb