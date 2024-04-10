
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.title("Getting Started")
st.markdown("""
Welcome to the Differential Privacy Deployment Simulation Tool.
You will be in the shoes of a trusted data curator examining privacy choices on public data. 
Keep in mind, that in a real deployment, you would not be able to test out and use the kind of visualization you see here as it would violate differential privacy gaurantees. 
This tool is meant to be a learning tool to help you understand the trade-offs of different choices you make when deploying differential privacy.

### Choose Your Dataset
In the Dataset View section you will be able to choose the dataset you are working with. You can choose from the following options:
- **Real Datasets**: We have uploaded a set of datasets that are common examples of differential privacy. These datasets examplify some of the challenges that come with real data.
- **Dummy Datasets**: If you want data that more closely mirrors your own, we allow you to create a synthetic dataset.
- **Upload Your Own Dataset**: You can upload your own dataset to see how differential privacy would work with your data. Do not upload sensitivite data to this site.

### Metadata Settings
Once you've selected your dataset, we'll have you choose the columns you want to analyze and the types of queries you want to release.            

### Visualizing Trade-offs
Once you've selected a dataset and queries, you'll be able to visualize the privacy-accuracy trade-offs of your chosen queries. The visualizations will include
- **Simulations**: Set different implementation parameters and see the effects on hypothetical data releases
- **One Query**: Visualize the privacy accuracy tradeoff charts for different parameter settings and error metrics.
- **Multiple Queries**: Balance your privacy usage across multiple queries and try different composition methods.
- **Data Export**: Create notebooks that allow you to run the private data releases on your own machine.

Let's get started by selecting your dataset.
""")

dataset_view = st.button("Datasets")
if dataset_view:
    switch_page("Dataset_View")
