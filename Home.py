import streamlit as st
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(
    page_title="Home",
)



st.markdown("""
# Visualizing Privacy-Accuracy Trade-offs in Differential Privacy

Welcome to our interactive tool designed to help you understand the privacy-accuracy tradeoffs inherent when deploying differential privacy (DP). 
We want this interface to be a playground that allows you try out different implementation strategies and parameter settings to get an intuitive understanding of how DP works.

### Background
Differential privacy is the gold standard for privacy-preserving data analysis, ensuring that individual contributions to datasets remain confidential. 
We find that though this is the case, it can be really difficult to understand how to impelment DP correctly. 
Explanations can often be hidden in complex mathematical notation and it can be difficult to understand the effects of different choices made. 
Our hope is that through visually seeing and interacting with a hyptothetical data release, you can better understand the trade-offs and make more informed decisions.

### Research
This interface is part of a broader effort to lower barriers to using DP correctly, promoting a wider and wiser application of this crucial privacy-preserving technology. 
To get started with our tool, please click the button below.

"""
)

getting_started = st.button("Let's get started!")
if getting_started:
    switch_page("Getting Started")

# Initialize 'pages' in session_state if it doesn't exist
if 'pages' not in st.session_state:
    st.session_state['pages'] = {'dataset': False, 'queries': False}


# Initialize 'queries' in session_state if it doesn't exist
if 'queries' not in st.session_state:
    st.session_state['queries'] = {}

