import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
# st.set_page_config(layout="wide")

#get data 
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df.columns = ['ASIN','title','date','volume', 'revenue', 'type','price']
    return df


st.header(":rainbow[Sales Analytics] ", divider='rainbow')
def byrevenue(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    bytype = df.groupby(['month', 'type'], as_index=False)['revenue'].sum()
    byproduct = df.groupby(['title'], as_index=False)['revenue'].sum()
    # top_12_products = byproduct.nlargest(12, 'revenue')['title']
    # st.write("top 12 products")
    # for title in top_12_products:
    #     st.write(title)
    # st.write(df.head())
    fig1 = px.line(bytype, x='month', y='revenue',color='type', hover_name='type')
    fig2 = px.bar(bytype, x='type', y='revenue', color='type')
    
    #layout
    row1 = st.container()
    row2 = st.container()
    # Display the first figure in the first column
    with row1:
        st.write(fig1)
    # Display the second figure in the second column
    with row2:
        st.write(fig2)
        

file = st.sidebar.file_uploader("Import File")
if file:
    df = load_data(file)  
    byrevenue(df)
else:
    st.write("Please upload your file") 

# # Create a list of page names
# pages = ["By Topic", "By Brand", "By Category"]

# # Render the sidebar navigation
# st.sidebar.title("Navigation")
# selected_page = st.sidebar.radio("Go to", pages)

# # Render the selected page
# if selected_page == "By Topic":
#     # st.title("Sales Analytics by Topic")
#     file = st.file_uploader("Import File")
#     if file:
#         df = load_data(file)  
#         byrevenue(df)
#     else:
#         st.write("Please upload your file")    

#     # Add content for the home page
# if selected_page == "By Brand":
#     file = st.file_uploader("Import File")
#     if file:
#         df = load_data(file)  
#         byrevenue(df)
#     else:
#         st.write("Please upload your file")    


# if selected_page == "By Category":
#     file = st.file_uploader("Import File")
#     if file:
#         df = load_data(file)  
#         byrevenue(df)
#     else:
#         st.write("Please upload your file")    
