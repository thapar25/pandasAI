import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

llm=OpenAI(api_token="API_KEY")

st.title('PandasAI using OpenAI')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,encoding='unicode_escape')
    print(df.head())
    sdf=SmartDataframe(df,config={"llm":llm})

# Ask User Query
user_query = st.text_input("Enter your query:")

# Convert User Query to Pandas Query
def convert_to_pandas_query(user_query):
    result = sdf.chat(user_query)
    print(result)
    return result

# Query DataFrame
if user_query and uploaded_file:
    pandas_query = convert_to_pandas_query(user_query)

    # Display Result in UI
    if pandas_query is not None:
        st.write(pandas_query)
