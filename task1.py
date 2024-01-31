import streamlit as st
import pandas as pd
from transformers import pipeline

st.title('PandasAI using OpenAI Key')
# Load the pre-trained model and tokenizer
nlp = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased")

# Step 3: Upload CSV File
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Step 4: Load CSV Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,encoding='unicode_escape')
    print(df)

# Step 5: Ask User Query
user_query = st.text_input("Enter your query:")

# Step 6: Convert User Query to Pandas Query
# Using Hugging Face Transformers for Question Answering
def convert_to_pandas_query(user_query, context):
    result = nlp(question=user_query, context=context)
    return result['answer']

# Step 7: Query DataFrame
if user_query and uploaded_file:
    context = " ".join(df.astype(str).values.flatten())
    pandas_query = convert_to_pandas_query(user_query, context)

    # Step 8: Display Result in UI
    if pandas_query is not None:
        result_df = df.query(pandas_query)
        st.dataframe(result_df)
