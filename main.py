from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv  
load_dotenv()
import streamlit as st

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "VERITAS AI"

prompt= ChatPromptTemplate.from_messages([
    "system","you are a helpful assistant, respond to the user's query as accurately as possible","user","{question}"
])
def generate_response(question,max_tokens,temperature):
    chat = ChatGroq(model="gemma2-9b-it")
    chain=prompt|chat|StrOutputParser()
    response = chain.invoke({"question":question})
    return response
st.title("VERITAS AI")


## Select the OpenAI model


## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")



if user_input :
    response=generate_response(user_input,max_tokens,temperature)
    st.write(response)
else:
    st.write("Please provide the user input")
