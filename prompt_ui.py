from dotenv import load_dotenv

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from langchain.core.prompts import PromptTemplate


load_dotenv()

st.title("Shivanshu-GPT")

st.header("Shailendra-chutiyapa :)")

llm = ChatGoogleGenerativeAI(
   api_key="",
    model="gemini-2.5-flash",
    temperature=0.7
)

user_input = st.text_input("Enter your message:")

if user_input and st.button("Send"):
    response = llm.invoke(user_input)
    st.write("Response:", response.content)

