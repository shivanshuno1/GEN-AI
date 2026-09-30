from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import streamlit as st
import os

load_dotenv()

st.header("Chat with Google Gemini 2.5")

subject = st.selectbox(
    "Choose Subject",
    ["Physics", "Chemistry", "Mathematics"]
)

user_input = st.text_input("You:")

if user_input:
    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash",
        temperature=0.7
    )

    prompt = PromptTemplate.from_template("""
You are an expert {subject} teacher.

Answer the following question in a simple and beginner-friendly way.

Question:
{question}
""")

    formatted_prompt = prompt.format(
        subject=subject,
        question=user_input
    )

    response = llm.invoke(formatted_prompt)

    st.write(response.content)