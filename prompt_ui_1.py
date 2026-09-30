from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    api_key="",
    model="gemini-2.5-flash",
    temperature=0
)

st.title("AI Career Advisor")

name = st.text_input("Name")
skills = st.text_input("Skills")
goal = st.text_input("Career Goal")

if st.button("Get Advice"):

    prompt = PromptTemplate.from_template("""
    You are an experienced career mentor.

    Student Name: {name}

    Skills:
    {skills}

    Goal:
    {goal}

    Give detailed advice.
    """)

    chain = prompt | llm

    response = chain.invoke({
        "name": name,
        "skills": skills,
        "goal": goal
    })

    st.write(response.content)