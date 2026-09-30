from dotenv import load_dotenv

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from langchain_core.prompts import PromptTemplate




load_dotenv()

st.title("RGB Color Generator")


# DEFINING LLM 
llm = ChatGoogleGenerativeAI(
    api_key="",
    model="gemini-2.5-flash",
    temperature=0.7
)


first_color = st.text_input("Enter the first color (e.g., red, #FF0000):")
second_color = st.text_input("Enter the second color (e.g., blue, #0000FF):")
third_color = st.text_input("Enter the third color (e.g., green, #00FF00):")


if(st.button("Generate RGB Color")):
    prompt = PromptTemplate.from_template("""
    You are an expert in color theory and design.

    Given the following three colors:
    1. {first_color}
    2. {second_color}
    3. {third_color}

    Generate a harmonious RGB color combination that complements these colors. 
    Provide the RGB values in the format (R, G, B) and explain why this combination works well with the given colors.
    """)

    chain = prompt | llm

    response = chain.invoke({
        "first_color": first_color,
        "second_color": second_color,
        "third_color": third_color
    })


    

    st.write(response.content)
