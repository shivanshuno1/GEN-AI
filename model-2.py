import langchain
import langchain_openai


from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
import os

key = os.getenv("")
llm = langchain_openai.OpenAI(
    model="gpt-3.5-turbo-instruct",
    openai_api_key="",
    temperature=0
)

result = llm.invoke("What is the capital of India?")
print(result)