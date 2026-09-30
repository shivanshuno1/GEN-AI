from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
# Load environment variables
load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=112,
    temperature=0.7,
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(llm=llm)



# prompt1 -> prompt for writing the detailed summary about the topic
prompt1 = PromptTemplate(
    template="Write a detailed summary about the following topic:\n{topic}",
    input_variables=["topic"]
)



prompt2 = PromptTemplate(
    template="Extract the single most important point from the following text:\n {summary}",
    input_variables=["summary"]
)



parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({
    "topic": "New Delhi"})

print(result)