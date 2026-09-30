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
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Explain the following concept in simple terms:\n{concept}",
    input_variables=["concept"]
)

template2 = PromptTemplate(
    template="Provide a 5-line explanation of the following text in Garhwali:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser


result = chain.invoke({
    "concept": "Black Hole"})

print(result)