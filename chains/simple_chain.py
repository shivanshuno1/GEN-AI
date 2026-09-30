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


parser = StrOutputParser()

prompt = PromptTemplate(
    template="Explain the following city in simple terms:\n{city}",
    input_variables=["city"]
)




chain = prompt | model | parser

result = chain.invoke({
    "city": "Dehradun"})


print(result)