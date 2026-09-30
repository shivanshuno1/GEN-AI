from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


# Prompt for writing Summary about somewhere

prompt1 = PromptTemplate(
    template="Write a detailed summary about the following city:\n{city}",
    input_variables=["city"]
)

summary_chain = prompt1 | model | parser

#Prompt for writing some keypoints about somewhere

prompt2 = PromptTemplate(
    template="Write some key one word points about the following city:\n{city}",
    input_variables=["city"]
)

keyword_chain = prompt2 | model | parser

parallel_chain = RunnableParallel(
    {
        "summary": summary_chain,
        "keywords": keyword_chain,
    }
)


result = parallel_chain.invoke({
    "city": "Dehradun"})


print("The summary of the city is :")
print(result["summary"])

print("The keywords for the city are :")
print(result["keywords"])
