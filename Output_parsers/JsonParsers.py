from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
Extract the information below into JSON.

Text:
{text}

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | model | parser

result = chain.invoke({
    "text": "Rahul is 25 years old and works as a Data Scientist.His hometown is Dehradun, Uttarakhand. He enjoys hiking and playing chess in his free time."
})

print(result)
print(type(result))