from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# -----------------------------
# HuggingFace Model
# -----------------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(llm=llm)


# -----------------------------
# Define Output Schema
# -----------------------------
class Facts(BaseModel):
    fact1: str = Field(description="First fact about the concept")
    fact2: str = Field(description="Second fact about the concept")
    fact3: str = Field(description="Third fact about the concept")


# -----------------------------
# Create Parser
# -----------------------------
parser = JsonOutputParser(pydantic_object=Facts)


# -----------------------------
# Prompt
# -----------------------------
prompt = PromptTemplate(
    template="""
Give exactly three facts about the following topic.

Topic:
{topic}

Return the answer in JSON format.

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# -----------------------------
# Create Chain
# -----------------------------
chain = prompt | model | parser


# -----------------------------
# Invoke
# -----------------------------
result = chain.invoke(
    {
        "topic": "Artificial Intelligence"
    }
)


# -----------------------------
# Output
# -----------------------------
print(result)

print("\nFact 1:", result["fact1"])
print("Fact 2:", result["fact2"])
print("Fact 3:", result["fact3"])