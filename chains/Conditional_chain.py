from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
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


# Positive response chain
positive_prompt = PromptTemplate(
    template="Respond politely to this positive review:\n{review}",
    input_variables=["review"]
)

positive_chain = positive_prompt | model | parser


# Negative response chain
negative_prompt = PromptTemplate(
    template="Respond professionally and apologize for this negative review:\n{review}",
    input_variables=["review"]
)

negative_chain = negative_prompt | model | parser


# Condition
def is_negative(x):
    review = x["review"].lower()

    negative_words = [
        "bad",
        "poor",
        "terrible",
        "worst",
        "awful",
        "disappointed"
    ]

    return any(word in review for word in negative_words)


# Conditional chain
chain = RunnableBranch(
    (is_negative, negative_chain),
    positive_chain
)


result = chain.invoke(
    {
        "review": "The service was terrible and the food was cold."
    }
)

print(result)