from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (make sure GOOGLE_API_KEY is set in your .env file)
load_dotenv()

# Initialize LLM
model = ChatGoogleGenerativeAI(
    api_key="",  # or use os.getenv("GOOGLE_API_KEY")
    model="gemini-2.5-flash",
    temperature=0.7
)

prompt = PromptTemplate(
    template="Explain the following topic in simple terms:\n{topic}",
    input_variables=["topic"]

)

parser = StrOutputParser()

uppercase = RunnableLambda(
    lambda x: {"topic": x["topic"].upper()}
)

chain = uppercase | prompt | model | parser

result = chain.invoke(
    {
        "topic":"artificial intelligence"
    }
)


print(result)
