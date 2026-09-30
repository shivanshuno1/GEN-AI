from pathlib import Path
from langchain_community.document_loaders import TextLoader
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
    model="gemini-2.5-flash",
    temperature=0.7
)



file_path = Path(__file__).parent / "ai_notes.txt"


loader = TextLoader(str(file_path))


documents = loader.load()


Prompt = PromptTemplate(
    template="Make a beautiful poem based on the document:\n{documents}",
    input_variables=["documents"]
)


parser = StrOutputParser()

chain = Prompt | model | parser

result = chain.invoke({
    "documents": documents[0].page_content
})
print("Poem Generated from the document:\n")
print("-"*50)
print(result)

