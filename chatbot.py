from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables (make sure GOOGLE_API_KEY is set in your .env file)
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    api_key="",  # or use os.getenv("GOOGLE_API_KEY")
    model="gemini-2.5-flash",
    temperature=0.7
)
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))

    # ✅ Pass the entire history to the model
    response = llm.invoke(chat_history)

    # Add assistant response to history
    chat_history.append(AIMessage(content=response.content))

    print("Assistant:", response.content)

print("\nChat History:")
