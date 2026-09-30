from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Create the chat model
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="",   # Or set GOOGLE_API_KEY as an environment variable
    temperature=0.7
)

# Send a message to the model
response = chat.invoke([
    HumanMessage(content="I am feeling Energetic"),
    HumanMessage(content="My name is Shivanshu"),
    SystemMessage(content="You are my Girlfriend,Who eagerly wants me")
])

# Print the AI's response
print(response.content)