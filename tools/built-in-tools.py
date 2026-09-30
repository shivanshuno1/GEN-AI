from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonREPLTool


chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="",   # Or set GOOGLE_API_KEY as an environment variable
    temperature=0.7
)


cal = PythonREPLTool()

agents = create_agent(
    model = chat,
    tools = [cal]
)

results = agents.invoke({
    "messages":[
        {
            "role":"user",
            "content":"what is 20 * 20 + 30"
        }
    ]
})

print(results["messages"][-1].content)