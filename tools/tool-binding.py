from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool



@tool
def multiply(a:int,b:int)->int:
    "Returns the product of the numbers"
    return a *b


print(multiply.invoke({'a':12,'b':12}))

# tool description analytics

print(multiply.name)
print(multiply.description)
print(multiply.args_schema)


# normal llm 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

llm_with_bind = llm.bind_tools([multiply])

