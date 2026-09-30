
'''reasons why chat models better than llms'''

#role based assignment

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = GoogleGenerativeAI(
    api_key="",
    model="gemini-2.5-flash",
    temperature= 1.5
)

response =  llm.invoke("write a poem about uttrakhand in hindi")


print(response)


'''from langchain.tools import tool
import requests

@tool
def get_weather(city: str):
    """Get current weather for a city"""

    url = f"https://wttr.in/{city}?format=3"

    response = requests.get(url)

    return response.text



print(get_weather.invoke({"city":"Delhi"}))'''