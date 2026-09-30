'''from google import genai

client = genai.Client(api_key="")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is Uttarakhand known in India?"
)

print(response.text)
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# LLM

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key="",
    temperature = 0
)


#PROMPT

prompt = PromptTemplate.from_template(
    """
    Summarize the text in Garhwali language of Uttarkhand:
    {text}
    """
)

# output parser

parser = StrOutputParser()

# chain

chain = prompt | model | parser

result = chain.invoke({
    "text":"I like Kritika Panwar and she likes me"
})


print(result)






print(result.content)


