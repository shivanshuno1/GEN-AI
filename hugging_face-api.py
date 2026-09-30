"""from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use a model that is supported
llm = HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",  # ✅ works
    task="conversational",
    huggingfacehub_api_token=token
)

chat = ChatHuggingFace(llm=llm)
response = chat.invoke("What is the capital of India?")
print(response.content)"""


from dotenv import load_dotenv
from huggingface_hub import InferenceClient, HfApi
import os

load_dotenv()

token = ""
if not token:
    raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN in environment.")

# Verify token
api = HfApi()
try:
    user = api.whoami(token=token)
    print(f"✅ Logged in as: {user['name']}")
except Exception as e:
    print(f"❌ Token validation failed: {e}")
    exit(1)

# Choose a working model
MODEL = "black-forest-labs/FLUX.1-schnell"  # or "runwayml/stable-diffusion-v1-5"

client = InferenceClient(api_key=token)

print(f"Generating image with {MODEL}...")
image = client.text_to_image(
    prompt="Goku in super saiyan blue form fighting jiren at base form in a rocky terrain with a sunset in the background, cinematic lighting, ultra-detailed, 8k, trending on artstation",
    model=MODEL
)

image.save("generated_image_.png")
print("✅ Image saved as generated_image_2.png")