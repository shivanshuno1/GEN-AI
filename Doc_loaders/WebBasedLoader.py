

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Uttarakhand")

docs = loader.load()



print(docs[0].page_content)  # Text from the first page of the website