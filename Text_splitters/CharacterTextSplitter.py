from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

documents = TextLoader(
    "Text_splitters/suka_file.txt"
).load()

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=4,
    chunk_overlap=2
)

chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    print("-----")
    print(chunk.page_content)