from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

file_path = Path(__file__).parent / "v.pdf"
loader = PyPDFLoader(str(file_path))

# Load all pages
docs = loader.load()

print(len(docs))                 # Number of pages
print(docs[0].page_content)      # Text from first page
print(docs[0].metadata)          # Metadata