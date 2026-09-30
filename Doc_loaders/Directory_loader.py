from langchain.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    "Doc_loaders/docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)

documents = loader.lazy_load()



for doc in documents:
    print(doc.metadata)