from langchain_community.document_loaders import SitemapLoader

loader = SitemapLoader(
    web_path="https://example.com/sitemap.xml"
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)