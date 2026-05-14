from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from typing import Any,Dict,List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyCrawl,TavilySearch,TavilyExtract
from langchain_community.document_loaders import PyPDFLoader ,TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import asyncio
import ssl
import certifi
load_dotenv()
google_api_key=os.getenv("GOOGLE_API_KEY")
pinecone_api_key=os.getenv("PINECONE_API_KEY")
tavily_api_key=os.getenv("TAVILY_API_KEY")
agent=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=google_api_key)
# tavily_agent=tavily_crawl(api_key=tavily_api_key)
llm=ChatOllama(model="gemma3:270m", temperature=0.9)

# ssl to avoid api issues
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# source
tavily=TavilyCrawl(api_key=tavily_api_key)
source=tavily.invoke({
    "url":"https://python.langchain.com",
    "max_depth":2,
    "extract_depth":"advanced"
})

docs=[Document(page_content=s["raw_content"],metadata={"source":s["url"]}) for s in source["results"]]
# text splitting
splitter=RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)
# splitting the Documents into chunks
splitted_docs=splitter.split_documents(docs)
# embedding 1024 dimensions
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

#vectore store
print("Connecting to Pinecone...")
vectorstore=PineconeVectorStore(pinecone_api_key=pinecone_api_key,index_name="travily01",embedding=embeddings)
print("✅ Connected to Pinecone")
vectorstore.aadd_documents
# ingestion to the model
async def ingestion(splitted_docs:List[Document],batch_size:int=250):
    # batching because 
    batches=[splitted_docs[i:i+batch_size] for i in range(0, len(splitted_docs), batch_size)]
    # adding batch wise to vector store [[batch 1,batch 2 ,batch 3],....] each batch at asynly
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
        except Exception as e:
           
            return False
        return True
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)
    print(f"Successfully indexed {successful}/{len(batches)} batches")

    
print("Starting ingestion...") 
asyncio.run(ingestion(splitted_docs,50))
