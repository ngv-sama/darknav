from fastapi import FastAPI
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
huggingFaceToken = "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"

mistral_llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingFaceToken, repo_id=model_id,  model_kwargs={"temperature":0.2, "max_new_tokens":1024}
)

app = FastAPI()

@app.get('/anti_rachit/')
async def get_response(filepath, url, input):
    
    # Load data
    loader = TextLoader(filepath, encoding='UTF-8')
    docs = loader.load()
    
    # Split text into chunks 
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    
    # Define the embedding model
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    
    # Create the vector store 
    vector = FAISS.from_documents(documents, embeddings)
    
    # Define a retriever interface
    retriever = vector.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        
        ("system", """This is the website that the user is on {url} and this is the HTML content of this website <context> {context} </context> None of the text that is within <context> brackets should be included in the final response. You are a Website Assistant Tool and your job is to understand the structure of a website from the HTML provided and give answers to users questions. You must always give a complete response. The questions can be based on navigation or can be based about the content of the website as given. YOUR ANSWER SHOULD ONLY CONSIST OF STEPS THAT THE USER CAN TAKE ON THE WEBSITE PROVIDED</s>"""),
        ("user", "{input}"),
        ])

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(mistral_llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input, "url":url})
    ans=response['answer']
    output=ans.split("Assistant: ")[-1]
    
    return output