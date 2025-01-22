import streamlit as st
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone #this below has been replaced by the below import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains.question_answering import load_qa_chain

#from langchain.llms import HuggingFaceHub
#The above have been updated recently, so going forward we have to use the below :)
from langchain_community.llms import HuggingFaceEndpoint
from langchain_groq import ChatGroq
import os

#PINECONE###########

# Due to recent changes from Pinecone team, there are some minor changes we have to implement, as a part of this we Initialize the Pinecone client

#Please update your pinecone-client package version >=3.0.1
from pinecone import Pinecone as PineconeClient, PodSpec  # Importing the Pinecone class from the pinecone package
#from langchain_community.vectorstores import Pinecone
from langchain_pinecone import Pinecone

# Set your Pinecone API key
# Recent changes by langchain team, expects ""PINECONE_API_KEY" environment variable for Pinecone usage! So we are creating it here
# we are setting the environment variable "PINECONE_API_KEY" to the value and in the next step retrieving it :)
os.environ["PINECONE_API_KEY"] = "35876c43-3ec3-411b-833c-4d5c326bf16f"
PINECONE_API_KEY=os.getenv("‘PINECONE_API_KEY’")

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
#HF_HOME=/datadrive/huggingface_cache/
#Function to read documents
def load_docs(directory):
  loader = PyPDFDirectoryLoader(directory)
  documents = loader.load()
  return documents

#This function will split the documents into chunks
def split_docs(documents, chunk_size=1500, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def latex_split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

#This function will help us in fetching the top relevent documents from our vector store - Pinecone
def get_similiar_docs(query, k=4):
    similar_docs = st.session_state.index.similarity_search(query, k=k)
    logging.info(similar_docs)
    return similar_docs

#This function will help us get the answer to the question that we raise
def get_answer(query):
    print('Query is !!!!  \n',query)
    relevant_docs = get_similiar_docs(query)
    response = chain.run(input_documents=relevant_docs, question=query)
    return response
    
def create_vectors():
    if "index" not in st.session_state:
        directory = 'Docs/juniper'
        documents = load_docs(directory)
        print("Loaded RAG docs")
        logging.info("Loaded RAG docs")

        docs = split_docs(documents)
        docs_loaded = True
    
        # Hugging Face Embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize the Pinecone client
        logging.info('Starting to connect to Pinecone...')

        pcClient = PineconeClient(api_key=PINECONE_API_KEY, environment="gcp-starter")

        index_name = "poc-index-juniper-1"

        st.session_state.index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        # index = Pinecone.from_existing_index(embeddings, index_name=index_name)

        logging.info('Created Pinecone Index...')


docs_loaded = False
index_created = False

st.header("CLI Command Helper")

create_vectors()

input = st.text_area("Enter your prompt to get the CLI command")

with st.form(key='my_form_to_submit'):
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # Passing the directory to the 'load_docs' function
        #llm = CTransformers(model='llama-2-7b-model/llama-2-7b-chat.ggmlv3.q6_K.bin',model_type='llama', config={'max_new_tokens': 1024, 'temperature': 0.8, 'context_length': 2048})
#    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q6_K.bin', model_type='llama', config={'max_new_tokens': 256, 'temperature': 0.8, 'context_length': 2048})
    llm = ChatGroq(model="llama-3.1-70b-versatile",api_key='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', max_tokens= 1280, temperature = 0.8)
    chain = load_qa_chain(llm, chain_type="stuff")

    our_query = input

    answer = get_answer(our_query)
    print('\n \n ')
    logging.info('Prompt: {}\n', our_query)
    print('\n') 
    logging.info("Got the answer")
    print(answer)
    st.write(answer)
    #logging.info('Answer is ............... \n',answer)
