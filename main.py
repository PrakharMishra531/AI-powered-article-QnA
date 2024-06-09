import streamlit as st
import unstructured
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import time
from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
from langchain_community.llms import Ollama

load_dotenv() 
access_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
repo_id="Intel/dynamic_tinybert"
# llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=500,temperature=0.7,token=access_token)

llm=Ollama(model="mistral")


st.title("Ask AI About Articles")
st.sidebar.title("Article URLs")
file_path = 'faiss_vector.pkl   '
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_click = st.sidebar.button("Process URL")

main_placeholder = st.empty()
if process_url_click:
    #load data

    loader = UnstructuredURLLoader(urls= urls)
    main_placeholder.text("Loading...")
    data = loader.load()
    main_placeholder.text(data)

    #split data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','],
            chunk_size= 1000,
    )
    docs = text_splitter.split_documents(data)
    main_placeholder.text("Text-Splitter Started")

    #create embeddings
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Vector Embedding Started")
    time.sleep(2)
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    # if vectorstore:
    #     query = main_placeholder.text_input("Question:")
    #     chain = RetrievalQAWithSourcesChain.from_llm(llm = llm,retriever = vectorstore.as_retriever())
    #     result = chain({"question":query},return_only_outputs = True)
    #     st.header("Answer")
    #     st.subheader(result["answer"])

    #     #sources
    #     sources = result.get("sources","")
    #     if sources:
    #         st.subheader("Sources:")
    #         sources_list = sources.split("\n")
    #         for source in sources_list:
    #             st.write(source)
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)






