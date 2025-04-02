
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_vecstore_from_url(url: str):
    """Scrapes a website and returns a vector store."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter()
    docs_chunks = splitter.split_documents(docs)
    chroma = Chroma.from_documents(documents=docs_chunks,
                                   embedding=OllamaEmbeddings(model="nomic-embed-text"),
                                   persist_directory="chroma_db")
    return chroma


def get_context_retriever_chain(vecstore):
    """Creates a context retriever chain."""
    llm = ChatOllama(model="mistral")
    retriever = vecstore.as_retriever()



    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to retrieve relevant information.")
    ])

    retrieve_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)
    return retrieve_chain


def get_conversational_rag_chain(retrieve_chain):
    """Creates a conversational RAG chain."""
    llm = ChatOllama(model="mistral")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retrieve_chain, stuff_doc_chain)


def get_response(user_input):
    chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(chain)



    response_text = ""  # Store the full response text

    def stream_answer():
        nonlocal response_text  # Allow modifying this variable inside generator
        for chunk in conversational_rag_chain.stream({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    }):
            if isinstance(chunk, dict) and "answer" in chunk:
                answer_part = chunk["answer"]
                response_text += answer_part  # Append to full response
                yield answer_part  # Stream answer part by part

    # Stream response in UI
    st.write_stream(stream_answer())

    return response_text