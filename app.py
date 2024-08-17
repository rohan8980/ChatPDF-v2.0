from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os


@st.cache_resource(ttl="1h", show_spinner='Processing File(s).') 
def get_docs_from_files(files):
    documents=[]
    for file in files:
        filepath = file.name
        with open(filepath,"wb") as f:
            f.write(file.getvalue())

        docs = PyPDFLoader(filepath).load()
        documents.extend(docs)

        if os.path.exists(filepath):
            os.remove(filepath)

    return documents

@st.cache_resource(ttl="1h", show_spinner='Processing File(s)..') 
def get_vectorstore_from_files(files, HF_Embed_Model,):
    pdf_docs = get_docs_from_files(files)             
    split_docs=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=150).split_documents(pdf_docs)
    embeddings = HuggingFaceEmbeddings(model_name = HF_Embed_Model)
    vectorestore = FAISS.from_documents(split_docs, embeddings)

    return vectorestore

@st.cache_resource(show_spinner='Processing File(s)...')
def get_llm(groq_api_key, model_name):
    llm=ChatGroq(groq_api_key=groq_api_key,model_name=model_name)
    return llm
    

def get_rag_chain(vectorstore, llm):
    # Create standalone question from current question + chat history using create_history_aware_retriever
    retriever = vectorstore.as_retriever()    
    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    # Create rag_chain to answer standalone question obtained from history_aware_retriever using create_retrieval_chain
    system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. Do not use any outside knowledge."
            "\n\n"
            "{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )  
    
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    return rag_chain



def get_current_session():
    return st.session_id
def get_new_session(session_id='default_chat'):
    st.session_id = session_id
def get_session_history(session_id:str)->BaseChatMessageHistory:
    session_id = get_current_session() if not session_id else session_id
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]
    
def show_chat_history(session_id=None):
    st.markdown("""
        <style>
        .human-message {
            text-align: right;
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
        .ai-message {
            text-align: left;
            background-color: #F1F0F0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    session_id = get_current_session() if not session_id else session_id
    chat_history = get_session_history(session_id).messages
    for message in chat_history:
        if isinstance(message, AIMessage):
            st.markdown(f"<div class='ai-message'>{message.content}</div>", unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            st.markdown(f"<div class='human-message'>{message.content}</div>", unsafe_allow_html=True)


# Env variables and global variables
LLM_Model="Gemma2-9b-It"
HF_Embed_Model = "all-MiniLM-L6-v2"
groq_api_key = st.secrets["GROQ_API_KEY"]

if 'store' not in st.session_state:
    st.session_state.store={}
if 'session_id' not in st.session_state:
    get_new_session()
if 'query' not in st.session_state:
    st.session_state.query="" 

# Streamlit UI
st.title("Chat PDF with Chat History")
st.write(f'<p style="font-size: medium">Chat with PDF using Langchain HuggingFace Groq</p>', unsafe_allow_html=True)


files=st.file_uploader('Choose PDF file(s)', type=['pdf'], accept_multiple_files=True)
if files:
    # Prepare rag_chain from uploaded files
    vectorstore = get_vectorstore_from_files(files=files, HF_Embed_Model=HF_Embed_Model)
    llm_model = get_llm(groq_api_key=groq_api_key,model_name=LLM_Model)
    rag_chain = get_rag_chain(vectorstore, llm_model)

    # Chat history
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        show_chat_history()
    def clear_input():
        st.session_state.query=st.session_state.text_input
        st.session_state.text_input=""

    # Loading response for input query
    st.text_input(placeholder="Ask your question here", label="Question", label_visibility="collapsed", key='text_input', on_change=clear_input)
    query = st.session_state.query
    if query:
        with st.spinner('Searching...'):
            config = {"configurable": {"session_id":get_current_session()}}
            conversational_rag_chain=RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")
            response = conversational_rag_chain.invoke({"input": query}, config=config)

        st.session_state.query_text_input=""
        with chat_placeholder.container():
            show_chat_history()