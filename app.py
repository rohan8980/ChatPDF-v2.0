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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    return st.session_state.session_id
def get_new_session(session_id='Chat 1'):
    st.session_state.session_id = session_id
def get_session_history(session_id:str)->BaseChatMessageHistory:
    session_id = get_current_session() if not session_id else session_id
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]
def start_new_chat():
    new_session_id = f'Chat {len(st.session_state.store) + 1}'
    get_new_session(new_session_id)
    st.session_state.store[new_session_id] = ChatMessageHistory()
    
def show_chat_history(session_id=None):
    st.markdown("""
        <style>
        .human-message {
            text-align: right;
            background: linear-gradient(135deg, #a8e063, #56ab2f);
            padding: 14px;
            border-radius: 20px 20px 0 20px;
            margin-bottom: 12px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.15);
            max-width: 65%;
            margin-left: auto;
            font-family: 'Roboto', sans-serif;
            font-size: 15px;
            color: #fff;
            animation: fade-slide-in 0.4s ease;
        }

        .ai-message {
            text-align: left;
            background: linear-gradient(135deg, #f0f0f0, #cccccc);
            padding: 14px;
            border-radius: 20px 20px 20px 0;
            margin-bottom: 12px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.15);
            max-width: 65%;
            margin-right: auto;
            font-family: 'Roboto', sans-serif;
            font-size: 15px;
            color: #333;
            animation: fade-slide-in 0.4s ease;
        }

        @keyframes fade-slide-in {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """, unsafe_allow_html=True)

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
if 'is_history_available' not in st.session_state:
    st.session_state.is_history_available = False  


# Streamlit UI
# Utility functions
def clear_input(): #Clears the input field to avoid rerunning with same query twice
        st.session_state.query=st.session_state.text_input
        st.session_state.text_input=""
def set_history_session_id(): #Update session_id to fetch older chats
    st.session_state.session_id = st.session_state.older_chat_id


# UI
st.title("Chat PDF with Chat History")
st.write(f'<p style="font-size: medium">Chat with PDF using Langchain HuggingFace Groq</p>', unsafe_allow_html=True)

# Files upload and Text input bar
files=st.file_uploader('Choose PDF file(s)', type=['pdf'], accept_multiple_files=True)
if files:
    # Prepare rag_chain from uploaded files
    vectorstore = get_vectorstore_from_files(files=files, HF_Embed_Model=HF_Embed_Model)
    llm_model = get_llm(groq_api_key=groq_api_key,model_name=LLM_Model)
    rag_chain = get_rag_chain(vectorstore, llm_model) 

    # Chat history container
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        show_chat_history()

    # Taking user Query and getting answer from LLM
    st.text_input(placeholder="Ask your question here", label="Question", label_visibility="collapsed", key='text_input', on_change=clear_input)
    if st.session_state.query:
        with st.spinner('Searching...'):
            config = {"configurable": {"session_id":get_current_session()}}
            conversational_rag_chain=RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")
            response = conversational_rag_chain.invoke({"input": st.session_state.query}, config=config)
        st.session_state.query = ""
        st.session_state.is_history_available = True

    with chat_placeholder.container():
        show_chat_history()

# New Chat and Older Chats buttons    
if st.session_state.is_history_available:
        
    col1, col2, col3, col4 = st.columns(spec=[1, 1, 2, 2])
    with col1:
        if st.button("New Chat"):
            start_new_chat()    
            chat_placeholder.empty()
    with col2:
        if st.button("Older Chats"): 
            sessions = list(st.session_state.store.keys())[::-1]
            index = sessions.index(st.session_state.session_id)
            with col3:
                st.selectbox(label="Choose a chat", options=sessions, index=index, key = "older_chat_id", on_change=set_history_session_id, label_visibility="collapsed")
                with chat_placeholder.container():
                    show_chat_history()