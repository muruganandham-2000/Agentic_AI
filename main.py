import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(page_title="Agentic AI", page_icon="🤖", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .big-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 1.1rem;
            text-align: center;
            color: #555;
        }
        .stTextInput > div > input {
            font-size: 1.1rem;
            padding: 0.5rem;
        }
        .response-box {
            background-color: #f0f2f6;
            border-left: 4px solid #4CAF50;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 0.5rem;
            font-size: 1.05rem;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title">🤖 Agentic AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything. Get thoughtful responses.</div>', unsafe_allow_html=True)

# Input
user_input = st.text_input("💬 What’s your question?", placeholder="e.g. What is agentic AI?")

# Session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if user_input:
    with st.spinner("Thinking... 🤔"):
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        prompt = PromptTemplate.from_template("You are a helpful AI assistant. Answer this: {question}")
        chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)

        response = chain.run({"question": user_input})
        st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
