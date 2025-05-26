import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# Page Config
st.set_page_config(page_title="Agentic AI", page_icon="🤖", layout="centered")

# Custom CSS for styling
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

# Header
st.markdown('<div class="big-title">🤖 Agentic AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything. Get thoughtful responses.</div>', unsafe_allow_html=True)

# Input box
user_input = st.text_input("💬 What’s your question?", placeholder="e.g. What is agentic AI?")

if user_input:
    with st.spinner("Thinking... 🤔"):
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        prompt = PromptTemplate.from_template("You are an AI assistant. Respond to: {question}")
        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run(user_input)

    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
