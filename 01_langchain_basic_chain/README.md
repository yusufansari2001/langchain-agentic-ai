# 01 - LangChain Basic Chain

This example demonstrates a basic LangChain pipeline using a Groq LLM.

## Concepts Covered

- PromptTemplate
- LangChain chains
- Groq LLM integration
- Environment variables

## Setup

Create a `.env` file in the project root:
GROQ_API_KEY=your_api_key_here

## Run the example

From the project root:


uv run 01_langchain_basic_chain/main.py


## Flow


Input text → PromptTemplate → Groq LLM → Response