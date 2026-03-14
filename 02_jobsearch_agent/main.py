import uuid

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from ddgs import DDGS

load_dotenv()


# ---------- Search Tool ----------

@tool
def search(query: str) -> str:
    """Search the internet for information."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return str(list(results))


# ---------- LLM ----------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ---------- Agent ----------

agent = create_agent(
    model=llm,
    tools=[search]
)


# ---------- Main ----------

def main():
    print("Running LangChain Search Agent...\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Find 3 AI engineer jobs using LangChain in the Bay Area."
                )
            ]
        }
    )

    print("\nAgent Response:\n")
    messages = result["messages"]

    jobs = []

    for msg in messages:
        if msg.type == "tool":
            jobs.append(msg.content)

    response = {
        "answer": messages[-1].content,
        "sources": jobs
    }

    print(response)


if __name__ == "__main__":
    main()