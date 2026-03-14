from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

# Force load .env and override any existing environment variables
load_dotenv(override=True)

# Debug: show whether variables are loaded
print("Groq:", os.getenv("GROQ_API_KEY"))
print("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))

# Debug: print all LangChain-related env vars
for key, value in os.environ.items():
    if "LANGCHAIN" in key:
        print(f"{key} = {value}")


def main():
    print("\nHello from langchain basic chain!")

    information = """
    Elon Musk is a technology entrepreneur known for founding SpaceX
    and leading Tesla. He played a major role in advancing electric
    vehicles, reusable rockets, and AI technologies.
    """

    summary_template = """
    Given the information {information} about a person, create:

    1. A short summary
    2. Two interesting facts about them
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatGroq(
        temperature=0,
        model="llama-3.1-8b-instant"
    )

    chain = prompt | llm

    response = chain.invoke({"information": information})

    print("\nResponse:\n")
    print(response.content)


if __name__ == "__main__":
    main()
    