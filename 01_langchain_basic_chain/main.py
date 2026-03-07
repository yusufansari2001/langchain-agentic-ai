from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def main():
    print("Hello from langchain basic chain!")

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