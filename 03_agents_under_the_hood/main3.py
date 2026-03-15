from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

MAX_ITERATIONS = 10


# -----------------------------
# Tools
# -----------------------------

def get_product_price(product: str) -> float:
    print(f">> Executing get_product_price({product})")

    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }

    return prices.get(product, 0)


def apply_discount(price: float, discount_tier: str) -> float:
    print(f">> Executing apply_discount({price}, {discount_tier})")

    discounts = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    percent = discounts.get(discount_tier, 0)

    return round(price * (1 - percent / 100), 2)


# Tool map
TOOLS = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}


# -----------------------------
# ReAct Prompt
# -----------------------------

REACT_PROMPT = """
You are a shopping assistant.

You can use the following tools:

get_product_price(product)
apply_discount(price, discount_tier)

Use the following format:

Question: {question}

Thought: think about what to do
Action: tool_name
Action Input: input

Observation: result

Repeat until you know the answer.

When finished:

Final Answer: your answer
"""


# -----------------------------
# Agent Loop
# -----------------------------

def run_agent(question):

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = REACT_PROMPT.format(question=question)

    scratchpad = ""

    for i in range(MAX_ITERATIONS):

        full_prompt = prompt + scratchpad

        response = llm.invoke(full_prompt)

        output = response.content

        print("\nLLM Output:")
        print(output)

        # Check final answer
        if "Final Answer:" in output:
            answer = output.split("Final Answer:")[-1].strip()
            print("\nFinal Answer:", answer)
            return answer

        # Parse action
        if "Action:" in output and "Action Input:" in output:

            action = output.split("Action:")[1].split("\n")[0].strip()
            action_input = output.split("Action Input:")[1].split("\n")[0].strip()

            print("\nTool Selected:", action)
            print("Tool Input:", action_input)

            tool = TOOLS.get(action)

            if tool is None:
                raise ValueError("Unknown tool")

            # Execute tool
            if action == "get_product_price":
                observation = tool(action_input)

            elif action == "apply_discount":

                parts = action_input.split(",")

                price = float(parts[0])
                tier = parts[1].strip()

                observation = tool(price, tier)

            print("Observation:", observation)

            scratchpad += f"""
Thought:
Action: {action}
Action Input: {action_input}
Observation: {observation}
"""

    print("Max iterations reached")


# -----------------------------
# Run Agent
# -----------------------------

if __name__ == "__main__":

    run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )