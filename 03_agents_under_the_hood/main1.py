from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10


# -----------------------------
# Tools
# -----------------------------

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")

    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }

    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


# -----------------------------
# Agent Loop
# -----------------------------

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a shopping assistant with access to tools.\n\n"
                "Available tools:\n"
                "1. get_product_price(product)\n"
                "2. apply_discount(price, discount_tier)\n\n"
                "RULES:\n"
                "1. ALWAYS call get_product_price first to get the real price.\n"
                "2. After receiving the price, you MUST call apply_discount.\n"
                "3. NEVER calculate discounts yourself.\n"
                "4. NEVER skip apply_discount when discount requested.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool call → final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)

        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )

    print("ERROR: Max iterations reached without final answer")

    return None


# -----------------------------
# Run Agent
# -----------------------------

if __name__ == "__main__":

    print("Hello LangChain Agent (.bind_tools)!")
    print()

    run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )