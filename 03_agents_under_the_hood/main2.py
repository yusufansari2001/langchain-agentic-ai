from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10


# -----------------------------
# Tool Implementations
# -----------------------------

def get_product_price(product: str) -> float:
    """Return the price of a product."""
    print(f"    >> Executing get_product_price(product='{product}')")

    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }

    return prices.get(product, 0)


def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount to a product price."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    discounts = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    percent = discounts.get(discount_tier, 0)

    return round(price * (1 - percent / 100), 2)


# -----------------------------
# JSON Tool Schema
# -----------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product from the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Name of the product"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply discount to a product price",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "Original product price"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "Discount tier (bronze, silver, gold)"
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]


# Dictionary to map tool names → python functions
tool_map = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}


# -----------------------------
# Agent Loop
# -----------------------------

@traceable(name="Manual JSON Agent Loop")
def run_agent(question: str):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant.\n"
                "You have access to tools to get product prices and apply discounts.\n"
                "Always call tools when needed before answering."
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm.invoke(messages, tools=tools)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_function = tool_map.get(tool_name)

        if tool_function is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_function(**tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)

        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )

    print("ERROR: Max iterations reached")
    return None


# -----------------------------
# Run Agent
# -----------------------------

if __name__ == "__main__":

    print("Manual JSON Tool Agent\n")

    run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )