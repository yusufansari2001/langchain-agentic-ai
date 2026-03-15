# Building AI Agents with Tool Use — From Abstraction to Manual

This README walks through three progressively lower-level implementations of the same AI agent: a shopping assistant that fetches product prices and applies discounts. Each version peels back one more layer of the framework, showing exactly what is happening under the hood.

---

## Part 1 — The High-Level Approach: LangChain Abstractions

The fastest way to get an agent running is with LangChain's `@tool` decorator and `.bind_tools()`. You write your Python functions, decorate them, and let the framework handle the schema generation, tool binding, and invocation plumbing.

```python
from langchain.tools import tool
from langchain_groq import ChatGroq

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price."""
    discounts = {"bronze": 5, "silver": 12, "gold": 23}
    return round(price * (1 - discounts.get(discount_tier, 0) / 100), 2)

tools = [get_product_price, apply_discount]
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)
```

**What LangChain does for you:**

- `@tool` introspects your function's signature, type hints, and docstring to automatically generate a JSON schema the LLM understands.
- `.bind_tools(tools)` attaches that schema to every call made to the LLM — you never manually write JSON tool definitions.
- `tool.invoke(args)` handles argument unpacking and calling the underlying Python function.
- `ToolMessage` construction is just one line — you pass the result and the tool call ID, and LangChain formats the message correctly for the conversation history.

**The agent loop** still runs explicitly — you write the `for` loop, check for `tool_calls`, dispatch, and append messages. LangChain is not doing "agentic" magic here; it is only handling the schema and invocation boilerplate.

**When to use this:** When you want to move fast, your tools are simple Python functions, and you trust the framework's schema generation to produce accurate descriptions.

---

## Part 2 — Under the Hood: What Actually Happens

Before looking at the manual implementations, it helps to understand what the LangChain abstractions are secretly doing at every step.

### Step 1 — Tool schema generation

The LLM does not know Python functions exist. It only knows what you tell it through the API. LangChain converts each `@tool` function into a JSON schema that describes its name, purpose, and parameters:

```json
{
  "type": "function",
  "function": {
    "name": "get_product_price",
    "description": "Look up the price of a product in the catalog.",
    "parameters": {
      "type": "object",
      "properties": {
        "product": { "type": "string", "description": "Name of the product" }
      },
      "required": ["product"]
    }
  }
}
```

This schema is passed to the LLM on every call. The LLM reads it and decides which tool to call and with what arguments.

### Step 2 — The LLM responds with a tool call

When the LLM decides to use a tool, it does not call it — it just returns a structured JSON response indicating *which* tool it wants to call and *what* arguments to pass:

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "name": "get_product_price",
      "args": { "product": "laptop" }
    }
  ]
}
```

Your code is responsible for reading this, finding the matching Python function, calling it, and returning the result.

### Step 3 — You execute the tool and return the result

You look up the function in a dictionary, call it with the parsed arguments, and wrap the result in a `ToolMessage`:

```python
tool_function = tool_map[tool_name]
observation = tool_function(**tool_args)

messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
```

The LLM expects to see this `ToolMessage` in the next call. It matches the `tool_call_id` to its earlier request and uses the result to continue reasoning.

### Step 4 — The loop continues

This exchange repeats until the LLM produces a response with no `tool_calls`. That signals it has gathered everything it needs and is ready to give a final answer in plain text.

```
User question
    → LLM sees tools, decides to call get_product_price
    → Your code calls the function, appends result
    → LLM sees result, decides to call apply_discount
    → Your code calls the function, appends result
    → LLM sees result, produces Final Answer
```

The entire "agentic loop" is just this message-passing cycle managed by a `for` loop with a max iteration guard.

---

## Part 3 — The Three Manual Implementations

Each implementation in this project removes one more layer of abstraction, exposing the mechanics described above.

### Implementation 1 — Manual JSON Schema (no `@tool`)

**File:** `manual_json_agent.py`

Instead of using `@tool`, the JSON schema is written by hand and passed directly to the LLM:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product from the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Name of the product"}
                },
                "required": ["product"]
            }
        }
    },
    ...
]

ai_message = llm.invoke(messages, tools=tools)
```

A `tool_map` dictionary connects tool names to their Python functions:

```python
tool_map = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}
```

**What changed vs the LangChain version:**
- No `@tool` decorator — the functions are plain Python.
- The JSON schema is defined manually instead of being generated from type hints and docstrings.
- `.bind_tools()` is replaced with passing `tools=tools` directly to `llm.invoke()`.
- Tool invocation is unchanged — you still call `tool_function(**tool_args)`.

**Why this matters:** You gain full control over the schema. You can write richer descriptions, add examples, constrain enum values, or structure parameters exactly as the LLM needs to see them — none of which `@tool` introspection gives you.

---

### Implementation 2 — LangChain `.bind_tools()` with `@tool`

**File:** `langchain_bind_tools_agent.py`

This is the highest-level implementation in the set, using both `@tool` and `.bind_tools()`:

```python
tools = [get_product_price, apply_discount]
llm_with_tools = llm.bind_tools(tools)
```

The `@tool` decorator handles schema generation. `.bind_tools()` attaches the schema so you do not pass `tools=` on each call. Tool invocation uses `tool.invoke(args)`:

```python
tool_to_use = tools_dict.get(tool_name)
observation = tool_to_use.invoke(tool_args)
```

**What changed vs the manual JSON version:**
- Schema is generated automatically from function signatures.
- `llm_with_tools` carries the bound tools — no need to pass them on each `invoke()` call.
- Tool dispatch uses LangChain's `BaseTool.invoke()` wrapper rather than calling the function directly.

**When this is the right choice:** Rapid prototyping with well-typed Python functions. The trade-off is less control over how the tool is described to the LLM.

---

### Implementation 3 — Pure ReAct with Text Parsing (no LangChain tool plumbing)

**File:** `react_agent.py`

This version abandons structured tool-call parsing entirely. The LLM is given a plain-text prompt that defines a reasoning format (Thought / Action / Action Input / Observation), and your code parses the raw text response to extract tool calls:

```python
REACT_PROMPT = """
You are a shopping assistant.
Use the following format:

Thought: think about what to do
Action: tool_name
Action Input: input

Observation: result

Final Answer: your answer
"""
```

The agent loop reads the LLM's text output and extracts the tool name and arguments with string parsing:

```python
action = output.split("Action:")[1].split("\n")[0].strip()
action_input = output.split("Action Input:")[1].split("\n")[0].strip()

if action == "apply_discount":
    parts = action_input.split(",")
    price = float(parts[0])
    tier = parts[1].strip()
    observation = tool(price, tier)
```

The result is appended to a scratchpad string, which is concatenated back into the next prompt — there is no `messages` list, no `ToolMessage`, no structured API objects.

**What changed vs the other two:**
- No structured `tool_calls` field in the response — the LLM writes its intent as plain text.
- Argument parsing is manual string splitting — brittle but fully transparent.
- The conversation history is a single concatenated string, not a list of message objects.
- Works with smaller or older models that may not support structured tool-call APIs.

**Why this matters:** This is how the original ReAct paper worked before LLMs had native function-calling support. It shows that the "agent loop" concept is fundamentally just prompt engineering plus string parsing — the structured API is a cleaner interface on top of the same idea.

---

## Summary

| | Manual JSON | LangChain `.bind_tools()` | ReAct Text Parsing |
|---|---|---|---|
| Schema defined by | Hand-written JSON | `@tool` introspection | Plain-text prompt |
| Tool call format | Structured API (`tool_calls`) | Structured API (`tool_calls`) | Parsed text output |
| Argument parsing | `**tool_args` dict unpacking | `tool.invoke(args)` | Manual `str.split()` |
| Conversation history | `messages` list | `messages` list | Concatenated string |
| Model requirement | Function-calling support | Function-calling support | Any text model |
| Control over schema | Full | Limited | N/A |

The progression from Part 3 → Part 1 in this codebase is a journey from raw text manipulation to structured API contracts. Understanding each layer makes you a better debugger when things go wrong at the boundary between your code and the LLM.