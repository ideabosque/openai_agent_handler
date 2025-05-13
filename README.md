# 🧠 OpenAIEventHandler

The `OpenAIEventHandler` is a concrete implementation of the `AIAgentEventHandler` base class designed to interface with OpenAI's GPT models. It orchestrates message formatting, model invocation, tool integration, streaming, and threading within the AI agent execution pipeline.

This handler enables a **stateless, multi-turn AI orchestration** system built to support tools like `web_search_preview`, and `get_weather_forecast`.

---

## 🖉 Inheritance

![AI Agent Event Handler Class Diagram](/images/ai_agent_event_handler_class_diagram.jpg)

```
AIAgentEventHandler
     ▲
     └── OpenAIEventHandler
```

---

## 📦 Module Features

### 🔧 Attributes

* `client`: OpenAI API client instance
* `model_settings`: A dictionary containing OpenAI model configuration (e.g., `model`, `temperature`, `tools`, etc.)

### 📞 Core Method: `invoke_model`

```python
def invoke_model(
    input: List[Dict[str, Any]],
    model: str,
    include: Optional[List[str]] = None,
    instructions: Optional[str] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
    parallel_tool_calls: Optional[bool] = True,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    service_tier: Optional[str] = None,
    store: Optional[bool] = True,
    stream: Optional[bool] = False,
    temperature: Optional[float] = 1.0,
    top_p: Optional[float] = 1.0,
    truncation: Optional[str] = "disabled",
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    user: Optional[str] = None
) -> None
```

---

## 📘 Sample Configuration

```json
{
  "endpoint_id": "openai",
  "agent_name": "Weather Assistant",
  "model": "gpt-4o",
  "temperature": 0,
  "tools": [
    {
      "type": "function",
      "name": "get_weather_forecast",
      "description": "Get the weather forecast for a given city and date",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The name of the city to retrieve the weather for."
          },
          "date": {
            "type": "string",
            "description": "The date to retrieve the forecast for (YYYY-MM-DD)."
          }
        },
        "additionalProperties": false,
        "required": ["city", "date"]
      }
    }
  ],
  "functions": {
    "get_weather_forecast": {
      "class_name": "WeatherForecastFunction",
      "module_name": "weather_funct",
      "configuration": {}
    }
  },
  "function_configuration": {
    "endpoint_id": "openai",
    "region_name": "${region_name}",
    "aws_access_key_id": "${aws_access_key_id}",
    "aws_secret_access_key": "${aws_secret_access_key}",
    "weather_provider": "open-meteo"
  },
  "instructions": """
Task:
- The OpenAI Assistant will handle user queries about weather by retrieving data using the `get_weather_forecast` function.

Role:
- You are a helpful AI weather assistant tasked with retrieving and presenting accurate forecasts.

Steps:
1. Call `get_weather_forecast`:
   - Use city and date from the user input to call the tool.
   - Structure the call as:
     get_weather_forecast({"city": <city>, "date": <date>})

2. Handle Errors:
   - If no forecast is found, respond politely and request clarification.

3. Process the Results:
   - Confirm weather details are clear and include temperature, conditions, or alerts.

4. Format the Response:
   - Use simple, concise sentences.
   - Example:
     - “In San Francisco on 2025-06-01, it will be sunny with a high of 72°F.”

5. Iterate if Needed:
   - If the user asks follow-ups (e.g., tomorrow's forecast), adapt appropriately.

Output Format:
- Weather Summary: State forecast clearly.
- Details: Add temperature, precipitation, and other context.
- Follow-Up: Suggest alternate days or cities if needed.
""",
  "num_of_messages": 30,
  "tool_call_role": "developer"
}
```

---

## 💬 Full-Scale Chatbot Scripts

### 🔁 Non-Streaming Chatbot Script

```python
import pendulum
from openai_agent_handler import OpenAIEventHandler

weather_agent = { ... }  # Configuration as defined above
handler = OpenAIEventHandler(logger=None, agent=None, **weather_agent)
handler.short_term_memory = []

def get_input_messages(messages, num_of_messages):
    return [msg["message"] for msg in sorted(messages, key=lambda x: x["created_at"], reverse=True)][:num_of_messages][::-1]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    message = {"role": "user", "content": user_input}
    handler.short_term_memory.append({"message": message, "created_at": pendulum.now("UTC")})
    messages = get_input_messages(handler.short_term_memory, weather_agent["num_of_messages"])
    run_id = handler.ask_model(messages)

    print("Chatbot:", handler.final_output["content"])
    handler.short_term_memory.append({
        "message": handler.final_output,
        "created_at": pendulum.now("UTC")
    })
```

### 🔁 Streaming Chatbot Script

```python
import pendulum
import threading
from queue import Queue
from openai_agent_handler import OpenAIEventHandler

weather_agent = { ... }  # Configuration as defined above
handler = OpenAIEventHandler(logger=None, agent=None, **weather_agent)
handler.short_term_memory = []

def get_input_messages(messages, num_of_messages):
    return [msg["message"] for msg in sorted(messages, key=lambda x: x["created_at"], reverse=True)][:num_of_messages][::-1]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    message = {"role": "user", "content": user_input}
    handler.short_term_memory.append({"message": message, "created_at": pendulum.now("UTC")})
    messages = get_input_messages(handler.short_term_memory, weather_agent["num_of_messages"])

    stream_queue = Queue()
    stream_event = threading.Event()
    stream_thread = threading.Thread(
        target=handler.ask_model,
        args=[messages, stream_queue, stream_event],
        daemon=True
    )
    stream_thread.start()

    result = stream_queue.get()
    if result["name"] == "run_id":
        print("Run ID:", result["value"])

    stream_event.wait()
    print("Chatbot:", handler.final_output["content"])
    handler.short_term_memory.append({
        "message": handler.final_output,
        "created_at": pendulum.now("UTC")
    })
```

---
