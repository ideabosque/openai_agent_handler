# 🧠 OpenAIEventHandler

The `OpenAIEventHandler` is a concrete implementation of the ![`AIAgentEventHandler`](https://github.com/ideabosque/ai_agent_handler) base class designed to interface with OpenAI's GPT models. It orchestrates message formatting, model invocation, tool integration, streaming, and threading within the AI agent execution pipeline.

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
    "weather_provider": "open-meteo"
  },
  "instructions": "You are a OpenAI-based AI Assistant responsible for providing accurate weather information using the `get_weather_forecast` function. Analyze user input to extract city and date information, and call the tool accordingly. Always clarify ambiguous input and offer detailed yet concise responses.",
  "num_of_messages": 30,
  "tool_call_role": "developer"
}
```

---

## 💬 Full-Scale Chatbot Scripts

### 🔁 Non-Streaming Chatbot Script

```python
import pendulum, os
from openai_agent_handler import OpenAIEventHandler
from dotenv import load_dotenv

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "funct_bucket_name": os.getenv("funct_bucket_name"),
    "funct_zip_path": os.getenv("funct_zip_path"),
    "funct_extract_path": os.getenv("funct_extract_path"),
    "connection_id": os.getenv("connection_id"),
    "endpoint_id": os.getenv("endpoint_id"),
    "test_mode": os.getenv("test_mode"),
}

weather_agent = { ... }  # Configuration as defined above
handler = OpenAIEventHandler(logger=None, agent=weather_agent, **setting)
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
