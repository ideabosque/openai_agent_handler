#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
import threading
from queue import Queue
from typing import Any, Dict, List, Optional

import openai

from ai_agent_handler import AIAgentEventHandler


# ----------------------------
# OpenAI Response Streaming with Function Handling and History
# ----------------------------
class OpenAIEventHandler(AIAgentEventHandler):
    """
    Manages conversations and function calls in real-time with OpenAI's API:
      - Streams partial text deltas.
      - Detects function calls embedded in the response.
      - Executes local functions as needed.
      - Maintains the conversation history (input_messages).
      - Stores the final generated message or function call outputs.
    """

    def __init__(
        self, logger: logging.Logger, agent: Dict[str, Any], **setting: Dict[str, Any]
    ) -> None:
        """
        :param logger: A logging instance for debug/info messages.
        :param client: An OpenAI client instance or a compatible object.
        :param model: Default model name to use for requests (defaults to "gpt-4o").
        :param tools: Optional list of tool definitions the model may call.
        """
        self.logger = logger
        self.client = openai.OpenAI(
            api_key=agent["llm_configuration"].get("openai_api_key")
        )
        self.model_setting = dict(
            agent["configuration"],
            **{
                "model": agent["llm_configuration"].get("model", "gpt-4o"),
                "instructions": agent["instructions"],
            },
        )

        # Will hold partial text from streaming
        self.accumulated_text: str = ""
        # Will hold the final output message data, if any
        self.final_output: Dict[str, Any] = {}
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Invokes the model with the provided arguments and returns the response.

        :param kwargs: Keyword arguments for the model invocation.
        :return: The response from the model.
        """
        variables = dict(self.model_setting, **kwargs)

        return self.client.responses.create(**variables)

    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Sends a request to the OpenAI API. If a queue is provided, we switch to streaming mode,
        otherwise, a simple (non-streaming) request is made.

        :param input_messages: Conversation history, including the latest user question.
        :param queue: An optional queue to receive streaming events. If provided, streaming is used.
        :param stream_event: An optional threading.Event to signal streaming completion.
        :return: The response ID if non-streaming, otherwise None.
        """
        if not self.client:
            self.logger.error("No OpenAI client provided.")
            return None

        should_stream = True if queue is not None else False

        # Add model-specific settings if provided
        if model_setting:
            self.model_setting.update(model_setting)

        response = self.invoke_model(
            **{
                "input": input_messages,
                "stream": should_stream,
            }
        )

        # If streaming is enabled, process chunks
        if should_stream:
            self.handle_stream(
                response,
                input_messages,
                queue=queue,
                stream_event=stream_event,
            )
            return None

        # Otherwise, handle a normal (non-stream) response
        for output in response.output:
            self.handle_output(
                output,
                input_messages,
            )
        return response.id

    def handle_function_call(
        self,
        tool_call: Any,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Extracts function-call details from the model output, invokes the corresponding local
        Python function, and updates the conversation history with the call + its output.

        :param tool_call: The function call data from a streaming or non-streaming response.
        :param input_messages: The conversation history to be appended with function call info.
        :param queue: Queue instance if streaming is in progress (optional).
        :param stream_event: Event instance to signal completion (optional).
        """
        function_call_data = {
            "id": tool_call.id,
            "arguments": tool_call.arguments,
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "type": tool_call.type,
            "status": tool_call.status,
        }

        # Parse arguments (typically JSON)
        try:
            arguments = json.loads(function_call_data.get("arguments", "{}"))
        except Exception as e:
            self.logger.error("Error parsing function arguments: %s", e)
            arguments = {}

        # Inject endpoint ID for contextual use
        arguments["endpoint_id"] = self.endpoint_id

        # Look up and execute the corresponding local function
        assistant_function = self.get_function(function_call_data["name"])
        function_output = None
        if assistant_function is not None:
            function_output = assistant_function(**arguments)
        else:
            self.logger.error(
                "Unsupported function requested: %s", function_call_data["name"]
            )

        # Append the function call + output to conversation history
        input_messages.append(function_call_data)
        input_messages.append(
            {
                "type": "function_call_output",
                "call_id": function_call_data["call_id"],
                "output": str(function_output),
            }
        )

        # Optionally continue the conversation with updated inputs (streaming or not)
        should_stream = True if queue else False
        response = self.invoke_model(
            **{
                "input": input_messages,
                "stream": should_stream,
            }
        )

        if should_stream:
            self.handle_stream(
                response,
                input_messages,
                queue=queue,
                stream_event=stream_event,
            )
        else:
            for output in response.output:
                self.handle_output(
                    output,
                    input_messages,
                )

    def handle_output(
        self,
        output: Any,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Processes a single output object. If it's a message, we store it as final output.
        If it's a function call, we route to handle_function_call.

        :param output: The model's output object.
        :param input_messages: Conversation history for potential updates.
        :param queue: Optional queue if streaming is in use.
        :param stream_event: Optional event to signal streaming completion.
        """
        self.logger.info("Processing output: %s", output)

        # If it's a normal message
        if output.type == "message":
            self.final_output = {
                "message_id": output.id,
                "role": output.role,
                "content": output.content[0].text,
            }

        # If it's a function call
        if output.type == "function_call":
            self.handle_function_call(
                output,
                input_messages,
                queue=queue,
                stream_event=stream_event,
            )

    def handle_stream(
        self,
        response_stream,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Iterates over each chunk in a streaming response:
          - Logs chunk details for debugging.
          - Detects 'response.created' to store a run ID in the queue.
          - Detects 'response.completed' to handle final outputs.
          - Captures partial text from 'response.output_text.delta'.
          - Notifies completion via stream_event at the end.

        :param response_stream: The streaming response object.
        :param input_messages: Conversation history for updates.
        :param queue: Optional queue to push events like 'response_id'.
        :param stream_event: Optional event to signal streaming completion.
        """
        self.accumulated_text = ""

        for chunk in response_stream:
            if chunk.type != "response.output_text.delta":
                self.logger.info(f"Chunk type: {getattr(chunk, 'type', 'N/A')}")
                self.logger.info(f"Chunk attributes: {vars(chunk)}")

            # If the model run has just started
            if chunk.type == "response.created":
                self.logger.info(f"Stream created, run_id={chunk.response.id}")
                if queue:
                    queue.put({"name": "run_id", "value": chunk.response.id})

            elif chunk.type == "response.output_item.added":
                pass
            elif chunk.type == "response.content_part.added":
                # TODO: Send the start signal to WSS.
                pass
            # If we received partial text data
            elif chunk.type == "response.output_text.delta":
                self.accumulated_text += chunk.delta
                print(chunk.delta, end="", flush=True)
                # TODO: Send the output_text.delta to WSS.
            elif chunk.type == "response.output_text.done":
                pass
            elif chunk.type == "response.content_part.done":
                # TODO: Send the complete signal to WSS.
                pass
            elif chunk.type == "response.output_item.done":
                pass
            # If streaming is completed
            elif chunk.type == "response.completed":
                self.logger.info(f"Stream completed, run_id={chunk.response.id}")

                # Process any final output objects in chunk.response
                if hasattr(chunk.response, "output") and chunk.response.output:
                    for output in chunk.response.output:
                        self.handle_output(
                            output,
                            input_messages,
                            queue=queue,
                            stream_event=stream_event,
                        )

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
