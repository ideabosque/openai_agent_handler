#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import logging
import threading
import traceback
from decimal import Decimal
from queue import Queue
from typing import Any, Dict, List, Optional

import openai
import pendulum

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Utility


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
      - Handles both streaming and non-streaming responses.
      - Supports function calling with argument validation.
      - Provides detailed logging and error handling.
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initializes the OpenAI event handler with required configuration.

        :param logger: A logging instance for debug/info messages.
        :param agent: Dictionary containing agent configuration including OpenAI API key and model settings.
        :param setting: Additional settings passed as keyword arguments.
        """
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

        self.logger = logger
        self.client = openai.OpenAI(
            api_key=agent["configuration"].get("openai_api_key")
        )
        self.model_setting = dict(
            {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in agent["configuration"].items()
                if k not in ["openai_api_key"]
            },
            **{
                "instructions": agent["instructions"],
            },
        )

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Makes an API call to OpenAI with provided arguments.

        :param kwargs: Dictionary of arguments to pass to the OpenAI API.
        :return: Response from OpenAI API.
        :raises: Exception if API call fails or returns error.
        """
        try:
            variables = dict(self.model_setting, **kwargs)
            return self.client.responses.create(**variables)
        except Exception as e:
            self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Sends a request to OpenAI API with support for both streaming and non-streaming responses.

        :param input_messages: List of message dictionaries representing conversation history.
        :param queue: Queue object for receiving streaming events.
        :param stream_event: Event object to signal when streaming is complete.
        :param model_setting: Optional model-specific settings to override defaults.
        :return: Response ID for non-streaming requests, None for streaming.
        :raises: Exception if request fails or client is not configured.
        """
        try:
            if not self.client:
                self.logger.error("No OpenAI client provided.")
                return None

            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            response = self.invoke_model(
                **{
                    "input": input_messages,
                    "stream": stream,
                }
            )

            # If streaming is enabled, process chunks
            if stream:
                self.handle_stream(
                    response,
                    input_messages,
                    queue=queue,
                    stream_event=stream_event,
                )
                return None

            # Otherwise, handle a normal (non-stream) response
            self.handle_response(response, input_messages)
            return response.id

        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")

    def handle_function_call(
        self,
        tool_call: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Processes function calls from the model including validation, execution and history updates.

        :param tool_call: Object containing function call details from model.
        :param input_messages: Conversation history to update with function results.
        :return: Updated input messages list.
        :raises: ValueError if tool_call is invalid.
                Exception if function execution fails.
        """
        # Validate tool call
        if not tool_call or not hasattr(tool_call, "id"):
            raise ValueError("Invalid tool_call object")

        try:
            # Extract function call metadata
            function_call_data = {
                "id": tool_call.id,
                "arguments": tool_call.arguments,
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "type": tool_call.type,
                "status": tool_call.status,
            }

            # Record initial function call
            self.logger.info(
                f"[handle_function_call] Starting function call recording for {function_call_data['name']}"
            )
            self._record_function_call_start(function_call_data)

            # Parse and process arguments
            self.logger.info(
                f"[handle_function_call] Processing arguments for function {function_call_data['name']}"
            )
            arguments = self._process_function_arguments(function_call_data)

            # Execute function and handle result
            self.logger.info(
                f"[handle_function_call] Executing function {function_call_data['name']} with arguments {arguments}"
            )
            function_output = self._execute_function(function_call_data, arguments)

            # Update conversation history
            self.logger.info(
                f"[handle_function_call][{function_call_data['name']}] Updating conversation history"
            )
            self._update_conversation_history(
                function_call_data, function_output, input_messages
            )

            # Continue conversation
            self.logger.info(
                f"[handle_function_call][{function_call_data['name']}] Continuing conversation"
            )

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Utility.json_dumps(
                                {
                                    "tool": {
                                        "tool_call_id": tool_call.id,
                                        "tool_type": tool_call.type,
                                        "name": tool_call.name,
                                        "arguments": arguments,
                                    },
                                    "output": function_output,
                                }
                            ),
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )
            return input_messages

        except Exception as e:
            self.logger.error(f"Error in handle_function_call: {e}")
            raise

    def _record_function_call_start(self, function_call_data: Dict[str, Any]) -> None:
        """
        Records the start of a function call execution in the system.

        :param function_call_data: Dictionary containing function call metadata.
        """
        self.invoke_async_funct(
            "async_insert_update_tool_call",
            **{
                "tool_call_id": function_call_data["id"],
                "tool_type": function_call_data["type"],
                "name": function_call_data["name"],
            },
        )

    def _process_function_arguments(
        self, function_call_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parses and validates function arguments from the model response.

        :param function_call_data: Dictionary containing function call details including arguments.
        :return: Processed and validated arguments dictionary.
        :raises: ValueError if argument parsing fails.
        """
        try:
            arguments = Utility.json_loads(function_call_data.get("arguments", "{}"))
            arguments["endpoint_id"] = self._endpoint_id

            return arguments

        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": function_call_data.get("arguments", "{}"),
                    "status": "failed",
                    "notes": log,
                },
            )
            self.logger.error("Error parsing function arguments: %s", e)
            raise ValueError(f"Failed to parse function arguments: {e}")

    def _execute_function(
        self, function_call_data: Dict[str, Any], arguments: Dict[str, Any]
    ) -> Any:
        """
        Executes the requested function with provided arguments and handles results.

        :param function_call_data: Dictionary containing function metadata.
        :param arguments: Processed arguments to pass to the function.
        :return: Function execution result or error message.
        :raises: ValueError if function is not supported.
        """
        agent_function = self.get_function(function_call_data["name"])
        if not agent_function:
            raise ValueError(
                f"Unsupported function requested: {function_call_data['name']}"
            )

        try:
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
                    "status": "in_progress",
                },
            )

            function_output = agent_function(**arguments)

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "content": Utility.json_dumps(function_output),
                    "status": "completed",
                },
            )
            return function_output

        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
                    "status": "failed",
                    "notes": log,
                },
            )
            return f"Function execution failed: {e}"

    def _update_conversation_history(
        self,
        function_call_data: Dict[str, Any],
        function_output: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Updates the conversation history with function call details and output.

        :param function_call_data: Dictionary containing function call metadata.
        :param function_output: Result returned from function execution.
        :param input_messages: List of messages to update with function details.
        """
        input_messages.append(function_call_data)
        input_messages.append(
            {
                "type": "function_call_output",
                "call_id": function_call_data["call_id"],
                "output": Utility.json_dumps(function_output),
            }
        )

    def handle_response(
        self, response: Any, input_messages: List[Dict[str, Any]]
    ) -> None:
        """
        Processes model responses and routes them to appropriate handlers.

        :param response: Response object from the model.
        :param input_messages: Current conversation history.
        :param queue: Optional queue for streaming responses.
        :param stream_event: Optional event to signal streaming completion.
        """

        for output in response.output:
            # If it's a normal message
            if output.type == "message":
                self.final_output = {
                    "message_id": output.id,
                    "role": output.role,
                    "content": output.content[0].text,
                }
                return

            # If it's a function call
            if output.type == "function_call":
                input_messages = self.handle_function_call(
                    output,
                    input_messages,
                )

        self.ask_model(input_messages)

    def handle_stream(
        self,
        response_stream,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Processes streaming responses from the model chunk by chunk.

        :param response_stream: Iterator of response chunks from the model.
        :param input_messages: Current conversation history.
        :param queue: Queue to receive streaming events.
        :param stream_event: Event to signal when streaming is complete.
        """
        message_id = None
        role = None
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        output_format = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )
        index = 0

        for chunk in response_stream:
            if chunk.type != "response.output_text.delta":
                self.logger.debug(f"Chunk type: {getattr(chunk, 'type', 'N/A')}")
                self.logger.debug(f"Chunk attributes: {vars(chunk)}")

            # If the model run has just started
            if chunk.type == "response.created":
                self.logger.info(f"Stream created, run_id={chunk.response.id}")
                if queue:
                    queue.put({"name": "run_id", "value": chunk.response.id})

            elif chunk.type == "response.output_item.added":
                pass
            elif chunk.type == "response.content_part.added":
                # Send initial message start signal to WebSocket server
                self.send_data_to_websocket(
                    index=index,
                    data_format=output_format,
                )
                index += 1
            # If we received partial text data
            elif chunk.type == "response.output_text.delta":
                print(chunk.delta, end="", flush=True)

                # For JSON formats, accumulate partial JSON text and process it
                # when complete JSON objects are detected. This ensures valid JSON
                # is sent to the WebSocket server.
                if output_format in ["json_object", "json_schema"]:
                    accumulated_partial_json += chunk.delta
                    index, self.accumulated_text, accumulated_partial_json = (
                        self.process_and_send_json(
                            index,
                            self.accumulated_text,
                            accumulated_partial_json,
                            output_format,
                        )
                    )
                else:
                    self.accumulated_text += chunk.delta
                    accumulated_partial_text += chunk.delta
                    # Send incremental text chunk to WebSocket server
                    if len(accumulated_partial_text) >= int(
                        self.setting.get("accumulated_partial_text_buffer", "10")
                    ):
                        self.send_data_to_websocket(
                            index=index,
                            data_format=output_format,
                            chunk_delta=accumulated_partial_text,
                        )
                        accumulated_partial_text = ""
                        index += 1
            elif chunk.type == "response.output_text.done":
                # Send message completion signal to WebSocket server
                if len(accumulated_partial_text) > 0:
                    self.send_data_to_websocket(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                    )
                    accumulated_partial_text = ""
                    index += 1
            elif chunk.type == "response.content_part.done":
                # Send message completion signal to WebSocket server
                self.send_data_to_websocket(
                    index=index,
                    data_format=output_format,
                    is_message_end=True,
                )
            elif chunk.type == "response.output_item.done":
                pass
            # If streaming is completed
            elif chunk.type == "response.completed":
                self.logger.info(f"Stream completed, run_id={chunk.response.id}")

                # Process any final output objects in chunk.response
                if hasattr(chunk.response, "output") and chunk.response.output:
                    # Check if output is a function call or message
                    if any(
                        output.type == "function_call"
                        for output in chunk.response.output
                        if hasattr(output, "type")
                    ):
                        for output in chunk.response.output:
                            # If it's a function call
                            if output.type == "function_call":
                                input_messages = self.handle_function_call(
                                    output, input_messages
                                )

                        self.ask_model(
                            input_messages, queue=queue, stream_event=stream_event
                        )
                        return

                    message_id = chunk.response.output[-1].id
                    role = chunk.response.output[-1].role

        self.final_output = {
            "message_id": message_id,
            "role": role,
            "content": self.accumulated_text,
        }

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
