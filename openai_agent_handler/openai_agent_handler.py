#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import base64
import logging
import threading
import traceback
from decimal import Decimal
from io import BytesIO
from queue import Queue
from typing import Any, Dict, List, Optional

import httpx
import openai
import pendulum
from httpx import Response

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
        input_files: List[str, Any] = [],
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

            if input_files:
                input_messages = self._process_input_files(input_files, input_messages)

            self._process_user_file_ids(input_messages[:-1])

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

    def _attach_files_into_code_interpreter(self, file_ids) -> bool:
        # Find existing code_interpreter tool if it exists
        code_interpreter_tool = next(
            (
                tool
                for tool in self.model_setting.get("tools", [])
                if tool.get("type") == "code_interpreter"
            ),
            None,
        )

        if not code_interpreter_tool:
            return False

        # Initialize file_ids list if it doesn't exist
        if "container" not in code_interpreter_tool:
            code_interpreter_tool["container"] = {"type": "auto"}
        if "file_ids" not in code_interpreter_tool["container"]:
            code_interpreter_tool["container"]["file_ids"] = []

        # Append file_ids to existing code_interpreter tool and ensure uniqueness
        code_interpreter_tool["container"]["file_ids"] = list(
            set(code_interpreter_tool["container"]["file_ids"] + file_ids)
        )

        return True

    def _process_input_files(
        self, input_files: List[Dict[str, Any]], input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and upload input files, attaching them to either code interpreter or user message.

        Args:
            input_files: List of file dictionaries containing file data
            input_messages: List of conversation messages

        Returns:
            Updated input_messages with file references
        """
        # Upload each file to OpenAI and store metadata
        uploaded_files = []
        for input_file in input_files:
            file_data = dict(input_file, purpose="user_data")
            uploaded_file = self.insert_file(**file_data)
            uploaded_files.append(uploaded_file)
            self.uploaded_files.append(uploaded_file)

        # Extract file IDs from uploaded files
        file_ids = [file["id"] for file in uploaded_files]

        # First try to attach files to code interpreter
        if self._attach_files_into_code_interpreter(file_ids):
            return input_messages

        # If code interpreter not available, attach to user message
        if input_messages and input_messages[-1]["role"] == "user":
            # Construct message content with original text and file references
            message_content = [
                {"type": "input_text", "text": input_messages[-1]["content"]}
            ]
            message_content.extend(
                {"type": "input_file", "file_id": file_id} for file_id in file_ids
            )

            # Update the last message with combined content
            input_messages[-1]["content"] = message_content

        return input_messages

    def _process_user_file_ids(self, input_messages: List[Dict[str, Any]]) -> None:
        """
        Process file IDs from user messages and attach them to code interpreter.
        Extracts file IDs from user messages and attempts to attach them to the code interpreter tool.
        Silently continues if message parsing fails.

        Args:
            input_messages: List of conversation messages to process
        """
        # Filter for only user messages
        user_messages = [msg for msg in input_messages if msg["role"] == "user"]

        for message in user_messages:
            try:
                # Parse message content and extract file IDs
                message_content = Utility.json_loads(message["content"])
                file_ids = [
                    content["file_id"]
                    for content in message_content
                    if content.get("type") == "input_file" and "file_id" in content
                ]

                # Attempt to attach files to code interpreter if any found
                if file_ids:
                    self._attach_files_into_code_interpreter(file_ids)

            except Exception:
                # Continue silently if message parsing fails
                continue

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

        message_id = None
        role = None
        content = ""

        for output in response.output:
            # If it's a function call
            if output.type == "function_call":
                input_messages = self.handle_function_call(
                    output,
                    input_messages,
                )
                self.ask_model(input_messages)
                return
            # If it's a normal message
            elif output.type == "message" and output.status == "completed":
                message_id = output.id if message_id is None else message_id
                role = output.role if role is None else role
                content = content + output.content[0].text
            elif output.type == "web_search_call" and output.status == "completed":
                continue
            elif (
                output.type == "code_interpreter_call" and output.status == "completed"
            ):
                continue
            elif output.type == "mcp_list_tools":
                continue
            elif output.type == "mcp_call":
                continue
            elif output.type == "mcp_approval_request":
                raise Exception("MCP Approval Request is not currently supported")
            else:
                raise Exception(
                    f"Unknown response type: {output.type} or status: {output.status}"
                )

        self.final_output = {
            "message_id": message_id,
            "role": role,
            "content": content,
        }

    def handle_stream(
        self,
        response_stream: Any,
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
                self.send_data_to_stream(
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
                    # Check if text contains XML-style tags and update format
                    index, accumulated_partial_text = self.process_text_content(
                        index, accumulated_partial_text, output_format
                    )
            elif chunk.type == "response.output_text.done":
                # Send message completion signal to WebSocket server
                if len(accumulated_partial_text) > 0:
                    self.send_data_to_stream(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                    )
                    accumulated_partial_text = ""
                    index += 1
            elif chunk.type == "response.content_part.done":
                # Send message completion signal to WebSocket server
                self.send_data_to_stream(
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

                    if any(
                        output.type == "mcp_approval_request"
                        for output in chunk.response.output
                        if hasattr(output, "type")
                    ):
                        raise Exception(
                            "MCP Approval Request is not currently supported"
                        )

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

    def get_file(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:

        file = self.client.files.retrieve(kwargs["file_id"])
        openai_file = {
            "id": file.id,
            "object": file.object,
            "filename": file.filename,
            "purpose": file.purpose,
            "created_at": pendulum.from_timestamp(file.created_at, tz="UTC"),
            "bytes": file.bytes,
        }
        if "encoded_content" in kwargs and kwargs["encoded_content"] == True:
            response: Response = self.client.files.content(kwargs["file_id"])
            content = response.content  # Get the actual bytes data)
            # Convert the content to a Base64-encoded string
            openai_file["encoded_content"] = base64.b64encode(content).decode("utf-8")

        return openai_file

    def get_file_list(self, **kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        if kwargs.get("purpose"):
            file_list = self.client.files.list(purpose=kwargs["purpose"])
        else:
            file_list = self.client.files.list()
        return [
            {
                "id": file.id,
                "object": file.object,
                "filename": file.filename,
                "purpose": file.purpose,
                "created_at": pendulum.from_timestamp(file.created_at, tz="UTC"),
                "bytes": file.bytes,
            }
            for file in file_list.data
        ]

    def insert_file(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        purpose = kwargs["purpose"]

        if "encoded_content" in kwargs:
            encoded_content = kwargs["encoded_content"]
            # Decode the Base64 string
            decoded_content = base64.b64decode(encoded_content)

            # Save the decoded content into a BytesIO object
            content_io = BytesIO(decoded_content)

            # Assign a filename to the BytesIO object
            content_io.name = kwargs["filename"]
        elif "file_uri" in kwargs:
            content_io = BytesIO(httpx.get(kwargs["file_uri"]).content)
        else:
            raise Exception("No file content provided")

        file = self.client.files.create(file=content_io, purpose=purpose)
        return {
            "id": file.id,
            "object": file.object,
            "filename": file.filename,
            "purpose": file.purpose,
            "created_at": pendulum.from_timestamp(file.created_at, tz="UTC"),
            "bytes": file.bytes,
        }

    def delete_file(self, **kwargs: Dict[str, Any]) -> None:
        result = self.client.files.delete(kwargs["file_id"])
        return result.deleted
