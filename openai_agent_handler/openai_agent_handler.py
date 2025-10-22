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
import requests

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

        # Configure HTTP client with connection pooling and keep-alive for better performance
        # This significantly reduces the connection setup time between consecutive API calls
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=100,  # Allow up to 100 concurrent connections
                max_keepalive_connections=20,  # Keep 20 connections alive for reuse
                keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
            ),
            timeout=httpx.Timeout(
                120.0, connect=10.0
            ),  # 10s connect, 120s total timeout
            http2=True,  # Enable HTTP/2 for better performance
        )

        self.client = openai.OpenAI(
            api_key=self.agent["configuration"].get("openai_api_key"),
            http_client=http_client,
        )

        # Build model settings with type conversions (performance optimization)
        self.model_setting = {"instructions": self.agent["instructions"]}
        for k, v in self.agent["configuration"].items():
            if k in ["openai_api_key"]:
                continue

            if k == "max_output_tokens":
                v = int(v)
            elif k in ["temperature", "top_p"]:
                v = float(v)
            # Convert Decimal to float for better performance
            elif isinstance(v, Decimal):
                v = float(v)

            self.model_setting[k] = v

        # Cache frequently accessed configuration values (performance optimization)
        self.output_format_type = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )

        # Validate reasoning configuration if present
        if "reasoning" in self.model_setting:
            if not isinstance(self.model_setting["reasoning"], dict):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning configuration should be a dictionary. "
                        "Reasoning features may not work correctly."
                    )
            elif self.model_setting["reasoning"].get("summary") is None:
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning summary is not enabled in configuration. "
                        "Reasoning events will be skipped during streaming."
                    )

        # Enable/disable timeline logging (default: enabled for backward compatibility)
        self.enable_timeline_log = setting.get("enable_timeline_log", True)

    def _check_retry_limit(self, retry_count: int) -> None:
        """
        Check if retry limit has been exceeded and raise exception if so.

        Args:
            retry_count: Current retry count

        Raises:
            Exception: If retry_count exceeds MAX_RETRIES
        """
        MAX_RETRIES = 3
        if retry_count > MAX_RETRIES:
            error_msg = (
                f"Maximum retry limit ({MAX_RETRIES}) exceeded for empty responses"
            )
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(error_msg)
            raise Exception(error_msg)

    def _has_valid_content(self, text: str) -> bool:
        """
        Check if response text contains valid content.

        Args:
            text: Response text to check

        Returns:
            True if text is not None/empty/whitespace-only, False otherwise
        """
        return bool(text and text.strip())

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time in milliseconds from the first ask_model call.

        Returns:
            Elapsed time in milliseconds, or 0 if global start time not set
        """
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        """
        Reset the global timeline for a new run.
        Should be called at the start of each new user interaction/run.
        """
        self._global_start_time = None
        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("[TIMELINE] Timeline reset for new run")

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Makes an API call to OpenAI with provided arguments.

        :param kwargs: Dictionary of arguments to pass to the OpenAI API.
        :return: Response from OpenAI API.
        :raises: Exception if API call fails or returns error.
        """
        try:
            invoke_start = pendulum.now("UTC")
            variables = dict(self.model_setting, **kwargs)

            result = self.client.responses.create(**variables)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    @Utility.performance_monitor.monitor_operation(operation_name="OpenAI")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        input_files: List[str] = [],
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
        # Track preparation time
        ask_model_start = pendulum.now("UTC")

        # Track recursion depth to identify top-level vs recursive calls
        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        # Initialize global start time only on top-level ask_model call
        # Recursive calls will use the same start time for the entire run timeline
        if is_top_level:
            self._global_start_time = ask_model_start
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                self.logger.info("[TIMELINE] T+0ms: Run started - First ask_model call")
        else:
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started"
                )

        try:
            if not self.client:
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error("No OpenAI client provided.")
                return None

            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            if input_files:
                input_messages = self._process_input_files(input_files, input_messages)

            self._process_user_file_ids(input_messages[:-1])

            # Clean up input messages to remove broken tool sequences (performance optimization)
            cleanup_start = pendulum.now("UTC")
            cleanup_end = pendulum.now("UTC")
            cleanup_time = (cleanup_end - cleanup_start).total_seconds() * 1000

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Track total preparation time before API call
                preparation_end = pendulum.now("UTC")
                preparation_time = (
                    preparation_end - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms, cleanup: {cleanup_time:.2f}ms)"
                )

            response = self.invoke_model(
                **{
                    "input": input_messages,
                    "stream": stream,
                }
            )

            # If streaming is enabled, process chunks
            if stream:
                # Note: run_id will be sent from handle_stream when response.created event is received
                self.handle_stream(
                    response,
                    input_messages,
                    queue=queue,
                    stream_event=stream_event,
                )
                result = None
            else:
                # Otherwise, handle a normal (non-stream) response
                self.handle_response(response, input_messages)
                result = response.id

            return result

        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")
        finally:
            # Decrement depth when exiting ask_model
            self._ask_model_depth -= 1

            # Reset timeline when returning to depth 0 (top-level call complete)
            if self._ask_model_depth == 0:
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Run complete - Resetting timeline"
                    )
                self._global_start_time = None

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
        file_ids = []
        for input_file in input_files:
            file_data = dict(input_file, purpose="user_data")
            uploaded_file = self.insert_file(**file_data)
            file_ids.append(uploaded_file["id"])
            self.uploaded_files.append({"file_id": uploaded_file["id"]})

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
        user_messages = [msg for msg in input_messages if msg.get("role") == "user"]

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
    ) -> List[Dict[str, Any]]:
        """
        Processes function calls from the model including validation, execution and history updates.

        :param tool_call: Object containing function call details from model.
        :param input_messages: Conversation history to update with function results.
        :return: Updated input messages list.
        :raises: ValueError if tool_call is invalid.
                Exception if function execution fails.
        """
        # Track function call timing
        function_call_start = pendulum.now("UTC")

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

            # Record initial function call (with conditional logging)
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Starting function call recording for {function_call_data['name']}"
                )
            self._record_function_call_start(function_call_data)

            # Parse and process arguments
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Processing arguments for function {function_call_data['name']}"
                )
            arguments = self._process_function_arguments(function_call_data)

            # Execute function and handle result
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Executing function {function_call_data['name']} with arguments {arguments}"
                )
            function_output = self._execute_function(function_call_data, arguments)

            # Update conversation history
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call][{function_call_data['name']}] Updating conversation history"
                )
            self._update_conversation_history(
                function_call_data, function_output, input_messages
            )

            # Continue conversation
            if self.logger.isEnabledFor(logging.INFO):
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

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Log function call execution time
                function_call_end = pendulum.now("UTC")
                function_call_time = (
                    function_call_end - function_call_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' complete (took {function_call_time:.2f}ms)"
                )

            return input_messages

        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
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
            if self.logger.isEnabledFor(logging.ERROR):
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
            # Cache JSON serialization to avoid duplicate work (performance optimization)
            arguments_json = Utility.json_dumps(arguments)

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "in_progress",
                },
            )

            # Track actual function execution time
            function_exec_start = pendulum.now("UTC")
            function_output = agent_function(**arguments)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                function_exec_end = pendulum.now("UTC")
                function_exec_time = (
                    function_exec_end - function_exec_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' executed (took {function_exec_time:.2f}ms)"
                )

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
            # Reuse cached arguments_json (performance optimization)
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
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
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> None:
        """
        Processes model responses and routes them to appropriate handlers.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty response → Retry up to 5 times
        3. Valid response → Set final_output

        :param response: Response object from the model.
        :param input_messages: Current conversation history.
        :param retry_count: Current retry count (max 5 retries).
        """
        self._check_retry_limit(retry_count)

        message_id = None
        role = None
        content = ""
        output_files = []
        reasoning_item = None

        for output in response.output:
            # Handle reasoning - store it and continue
            if output.type == "reasoning":
                reasoning_item = {
                    "type": "reasoning",
                    "id": output.id,
                    "summary": output.summary,
                }

                try:
                    self.final_output["reasoning_summary"] = "\n".join(
                        [summary.text for summary in output.summary]
                    )
                except Exception as e:
                    if self.logger.isEnabledFor(logging.ERROR):
                        self.logger.error(f"Failed to process reasoning summary: {e}")
                    self.final_output["reasoning_summary"] = (
                        "Error processing reasoning summary"
                    )
                continue

            # Handle function_call - add reasoning if exists, then process
            if output.type == "function_call":
                if reasoning_item is not None:
                    input_messages.append(reasoning_item)
                reasoning_item = None

                input_messages = self.handle_function_call(
                    output,
                    input_messages,
                )
                self.ask_model(input_messages)
                return

            # For all other types, add reasoning if exists then reset
            if reasoning_item is not None:
                input_messages.append(reasoning_item)
            reasoning_item = None

            if output.type == "message" and output.status == "completed":
                message_id = output.id if message_id is None else message_id
                role = output.role if role is None else role
                content = content + output.content[0].text
                for ann in getattr(output.content[0], "annotations", []):
                    if ann.type == "container_file_citation":
                        output_files.append(
                            {
                                "filename": ann.filename,
                                "file_id": ann.file_id,
                                "container_id": ann.container_id,
                            }
                        )
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

        # Scenario 2: Empty response - retry (performance optimization)
        if not self._has_valid_content(content):
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
                )
            next_response = self.invoke_model(
                **{"input": input_messages, "stream": False}
            )
            self.handle_response(
                next_response, input_messages, retry_count=retry_count + 1
            )
            return

        # Scenario 3: Valid response - set final output
        self.final_output = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "output_files": output_files,
        }

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        retry_count: int = 0,
    ) -> None:
        """
        Processes streaming responses from the model chunk by chunk.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty stream → Retry up to 5 times
        3. Valid stream → Accumulate and set final_output

        :param response_stream: Iterator of response chunks from the model.
        :param input_messages: Current conversation history.
        :param stream_event: Event to signal when streaming is complete.
        :param retry_count: Current retry count (max 5 retries).
        """
        self._check_retry_limit(retry_count)

        message_id = None
        role = None
        accumulated_partial_reasoning_text = ""
        # Use list for efficient string concatenation (performance optimization)
        accumulated_text_parts = []
        output_files = []
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        received_any_content = False
        # Use cached output format type (performance optimization)
        output_format = self.output_format_type

        # Index variables for tracking stream positions:
        # - reasoning_no: Unique ID for each complete reasoning block (increments after reasoning_summary_part.done)
        # - reasoning_index: Chunk position within the current reasoning block (resets at reasoning_summary_part.added)
        # - index: Chunk position for regular message content (increments with content_part events)
        reasoning_no = 0
        reasoning_index = 0
        index = 0

        stream_start_time = pendulum.now("UTC")

        for chunk in response_stream:
            if chunk.type != "response.output_text.delta":
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Chunk type: {getattr(chunk, 'type', 'N/A')}")
                    self.logger.debug(f"Chunk attributes: {vars(chunk)}")

                # Track reasoning events timing
                if "reasoning" in chunk.type.lower():
                    if self.model_setting.get("reasoning", {}).get("summary") is None:
                        self.logger.warning(
                            "Reasoning summary is not enabled, skipping reasoning events"
                        )
                        continue

                    if self.logger.isEnabledFor(logging.DEBUG):
                        reasoning_event_time = pendulum.now("UTC")
                        time_to_reasoning = (
                            reasoning_event_time - stream_start_time
                        ).total_seconds() * 1000
                        self.logger.info(
                            f"[handle_stream] Reasoning event '{chunk.type}' received at: {time_to_reasoning:.2f}ms"
                        )

                    if chunk.type == "response.reasoning_summary_part.added":
                        # Send initial message start signal to WebSocket server
                        if reasoning_index != 0:
                            reasoning_index = 0
                        self.send_data_to_stream(
                            index=reasoning_index,
                            data_format=output_format,
                            chunk_delta=f"<ReasoningStart Id={reasoning_no}/>",
                            suffix=f"rs#{reasoning_no}",
                        )
                        reasoning_index += 1

                        if self.enable_timeline_log and self.logger.isEnabledFor(
                            logging.INFO
                        ):
                            elapsed = self._get_elapsed_time()
                            self.logger.info(
                                f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning added"
                            )
                    elif chunk.type == "response.reasoning_summary_text.delta":
                        print(chunk.delta, end="", flush=True)

                        accumulated_partial_reasoning_text += chunk.delta
                        # Check if text contains XML-style tags and update format
                        reasoning_index, accumulated_partial_reasoning_text = (
                            self.process_text_content(
                                reasoning_index,
                                accumulated_partial_reasoning_text,
                                output_format,
                                suffix=f"rs#{reasoning_no}",
                            )
                        )

                    elif chunk.type == "response.reasoning_summary_text.done":
                        # Send message completion signal to WebSocket server
                        if len(accumulated_partial_reasoning_text) > 0:
                            self.send_data_to_stream(
                                index=reasoning_index,
                                data_format=output_format,
                                chunk_delta=accumulated_partial_reasoning_text,
                                suffix=f"rs#{reasoning_no}",
                            )
                            accumulated_partial_reasoning_text = ""
                            reasoning_index += 1
                    elif chunk.type == "response.reasoning_summary_part.done":
                        # Send message completion signal to WebSocket server
                        self.send_data_to_stream(
                            index=reasoning_index,
                            data_format=output_format,
                            chunk_delta=f"<ReasoningEnd Id={reasoning_no}/>",
                            is_message_end=True,
                            suffix=f"rs#{reasoning_no}",
                        )
                        reasoning_no += 1

                        if self.enable_timeline_log and self.logger.isEnabledFor(
                            logging.INFO
                        ):
                            elapsed = self._get_elapsed_time()
                            self.logger.info(
                                f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning done"
                            )

            # If the model run has just started
            if chunk.type == "response.created":
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Stream created, run_id={chunk.response.id}"
                    )
                # Send run_id to queue for client notification
                if queue:
                    queue.put({"name": "run_id", "value": chunk.response.id})

            elif chunk.type == "response.output_item.added":
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(f"[TIMELINE] T+{elapsed:.2f}ms: Output item added")
            elif chunk.type == "response.content_part.added":
                # Send initial message start signal to WebSocket server
                self.send_data_to_stream(
                    index=index,
                    data_format=output_format,
                )
                index += 1
            # If we received partial text data
            elif chunk.type == "response.output_text.delta":
                received_any_content = True

                print(chunk.delta, end="", flush=True)

                # Accumulate in list for efficient concatenation (performance optimization)
                accumulated_text_parts.append(chunk.delta)

                # For JSON formats, accumulate partial JSON text and process it
                # when complete JSON objects are detected. This ensures valid JSON
                # is sent to the WebSocket server.
                if output_format in ["json_object", "json_schema"]:
                    accumulated_partial_json += chunk.delta
                    # Temporarily build accumulated_text for processing
                    temp_accumulated_text = "".join(accumulated_text_parts)
                    index, temp_accumulated_text, accumulated_partial_json = (
                        self.process_and_send_json(
                            index,
                            temp_accumulated_text,
                            accumulated_partial_json,
                            output_format,
                        )
                    )
                else:
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
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(f"[TIMELINE] T+{elapsed:.2f}ms: Output item done")
            elif chunk.type == "response.completed":
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    # Log when response.completed event is received
                    response_completed_time = pendulum.now("UTC")
                    time_to_completion = (
                        response_completed_time - stream_start_time
                    ).total_seconds() * 1000
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Stream completed, run_id={chunk.response.id} (took {time_to_completion:.2f}ms from stream start)"
                    )

                # Process any final output objects in chunk.response
                if hasattr(chunk.response, "output") and chunk.response.output:
                    # Check if output is a function call or message
                    if any(
                        output.type == "function_call"
                        for output in chunk.response.output
                        if hasattr(output, "type")
                    ):
                        reasoning_item = None
                        for output in chunk.response.output:
                            # Handle reasoning - store it and continue
                            if output.type == "reasoning":
                                reasoning_item = {
                                    "type": "reasoning",
                                    "id": output.id,
                                    "summary": output.summary,
                                }

                                try:
                                    reasoning_summary = "\n".join(
                                        [summary.text for summary in output.summary]
                                    )

                                    # Accumulate reasoning summaries from multiple function call rounds
                                    # Note: Individual reasoning chunks are already sent via send_data_to_stream
                                    # This accumulation is for the final_output record and function call context
                                    if self.final_output.get("reasoning_summary"):
                                        self.final_output["reasoning_summary"] = (
                                            self.final_output["reasoning_summary"]
                                            + "\n"
                                            + reasoning_summary
                                        )
                                    else:
                                        self.final_output["reasoning_summary"] = (
                                            reasoning_summary
                                        )
                                except Exception as e:
                                    if self.logger.isEnabledFor(logging.ERROR):
                                        self.logger.error(
                                            f"Failed to process reasoning summary in stream: {e}"
                                        )
                                    if not self.final_output.get("reasoning_summary"):
                                        self.final_output["reasoning_summary"] = (
                                            "Error processing reasoning summary"
                                        )
                                continue

                            # Handle function_call - add reasoning if exists, then process
                            if output.type == "function_call":
                                if reasoning_item is not None:
                                    input_messages.append(reasoning_item)
                                reasoning_item = None

                                input_messages = self.handle_function_call(
                                    output, input_messages
                                )
                                continue

                            # For all other types, reset reasoning
                            reasoning_item = None

                        if self.enable_timeline_log and self.logger.isEnabledFor(
                            logging.INFO
                        ):
                            # Log time before recursive ask_model call after function execution
                            recursive_call_start = pendulum.now("UTC")
                            time_from_stream_start = (
                                recursive_call_start - stream_start_time
                            ).total_seconds() * 1000
                            elapsed = self._get_elapsed_time()
                            self.logger.info(
                                f"[TIMELINE] T+{elapsed:.2f}ms: Starting recursive ask_model ({time_from_stream_start:.2f}ms after stream start)"
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

                    if hasattr(chunk.response.output[-1], "content"):
                        for ann in getattr(
                            chunk.response.output[-1].content[-1], "annotations", []
                        ):
                            if ann.type == "container_file_citation":
                                output_files.append(
                                    {
                                        "filename": ann.filename,
                                        "file_id": ann.file_id,
                                        "container_id": ann.container_id,
                                    }
                                )

                    message_id = chunk.response.output[-1].id
                    role = chunk.response.output[-1].role

        # Log start of post-processing
        post_processing_start = pendulum.now("UTC")

        # Build final accumulated text from parts (performance optimization)
        final_accumulated_text = "".join(accumulated_text_parts)

        # Scenario 2: Empty stream - retry (performance optimization)
        if not received_any_content:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
                )
            next_response = self.invoke_model(
                **{"input": input_messages, "stream": True}
            )
            self.handle_stream(
                next_response,
                input_messages,
                queue=queue,
                stream_event=stream_event,
                retry_count=retry_count + 1,
            )
            return

        # Scenario 3: Valid stream - set final output
        self.final_output = dict(
            self.final_output,
            **{
                "message_id": message_id,
                "role": role,
                "content": final_accumulated_text,
                "output_files": output_files,
            },
        )

        # Store accumulated_text for backward compatibility
        self.accumulated_text = final_accumulated_text

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()

        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            # Log post-processing time
            post_processing_end = pendulum.now("UTC")
            post_processing_time = (
                post_processing_end - post_processing_start
            ).total_seconds() * 1000
            elapsed = self._get_elapsed_time()
            self.logger.info(
                f"[TIMELINE] T+{elapsed:.2f}ms: Post-processing complete (took {post_processing_time:.2f}ms)"
            )

    def get_file(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        file = self.client.files.retrieve(kwargs["file_id"])
        uploaded_file = {
            "id": file.id,
            "object": file.object,
            "filename": file.filename,
            "purpose": file.purpose,
            "created_at": pendulum.from_timestamp(file.created_at, tz="UTC"),
            "bytes": file.bytes,
        }
        if "encoded_content" in kwargs and kwargs["encoded_content"]:
            response = self.client.files.content(kwargs["file_id"])
            content = response.content  # Get the actual bytes data)
            # Convert the content to a Base64-encoded string
            uploaded_file["encoded_content"] = base64.b64encode(content).decode("utf-8")

        return uploaded_file

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
            content_io.name = kwargs["filename"]
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

    def delete_file(self, **kwargs: Dict[str, Any]) -> bool:
        result = self.client.files.delete(kwargs["file_id"])
        return result.deleted

    def get_output_file(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get container file metadata and optionally content"""
        container_id = kwargs["container_id"]
        file_id = kwargs["file_id"]
        api_key = self.agent["configuration"].get("openai_api_key")
        headers = {"Authorization": f"Bearer {api_key}"}

        # Get file metadata
        metadata_url = (
            f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}"
        )
        metadata_response = requests.get(metadata_url, headers=headers)
        metadata_response.raise_for_status()
        file = metadata_response.json()

        file_data = {
            "id": file["id"],
            "object": file["object"],
            "created_at": pendulum.from_timestamp(file["created_at"], tz="UTC"),
            "bytes": file["bytes"],
            "container_id": file["container_id"],
            "path": file["path"],
            "source": file["source"],
        }

        # Get file content if requested
        if "encoded_content" in kwargs and kwargs["encoded_content"]:
            content_url = f"{metadata_url}/content"
            content_response = requests.get(content_url, headers=headers)
            content_response.raise_for_status()
            content = content_response.content
            file_data["encoded_content"] = base64.b64encode(content).decode("utf-8")

        return file_data
