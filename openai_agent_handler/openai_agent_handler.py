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

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Debugger, Serializer
from silvaengine_utility.performance_monitor import performance_monitor


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
        try:
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

            # Persist for reuse in file operations (insert_file, get_output_file)
            self.http_client = http_client

            self.client = openai.OpenAI(
                api_key=self.agent.get("configuration", {}).get("openai_api_key"),
                http_client=http_client,
            )
            if self.agent.get("configuration", {}).get("base_url"):
                self.client.base_url = self.agent["configuration"]["base_url"]

            if "enabled_tools" in self.agent["configuration"]:
                # Filter tools using set-based membership for O(1) lookups
                enabled_set = set(self.agent["configuration"]["enabled_tools"])
                if "tools" in self.agent["configuration"]:
                    self.agent["configuration"]["tools"] = [
                        tool
                        for tool in self.agent["configuration"]["tools"]
                        if tool["name"] in enabled_set
                    ]

            # Build model settings with type conversions (performance optimization)
            self.model_setting = {
                "instructions": self.agent["instructions"],
            }
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

            # Build shell tool with skill references if skills are configured.
            # Supports "local" (local shell execution) and "container_auto" (hosted).
            # Inline skills are not supported — use skill_reference with skill_id.
            if "skills" in self.model_setting:
                skills = self.model_setting.pop("skills")
                env_type = self.model_setting.pop(
                    "skills_environment", "container_auto"
                )

                if not isinstance(skills, list):
                    if self.logger and self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            "Skills configuration should be a list. "
                            "Skills features may not work correctly."
                        )
                elif env_type not in ("local", "container_auto"):
                    if self.logger and self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            f"Unknown skills_environment '{env_type}'. "
                            "Must be 'local' or 'container_auto'. "
                            "Defaulting to 'container_auto'."
                        )
                    env_type = "container_auto"
                else:
                    # Build skill_reference entries (drop any that lack skill_id)
                    skill_refs = []
                    for entry in skills:
                        if not isinstance(entry, dict) or "skill_id" not in entry:
                            if self.logger and self.logger.isEnabledFor(
                                logging.WARNING
                            ):
                                self.logger.warning(
                                    f"Skipping invalid skill entry (missing skill_id): {entry}"
                                )
                            continue
                        ref = {"type": "skill_reference", "skill_id": entry["skill_id"]}
                        if "version" in entry:
                            ref["version"] = entry["version"]
                        skill_refs.append(ref)

                    if skill_refs:
                        shell_tool = {
                            "type": "shell",
                            "environment": {
                                "type": env_type,
                                "skills": skill_refs,
                            },
                        }
                        self.model_setting.setdefault("tools", []).append(shell_tool)

            # Enable/disable timeline logging (default: enabled for backward compatibility)
            self.enable_timeline_log = setting.get("enable_timeline_log", False)

            # Cached code_interpreter tool reference (invalidated on model_setting update)
            self._cached_code_interpreter_tool = None
            self._code_interpreter_cache_valid = False
        except Exception as e:
            Debugger.info(variable=e, stage=f"{__name__}:__init__")
            raise

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
            if self.enable_timeline_log:
                invoke_start = pendulum.now("UTC")

            variables = dict(self.model_setting, **kwargs)

            result = self.client.responses.create(**variables)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_time = (
                    pendulum.now("UTC") - invoke_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    @performance_monitor.monitor_operation(operation_name="OpenAI")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        input_files: Optional[List[Dict[str, Any]]] = None,
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
        # Track preparation time (only when timeline logging is enabled)
        if self.enable_timeline_log:
            ask_model_start = pendulum.now("UTC")

        # Track recursion depth to identify top-level vs recursive calls
        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        # Initialize global start time only on top-level ask_model call
        # Recursive calls will use the same start time for the entire run timeline
        if is_top_level and self.enable_timeline_log:
            self._global_start_time = ask_model_start

            # Reset reasoning_summary for new conversation turn
            # Recursive calls (function call loops) will continue accumulating
            if (
                hasattr(self, "final_output")
                and isinstance(self.final_output, dict)
                and "reasoning_summary" in self.final_output
            ):
                del self.final_output["reasoning_summary"]

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
            input_files = input_files or []

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)
                # Invalidate tool cache since tools may have changed
                self._code_interpreter_cache_valid = False

            if input_files:
                input_messages = self._process_input_files(input_files, input_messages)

            self._process_user_file_ids(input_messages[:-1])

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Track total preparation time before API call
                preparation_time = (
                    pendulum.now("UTC") - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms)"
                )

            response = self.invoke_model(
                **{
                    "input": input_messages,
                    "stream": stream,
                }
            )

            run_id = None

            # If streaming is enabled, process chunks
            if stream:
                # Note: run_id will be sent from handle_stream when response.created event is received
                run_id = self.handle_stream(
                    response,
                    input_messages,
                    queue=queue,
                    stream_event=stream_event,
                )
            else:
                # Otherwise, handle a normal (non-stream) response
                run_id = self.handle_response(response, input_messages)

            return run_id
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

    def _get_code_interpreter_tool(self) -> Optional[Dict[str, Any]]:
        """Return cached code_interpreter tool reference, scanning only on cache miss."""
        if not self._code_interpreter_cache_valid:
            self._cached_code_interpreter_tool = next(
                (
                    tool
                    for tool in self.model_setting.get("tools", [])
                    if tool.get("type") == "code_interpreter"
                ),
                None,
            )
            self._code_interpreter_cache_valid = True
        return self._cached_code_interpreter_tool

    def _attach_files_into_code_interpreter(self, file_ids) -> bool:
        code_interpreter_tool = self._get_code_interpreter_tool()

        if not code_interpreter_tool:
            return False

        # Initialize file_ids list if it doesn't exist
        if "container" not in code_interpreter_tool:
            code_interpreter_tool["container"] = {"type": "auto"}
        if "file_ids" not in code_interpreter_tool["container"]:
            code_interpreter_tool["container"]["file_ids"] = []

        # Append file_ids using incremental set-based dedup (avoids list→set→list rebuild)
        existing = set(code_interpreter_tool["container"]["file_ids"])
        for fid in file_ids:
            if fid not in existing:
                existing.add(fid)
                code_interpreter_tool["container"]["file_ids"].append(fid)

        return True

    def _trim_messages_for_recursion(
        self, input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply sliding window to prevent unbounded message growth during recursive calls.
        Preserves function_call / function_call_output pair integrity at the trim boundary.

        Uses the agent's num_of_messages setting as the window size.
        If not set or messages are within the limit, returns the list unchanged.
        """
        max_messages = self.agent.get("num_of_messages")
        if not max_messages or len(input_messages) <= max_messages:
            return input_messages

        # Trim from the beginning, keeping the most recent messages
        trimmed = input_messages[-max_messages:]

        # If we split a function_call_output from its function_call, drop the orphan
        if trimmed and trimmed[0].get("type") == "function_call_output":
            trimmed = trimmed[1:]

        return trimmed

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
        image_urls = []
        for input_file in input_files:
            if "encoded_image" in input_file:
                image_urls.append(
                    f"data:image/jpeg;base64,{input_file['encoded_image']}"
                )
                continue
            if "image_url" in input_file:
                image_urls.append(input_file["image_url"])
                continue

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
            if file_ids:
                message_content.extend(
                    {"type": "input_file", "file_id": file_id} for file_id in file_ids
                )
            if image_urls:
                message_content.extend(
                    {"type": "input_image", "image_url": image_url}
                    for image_url in image_urls
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
                message_content = Serializer.json_loads(message["content"])
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
        # Track function call timing (only when timeline logging is enabled)
        if self.enable_timeline_log:
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
            function_output, serialized_output = self._execute_function(
                function_call_data, arguments
            )

            # Update conversation history
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call][{function_call_data['name']}] Updating conversation history"
                )
            self._update_conversation_history(
                function_call_data, function_output, input_messages, serialized_output
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
                            "content": Serializer.json_dumps(
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
            arguments = Serializer.json_loads(function_call_data.get("arguments", "{}"))

            return arguments

        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
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
            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "tool_type": function_call_data["type"],
                    "name": function_call_data["name"],
                },
            )

            # Cache JSON serialization to avoid duplicate work (performance optimization)
            arguments_json = Serializer.json_dumps(arguments)

            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "tool_type": function_call_data["type"],
                    "name": function_call_data["name"],
                    "arguments": arguments_json,
                    "status": "in_progress",
                },
            )

            if self.enable_timeline_log:
                function_exec_start = pendulum.now("UTC")

            function_output = agent_function(**arguments)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                function_exec_time = (
                    pendulum.now("UTC") - function_exec_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' executed (took {function_exec_time:.2f}ms)"
                )

            # Serialize once and reuse in both DB write and conversation history
            serialized_output = Serializer.json_dumps(function_output)

            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "content": serialized_output,
                    "status": "completed",
                },
            )
            return function_output, serialized_output

        except Exception as e:
            log = traceback.format_exc()
            # Reuse cached arguments_json (performance optimization)
            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "failed",
                    "notes": log,
                },
            )
            error_msg = f"Function execution failed: {e}"
            return error_msg, Serializer.json_dumps(error_msg)

    def _update_conversation_history(
        self,
        function_call_data: Dict[str, Any],
        function_output: Any,
        input_messages: List[Dict[str, Any]],
        serialized_output: Optional[str] = None,
    ) -> None:
        """
        Updates the conversation history with function call details and output.

        :param function_call_data: Dictionary containing function call metadata.
        :param function_output: Result returned from function execution.
        :param input_messages: List of messages to update with function details.
        :param serialized_output: Pre-serialized output string to avoid re-serialization.
        """
        input_messages.append(function_call_data)
        input_messages.append(
            {
                "type": "function_call_output",
                "call_id": function_call_data["call_id"],
                "output": (
                    serialized_output
                    if serialized_output is not None
                    else Serializer.json_dumps(function_output)
                ),
            }
        )

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> str:
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
                input_messages = self._trim_messages_for_recursion(input_messages)
                self.ask_model(input_messages)
                return response.id

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
            return response.id

        # Scenario 3: Valid response - set final output
        self.final_output.update(
            {
                "message_id": message_id,
                "role": role,
                "content": content,
                "output_files": output_files,
            }
        )

        return response.id

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        retry_count: int = 0,
    ) -> str | None:
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

        run_id = None
        message_id = None
        role = None
        # Use list-based accumulation to avoid O(n^2) string concatenation
        accumulated_partial_reasoning_parts = []
        # Accumulate complete reasoning for the current block
        accumulated_reasoning_block = []
        # Use list for efficient string concatenation (performance optimization)
        accumulated_text_parts = []
        output_files = []
        accumulated_partial_json_parts = []
        accumulated_partial_text_parts = []
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

        if self.enable_timeline_log:
            stream_start_time = pendulum.now("UTC")

        for chunk in response_stream:
            if run_id is None:
                run_id = chunk.response.id
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
                        # Reset accumulated reasoning for new block
                        accumulated_reasoning_block = []

                        if self.enable_timeline_log and self.logger.isEnabledFor(
                            logging.INFO
                        ):
                            elapsed = self._get_elapsed_time()
                            self.logger.info(
                                f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning added"
                            )
                    elif chunk.type == "response.reasoning_summary_text.delta":
                        if self.logger.isEnabledFor(logging.DEBUG):
                            print(chunk.delta, end="", flush=True)

                        # Accumulate for final summary
                        accumulated_reasoning_block.append(chunk.delta)

                        accumulated_partial_reasoning_parts.append(chunk.delta)
                        # Check if text contains XML-style tags and update format
                        reasoning_index, remaining = self.process_text_content(
                            reasoning_index,
                            "".join(accumulated_partial_reasoning_parts),
                            output_format,
                            suffix=f"rs#{reasoning_no}",
                        )
                        accumulated_partial_reasoning_parts = (
                            [remaining] if remaining else []
                        )

                    elif chunk.type == "response.reasoning_summary_text.done":
                        # Send message completion signal to WebSocket server
                        if accumulated_partial_reasoning_parts:
                            self.send_data_to_stream(
                                index=reasoning_index,
                                data_format=output_format,
                                chunk_delta="".join(
                                    accumulated_partial_reasoning_parts
                                ),
                                suffix=f"rs#{reasoning_no}",
                            )
                            accumulated_partial_reasoning_parts = []
                            # reasoning_index += 1
                    elif chunk.type == "response.reasoning_summary_part.done":
                        # Save accumulated reasoning text to final_output
                        # Build the reasoning text from accumulated_reasoning_block list
                        # which has been collecting text from response.reasoning_summary_text.delta events
                        if accumulated_reasoning_block:
                            reasoning_summary = "".join(accumulated_reasoning_block)

                            # Accumulate reasoning summaries from multiple reasoning blocks
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

                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug(
                                    f"Captured reasoning summary: {reasoning_summary[:100]}..."
                                )

                            # Reset for next reasoning block
                            accumulated_reasoning_block = []

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
                if index == 0 and reasoning_index > 0:
                    index = reasoning_index + 1

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

                if self.logger.isEnabledFor(logging.DEBUG):
                    print(chunk.delta, end="", flush=True)

                # Accumulate in list for efficient concatenation (performance optimization)
                accumulated_text_parts.append(chunk.delta)

                # For JSON formats, accumulate partial JSON text and process it
                # when complete JSON objects are detected. This ensures valid JSON
                # is sent to the WebSocket server.
                if output_format in ["json_object", "json_schema"]:
                    accumulated_partial_json_parts.append(chunk.delta)
                    # Temporarily build accumulated_text for processing
                    temp_accumulated_text = "".join(accumulated_text_parts)
                    index, temp_accumulated_text, remaining_json = (
                        self.process_and_send_json(
                            index,
                            temp_accumulated_text,
                            "".join(accumulated_partial_json_parts),
                            output_format,
                        )
                    )
                    accumulated_partial_json_parts = (
                        [remaining_json] if remaining_json else []
                    )
                else:
                    accumulated_partial_text_parts.append(chunk.delta)
                    # Check if text contains XML-style tags and update format
                    index, remaining_text = self.process_text_content(
                        index, "".join(accumulated_partial_text_parts), output_format
                    )
                    accumulated_partial_text_parts = (
                        [remaining_text] if remaining_text else []
                    )
            elif chunk.type == "response.output_text.done":
                # Send message completion signal to WebSocket server
                if accumulated_partial_text_parts:
                    self.send_data_to_stream(
                        index=index,
                        data_format=output_format,
                        chunk_delta="".join(accumulated_partial_text_parts),
                    )
                    accumulated_partial_text_parts = []
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

                # Single-pass processing of final output objects
                if hasattr(chunk.response, "output") and chunk.response.output:
                    has_function_call = False
                    reasoning_item = None
                    last_message_output = None

                    for output in chunk.response.output:
                        output_type = getattr(output, "type", None)

                        if output_type == "mcp_approval_request":
                            raise Exception(
                                "MCP Approval Request is not currently supported"
                            )

                        if output_type == "reasoning":
                            reasoning_item = {
                                "type": "reasoning",
                                "id": output.id,
                                "summary": output.summary,
                            }
                            continue

                        if output_type == "function_call":
                            has_function_call = True
                            if reasoning_item is not None:
                                input_messages.append(reasoning_item)
                            reasoning_item = None

                            input_messages = self.handle_function_call(
                                output, input_messages
                            )
                            continue

                        # Track last message output for metadata extraction
                        if output_type == "message":
                            last_message_output = output

                        # Reset reasoning for non-function-call, non-reasoning types
                        reasoning_item = None

                    if has_function_call:
                        if self.enable_timeline_log and self.logger.isEnabledFor(
                            logging.INFO
                        ):
                            recursive_call_start = pendulum.now("UTC")
                            time_from_stream_start = (
                                recursive_call_start - stream_start_time
                            ).total_seconds() * 1000
                            elapsed = self._get_elapsed_time()
                            self.logger.info(
                                f"[TIMELINE] T+{elapsed:.2f}ms: Starting recursive ask_model ({time_from_stream_start:.2f}ms after stream start)"
                            )

                        input_messages = self._trim_messages_for_recursion(
                            input_messages
                        )
                        self.ask_model(
                            input_messages, queue=queue, stream_event=stream_event
                        )
                        return run_id

                    # Extract metadata from last output item
                    final_output_item = last_message_output or chunk.response.output[-1]
                    if hasattr(final_output_item, "content"):
                        for ann in getattr(
                            final_output_item.content[-1], "annotations", []
                        ):
                            if ann.type == "container_file_citation":
                                output_files.append(
                                    {
                                        "filename": ann.filename,
                                        "file_id": ann.file_id,
                                        "container_id": ann.container_id,
                                    }
                                )

                    message_id = final_output_item.id
                    role = final_output_item.role

        if self.enable_timeline_log:
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
            return run_id

        # Scenario 3: Valid stream - set final output
        self.final_output.update(
            {
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

        return run_id

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
            content_io = BytesIO(self.http_client.get(kwargs["file_uri"]).content)
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
        metadata_response = self.http_client.get(metadata_url, headers=headers)
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
            content_response = self.http_client.get(content_url, headers=headers)
            content_response.raise_for_status()
            content = content_response.content
            file_data["encoded_content"] = base64.b64encode(content).decode("utf-8")

        return file_data

    # ----------------------------
    # Skill Management Methods
    # ----------------------------

    def list_skills(
        self, limit: int = 20, after: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all available skills.

        Args:
            limit: Maximum number of skills to return (default 20)
            after: Cursor for pagination

        Returns:
            Dictionary containing:
                - data: List of skill objects
                - has_more: Boolean indicating if more results exist
                - first_id: ID of first skill in results
                - last_id: ID of last skill in results
        """
        params = {"limit": limit}
        if after:
            params["after"] = after

        skills = self.client.skills.list(**params)

        return {
            "data": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "default_version": skill.default_version,
                    "latest_version": skill.latest_version,
                    "created_at": skill.created_at,
                }
                for skill in skills.data
            ],
            "has_more": skills.has_more,
            "first_id": skills.first_id,
            "last_id": skills.last_id,
        }

    def get_skill(self, skill_id: str) -> Dict[str, Any]:
        """
        Retrieve details about a specific skill.

        Args:
            skill_id: The ID of the skill to retrieve

        Returns:
            Dictionary containing skill details
        """
        skill = self.client.skills.retrieve(skill_id)

        return {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "default_version": skill.default_version,
            "latest_version": skill.latest_version,
            "created_at": skill.created_at,
        }

    def create_skill(
        self,
        name: str,
        files: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new custom skill.

        Args:
            name: Name for the skill
            files: List of file dictionaries with keys:
                - filename: Name of the file
                - encoded_content: Base64-encoded content, or
                - content: Raw bytes content
                - mime_type: Optional MIME type
            description: Optional description for the skill

        Returns:
            Dictionary containing created skill details
        """
        file_tuples = []
        for file_dict in files:
            if "encoded_content" in file_dict:
                content = base64.b64decode(file_dict["encoded_content"])
            else:
                content = file_dict.get("content", b"")

            file_tuple = (
                file_dict["filename"],
                BytesIO(content) if isinstance(content, bytes) else content,
            )
            if "mime_type" in file_dict:
                file_tuple = file_tuple + (file_dict["mime_type"],)
            file_tuples.append(file_tuple)

        params = {"name": name, "files": file_tuples}
        if description:
            params["description"] = description

        skill = self.client.skills.create(**params)

        return {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "default_version": skill.default_version,
            "latest_version": skill.latest_version,
            "created_at": skill.created_at,
        }

    def update_skill(self, skill_id: str, default_version: str) -> Dict[str, Any]:
        """
        Update the default version pointer for a skill.

        Args:
            skill_id: The ID of the skill to update
            default_version: The version identifier to set as default

        Returns:
            Dictionary containing updated skill details
        """
        skill = self.client.skills.update(skill_id, default_version=default_version)

        return {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "default_version": skill.default_version,
            "latest_version": skill.latest_version,
            "created_at": skill.created_at,
        }

    def delete_skill(self, skill_id: str) -> Dict[str, Any]:
        """
        Delete a custom skill.

        Args:
            skill_id: The ID of the skill to delete

        Returns:
            Dictionary confirming deletion

        Note:
            All versions of the skill must be deleted before the skill can be deleted.
        """
        # First delete all versions
        versions = self.client.skills.versions.list(skill_id)
        for version in versions.data:
            self.client.skills.versions.delete(version.version, skill_id=skill_id)

        # Then delete the skill
        self.client.skills.delete(skill_id)

        return {"id": skill_id, "deleted": True}

    def list_skill_versions(
        self, skill_id: str, limit: int = 20, after: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all versions of a skill.

        Args:
            skill_id: The ID of the skill
            limit: Maximum number of versions to return (default 20)
            after: Cursor for pagination

        Returns:
            Dictionary containing:
                - data: List of version objects
                - has_more: Boolean indicating if more results exist
        """
        params = {"limit": limit}
        if after:
            params["after"] = after

        versions = self.client.skills.versions.list(skill_id, **params)

        return {
            "data": [
                {
                    "id": version.id,
                    "version": version.version,
                    "name": version.name,
                    "description": version.description,
                    "skill_id": version.skill_id,
                    "created_at": version.created_at,
                }
                for version in versions.data
            ],
            "has_more": versions.has_more,
        }

    def get_skill_version(self, skill_id: str, version: str) -> Dict[str, Any]:
        """
        Retrieve details about a specific skill version.

        Args:
            skill_id: The ID of the skill
            version: The version identifier

        Returns:
            Dictionary containing version details
        """
        skill_version = self.client.skills.versions.retrieve(version, skill_id=skill_id)

        return {
            "id": skill_version.id,
            "version": skill_version.version,
            "name": skill_version.name,
            "description": skill_version.description,
            "skill_id": skill_version.skill_id,
            "created_at": skill_version.created_at,
        }

    def create_skill_version(
        self, skill_id: str, files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new immutable version of an existing skill.

        Args:
            skill_id: The ID of the skill to update
            files: List of file dictionaries with keys:
                - filename: Name of the file
                - encoded_content: Base64-encoded content, or
                - content: Raw bytes content
                - mime_type: Optional MIME type

        Returns:
            Dictionary containing created version details
        """
        file_tuples = []
        for file_dict in files:
            if "encoded_content" in file_dict:
                content = base64.b64decode(file_dict["encoded_content"])
            else:
                content = file_dict.get("content", b"")

            file_tuple = (
                file_dict["filename"],
                BytesIO(content) if isinstance(content, bytes) else content,
            )
            if "mime_type" in file_dict:
                file_tuple = file_tuple + (file_dict["mime_type"],)
            file_tuples.append(file_tuple)

        version = self.client.skills.versions.create(skill_id, files=file_tuples)

        return {
            "id": version.id,
            "version": version.version,
            "name": version.name,
            "description": version.description,
            "skill_id": version.skill_id,
            "created_at": version.created_at,
        }

    def delete_skill_version(self, skill_id: str, version: str) -> Dict[str, Any]:
        """
        Delete a specific version of a skill.

        Args:
            skill_id: The ID of the skill
            version: The version to delete

        Returns:
            Dictionary confirming deletion
        """
        self.client.skills.versions.delete(version, skill_id=skill_id)

        return {"skill_id": skill_id, "version": version, "deleted": True}

    def get_skill_content(self, skill_id: str) -> Dict[str, Any]:
        """
        Download the binary bundle for a skill.

        Args:
            skill_id: The ID of the skill

        Returns:
            Dictionary containing:
                - skill_id: The skill ID
                - encoded_content: Base64-encoded skill bundle
        """
        content = self.client.skills.content.retrieve(skill_id)

        return {
            "skill_id": skill_id,
            "encoded_content": base64.b64encode(content).decode("utf-8"),
        }

    def get_skill_version_content(self, skill_id: str, version: str) -> Dict[str, Any]:
        """
        Download the binary bundle for a specific skill version.

        Args:
            skill_id: The ID of the skill
            version: The version identifier

        Returns:
            Dictionary containing:
                - skill_id: The skill ID
                - version: The version identifier
                - encoded_content: Base64-encoded version bundle
        """
        content = self.client.skills.versions.content.retrieve(
            version, skill_id=skill_id
        )

        return {
            "skill_id": skill_id,
            "version": version,
            "encoded_content": base64.b64encode(content).decode("utf-8"),
        }
