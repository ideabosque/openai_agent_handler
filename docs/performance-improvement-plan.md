# OpenAI Agent Handler - Performance Improvement Development Plan

## 1. Overview and Context

**Module**: `openai_agent_handler/openai_agent_handler.py`  
**Class**: `OpenAIEventHandler` (extends `AIAgentEventHandler`)  
**Purpose**: Stateless multi-turn AI orchestration with tool integration, streaming, and threading for OpenAI GPT models.

### Why these optimizations matter

The handler is in the critical path for every interaction. A request flows through `ask_model()` -> `invoke_model()` -> `handle_stream()` or `handle_response()`. Tool-call loops recurse back into `ask_model()`, which amplifies inefficiencies.

Top concerns now are not only micro-performance, but also correctness and operability:
- Recursion depth state (`_ask_model_depth`) is incremented but not decremented.
- The current regression test guidance depends on interactive tests.
- Streaming prints each chunk with `flush=True`, which can dominate latency.
- Several hot-path allocations and repeated scans are still present.

---

## 2. Phase 0 - Correctness and Reliability First

### 2.1 Fix `_ask_model_depth` lifecycle

**Problem**: `ask_model()` increments `_ask_model_depth` but the decrement/reset logic is commented out.  
**Current locations**:
- Increment: `openai_agent_handler/openai_agent_handler.py:320`
- Decrement/reset block commented: `openai_agent_handler/openai_agent_handler.py:326`

**Plan**:
- Restore a `finally` block to decrement depth.
- Reset timeline only when returning to top-level.
- Ensure recursive calls preserve the same run timeline.

**Expected impact**:
- Correct top-level vs recursive classification.
- Prevent depth drift across long-lived handler instances.

### 2.2 Remove mutable default argument in `ask_model`

**Problem**: `input_files: List[str] = []` is shared across calls.  
**Location**: `openai_agent_handler/openai_agent_handler.py:219`

**Plan**:
- Change signature to `input_files: Optional[List[Dict[str, Any]]] = None`.
- Inside method: `input_files = input_files or []`.

**Expected impact**:
- Eliminates state leakage risk and hard-to-debug behavior.

### 2.3 Replace interactive test guidance with deterministic tests

**Problem**: Existing regression tests include interactive loops and are not CI-safe.  
**Location**: `openai_agent_handler/tests/test_openai_agent_handler.py`

**Plan**:
- Add non-interactive unit tests with mocked client responses.
- Keep interactive scripts as manual examples only.
- Update verification section to reference deterministic tests first.

**Expected impact**:
- Repeatable validation and safer refactors.

---

## 3. Phase 1 - High Impact Performance Optimizations

### 3.1 Reduce async DB write amplification in tool calls

**Problem**: Function call lifecycle performs separate write calls for start and in-progress states.  
**Current locations**:
- Start record: `_record_function_call_start` at `openai_agent_handler/openai_agent_handler.py:561`
- In-progress record: `_execute_function` at `openai_agent_handler/openai_agent_handler.py:631`

**Plan**:
- Merge start + in-progress into one `async_insert_update_tool_call` invocation.
- Remove standalone `_record_function_call_start()` call from `handle_function_call`.

**Expected impact**:
- Happy path DB writes per function call reduced from 3 to 2.

### 3.2 Eliminate per-chunk synchronous console flush in streaming

**Problem**: Streaming path prints every chunk with `flush=True`.  
**Current locations**:
- Reasoning chunk: `openai_agent_handler/openai_agent_handler.py:915`
- Text chunk: `openai_agent_handler/openai_agent_handler.py:1006`

**Plan**:
- Remove unconditional prints, or gate under explicit debug setting.
- Keep normal output delivery through `send_data_to_stream()`.

**Expected impact**:
- Lower latency and less CPU overhead in streaming mode.

### 3.3 Fix string concatenation in streaming hot path

**Problem**: `+=` is used for partial accumulators:
- `accumulated_partial_reasoning_text`
- `accumulated_partial_json`
- `accumulated_partial_text`

**Current locations**:
- `openai_agent_handler/openai_agent_handler.py:920`
- `openai_agent_handler/openai_agent_handler.py:1015`
- `openai_agent_handler/openai_agent_handler.py:1027`

**Plan**:
- Replace with list append + join strategy where safe.
- Validate compatibility with parent methods:
  - `process_text_content`
  - `process_and_send_json`

**Expected impact**:
- Avoid O(n^2) copying behavior for long streams.

### 3.4 Single-pass processing for `response.completed`

**Problem**: Output list is scanned multiple times in completion handling.  
**Current block**: `openai_agent_handler/openai_agent_handler.py:1066-1125`

**Plan**:
- Use one pass with flags for:
  - function calls
  - MCP approval request
  - reasoning carryover

**Expected impact**:
- Lower iteration overhead and cleaner control flow.

---

## 4. Phase 2 - Medium Impact Improvements

### 4.1 Move timing capture behind timeline flag where possible

**Problem**: `pendulum.now("UTC")` is called even when timeline logging is disabled.

**Plan**:
- Guard timing-only timestamps in:
  - `invoke_model`
  - `handle_function_call`
  - `_execute_function`
  - `handle_stream` timing-only checkpoints
- Keep timestamps that are functionally required.

### 4.2 Replace mixed HTTP usage with pooled `httpx.Client`

**Problem**:
- `requests.get` in `get_output_file`
- one-off `httpx.get` in `insert_file`

**Current locations**:
- `openai_agent_handler/openai_agent_handler.py:1247`
- `openai_agent_handler/openai_agent_handler.py:1277`
- `openai_agent_handler/openai_agent_handler.py:1294`

**Plan**:
- Persist the existing `httpx.Client` as `self.http_client`.
- Use it for all file fetches and metadata/content retrieval.
- Remove `requests` import.

### 4.3 Reduce repeated JSON serialization for function output

**Problem**: `function_output` is JSON-serialized more than once across lifecycle.

**Plan**:
- Serialize once in `_execute_function`.
- Reuse serialized value when updating conversation history.

### 4.4 Improve file-id dedup strategy

**Problem**: list -> set -> list recreation on every attach.  
**Location**: `openai_agent_handler/openai_agent_handler.py:360`

**Plan**:
- Use incremental dedup with an `existing_set`.

### 4.5 Improve `enabled_tools` filtering to set-based membership

**Problem**: repeated list membership checks in initialization.  
**Location**: `openai_agent_handler/openai_agent_handler.py:79`

**Plan**:
- Convert enabled names to a set once.

---

## 5. Phase 3 - Design-Level and Conditional Optimizations

### 5.1 Tool lookup caching with invalidation rules

**Problem**: repeated linear scan for `code_interpreter` tool.

**Important constraint**:
- `self.model_setting` may be mutated at runtime via `model_setting.update(...)` in `ask_model`, so a one-time cache can become stale.

**Plan options**:
- Option A: no cache, keep scan (safe baseline).
- Option B: cache with explicit rebuild whenever `tools` is updated.

### 5.2 Message history growth strategy

**Problem**: recursive loops grow `input_messages`; copy-on-update remains linear.

**Plan**:
- Define a sliding-window policy before recursive calls.
- Ensure tool-call integrity is preserved when truncating.

### 5.3 Async file I/O migration (optional)

**Problem**: file operations are synchronous.

**Plan**:
- Defer full `AsyncClient` migration unless profiling shows this is a bottleneck.

---

## 6. Updated Implementation Sequence

| Order | Item | Risk | Effort | Notes |
|------|------|------|------|------|
| 1 | Fix `_ask_model_depth` lifecycle (2.1) | Medium | 30-45 min | Correctness blocker |
| 2 | Fix mutable default arg (2.2) | Low | 10 min | Correctness blocker |
| 3 | Remove/gate chunk `print(..., flush=True)` (3.2) | Low | 15-30 min | High runtime impact |
| 4 | Add deterministic unit tests (2.3) | Medium | 1-2 hr | Required for safe refactor |
| 5 | Reduce DB write amplification (3.1) | Medium | 45-75 min | Validate downstream expectations |
| 6 | String accumulation improvements (3.3) | Medium | 45-75 min | Validate parent API behavior |
| 7 | Single-pass completion processing (3.4) | Medium | 45 min | Behavior-sensitive |
| 8 | Replace mixed HTTP calls with pooled `httpx` (4.2) | Low | 30-45 min | Also removes `requests` |
| 9 | Timing capture guards (4.1) | Low | 20-30 min | Micro-optimization |
| 10 | Serialization and dedup cleanups (4.3, 4.4, 4.5) | Low | 30-45 min | Independent |
| 11 | Tool cache with invalidation (5.1) | Medium | 30-60 min | Only if profiling justifies |
| 12 | Message-window strategy (5.2) | Medium | 1-2 hr | Design decision |

---

## 7. Verification and Testing Strategy

### 7.1 Deterministic unit tests (primary)

Add non-interactive tests that mock model events and tool execution:
1. Depth lifecycle test:
   - ensure `_ask_model_depth` returns to original value after success and exception paths.
2. Streaming output test:
   - verify final content and reasoning handling with mocked chunk sequences.
3. Tool-call persistence test:
   - assert expected `invoke_async_funct` call count and payload transitions.
4. Completion parser test:
   - mixed outputs: reasoning, function_call, mcp_approval_request.
5. File operation test:
   - verify `self.http_client` usage and response handling.

### 7.2 Performance benchmark harness (secondary)

Use a fixed synthetic stream and fixed tool-call workload:
- Measure median/p95 wall time across N runs.
- Count async tool-call persistence writes.
- Track allocations with `tracemalloc` around `handle_stream`.

### 7.3 Manual smoke checks

Keep interactive test scripts for manual validation only; do not treat them as regression gates.

---

## 8. Risk Assessment

| Change | Risk | Mitigation |
|------|------|------|
| Depth lifecycle fix | Recursive flow/timeline behavior change | Add explicit tests for top-level and nested calls |
| DB write consolidation | Downstream may depend on separate start event | Validate consumers of `async_insert_update_tool_call` |
| String accumulation refactor | Parent methods may assume string semantics | Refactor incrementally with unit coverage |
| Single-pass completion parser | Mixed output edge cases | Test all known output combinations |
| HTTP client unification | Behavioral differences across clients | Validate status/error/content handling paths |
| Tool cache | Stale cache after runtime config updates | Add invalidation or avoid cache |
