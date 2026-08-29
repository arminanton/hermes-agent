# Recorded Copilot response fixtures

These files are verbatim captures from `https://api.githubcopilot.com/chat/completions`,
written directly from the parsed HTTP response with no hand editing. They exist to pin
an undocumented wire contract that Hermes depends on.

## Why they exist

GitHub Copilot returns Gemini's reasoning under a key named `reasoning_text`, which is
not part of the OpenAI schema and is not documented by GitHub. Because the OpenAI SDK
does not know the field, it lands in `model_extra`, and every Hermes reader (which looked
only at `reasoning`, `reasoning_content`, and `reasoning_details`) silently discarded it.
The model produced reasoning, the account was billed for the tokens (`usage.reasoning_tokens`
is non-zero in every capture here), and nothing was surfaced or persisted.

Note what the captures prove: in each recorded message the standard `reasoning` and
`reasoning_content` keys are entirely absent, and only `reasoning_text` is present. That
absence is the bug.

## Provenance

Captured on 2026-08-29 against model `gemini-3.5-flash` with
`reasoning: {effort: high, summary: detailed}`. That model id was confirmed present in the
live `GET https://api.githubcopilot.com/models` catalog at capture time, alongside
`gemini-3.1-pro-preview`, `gemini-3.6-flash`, and `gemini-3.7-flash`. Request headers used
the standard Hermes Copilot identity (`copilot-integration-id: copilot-developer-cli`,
`editor-version: copilot/1.0.81-6`, `x-github-api-version: 2026-08-01`).

`gemini_reasoning_text_response.json` holds two complete non-streaming responses:

- `non_streaming`: a plain answer. Message keys are content, reasoning_opaque,
  reasoning_text, role. `reasoning_text` is 2158 characters, `usage.reasoning_tokens` is 477.
- `non_streaming_tool_call`: the same shape plus a populated `tool_calls` array, so the
  tool-call path is covered and not only the plain-answer path. `reasoning_text` is 622
  characters, `usage.reasoning_tokens` is 69.

`gemini_reasoning_text_stream.json` holds a real SSE capture. `delta_keys_observed` records
every delta key seen across the full stream (content, reasoning_text, role), and `chunks`
retains the first 12 chunks so the file stays small while still containing reasoning deltas.

## What the canaries do, and what they cannot do

`tests/agent/test_copilot_reasoning_text_capture.py` asserts these recordings still carry
`reasoning_text`. The production fallback fails silently by design: if Copilot renames the
key, reasoning simply stops being captured and nothing raises. The canaries turn a change
in the recorded contract into a red test rather than quiet data loss, and their failure
messages name the production files to update.

They cannot detect drift that only appears on a different account, in a different region,
or in a future API version, because a committed recording cannot observe the live endpoint.
Closing that gap needs a periodic live probe.

## TODO: periodic live drift probe

Add a scheduled (not per-PR) job that issues one real request to
`api.githubcopilot.com/chat/completions` for a reasoning-capable Gemini model and asserts
`reasoning_text` is still present, then refreshes these fixtures when it is not. Keep it off
the PR path: it needs live credentials and would be flaky as a merge gate. Until that exists,
account and region portability of this key is unverified.

## Refreshing

Re-capture and update the fixtures and the extraction branches together. The extraction
points are `agent/transports/chat_completions.py` (normalize_response),
`agent/chat_completion_helpers.py` (streaming accumulator), and
`agent/agent_runtime_helpers.py` (extract_reasoning), which carries the long CONTRACT RISK
note the other two reference.
