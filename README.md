# provider-switch

A Claude Code mod that lets you switch, from a pane beside the chat, between
Claude and the models of an OpenAI-compatible proxy such as
[IETI Agents Proxy](https://github.com/optimisme/IETI-AgentsProxy).

- **Model provider pane**: shows which backend answers, a picker with
  *Claude Code (Anthropic)* plus every model the proxy publishes, the proxy
  URL and API key fields, and a **Reload models** button.
- **Model list**: fetched from `GET <proxy>/v1/models` every time Claude Code
  starts, and again whenever you press **Reload models** or save a new URL or
  key. If the selected proxy model disappears, the mod switches back to Claude.
- **Capabilities**: read from `GET <proxy>/v1/model-capabilities` (IETI's
  format: `context_window`, `max_output_tokens`, `capabilities.image/tools/reasoning`)
  and shown in the pane, e.g. `90k context · 8k output · vision · tools · reasoning`.
  A proxy without that endpoint still works; capabilities just show as unknown
  and everything is sent as usual.
  - A model without vision gets each image replaced by a short note.
  - A model without tools gets no tools and can only answer in text.
  - After each answer the pane and status line show how full the context is
    (`⇄ deepseek-flash (proxy) · 14% ctx`); at 80% a toast suggests `/compact`
    or switching to Claude. A conversation already larger than the window is
    stopped before it is sent, with the same advice.
- **Status line**: `⇄ Claude` or `⇄ <model> (proxy) · <n>% ctx`.
- **`/provider`**: opens the pane. `/provider <model>` selects a proxy model,
  `/provider claude` goes back to Claude, `/provider reload` lists the proxy's
  models (handy in `claude -p`, where there is no pane).

## Install

```
/plugin install provider-switch --marketplace optimisme/claude-provider-switch
```

Or, while developing, start Claude Code with the folder loaded directly:

```bash
claude --plugin-dir /path/to/claude-provider-switch
```

## Configure the proxy

Either:

- in the pane, type the proxy URL (e.g. `https://agents.example.com`; `/v1` is
  added when missing) and the API key (`ieti_sk_…`) and press Enter on each, or
- set `proxyUrl` and `apiKey` in the plugin's options when installing. The
  `apiKey` option is marked sensitive, so Claude Code keeps it in secure storage.

A key typed in the pane is saved in the plugin's own local store (plain JSON on
this machine) and takes precedence over the configured one; **Forget key**
removes it. The key is also visible in the field while you type it.

## How it works

When a proxy model is selected, the mod answers each model request of a turn
(the `turn.step` hook) itself instead of sending it to Anthropic:

1. reads the conversation in Messages API form and the system prompt Claude
   Code composed,
2. translates them to the chat-completions format (tool uses → `tool_calls`,
   tool results → `tool` messages),
3. `POST`s them to `<proxy>/v1/chat/completions` with `Authorization: Bearer <key>`,
4. turns the reply back into text and tool calls that Claude Code runs as usual
   (with its normal permission prompts).

With Claude selected, requests pass through untouched.

## Limitations

- Replies arrive whole, not streamed.
- The proxy model is offered these tools only: Bash, Read, Write, Edit, Glob,
  Grep, WebFetch, WebSearch and Agent. The mod API does not expose built-in tool
  schemas, so they are written out in `hooks/schemas.ts`. MCP and other tools
  are not offered to proxy models.
- Thinking blocks, documents and prompt caching are not forwarded.
- Claude Code's other model calls (compaction, titles, helper classifiers) still
  go to Claude.
- How well tools work depends on the proxy model's function-calling quality.

## Development

```bash
claude plugin validate .
```

```bash
claude plugin test .
```
