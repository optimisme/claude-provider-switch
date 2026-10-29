# provider-switch

A Claude Code mod that lets you switch, from a pane beside the chat, between
Claude and the models of any number of OpenAI-compatible providers: a proxy
such as [IETI Agents Proxy](https://github.com/optimisme/IETI-AgentsProxy),
DeepSeek, OpenAI, OpenRouter, Groq, Mistral, a local Ollama or vLLM, and so on.

- **Model provider pane**: shows which backend answers, one picker with
  *Claude Code (Anthropic)* plus every chat model of every provider, the
  providers with their URL and key status, and a form to add or update one.
- **Named providers**: each has a name, a base URL and an optional API key.
  Saving a provider with an existing name replaces it (an empty key keeps the
  saved one); **Forget key** and **Remove** are on each row.
- **Model lists**: fetched from each provider's `GET <base>/models` every time
  Claude Code starts, on **Reload all**, and when a provider is saved.
  Embedding, speech, image and moderation models are left out. If the selected
  model disappears, or its provider is removed, the mod switches back to Claude.
- **Capabilities**, merged from what the provider publishes:
  - `GET <base>/model-capabilities` in IETI's format (`context_window`,
    `max_output_tokens`, `capabilities.image/tools/reasoning`),
  - OpenRouter-style fields in `/models` (`context_length`,
    `architecture.input_modalities`, `supported_parameters`) and vLLM's `max_model_len`.

  Shown in the pane, e.g. `90k context · 8k output · vision · tools · reasoning`.
  Where a provider says nothing, set **Vision** and **Context window** by hand
  for the selected model; your setting wins over what the provider says.
  - A model without vision gets each image replaced by a short note.
  - A model without tools gets no tools and can only answer in text.
  - After each answer the pane and the button show how full the context is;
    at 80% a toast suggests `/compact` or switching to Claude. A conversation
    already larger than the window is stopped before it is sent.
- **Context**: a folded section at the bottom of the pane, for whichever
  provider is selected, Claude included. Its header shows about how many
  tokens every request carries before the conversation. Unfolded, it lists
  each item on its own row with its cost and a toggle, plus **All** and **None** per group:
  - **Tools**: for a provider, the tools its model is offered; for Claude, the
    tool schemas in the prompt (one switched off waits behind ToolSearch).
    MCP tools (`mcp__<server>__<tool>`) fold into one row per server that
    switches all of its tools; ▸ unfolds it to switch them one by one.
  - **System prompt**: each section of Claude Code's system prompt.
  - **Injected context**: reminders and listings the engine adds (skills,
    deferred tools, ...), hooks' context (`SessionStart` and others) and the
    first message's context blocks (`claudeMd`, `userEmail`, ...). Mentioned
    files and queued prompts are never offered.

  While everything is on, **Lean** switches off what a coding turn can do
  without: tools other than Bash, Read, Write, Edit, Glob, Grep and ToolSearch,
  hooks' and plugins' context, context blocks other than `claudeMd` and
  `currentDate`, and, for a provider, the engine's reminders and listings.
  System prompt sections stay on. Once anything is off, **Restore all** switches
  everything back on.

  The choice is kept per provider, across sessions. System prompt sections and
  injected context are listed once a request has carried them.
- **Button above the prompt**: `⇄ Claude` or `⇄ <model> (<provider>) · <n>% ctx`;
  press it to open the pane.
- **`/provider`**: opens the pane. `/provider <model>` selects the first
  provider's model of that name, `/provider <provider>/<model>` a given
  provider's (`/provider openrouter/openai/gpt-4o`), `/provider claude` goes
  back to Claude, and `/provider reload` lists every provider's models.

## Install

**Claude desktop app (Code tab):** click **+** next to the prompt box, then
**Plugins** → **Add marketplace**, enter `optimisme/claude-provider-switch`,
then **Add plugin** next to *provider-switch*. Start a new session and type
`/provider` to open the pane.

**Claude Code in a terminal:**

```
/plugin install provider-switch --marketplace optimisme/claude-provider-switch
```

Both install it for your user, so the desktop app and the terminal share it.

Or, while developing, start Claude Code with the folder loaded directly:

```bash
claude --plugin-dir /path/to/claude-provider-switch
```

## Add providers

In the pane, under **Add or update a provider**, type a name, the base URL and
the key, then press **Save provider** (or Enter in the key field).

| Provider | URL | Key |
| --- | --- | --- |
| IETI Agents Proxy | `https://agents.ieti.site/v1` | `ieti_sk_…` |
| DeepSeek | `https://api.deepseek.com` | `sk-…` |
| OpenAI | `https://api.openai.com` | `sk-…` |
| OpenRouter | `https://openrouter.ai/api/v1` | `sk-or-…` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | Google AI key |
| Ollama | `http://localhost:11434` | none |

A URL with no path gets `/v1` added; a URL with a path is used as given.

You can also set one provider in the plugin's options when installing
(`proxyUrl`, `apiKey`). It shows as **Configured**, and its key is kept in
Claude Code's secure storage. Keys typed in the pane are saved in the plugin's
own local store (plain JSON on this machine), and are visible in the field
while you type them.

## How it works

When a provider's model is selected, the mod answers each model request of a
turn (the `turn.step` hook) itself instead of sending it to Anthropic:

1. reads the conversation in Messages API form and the system prompt Claude
   Code composed,
2. translates them to the chat-completions format (tool uses → `tool_calls`,
   tool results → `tool` messages),
3. `POST`s them to `<base>/chat/completions` with `curl`, with
   `Authorization: Bearer <key>` when the provider has a key,
4. turns the reply back into text and tool calls that Claude Code runs as usual
   (with its normal permission prompts).

With Claude selected, requests pass through untouched.

## Limitations

- Replies arrive whole, not streamed. Chat requests go through `curl` (included
  in macOS, Windows 10 and later, and most Linux installs), allowed 5 minutes;
  where curl is missing they use Claude Code's own HTTP call, which gives up
  after 30 seconds.
- A provider's model is offered these tools only: Bash, Read, Write, Edit, Glob,
  Grep, WebFetch, WebSearch and Agent. The mod API does not expose built-in tool
  schemas, so they are written out in `hooks/schemas.ts`. MCP and other tools
  are not offered, and any of these can be switched off under **Context**.
- Thinking blocks, documents and prompt caching are not forwarded. Reasoning
  models that expect their reasoning sent back during a tool loop may fail.
- Models only reachable through OpenAI's Responses API, and Azure OpenAI's URL
  scheme, are not supported.
- Claude Code's other model calls (compaction, titles, helper classifiers) still
  go to Claude.
- How well tools work depends on the model's function-calling quality.

## Development

```bash
claude plugin validate .
```

```bash
claude plugin test .
```
