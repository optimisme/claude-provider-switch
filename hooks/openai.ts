// Translation between Claude Code's Messages API form and the OpenAI
// chat-completions form an OpenAI-compatible proxy speaks. Pure functions:
// no `$`, so tests can call them directly.

/** A content block as `$.session.messages({ as: 'api' })` hands it over. */
type Block = { type: string; [field: string]: unknown }
export type AnthropicMessage = { role: 'user' | 'assistant'; content: readonly Block[] }

export type OpenAiToolCall = {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

export type OpenAiMessage =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string | OpenAiUserPart[] }
  | { role: 'assistant'; content: string | null; tool_calls?: OpenAiToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string }

type OpenAiUserPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }

export type OpenAiTool = {
  type: 'function'
  function: { name: string; description: string; parameters: Record<string, unknown> }
}

/** What the proxy answered, reduced to what a turn step needs. */
export type Reply = {
  text: string
  toolCalls: { id: string; name: string; arguments: string }[]
  stopReason: 'end_turn' | 'max_tokens' | 'tool_use'
  usage: { input_tokens: number; output_tokens: number }
  model: string
}

/** `https://host` or `https://host/v1/` → `https://host/v1`. */
export function apiBase(url: string): string {
  const trimmed = url.trim().replace(/\/+$/, '')
  return /\/v1$/.test(trimmed) ? trimmed : `${trimmed}/v1`
}

/** The ids of a `GET /v1/models` body, sorted; throws on a body of another shape. */
export function parseModels(body: string): string[] {
  const json = JSON.parse(body) as { data?: { id?: unknown }[] }
  if (!Array.isArray(json.data)) throw new Error('the reply has no "data" list')
  const ids = json.data.map(m => m.id).filter((id): id is string => typeof id === 'string')
  return [...new Set(ids)].sort()
}

function textOf(value: unknown): string {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''
  return value
    .map(part => {
      const block = part as Block
      if (block.type === 'text') return String(block.text ?? '')
      if (block.type === 'image') return '[image omitted]'
      return ''
    })
    .filter(Boolean)
    .join('\n')
}

function imageUrl(block: Block): string | undefined {
  const source = block.source as { type?: string; media_type?: string; data?: string; url?: string } | undefined
  if (source?.type === 'base64' && source.data) return `data:${source.media_type ?? 'image/png'};base64,${source.data}`
  if (source?.type === 'url' && source.url) return source.url
  return undefined
}

/** What the proxy says each model can do, as `GET /v1/model-capabilities` lists it. */
export type Capabilities = Record<
  string,
  { contextWindow?: number; maxOutputTokens?: number; image?: boolean; tools?: boolean; reasoning?: boolean }
>

/**
 * The capabilities of an IETI-style `/v1/model-capabilities` body, keyed by
 * model id. Fields the body leaves out stay absent; throws on another shape.
 */
export function parseCapabilities(body: string): Capabilities {
  const json = JSON.parse(body) as {
    data?: {
      id?: unknown
      context_window?: unknown
      max_output_tokens?: unknown
      capabilities?: { image?: unknown; tools?: unknown; reasoning?: unknown }
      modalities?: { input?: unknown }
    }[]
  }
  if (!Array.isArray(json.data)) throw new Error('the reply has no "data" list')

  const numberOr = (v: unknown) => (typeof v === 'number' && v > 0 ? v : undefined)
  const boolOr = (v: unknown) => (typeof v === 'boolean' ? v : undefined)
  const out: Capabilities = {}
  for (const entry of json.data) {
    if (typeof entry.id !== 'string') continue
    const inputs = Array.isArray(entry.modalities?.input) ? (entry.modalities.input as unknown[]) : undefined
    const image = boolOr(entry.capabilities?.image) ?? (inputs ? inputs.includes('image') : undefined)
    const info = {
      contextWindow: numberOr(entry.context_window),
      maxOutputTokens: numberOr(entry.max_output_tokens),
      image,
      tools: boolOr(entry.capabilities?.tools),
      reasoning: boolOr(entry.capabilities?.reasoning),
    }
    // Leave unknown fields out rather than undefined, so state stays plain JSON.
    out[entry.id] = Object.fromEntries(Object.entries(info).filter(([, v]) => v !== undefined))
  }
  return out
}

/** A rough token count for a request body: about four characters a token. */
export function estimateTokens(body: string): number {
  return Math.ceil(body.length / 4)
}

/**
 * The conversation, with `system` first, in chat-completions form. With
 * `images: false` each image becomes a short note, for a model without vision.
 */
export function toOpenAiMessages(
  system: string,
  messages: readonly AnthropicMessage[],
  { images = true }: { images?: boolean } = {},
): OpenAiMessage[] {
  const out: OpenAiMessage[] = []
  if (system) out.push({ role: 'system', content: system })

  for (const message of messages) {
    if (message.role === 'assistant') {
      const text = message.content
        .filter(b => b.type === 'text')
        .map(b => String(b.text ?? ''))
        .join('')
      const calls: OpenAiToolCall[] = message.content
        .filter(b => b.type === 'tool_use')
        .map(b => ({
          id: String(b.id),
          type: 'function',
          function: { name: String(b.name), arguments: JSON.stringify(b.input ?? {}) },
        }))
      out.push({ role: 'assistant', content: text || null, ...(calls.length ? { tool_calls: calls } : {}) })
      continue
    }

    // Tool results answer the previous assistant's calls, so they go first.
    for (const block of message.content) {
      if (block.type !== 'tool_result') continue
      const content = textOf(block.content)
      out.push({
        role: 'tool',
        tool_call_id: String(block.tool_use_id),
        content: block.is_error ? `Error: ${content}` : content || '(no output)',
      })
    }

    const parts: OpenAiUserPart[] = []
    for (const block of message.content) {
      if (block.type === 'text' && block.text) parts.push({ type: 'text', text: String(block.text) })
      if (block.type === 'image') {
        const url = images ? imageUrl(block) : undefined
        if (url) parts.push({ type: 'image_url', image_url: { url } })
        else if (!images) parts.push({ type: 'text', text: '[image omitted: this model has no image input]' })
      }
      if (block.type === 'document') parts.push({ type: 'text', text: '[document omitted]' })
    }
    if (parts.length === 0) continue
    const onlyText = parts.every(p => p.type === 'text')
    out.push({
      role: 'user',
      content: onlyText ? parts.map(p => (p as { text: string }).text).join('\n\n') : parts,
    })
  }

  return out
}

/** The chat-completions reply body, read into a Reply; throws on another shape. */
export function parseReply(body: string, requestedModel: string): Reply {
  const json = JSON.parse(body) as {
    model?: string
    choices?: {
      finish_reason?: string
      message?: {
        content?: string | null
        tool_calls?: { id?: string; function?: { name?: string; arguments?: string } }[]
      }
    }[]
    usage?: { prompt_tokens?: number; completion_tokens?: number }
  }
  const choice = json.choices?.[0]
  if (!choice?.message) throw new Error('the reply has no choices[0].message')

  const toolCalls = (choice.message.tool_calls ?? [])
    .filter(c => c.function?.name)
    .map((c, i) => ({
      id: c.id || `call_${Math.random().toString(36).slice(2, 10)}_${i}`,
      name: String(c.function?.name),
      arguments: c.function?.arguments || '{}',
    }))

  const stopReason = toolCalls.length > 0 ? 'tool_use' : choice.finish_reason === 'length' ? 'max_tokens' : 'end_turn'

  return {
    text: choice.message.content ?? '',
    toolCalls,
    stopReason,
    usage: {
      input_tokens: json.usage?.prompt_tokens ?? 0,
      output_tokens: json.usage?.completion_tokens ?? 0,
    },
    model: json.model ?? requestedModel,
  }
}

/** The error text of a failed proxy call, short enough for a transcript line. */
export function errorText(status: number, body: string): string {
  let detail = body
  try {
    const json = JSON.parse(body) as { error?: { message?: string } | string }
    detail = typeof json.error === 'string' ? json.error : (json.error?.message ?? body)
  } catch {}
  return `HTTP ${status}${detail ? `: ${detail.slice(0, 300)}` : ''}`
}
