// Translation between Claude Code's Messages API form and the OpenAI
// chat-completions form an OpenAI-compatible proxy speaks. Pure functions:
// no `$`, so tests can call them directly.

import type { ModelInfo } from '../types'

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
  | { role: 'assistant'; content: string | null; tool_calls?: OpenAiToolCall[]; reasoning_content?: string }
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
  /** A thinking model's `reasoning_content` (DeepSeek and those like it); empty when none. */
  reasoning: string
  toolCalls: { id: string; name: string; arguments: string }[]
  stopReason: 'end_turn' | 'max_tokens' | 'tool_use'
  usage: { input_tokens: number; output_tokens: number }
  model: string
}

/**
 * The base the API paths hang off: a URL with a path is kept as given
 * (`https://host/v1beta/openai`), a bare host gets `/v1` (`https://host` → `https://host/v1`).
 */
export function apiBase(url: string): string {
  const trimmed = url.trim().replace(/\/+$/, '')
  const path = trimmed.replace(/^[a-z]+:\/\/[^/]+/i, '')
  return path ? trimmed : `${trimmed}/v1`
}

/** The request headers for `key`: none when the provider takes no key. */
export function authHeaders(key: string): Record<string, string> {
  return key ? { authorization: `Bearer ${key}` } : {}
}

/** Model ids that name an embedding, speech, image or moderation model, not a chat one. */
const NOT_CHAT = /(embed|whisper|tts|dall-e|gpt-image|moderation|transcribe|realtime|audio|davinci|babbage|rerank|sora|imagen|veo)/i

/** Whether `id` can answer a chat completion, by its name and what its provider says it outputs. */
export function isChatModel(id: string, info: ModelInfo | undefined): boolean {
  return info?.text !== false && !NOT_CHAT.test(id)
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

/**
 * An image as an inline `data:` URL; undefined for one only reachable at a remote URL,
 * which providers like IETI Agents Proxy refuse, failing the whole request.
 */
function imageUrl(block: Block): string | undefined {
  const source = block.source as { type?: string; media_type?: string; data?: string } | undefined
  if (source?.type === 'base64' && source.data) return `data:${source.media_type ?? 'image/png'};base64,${source.data}`
  return undefined
}

export type Capabilities = Record<string, ModelInfo>

type ModelEntry = {
  id?: unknown
  // IETI's /v1/model-capabilities
  context_window?: unknown
  max_output_tokens?: unknown
  capabilities?: { text?: unknown; image?: unknown; tools?: unknown; reasoning?: unknown }
  modalities?: { input?: unknown; output?: unknown }
  // OpenRouter-style /models
  context_length?: unknown
  architecture?: { input_modalities?: unknown; output_modalities?: unknown }
  top_provider?: { max_completion_tokens?: unknown }
  supported_parameters?: unknown
  // vLLM's /v1/models
  max_model_len?: unknown
}

const numberOr = (v: unknown) => (typeof v === 'number' && v > 0 ? v : undefined)
const boolOr = (v: unknown) => (typeof v === 'boolean' ? v : undefined)
const listOr = (v: unknown) => (Array.isArray(v) ? (v as unknown[]) : undefined)

/** What one model entry says, in any of the shapes above; unknown fields left out. */
function infoOf(entry: ModelEntry): ModelInfo {
  const inputs = listOr(entry.modalities?.input) ?? listOr(entry.architecture?.input_modalities)
  const outputs = listOr(entry.modalities?.output) ?? listOr(entry.architecture?.output_modalities)
  const params = listOr(entry.supported_parameters)
  const info: ModelInfo = {
    contextWindow: numberOr(entry.context_window) ?? numberOr(entry.context_length) ?? numberOr(entry.max_model_len),
    maxOutputTokens: numberOr(entry.max_output_tokens) ?? numberOr(entry.top_provider?.max_completion_tokens),
    image: boolOr(entry.capabilities?.image) ?? (inputs ? inputs.includes('image') : undefined),
    tools: boolOr(entry.capabilities?.tools) ?? (params ? params.includes('tools') : undefined),
    reasoning: boolOr(entry.capabilities?.reasoning) ?? (params ? params.includes('reasoning') : undefined),
    text: boolOr(entry.capabilities?.text) ?? (outputs ? outputs.includes('text') : undefined),
  }
  // Leave unknown fields out rather than undefined, so state stays plain JSON.
  return Object.fromEntries(Object.entries(info).filter(([, v]) => v !== undefined)) as ModelInfo
}

function entriesOf(body: string): ModelEntry[] {
  const json = JSON.parse(body) as { data?: ModelEntry[] }
  if (!Array.isArray(json.data)) throw new Error('the reply has no "data" list')
  return json.data.filter(e => typeof e?.id === 'string')
}

/**
 * The model ids of a `GET /models` body, sorted, and whatever each entry says
 * about the model (OpenRouter and vLLM add context sizes there); throws on another shape.
 */
export function parseModels(body: string): { ids: string[]; capabilities: Capabilities } {
  const entries = entriesOf(body)
  const capabilities: Capabilities = {}
  for (const entry of entries) {
    const info = infoOf(entry)
    if (Object.keys(info).length) capabilities[entry.id as string] = info
  }
  return { ids: [...new Set(entries.map(e => e.id as string))].sort(), capabilities }
}

/**
 * The capabilities of an IETI-style `/model-capabilities` body, keyed by
 * model id. Fields the body leaves out stay absent; throws on another shape.
 */
export function parseCapabilities(body: string): Capabilities {
  return Object.fromEntries(entriesOf(body).map(entry => [entry.id as string, infoOf(entry)]))
}

/** "128k", "128000" or "1m" as a token count; undefined for anything else. */
export function parseTokens(text: string): number | undefined {
  const match = /^\s*(\d+(?:\.\d+)?)\s*([km]?)\s*$/i.exec(text)
  if (!match) return undefined
  const scale = { '': 1, k: 1000, m: 1_000_000 }[match[2]!.toLowerCase() as '' | 'k' | 'm']
  const n = Math.round(Number(match[1]) * scale)
  return n > 0 ? n : undefined
}

// What one image costs a vision model, which counts pixels, not bytes: a ceiling
// that covers a full-screen screenshot.
const IMAGE_TOKENS = 1500

/**
 * A rough token count for a request body: about four characters a token, and
 * each inline image at IMAGE_TOKENS rather than by the length of its base64.
 */
export function estimateTokens(body: string): number {
  let images = 0
  const text = body.replace(/data:[\w.+-]+\/[\w.+-]+;base64,[A-Za-z0-9+/=]+/g, () => {
    images++
    return ''
  })
  return Math.ceil(text.length / 4) + images * IMAGE_TOKENS
}

/**
 * The conversation, with `system` first, in chat-completions form. With
 * `images: false` each image becomes a short note, for a model without vision.
 *
 * `reasoning` gives back, by tool call id, the `reasoning_content` a thinking
 * model sent with the reply that made the call: DeepSeek's thinking mode refuses
 * a tool call sent back without it. With `fillReasoning`, a call whose reasoning
 * is unknown (a reply from before, another model) gets an empty one.
 */
export function toOpenAiMessages(
  system: string,
  messages: readonly AnthropicMessage[],
  {
    images = true,
    reasoning = {},
    fillReasoning = false,
  }: { images?: boolean; reasoning?: Readonly<Record<string, string>>; fillReasoning?: boolean } = {},
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
      const thought = calls.map(c => reasoning[c.id]).find(r => r !== undefined) ?? (fillReasoning && calls.length ? '' : undefined)
      out.push({
        role: 'assistant',
        content: text || null,
        ...(calls.length ? { tool_calls: calls } : {}),
        ...(thought !== undefined ? { reasoning_content: thought } : {}),
      })
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
        else parts.push({ type: 'text', text: '[image omitted: only inline images are sent]' })
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
        reasoning_content?: string | null
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
    reasoning: typeof choice.message.reasoning_content === 'string' ? choice.message.reasoning_content : '',
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

/** Whether a failed call's error is a thinking model asking for its `reasoning_content` back. */
export function wantsReasoning(status: number, body: string): boolean {
  return status === 400 && body.includes('reasoning_content')
}
