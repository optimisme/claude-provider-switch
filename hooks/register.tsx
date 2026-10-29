import { atom, read, update } from 'claude-code'
import type { EngineInterface, PluginOptions, Register } from 'claude-code'

import type { Catalog, Connection, ContextUse, ModelInfo, Selection } from '../types'
import { apiBase, errorText, estimateTokens, parseCapabilities, parseModels, parseReply, toOpenAiMessages } from './openai'
import type { AnthropicMessage, OpenAiTool } from './openai'
import { KNOWN_SCHEMAS } from './schemas'

const PANE = 'provider-switch'
const COMMAND = 'provider'

const selection = atom({ plugin: 'provider-switch', key: 'selection' } as const, { provider: 'claude' } as Selection)
const catalog = atom(
  { plugin: 'provider-switch', key: 'catalog' } as const,
  { models: [], capabilities: {}, isLoading: false } as Catalog,
)
const connection = atom(
  { plugin: 'provider-switch', key: 'connection' } as const,
  { url: '', keyHint: '', keySource: 'none' } as Connection,
)
const context = atom({ plugin: 'provider-switch', key: 'context' } as const, null as ContextUse | null)

// Kept across sessions in the plugin's store. A key typed in the pane lands
// here; one set in the plugin's config stays in Claude Code's secure storage.
const STORE_URL = 'proxyUrl'
const STORE_KEY = 'apiKey'
const STORE_SELECTION = 'selection'

const CLAUDE = 'claude'
const PROXY_PREFIX = 'proxy:'

// Share of a model's context window at which the mod warns, once per model.
const CONTEXT_WARN = 0.8

// The plugin's config values, as `register` last received them.
let pluginOptions: PluginOptions = {}
// The system prompt Claude Code composed last; sent to the proxy as its system message.
let systemPrompt = ''
// The model whose context the mod last warned about, so each crossing warns once.
let warnedModel = ''

/** `90000` → `90k`. */
function kTokens(n: number): string {
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n)
}

/** One line of what a model can do: `90k context · 8k output · vision · tools`. */
function describe(info: ModelInfo | undefined): string {
  if (!info || Object.keys(info).length === 0) return 'capabilities unknown'
  const parts = []
  if (info.contextWindow) parts.push(`${kTokens(info.contextWindow)} context`)
  if (info.maxOutputTokens) parts.push(`${kTokens(info.maxOutputTokens)} output`)
  if (info.image !== undefined) parts.push(info.image ? 'vision' : 'no vision')
  if (info.tools !== undefined) parts.push(info.tools ? 'tools' : 'no tools')
  if (info.reasoning) parts.push('reasoning')
  return parts.join(' · ')
}

async function credentials($: EngineInterface) {
  const paneUrl = (await $.store.get(STORE_URL)) as string | undefined
  const paneKey = (await $.store.get(STORE_KEY)) as string | undefined
  const configKey = typeof pluginOptions.apiKey === 'string' ? pluginOptions.apiKey : ''
  const configUrl = typeof pluginOptions.proxyUrl === 'string' ? pluginOptions.proxyUrl : ''
  const key = paneKey || configKey
  const source: Connection['keySource'] = paneKey ? 'pane' : configKey ? 'config' : 'none'

  return { url: paneUrl || configUrl, key, source }
}

async function syncConnection($: EngineInterface) {
  const { url, key, source } = await credentials($)
  await update($, connection, () => ({ url, keyHint: key.slice(-4), keySource: source }))
}

async function showStatus($: EngineInterface) {
  const sel = await read($, selection)
  if (sel.provider !== 'proxy') return $.ui.status('⇄ Claude')

  const window = (await read($, catalog)).capabilities[sel.model]?.contextWindow
  const used = await read($, context)
  const share = window && used && used.model === sel.model ? ` · ${Math.round((used.inputTokens / window) * 100)}% ctx` : ''
  $.ui.status(`⇄ ${sel.model} (proxy)${share}`)
}

/** Record how full the context is and warn once when it nears the window. */
async function noteContext($: EngineInterface, model: string, inputTokens: number) {
  await update($, context, () => ({ model, inputTokens }))
  await showStatus($)

  const window = (await read($, catalog)).capabilities[model]?.contextWindow
  if (!window) return
  if (inputTokens < window * CONTEXT_WARN) {
    if (warnedModel === model) warnedModel = ''
    return
  }
  if (warnedModel === model) return
  warnedModel = model
  $.ui.toast(
    `${model} has used ${kTokens(inputTokens)} of its ${kTokens(window)} context. Run /compact or switch to Claude before it fills up.`,
  )
}

async function choose($: EngineInterface, value: string) {
  const next: Selection =
    value.startsWith(PROXY_PREFIX) ? { provider: 'proxy', model: value.slice(PROXY_PREFIX.length) } : { provider: 'claude' }
  await update($, selection, () => next)
  await $.store.set(STORE_SELECTION, next)
  await showStatus($)
  $.ui.toast(next.provider === 'proxy' ? `Model requests now go to ${next.model} via the proxy` : 'Model requests now go to Claude')
}

async function reloadModels($: EngineInterface) {
  const { url, key } = await credentials($)
  if (!url || !key) {
    await update($, catalog, c => ({ ...c, isLoading: false, error: 'Set the proxy URL and API key first' }))
    return
  }

  await update($, catalog, c => ({ ...c, isLoading: true, error: undefined }))
  let models: string[]
  try {
    const res = await $.http.fetch(`${apiBase(url)}/models`, { headers: { authorization: `Bearer ${key}` } })
    if (!res.ok) throw new Error(errorText(res.status, res.text))
    models = parseModels(res.text)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    await update($, catalog, c => ({ ...c, isLoading: false, error: `Could not list models: ${message}` }))
    return
  }

  // Optional: a proxy without the endpoint just leaves capabilities unknown.
  let capabilities: Catalog['capabilities'] = {}
  try {
    const res = await $.http.fetch(`${apiBase(url)}/model-capabilities`, { headers: { authorization: `Bearer ${key}` } })
    if (res.ok) capabilities = parseCapabilities(res.text)
  } catch {}

  const loadedAt = await $.clock.now()
  await update($, catalog, () => ({ models, capabilities, isLoading: false, loadedAt }))
  await showStatus($)

  // A proxy model that is gone can't answer: fall back to Claude.
  const sel = await read($, selection)
  if (sel.provider === 'proxy' && !models.includes(sel.model)) {
    await choose($, CLAUDE)
    $.ui.toast(`${sel.model} is no longer offered by the proxy; switched back to Claude`)
  }
}

async function saveUrl($: EngineInterface, value: string) {
  const url = value.trim()
  if (url) await $.store.set(STORE_URL, url)
  else await $.store.delete(STORE_URL)
  await syncConnection($)
  await reloadModels($)
}

async function saveKey($: EngineInterface, value: string) {
  const key = value.trim()
  if (!key) return
  await $.store.set(STORE_KEY, key)
  await syncConnection($)
  await reloadModels($)
}

async function forgetKey($: EngineInterface) {
  await $.store.delete(STORE_KEY)
  await syncConnection($)
  await reloadModels($)
}

export const register: Register = (on, options) => {
  pluginOptions = options

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: COMMAND,
      description: 'Choose between Claude and the proxy models',
      argumentHint: '[claude | <model> | reload]',
    })

    const saved = (await $.store.get(STORE_SELECTION)) as Selection | undefined
    if (saved?.provider === 'proxy' && typeof saved.model === 'string') await update($, selection, () => saved)
    await syncConnection($)
    await showStatus($)
    // The model list is refreshed at every start; a failure shows in the pane.
    void reloadModels($)
    void $.ui.open({ id: PANE, title: 'Model provider' })

    return next(e)
  })

  on('command.run', { command: COMMAND }, async ($, e) => {
    const arg = e.args.trim()
    if (!arg) {
      await $.ui.open({ id: PANE, title: 'Model provider', focus: true })
      return { text: 'Model provider pane opened.' }
    }

    if (arg === CLAUDE) {
      await choose($, CLAUDE)
      return { text: 'Model requests now go to Claude.' }
    }

    // `reload`, or a model the list may not hold yet: refresh it first.
    let cat = await read($, catalog)
    if (arg === 'reload' || !cat.models.includes(arg)) {
      await reloadModels($)
      cat = await read($, catalog)
    }
    if (cat.error) return { text: `⚠ ${cat.error}` }
    const list = cat.models.length ? cat.models.join(', ') : 'none'
    if (arg === 'reload') return { text: `Proxy models: ${list}` }
    if (!cat.models.includes(arg)) return { text: `The proxy offers no model "${arg}". Proxy models: ${list}` }

    await choose($, `${PROXY_PREFIX}${arg}`)
    return { text: `Model requests now go to ${arg} via the proxy.` }
  })

  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    systemPrompt = composed.sections.map(s => s.text).join('\n\n')

    return composed
  })

  on('turn.step', async function* ($, e, next) {
    const sel = await read($, selection)
    if (sel.provider !== 'proxy') return yield* next(e)

    const usage = (input: number, output: number, model: string) => ({
      input_tokens: input,
      output_tokens: output,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
      model,
    })
    const fail = (why: string) => ({
      turnId: e.turnId,
      index: e.index,
      answer: why,
      toolUses: [],
      stopReason: 'end_turn' as const,
      usage: null,
    })

    const { url, key } = await credentials($)
    if (!url || !key) {
      const why = '⚠ provider-switch: the proxy URL or API key is missing. Set them with /provider or switch back to Claude.'
      yield { kind: 'text', index: 0, text: why }
      yield { kind: 'stop', stopReason: 'end_turn', usage: null }
      return fail(why)
    }

    const history =
      e.agentId === undefined
        ? await $.session.messages({ as: 'api' })
        : await $.session.messages({ agentId: e.agentId, as: 'api' })
    if (!Array.isArray(history)) {
      const why = `⚠ provider-switch: could not read the conversation (${history.deny})`
      yield { kind: 'text', index: 0, text: why }
      yield { kind: 'stop', stopReason: 'end_turn', usage: null }
      return fail(why)
    }

    const info = (await read($, catalog)).capabilities[sel.model] ?? {}
    // Only a model the proxy says lacks tools gets none; unknown means try.
    const tools: OpenAiTool[] = (info.tools === false ? [] : await $.tool.list())
      .filter(t => KNOWN_SCHEMAS[t.name])
      .map(t => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: KNOWN_SCHEMAS[t.name]! },
      }))

    const body = {
      model: sel.model,
      messages: toOpenAiMessages(systemPrompt, history as unknown as AnthropicMessage[], { images: info.image !== false }),
      ...(tools.length ? { tools } : {}),
      stream: false,
    }

    const payload = JSON.stringify(body)
    // Before sending: a conversation already past the window can only fail.
    const estimate = estimateTokens(payload)
    if (info.contextWindow && estimate > info.contextWindow) {
      const why =
        `⚠ provider-switch: this conversation (~${kTokens(estimate)} tokens) is larger than ${sel.model}'s ` +
        `${kTokens(info.contextWindow)} context. Run /compact, start a new conversation, or switch to Claude with /provider claude.`
      yield { kind: 'text', index: 0, text: why }
      yield { kind: 'stop', stopReason: 'end_turn', usage: null }
      return fail(why)
    }

    let reply
    try {
      const res = await $.http.fetch(`${apiBase(url)}/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
        body: payload,
      })
      if (!res.ok) throw new Error(errorText(res.status, res.text))
      reply = parseReply(res.text, sel.model)
    } catch (error) {
      const why = `⚠ provider-switch: ${sel.model} failed: ${error instanceof Error ? error.message : String(error)}`
      yield { kind: 'text', index: 0, text: why }
      yield { kind: 'stop', stopReason: 'end_turn', usage: null }
      return fail(why)
    }

    await noteContext($, sel.model, reply.usage.input_tokens || estimate)

    let block = 0
    if (reply.text) yield { kind: 'text', index: block++, text: reply.text }
    const toolUses = []
    for (const call of reply.toolCalls) {
      const index = block++
      yield { kind: 'tool', index, id: call.id, name: call.name }
      yield { kind: 'input', index, json: call.arguments }
      let input: unknown = {}
      try {
        input = JSON.parse(call.arguments)
      } catch {}
      toolUses.push({ name: call.name, input })
    }
    const cost = usage(reply.usage.input_tokens, reply.usage.output_tokens, reply.model)
    yield { kind: 'stop', stopReason: reply.stopReason, usage: cost }

    return {
      turnId: e.turnId,
      index: e.index,
      answer: reply.text,
      toolUses,
      stopReason: reply.stopReason,
      usage: cost,
    }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const elements = $.ui.resolve(e)
    const { Box, Text, Button } = elements
    const sel = await read($, selection)
    const cat = await read($, catalog)
    const conn = await read($, connection)
    const used = await read($, context)

    const choices = [
      { value: CLAUDE, label: 'Claude Code (Anthropic)' },
      ...cat.models.map(model => {
        const window = cat.capabilities[model]?.contextWindow
        return { value: `${PROXY_PREFIX}${model}`, label: `${model} · proxy${window ? ` · ${kTokens(window)}` : ''}` }
      }),
    ]
    const current = sel.provider === 'proxy' ? `${PROXY_PREFIX}${sel.model}` : CLAUDE
    const keyLabel =
      conn.keySource === 'none' ? 'no key set' : `key …${conn.keyHint} (${conn.keySource === 'pane' ? 'set here' : 'from plugin config'})`
    const listLabel = cat.isLoading
      ? 'Loading models…'
      : cat.error
        ? `⚠ ${cat.error}`
        : `${cat.models.length} proxy model${cat.models.length === 1 ? '' : 's'} available`

    const info = sel.provider === 'proxy' ? cat.capabilities[sel.model] : undefined
    const window = info?.contextWindow
    const usedLine =
      sel.provider === 'proxy' && used && used.model === sel.model
        ? `Context: ${kTokens(used.inputTokens)}${window ? ` of ${kTokens(window)} (${Math.round((used.inputTokens / window) * 100)}%)` : ''} at the last answer`
        : undefined
    const isNearFull = Boolean(window && used && used.model === sel.model && used.inputTokens >= window * CONTEXT_WARN)
    const header = (
      <Box flexDirection="column">
        <Text bold>{sel.provider === 'proxy' ? `Using ${sel.model} via the proxy` : 'Using Claude'}</Text>
        {sel.provider === 'proxy' && <Text dimColor>{describe(info)}</Text>}
        {usedLine && <Text dimColor={!isNearFull}>{isNearFull ? `⚠ ${usedLine}` : usedLine}</Text>}
        {info?.image === false && <Text dimColor>Images in the conversation are left out for this model.</Text>}
        {info?.tools === false && <Text dimColor>This model gets no tools: it can only answer in text.</Text>}
      </Box>
    )
    const reload = (
      <Button key="reload" label={cat.isLoading ? 'Reloading…' : 'Reload models'} onPress={() => void reloadModels($)} />
    )

    // Mobile draws no Input or Select: one button per model, and the connection is set elsewhere.
    if (!('Select' in elements) || !('Input' in elements)) {
      return (
        <Box flexDirection="column">
          {header}
          {choices.map(choice => (
            <Button
              key={choice.value}
              label={choice.value === current ? `● ${choice.label}` : choice.label}
              onPress={() => void choose($, choice.value)}
            />
          ))}
          <Text dimColor={!cat.error}>{listLabel}</Text>
          {reload}
        </Box>
      )
    }
    const { Input, Select } = elements

    return (
      <Box flexDirection="column">
        {header}
        <Select key="model" label="Model" options={choices} value={current} onSelect={value => void choose($, value)} />
        <Text dimColor={!cat.error}>{listLabel}</Text>
        <Text> </Text>
        <Text bold>Proxy connection</Text>
        <Input
          key="url"
          label="URL"
          placeholder="https://your-proxy.example"
          value={conn.url}
          submitLabel="save"
          onSubmit={value => void saveUrl($, value)}
        />
        <Input
          key="apikey"
          label="API key"
          placeholder="ieti_sk_… (Enter to save)"
          value=""
          submitLabel="save"
          onSubmit={value => void saveKey($, value)}
        />
        <Text dimColor>{keyLabel}</Text>
        <Box flexDirection="row">
          {reload}
          {conn.keySource === 'pane' && (
            <Button key="forget" label="Forget key" dimColor onPress={() => void forgetKey($)} />
          )}
        </Box>
      </Box>
    )
  })
}
