import { atom, read, update } from 'claude-code'
import type { EngineInterface, PluginOptions, Register } from 'claude-code'

import type { Catalog, ContextUse, Draft, ModelInfo, Override, ProviderView, Selection } from '../types'
import {
  apiBase,
  authHeaders,
  errorText,
  estimateTokens,
  isChatModel,
  parseCapabilities,
  parseModels,
  parseReply,
  parseTokens,
  toOpenAiMessages,
} from './openai'
import type { AnthropicMessage, OpenAiTool } from './openai'
import { KNOWN_SCHEMAS } from './schemas'

const PANE = 'provider-switch'
const COMMAND = 'provider'

const selection = atom({ plugin: 'provider-switch', key: 'selection' } as const, { provider: 'claude' } as Selection)
const providers = atom({ plugin: 'provider-switch', key: 'providers' } as const, [] as ProviderView[])
const catalogs = atom({ plugin: 'provider-switch', key: 'catalogs' } as const, {} as Record<string, Catalog>)
const overrides = atom({ plugin: 'provider-switch', key: 'overrides' } as const, {} as Record<string, Override>)
const context = atom({ plugin: 'provider-switch', key: 'context' } as const, null as ContextUse | null)
const draft = atom({ plugin: 'provider-switch', key: 'draft' } as const, { name: '', url: '', key: '' } as Draft)

// Kept across sessions in the plugin's store (plain JSON on this machine).
// The provider set in the plugin's config keeps its key in Claude Code's secure storage.
const STORE_PROVIDERS = 'providers' // { id, name, url }[]
const STORE_KEYS = 'keys' // { [providerId]: key }
const STORE_OVERRIDES = 'overrides' // { [providerId/model]: Override }
const STORE_SELECTION = 'selection'
// Before named providers: one URL and key typed in the pane.
const LEGACY_URL = 'proxyUrl'
const LEGACY_KEY = 'apiKey'

const CLAUDE = 'claude'
const PROXY_PREFIX = 'proxy:'
const CONFIG_ID = 'config'

// Share of a model's context window at which the mod warns, once per model.
const CONTEXT_WARN = 0.8

type StoredProvider = { id: string; name: string; url: string }
type Provider = StoredProvider & { key: string; source: ProviderView['source'] }

// The plugin's config values, as `register` last received them.
let pluginOptions: PluginOptions = {}
// The system prompt Claude Code composed last; sent to the provider as its system message.
let systemPrompt = ''
// The model whose context the mod last warned about, so each crossing warns once.
let warnedModel = ''

/** `90000` → `90k`. */
function kTokens(n: number): string {
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n)
}

/** `My DeepSeek` → `my-deepseek`; never `config`, which the plugin's config holds. */
function slug(name: string): string {
  const base = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'provider'
  return base === CONFIG_ID ? `${base}-2` : base
}

/** `<providerId>/<model>`: what overrides and context use are keyed by. */
function modelKey(providerId: string, model: string): string {
  return `${providerId}/${model}`
}

/** `proxy:<providerId>/<model>` → its parts; model ids may hold `/` themselves. */
function parseChoice(value: string): Selection {
  if (!value.startsWith(PROXY_PREFIX)) return { provider: 'claude' }
  const rest = value.slice(PROXY_PREFIX.length)
  const cut = rest.indexOf('/')
  return { provider: 'proxy', providerId: rest.slice(0, cut), model: rest.slice(cut + 1) }
}

/** What the provider says about a model, with the person's overrides on top. */
function effectiveInfo(cat: Catalog | undefined, set: Override | undefined, model: string): ModelInfo {
  return { ...cat?.capabilities[model], ...set }
}

/** One line of what a model can do: `90k context · 8k output · vision · tools`. */
function describe(info: ModelInfo): string {
  const parts = []
  if (info.contextWindow) parts.push(`${kTokens(info.contextWindow)} context`)
  if (info.maxOutputTokens) parts.push(`${kTokens(info.maxOutputTokens)} output`)
  if (info.image !== undefined) parts.push(info.image ? 'vision' : 'no vision')
  if (info.tools !== undefined) parts.push(info.tools ? 'tools' : 'no tools')
  if (info.reasoning) parts.push('reasoning')
  return parts.length ? parts.join(' · ') : 'capabilities unknown'
}

async function storedProviders($: EngineInterface): Promise<StoredProvider[]> {
  return ((await $.store.get(STORE_PROVIDERS)) as StoredProvider[] | undefined) ?? []
}

async function storedKeys($: EngineInterface): Promise<Record<string, string>> {
  return ((await $.store.get(STORE_KEYS)) as Record<string, string> | undefined) ?? {}
}

/** Every provider with its key: the configured one first, then those added in the pane. */
async function loadProviders($: EngineInterface): Promise<Provider[]> {
  const keys = await storedKeys($)
  const list: Provider[] = []
  const configUrl = typeof pluginOptions.proxyUrl === 'string' ? pluginOptions.proxyUrl.trim() : ''
  if (configUrl) {
    const key = typeof pluginOptions.apiKey === 'string' ? pluginOptions.apiKey : ''
    list.push({ id: CONFIG_ID, name: 'Configured', url: configUrl, key, source: 'config' })
  }
  for (const p of await storedProviders($)) list.push({ ...p, key: keys[p.id] ?? '', source: 'pane' })
  return list
}

/** Show the providers in the pane; returns them with their keys. */
async function syncProviders($: EngineInterface): Promise<Provider[]> {
  const list = await loadProviders($)
  await update($, providers, () => list.map(({ key, ...p }) => ({ ...p, keyHint: key.slice(-4) })))
  return list
}

/** Moves a URL and key typed in the pane before named providers into a provider named Proxy. */
async function migrateLegacy($: EngineInterface) {
  const url = (await $.store.get(LEGACY_URL)) as string | undefined
  const key = (await $.store.get(LEGACY_KEY)) as string | undefined
  if (url) {
    const list = await storedProviders($)
    if (!list.some(p => p.id === 'proxy')) await $.store.set(STORE_PROVIDERS, [...list, { id: 'proxy', name: 'Proxy', url }])
    if (key) await $.store.set(STORE_KEYS, { ...(await storedKeys($)), proxy: key })
  }
  await $.store.delete(LEGACY_URL)
  await $.store.delete(LEGACY_KEY)

  // A selection from then names no provider: it was the pane's, or the configured one.
  const saved = (await $.store.get(STORE_SELECTION)) as { provider?: string; providerId?: string } | undefined
  if (saved?.provider === 'proxy' && !saved.providerId) {
    await $.store.set(STORE_SELECTION, { ...saved, providerId: url ? 'proxy' : CONFIG_ID })
  }
}

async function showStatus($: EngineInterface) {
  const sel = await read($, selection)
  if (sel.provider !== 'proxy') return $.ui.status('⇄ Claude')

  const name = (await read($, providers)).find(p => p.id === sel.providerId)?.name ?? sel.providerId
  const key = modelKey(sel.providerId, sel.model)
  const info = effectiveInfo((await read($, catalogs))[sel.providerId], (await read($, overrides))[key], sel.model)
  const used = await read($, context)
  const share =
    info.contextWindow && used && used.model === key
      ? ` · ${Math.round((used.inputTokens / info.contextWindow) * 100)}% ctx`
      : ''
  $.ui.status(`⇄ ${sel.model} (${name})${share}`)
}

async function choose($: EngineInterface, value: string) {
  const next = parseChoice(value)
  await update($, selection, () => next)
  await $.store.set(STORE_SELECTION, next)
  await showStatus($)
  if (next.provider === 'claude') return $.ui.toast('Model requests now go to Claude')
  const name = (await read($, providers)).find(p => p.id === next.providerId)?.name ?? next.providerId
  $.ui.toast(`Model requests now go to ${next.model} via ${name}`)
}

/** Record how full the context is and warn once when it nears the window. */
async function noteContext($: EngineInterface, key: string, inputTokens: number, window: number | undefined) {
  await update($, context, () => ({ model: key, inputTokens }))
  await showStatus($)

  if (!window) return
  if (inputTokens < window * CONTEXT_WARN) {
    if (warnedModel === key) warnedModel = ''
    return
  }
  if (warnedModel === key) return
  warnedModel = key
  $.ui.toast(
    `${key} has used ${kTokens(inputTokens)} of its ${kTokens(window)} context. Run /compact or switch to Claude before it fills up.`,
  )
}

/** List one provider's chat models and whatever it says about them. */
async function reloadProvider($: EngineInterface, p: Provider) {
  await update($, catalogs, all => ({
    ...all,
    [p.id]: { models: [], capabilities: {}, ...all[p.id], isLoading: true, error: undefined },
  }))

  const base = apiBase(p.url)
  let listed: ReturnType<typeof parseModels>
  try {
    const res = await $.http.fetch(`${base}/models`, { headers: authHeaders(p.key) })
    if (!res.ok) throw new Error(errorText(res.status, res.text))
    listed = parseModels(res.text)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    await update($, catalogs, all => ({
      ...all,
      [p.id]: { models: [], capabilities: {}, ...all[p.id], isLoading: false, error: `Could not list models: ${message}` },
    }))
    return
  }

  // Optional (IETI Agents Proxy has it): a provider without it keeps what /models said.
  let detailed: Catalog['capabilities'] = {}
  try {
    const res = await $.http.fetch(`${base}/model-capabilities`, { headers: authHeaders(p.key) })
    if (res.ok) detailed = parseCapabilities(res.text)
  } catch {}

  const capabilities: Catalog['capabilities'] = { ...listed.capabilities }
  for (const [id, info] of Object.entries(detailed)) capabilities[id] = { ...capabilities[id], ...info }
  const models = listed.ids.filter(id => isChatModel(id, capabilities[id]))
  const loadedAt = await $.clock.now()
  await update($, catalogs, all => ({ ...all, [p.id]: { models, capabilities, isLoading: false, loadedAt } }))
}

/** Reload every provider, drop catalogs of removed ones, and leave a model that is gone. */
async function reloadAll($: EngineInterface) {
  const list = await syncProviders($)
  await update($, catalogs, all => Object.fromEntries(Object.entries(all).filter(([id]) => list.some(p => p.id === id))))
  await Promise.all(list.map(p => reloadProvider($, p)))
  await checkSelection($)
  await showStatus($)
}

/** Back to Claude when the selected provider was removed or no longer lists the model. */
async function checkSelection($: EngineInterface) {
  const sel = await read($, selection)
  if (sel.provider !== 'proxy') return
  const isKnown = (await read($, providers)).some(p => p.id === sel.providerId)
  const cat = (await read($, catalogs))[sel.providerId]
  // A failed reload says nothing about the model: keep it.
  const isGone = !isKnown || (cat !== undefined && !cat.isLoading && !cat.error && !cat.models.includes(sel.model))
  if (!isGone) return
  await choose($, CLAUDE)
  $.ui.toast(`${sel.model} is no longer offered by ${isKnown ? sel.providerId : 'a removed provider'}; switched back to Claude`)
}

async function setDraft($: EngineInterface, field: keyof Draft, value: string) {
  await update($, draft, d => ({ ...d, [field]: value }))
}

/** Add the provider in the form, or replace the one with the same name; an empty key keeps the old one. */
async function saveDraft($: EngineInterface) {
  const d = await read($, draft)
  const name = d.name.trim()
  const url = d.url.trim()
  if (!name || !url) return $.ui.toast('A provider needs a name and a URL')

  const id = slug(name)
  const list = (await storedProviders($)).filter(p => p.id !== id)
  await $.store.set(STORE_PROVIDERS, [...list, { id, name, url }])
  if (d.key.trim()) await $.store.set(STORE_KEYS, { ...(await storedKeys($)), [id]: d.key.trim() })
  await update($, draft, () => ({ name: '', url: '', key: '' }))

  const added = (await syncProviders($)).find(p => p.id === id)
  if (added) await reloadProvider($, added)
  await checkSelection($)
  $.ui.toast(`Saved provider ${name}`)
}

async function removeProvider($: EngineInterface, id: string) {
  await $.store.set(STORE_PROVIDERS, (await storedProviders($)).filter(p => p.id !== id))
  const { [id]: _, ...keys } = await storedKeys($)
  await $.store.set(STORE_KEYS, keys)
  await syncProviders($)
  await update($, catalogs, ({ [id]: __, ...rest }) => rest)
  await checkSelection($)
}

async function forgetKey($: EngineInterface, id: string) {
  const { [id]: _, ...keys } = await storedKeys($)
  await $.store.set(STORE_KEYS, keys)
  const p = (await syncProviders($)).find(x => x.id === id)
  if (p) await reloadProvider($, p)
}

/** Set or clear one field of a model's override; an empty override is removed. */
async function setOverride($: EngineInterface, key: string, patch: Override) {
  const all = ((await $.store.get(STORE_OVERRIDES)) as Record<string, Override> | undefined) ?? {}
  const merged = { ...all[key], ...patch }
  const cleaned = Object.fromEntries(Object.entries(merged).filter(([, v]) => v !== undefined)) as Override
  const { [key]: _, ...rest } = all
  const next = Object.keys(cleaned).length ? { ...rest, [key]: cleaned } : rest
  await $.store.set(STORE_OVERRIDES, next)
  await update($, overrides, () => next)
  await showStatus($)
}

async function setVision($: EngineInterface, key: string, value: string) {
  await setOverride($, key, { image: value === 'yes' ? true : value === 'no' ? false : undefined })
}

async function setContextWindow($: EngineInterface, key: string, value: string) {
  if (value.trim() && parseTokens(value) === undefined) return $.ui.toast(`"${value}" is not a token count (try 128k)`)
  await setOverride($, key, { contextWindow: parseTokens(value) })
}

/** The model a `/provider` argument names: `<providerId>/<model>`, or a model any provider lists. */
function findModel(arg: string, list: ProviderView[], all: Record<string, Catalog>): string | undefined {
  const cut = arg.indexOf('/')
  if (cut > 0) {
    const id = arg.slice(0, cut)
    const model = arg.slice(cut + 1)
    if (list.some(p => p.id === id) && all[id]?.models.includes(model)) return `${PROXY_PREFIX}${id}/${model}`
  }
  const owner = list.find(p => all[p.id]?.models.includes(arg))
  return owner ? `${PROXY_PREFIX}${owner.id}/${arg}` : undefined
}

/** `IETI (ieti): deepseek-flash, Qwen38` per provider, or its error. */
function catalogText(list: ProviderView[], all: Record<string, Catalog>): string {
  if (!list.length) return 'No providers yet: add one in the pane (/provider).'
  return list
    .map(p => {
      const cat = all[p.id]
      if (cat?.error) return `${p.name} (${p.id}): ⚠ ${cat.error}`
      return `${p.name} (${p.id}): ${cat?.models.length ? cat.models.join(', ') : 'no chat models'}`
    })
    .join('\n')
}

export const register: Register = (on, options) => {
  pluginOptions = options

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: COMMAND,
      description: 'Choose between Claude and the models of your OpenAI-compatible providers',
      argumentHint: '[claude | <model> | <provider>/<model> | reload]',
    })

    await migrateLegacy($)
    const saved = (await $.store.get(STORE_SELECTION)) as Selection | undefined
    if (saved?.provider === 'proxy' && typeof saved.model === 'string' && typeof saved.providerId === 'string') {
      await update($, selection, () => saved)
    }
    const stored = ((await $.store.get(STORE_OVERRIDES)) as Record<string, Override> | undefined) ?? {}
    await update($, overrides, () => stored)
    await syncProviders($)
    await showStatus($)
    // The model lists are refreshed at every start; a failed fetch shows in the pane,
    // and one cut short by the session ending has no one left to tell.
    reloadAll($).catch(() => {})
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

    // `reload`, or a model the lists may not hold yet: refresh them first.
    let choice = findModel(arg, await read($, providers), await read($, catalogs))
    if (arg === 'reload' || !choice) {
      await reloadAll($)
      choice = findModel(arg, await read($, providers), await read($, catalogs))
    }
    const listing = catalogText(await read($, providers), await read($, catalogs))
    if (arg === 'reload') return { text: listing }
    if (!choice) return { text: `No provider offers a model "${arg}".\n${listing}` }

    await choose($, choice)
    const sel = parseChoice(choice)
    const id = sel.provider === 'proxy' ? sel.providerId : ''
    const name = (await read($, providers)).find(p => p.id === id)?.name ?? id
    return { text: `Model requests now go to ${sel.provider === 'proxy' ? sel.model : arg} via ${name}.` }
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

    const provider = (await loadProviders($)).find(p => p.id === sel.providerId)
    if (!provider) {
      const why = `⚠ provider-switch: the provider "${sel.providerId}" is gone. Pick another model with /provider or switch back to Claude.`
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

    const key = modelKey(provider.id, sel.model)
    const info = effectiveInfo((await read($, catalogs))[provider.id], (await read($, overrides))[key], sel.model)
    // Only a model its provider says lacks tools gets none; unknown means try.
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
      const res = await $.http.fetch(`${apiBase(provider.url)}/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...authHeaders(provider.key) },
        body: payload,
      })
      if (!res.ok) throw new Error(errorText(res.status, res.text))
      reply = parseReply(res.text, sel.model)
    } catch (error) {
      const why = `⚠ provider-switch: ${sel.model} (${provider.name}) failed: ${error instanceof Error ? error.message : String(error)}`
      yield { kind: 'text', index: 0, text: why }
      yield { kind: 'stop', stopReason: 'end_turn', usage: null }
      return fail(why)
    }

    await noteContext($, key, reply.usage.input_tokens || estimate, info.contextWindow)

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
    const list = await read($, providers)
    const all = await read($, catalogs)
    const set = await read($, overrides)
    const used = await read($, context)
    const d = await read($, draft)

    const choices = [
      { value: CLAUDE, label: 'Claude Code (Anthropic)' },
      ...list.flatMap(p =>
        (all[p.id]?.models ?? []).map(model => {
          const window = effectiveInfo(all[p.id], set[modelKey(p.id, model)], model).contextWindow
          return {
            value: `${PROXY_PREFIX}${p.id}/${model}`,
            label: `${model} · ${p.name}${window ? ` · ${kTokens(window)}` : ''}`,
          }
        }),
      ),
    ]
    const current = sel.provider === 'proxy' ? `${PROXY_PREFIX}${sel.providerId}/${sel.model}` : CLAUDE
    const isLoading = list.some(p => all[p.id]?.isLoading)
    const reload = (
      <Button key="reload" label={isLoading ? 'Reloading…' : 'Reload all'} onPress={() => void reloadAll($)} />
    )

    const selKey = sel.provider === 'proxy' ? modelKey(sel.providerId, sel.model) : ''
    const selName = sel.provider === 'proxy' ? (list.find(p => p.id === sel.providerId)?.name ?? sel.providerId) : ''
    const reported: ModelInfo = sel.provider === 'proxy' ? (all[sel.providerId]?.capabilities[sel.model] ?? {}) : {}
    const info: ModelInfo = sel.provider === 'proxy' ? effectiveInfo(all[sel.providerId], set[selKey], sel.model) : {}
    const window = info.contextWindow
    const usedLine =
      selKey && used && used.model === selKey
        ? `Context: ${kTokens(used.inputTokens)}${window ? ` of ${kTokens(window)} (${Math.round((used.inputTokens / window) * 100)}%)` : ''} at the last answer`
        : undefined
    const isNearFull = Boolean(window && used && used.model === selKey && used.inputTokens >= window * CONTEXT_WARN)

    const header = (
      <Box flexDirection="column">
        <Text bold>{sel.provider === 'proxy' ? `Using ${sel.model} via ${selName}` : 'Using Claude'}</Text>
        {sel.provider === 'proxy' && <Text dimColor>{describe(info)}</Text>}
        {usedLine && <Text dimColor={!isNearFull}>{isNearFull ? `⚠ ${usedLine}` : usedLine}</Text>}
        {info.image === false && <Text dimColor>Images in the conversation are left out for this model.</Text>}
        {info.tools === false && <Text dimColor>This model gets no tools: it can only answer in text.</Text>}
      </Box>
    )

    const providerRows = list.map(p => {
      const cat = all[p.id]
      const state = cat?.isLoading
        ? 'loading…'
        : cat?.error
          ? `⚠ ${cat.error}`
          : `${cat?.models.length ?? 0} chat model${cat?.models.length === 1 ? '' : 's'}`
      const keyText = p.keyHint ? `key …${p.keyHint}` : 'no key'
      return (
        <Box key={`provider:${p.id}`} flexDirection="column">
          <Text>
            {p.name} <Text dimColor>· {p.url} · {keyText}{p.source === 'config' ? ' · from plugin config' : ''}</Text>
          </Text>
          <Text dimColor={!cat?.error}>{state}</Text>
          {p.source === 'pane' && (
            <Box flexDirection="row">
              {p.keyHint && (
                <Button key={`forget:${p.id}`} label="Forget key" dimColor onPress={() => void forgetKey($, p.id)} />
              )}
              <Button key={`remove:${p.id}`} label="Remove" dimColor onPress={() => void removeProvider($, p.id)} />
            </Box>
          )}
        </Box>
      )
    })

    // Mobile draws no Input or Select: one button per model, and providers are set elsewhere.
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
          {reload}
        </Box>
      )
    }
    const { Input, Select } = elements
    const override = set[selKey]

    return (
      <Box flexDirection="column">
        {header}
        <Select key="model" label="Model" options={choices} value={current} onSelect={value => void choose($, value)} />
        {sel.provider === 'proxy' && (
          <Box flexDirection="column">
            <Select
              key="vision"
              label="Vision"
              options={[
                {
                  value: 'auto',
                  label: `as reported (${reported.image === undefined ? 'unknown' : reported.image ? 'yes' : 'no'})`,
                },
                { value: 'yes', label: 'yes' },
                { value: 'no', label: 'no' },
              ]}
              value={override?.image === undefined ? 'auto' : override.image ? 'yes' : 'no'}
              onSelect={value => void setVision($, selKey, value)}
            />
            <Input
              key="context"
              label="Context window"
              placeholder={`as reported (${reported.contextWindow ? kTokens(reported.contextWindow) : 'unknown'}), e.g. 128k`}
              value={override?.contextWindow ? kTokens(override.contextWindow) : ''}
              submitLabel="save"
              onSubmit={value => void setContextWindow($, selKey, value)}
            />
          </Box>
        )}
        <Text> </Text>
        <Text bold>Providers</Text>
        {providerRows.length ? providerRows : <Text dimColor>None yet: add one below.</Text>}
        {reload}
        <Text> </Text>
        <Text bold>Add or update a provider</Text>
        <Input
          key="new-name"
          label="Name"
          placeholder="e.g. DeepSeek (same name replaces)"
          value={d.name}
          onInput={value => void setDraft($, 'name', value)}
          onSubmit={value => void setDraft($, 'name', value)}
        />
        <Input
          key="new-url"
          label="URL"
          placeholder="https://api.deepseek.com (no path → /v1 is added)"
          value={d.url}
          onInput={value => void setDraft($, 'url', value)}
          onSubmit={value => void setDraft($, 'url', value)}
        />
        <Input
          key="new-key"
          label="API key"
          placeholder="optional; empty keeps the saved key"
          value={d.key}
          submitLabel="save"
          onInput={value => void setDraft($, 'key', value)}
          onSubmit={value => void submitKey($, value)}
        />
        <Button key="save-provider" label="Save provider" onPress={() => void saveDraft($)} />
      </Box>
    )
  })
}

/** Enter in the key field: take the key and save the provider. */
async function submitKey($: EngineInterface, value: string) {
  await setDraft($, 'key', value)
  await saveDraft($)
}
