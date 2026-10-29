import { atom, read, update } from 'claude-code'
import type { EngineInterface, PluginOptions, Register, RenderElement } from 'claude-code'

import type { Catalog, ContextItem, ContextSeen, ContextUse, Draft, ModelInfo, Override, ProviderView, Selection } from '../types'
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
  wantsReasoning,
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
const contextOff = atom({ plugin: 'provider-switch', key: 'contextOff' } as const, {} as Record<string, string[]>)
const contextSeen = atom({ plugin: 'provider-switch', key: 'contextSeen' } as const, { tools: [], sections: [], injected: [] } as ContextSeen)
const isContextOpen = atom({ plugin: 'provider-switch', key: 'isContextOpen' } as const, false)
const openServers = atom({ plugin: 'provider-switch', key: 'openServers' } as const, [] as string[])
const draft = atom({ plugin: 'provider-switch', key: 'draft' } as const, { name: '', url: '', key: '' } as Draft)
const editing = atom({ plugin: 'provider-switch', key: 'editing' } as const, null as string | null)
const removing = atom({ plugin: 'provider-switch', key: 'removing' } as const, null as string | null)

// Kept across sessions in the plugin's store (plain JSON on this machine).
// The provider set in the plugin's config keeps its key in Claude Code's secure storage.
const STORE_PROVIDERS = 'providers' // { id, name, url }[]
const STORE_KEYS = 'keys' // { [providerId]: key }
const STORE_OVERRIDES = 'overrides' // { [providerId/model]: Override }
const STORE_SELECTION = 'selection'
const STORE_LAST_MODELS = 'lastModels' // { [providerId]: model }, what picking the provider again selects
const STORE_CONTEXT_OFF = 'contextOff' // { [providerId | 'claude']: ContextItem ids switched off }
const STORE_CONTEXT_OPEN = 'contextOpen'
// Before the Context section: { [providerId]: tool names switched off }.
const LEGACY_DISABLED_TOOLS = 'disabledTools'
// Before named providers: one URL and key typed in the pane.
const LEGACY_URL = 'proxyUrl'
const LEGACY_KEY = 'apiKey'

const CLAUDE = 'claude'
const PROXY_PREFIX = 'proxy:'
const CONFIG_ID = 'config'
// `editing` while the add form is open; no slug can spell it.
const NEW = '+new'
const EMPTY_DRAFT: Draft = { name: '', url: '', key: '' }

// Injected kinds that are the conversation itself (a mentioned file, a queued prompt, a mode
// the person chose), never offered to switch off.
const KEPT_INJECTIONS = /^(file|edited_text_file|queued_command|plan_mode.*|auto_mode.*)$/

// What the Lean preset keeps: the tools a coding turn needs (ToolSearch, so Claude can still
// load the rest), and the context blocks worth their tokens. System prompt sections all stay.
const LEAN_TOOLS = new Set(['Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep', 'ToolSearch'])
const LEAN_BLOCKS = new Set(['claudeMd', 'currentDate'])

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
// A thinking model's `reasoning_content`, by the id of each tool call its reply made: the
// transcript keeps no thinking a hook streams, so the next request gets it back from here.
const reasonings = new Map<string, string>()
// The most tool calls whose reasoning is kept; older ones are dropped first.
const REASONINGS_KEPT = 500
// Providers whose URL or key changed in their open form: their models reload when it closes.
const staleIds = new Set<string>()
// Below this, a context window being typed is taken as not finished yet (`12` on the way to `128k`).
const MIN_CONTEXT = 1000
// The pane's label column, wide enough for "Context window", and the least a control
// beside it gets before it moves under its label.
const LABEL_WIDTH = 15
const CONTROL_MIN_WIDTH = 20

/** `90000` → `90k`. */
function kTokens(n: number): string {
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n)
}

/** `My DeepSeek` → `my-deepseek`; never `config` (the plugin's config) or `claude` (Claude's context settings). */
function slug(name: string): string {
  const base = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'provider'
  return base === CONFIG_ID || base === CLAUDE ? `${base}-2` : base
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

/**
 * The tools of this session a provider's model can call, in chat-completions form:
 * those with a known schema, less the ones switched off for the provider.
 */
async function offeredTools($: EngineInterface, off: readonly string[] = []): Promise<OpenAiTool[]> {
  return (await $.tool.list())
    .filter(t => KNOWN_SCHEMAS[t.name] && !off.includes(`tool:${t.name}`))
    .map(t => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: KNOWN_SCHEMAS[t.name]! },
    }))
}

/** About how many prompt tokens offering `tool` costs each request. */
function toolTokens(tool: OpenAiTool): number {
  return estimateTokens(JSON.stringify(tool))
}

/** Whose context settings apply now: the selected provider's id, or `claude`. */
function targetOf(sel: Selection): string {
  return sel.provider === 'proxy' ? sel.providerId : CLAUDE
}

async function offNow($: EngineInterface): Promise<readonly string[]> {
  return (await read($, contextOff))[targetOf(await read($, selection))] ?? []
}

/** Switch context items on or off for `target`: `ids` flip to `isOff`, from the next request. */
async function setContextOff($: EngineInterface, target: string, ids: readonly string[], isOff: boolean) {
  const all = ((await $.store.get(STORE_CONTEXT_OFF)) as Record<string, string[]> | undefined) ?? {}
  const kept = (all[target] ?? []).filter(n => !ids.includes(n))
  const off = isOff ? [...kept, ...ids] : kept
  const { [target]: _, ...rest } = all
  const next = off.length ? { ...rest, [target]: off } : rest
  await $.store.set(STORE_CONTEXT_OFF, next)
  await update($, contextOff, () => next)
  // The engine keeps these answers until asked again; system prompt sections are asked every request.
  if (ids.some(id => id.startsWith('inject:'))) $.ui.invalidate('prompt.attachment')
  if (ids.some(id => id.startsWith('block:'))) $.ui.invalidate('prompt.context')
  if (target === CLAUDE && ids.some(id => id.startsWith('tool:'))) $.ui.invalidate('tool.describe')
}

/** Record one item a request carried, keeping the pane's lists in step without needless redraws. */
/**
 * Items in their order, each MCP server's tools (`mcp__<server>__<tool>`) gathered at the
 * place of its first one: `['', [item]]` for an item of its own, `[server, items]` for a server.
 */
function byServer(items: readonly ContextItem[]): [string, ContextItem[]][] {
  const out: [string, ContextItem[]][] = []
  const servers = new Map<string, ContextItem[]>()
  for (const item of items) {
    const server = /^mcp__(.+?)__./.exec(item.label)?.[1]
    if (!server) {
      out.push(['', [item]])
      continue
    }
    const known = servers.get(server)
    if (known) known.push(item)
    else {
      const list = [item]
      servers.set(server, list)
      out.push([server, list])
    }
  }
  return out
}

/**
 * The items the Lean preset switches off: tools beyond LEAN_TOOLS, context blocks beyond
 * LEAN_BLOCKS, hooks' and plugins' context, and, for a provider (which gets no Skill or
 * ToolSearch), the engine's reminders and listings too.
 */
function leanOff(isClaude: boolean, tools: readonly ContextItem[], injected: readonly ContextItem[]): string[] {
  const isKept = (id: string) => {
    if (id.startsWith('block:')) return LEAN_BLOCKS.has(id.slice('block:'.length))
    return isClaude && id.startsWith('inject:') && !/^inject:(hook|plugin):/.test(id)
  }
  return [
    ...tools.filter(t => !LEAN_TOOLS.has(t.label)).map(t => t.id),
    ...injected.filter(i => !isKept(i.id)).map(i => i.id),
  ]
}

async function noteSeen($: EngineInterface, group: 'tools' | 'injected', item: ContextItem) {
  const seen = await read($, contextSeen)
  const before = seen[group].find(i => i.id === item.id)
  if (before && before.tokens === item.tokens && before.label === item.label) return
  await update($, contextSeen, all => ({
    ...all,
    [group]: before ? all[group].map(i => (i.id === item.id ? item : i)) : [...all[group], item],
  }))
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

/** What the bar above the prompt shows: the provider's name and small labels for the model. */
async function barView($: EngineInterface): Promise<{ name: string; labels: string[]; isNearFull: boolean }> {
  const sel = await read($, selection)
  if (sel.provider !== 'proxy') return { name: 'Claude', labels: [], isNearFull: false }

  const name = (await read($, providers)).find(p => p.id === sel.providerId)?.name ?? sel.providerId
  const key = modelKey(sel.providerId, sel.model)
  const info = effectiveInfo((await read($, catalogs))[sel.providerId], (await read($, overrides))[key], sel.model)
  const used = await read($, context)
  const labels = [sel.model]
  if (info.image) labels.push('vision')
  if (info.tools) labels.push('tools')
  if (info.reasoning) labels.push('reasoning')
  const window = info.contextWindow
  const isUsed = Boolean(window && used && used.model === key)
  if (window) {
    labels.push(isUsed ? `${Math.round((used!.inputTokens / window) * 100)}% of ${kTokens(window)}` : `${kTokens(window)} ctx`)
  }
  return { name, labels, isNearFull: isUsed && used!.inputTokens >= window! * CONTEXT_WARN }
}

async function choose($: EngineInterface, value: string) {
  const next = parseChoice(value)
  await update($, selection, () => next)
  await $.store.set(STORE_SELECTION, next)
  // Each provider has its own context settings: the engine asks again for what it injects.
  $.ui.invalidate('prompt.attachment')
  $.ui.invalidate('prompt.context')
  if (next.provider === 'claude') return $.ui.toast('Model requests now go to Claude')
  const last = ((await $.store.get(STORE_LAST_MODELS)) as Record<string, string> | undefined) ?? {}
  await $.store.set(STORE_LAST_MODELS, { ...last, [next.providerId]: next.model })
  const name = (await read($, providers)).find(p => p.id === next.providerId)?.name ?? next.providerId
  $.ui.toast(`Model requests now go to ${next.model} via ${name}`)
}

/** Select a provider: the model last used with it, else its first. */
async function pickProvider($: EngineInterface, id: string) {
  if (id === CLAUDE) return choose($, CLAUDE)
  const sel = await read($, selection)
  if (sel.provider === 'proxy' && sel.providerId === id) return
  const cat = (await read($, catalogs))[id]
  const name = (await read($, providers)).find(p => p.id === id)?.name ?? id
  if (!cat?.models.length) return $.ui.toast(`${name} lists no chat models${cat?.error ? ': fix it with Edit' : ' yet'}`)
  const last = ((await $.store.get(STORE_LAST_MODELS)) as Record<string, string> | undefined)?.[id]
  await choose($, `${PROXY_PREFIX}${id}/${last && cat.models.includes(last) ? last : cat.models[0]}`)
}

/** Open the form of a provider (or the add form, `NEW`), or close it when it is open. */
async function toggleForm($: EngineInterface, id: string) {
  const open = await read($, editing)
  await update($, removing, () => null)
  if (open === id) {
    await update($, editing, () => null)
    await update($, draft, () => EMPTY_DRAFT)
  } else {
    const p = (await read($, providers)).find(x => x.id === id)
    await update($, draft, () => (p ? { name: p.name, url: p.url, key: '' } : EMPTY_DRAFT))
    await update($, editing, () => id)
  }
  // The form it leaves was saved as typed; its models follow a new URL or key now.
  if (open && staleIds.delete(open)) await reloadOne($, open)
}

/** Record how full the context is and warn once when it nears the window. */
async function noteContext($: EngineInterface, key: string, inputTokens: number, window: number | undefined) {
  await update($, context, () => ({ model: key, inputTokens }))

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

/**
 * A field of an existing provider's form, as typed: kept at once, so the form needs no Save.
 * A name or URL left empty keeps the last one; an empty key keeps the stored key.
 */
async function editField($: EngineInterface, field: keyof Draft, value: string) {
  await setDraft($, field, value)
  const id = await read($, editing)
  if (!id || id === NEW) return
  const stored = await storedProviders($)
  const before = stored.find(p => p.id === id)
  if (!before) return

  const d = await read($, draft)
  const name = d.name.trim() || before.name
  const url = d.url.trim() || before.url
  await $.store.set(STORE_PROVIDERS, stored.map(p => (p.id === id ? { id, name, url } : p)))
  const key = field === 'key' ? value.trim() : ''
  if (key) await $.store.set(STORE_KEYS, { ...(await storedKeys($)), [id]: key })
  if (url !== before.url || key) staleIds.add(id)
  await syncProviders($)
}

/** Save the open form: a new provider, or the one being edited; an empty key keeps the old one. */
async function saveDraft($: EngineInterface) {
  const d = await read($, draft)
  const name = d.name.trim()
  const url = d.url.trim()
  if (!name || !url) return $.ui.toast('A provider needs a name and a URL')

  const stored = await storedProviders($)
  const open = await read($, editing)
  let id = open && open !== NEW ? open : slug(name)
  if (open === NEW || !open) {
    // A new provider never replaces one: `deepseek`, `deepseek-2`, …
    const base = id
    for (let n = 2; stored.some(p => p.id === id); n++) id = `${base}-${n}`
  }
  const at = stored.findIndex(p => p.id === id)
  const list = at < 0 ? [...stored, { id, name, url }] : stored.map(p => (p.id === id ? { id, name, url } : p))
  await $.store.set(STORE_PROVIDERS, list)
  if (d.key.trim()) await $.store.set(STORE_KEYS, { ...(await storedKeys($)), [id]: d.key.trim() })
  await update($, draft, () => EMPTY_DRAFT)
  await update($, editing, () => null)

  const added = (await syncProviders($)).find(p => p.id === id)
  if (added) await reloadProvider($, added)
  await checkSelection($)
  $.ui.toast(`Saved provider ${name}`)
}

/** Ask before removing `id`, or stop asking (`null`). */
async function askRemove($: EngineInterface, id: string | null) {
  await update($, removing, () => id)
}

async function removeProvider($: EngineInterface, id: string) {
  await update($, removing, () => null)
  await $.store.set(STORE_PROVIDERS, (await storedProviders($)).filter(p => p.id !== id))
  const { [id]: _, ...keys } = await storedKeys($)
  await $.store.set(STORE_KEYS, keys)
  if ((await read($, editing)) === id) await update($, editing, () => null)
  await syncProviders($)
  await update($, catalogs, ({ [id]: __, ...rest }) => rest)
  await checkSelection($)
}

async function forgetKey($: EngineInterface, id: string) {
  const { [id]: _, ...keys } = await storedKeys($)
  await $.store.set(STORE_KEYS, keys)
  await setDraft($, 'key', '')
  staleIds.delete(id)
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
}

async function setVision($: EngineInterface, key: string, value: string) {
  await setOverride($, key, { image: value === 'yes' ? true : value === 'no' ? false : undefined })
}

/** The context window as typed: kept once it reads as a whole window, or cleared when emptied. */
async function typeContextWindow($: EngineInterface, key: string, value: string) {
  const n = parseTokens(value)
  if (!value.trim() || (n !== undefined && n >= MIN_CONTEXT)) await setOverride($, key, { contextWindow: n })
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
    const legacyTools = (await $.store.get(LEGACY_DISABLED_TOOLS)) as Record<string, string[]> | undefined
    if (legacyTools) {
      const migrated = Object.fromEntries(Object.entries(legacyTools).map(([id, names]) => [id, names.map(n => `tool:${n}`)]))
      await $.store.set(STORE_CONTEXT_OFF, { ...migrated, ...((await $.store.get(STORE_CONTEXT_OFF)) as object | undefined) })
      await $.store.delete(LEGACY_DISABLED_TOOLS)
    }
    const off = ((await $.store.get(STORE_CONTEXT_OFF)) as Record<string, string[]> | undefined) ?? {}
    await update($, contextOff, () => off)
    const isOpen = ((await $.store.get(STORE_CONTEXT_OPEN)) as boolean | undefined) ?? false
    await update($, isContextOpen, () => isOpen)
    // A resumed thread sends the tool placement it recorded: ask again so Claude's switched-off tools stay deferred.
    if (off[CLAUDE]?.length) $.ui.invalidate('tool.describe')
    await syncProviders($)
    // The backend now shows as a button above the prompt; clear the line an older version set.
    $.ui.status(undefined)
    // The model lists are refreshed at every start; a failed fetch shows in the pane,
    // and one cut short by the session ending has no one left to tell.
    // The pane opens only when asked for: the ⚙ above the prompt or the command.
    reloadAll($).catch(() => {})

    return next(e)
  })

  on('command.run', { command: COMMAND }, async ($, e) => {
    const arg = e.args.trim()
    if (!arg) {
      await $.ui.open({ id: PANE, title: 'Provider Switch', focus: true })
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

  // The system prompt, less the sections switched off for whoever answers now.
  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    const sections = composed.sections.map(s => ({ id: `section:${s.id}`, label: s.id, tokens: estimateTokens(s.text) }))
    const seen = await read($, contextSeen)
    if (JSON.stringify(seen.sections) !== JSON.stringify(sections)) await update($, contextSeen, all => ({ ...all, sections }))

    const off = await offNow($)
    const kept = composed.sections.filter(s => !off.includes(`section:${s.id}`))
    systemPrompt = kept.map(s => s.text).join('\n\n')
    return { sections: kept }
  })

  // Reminders, listings and hook context the engine adds to the conversation on its own.
  on('prompt.attachment', async ($, e, next) => {
    const answer = await next(e)
    if (answer.text === null || KEPT_INJECTIONS.test(e.type)) return answer
    const kind = e.origin.kind === 'engine' ? e.type : `${e.origin.kind}:${e.origin.event}`
    const label = e.origin.kind === 'hook' ? `${e.origin.event} hook` : e.origin.kind === 'plugin' ? `plugin (${e.origin.event})` : e.type
    const id = `inject:${kind}`
    if (e.agentId === undefined) await noteSeen($, 'injected', { id, label, tokens: estimateTokens(answer.text) })
    return (await offNow($)).includes(id) ? { text: null } : answer
  })

  // The blocks the first user message carries: CLAUDE.md, the user's email, the date, ...
  on('prompt.context', async ($, e, next) => {
    const answer = await next(e)
    for (const b of answer.blocks) {
      await noteSeen($, 'injected', { id: `block:${b.name}`, label: b.name, tokens: estimateTokens(b.text) })
    }
    const off = await offNow($)
    return { ...answer, blocks: answer.blocks.filter(b => !off.includes(`block:${b.name}`)) }
  })

  // Claude's tool schemas: one switched off waits behind ToolSearch, loaded only when asked for.
  on('tool.describe', async ($, e, next) => {
    const answer = await next(e)
    if (e.isDeferred || answer.isDeferred === true) return answer
    const schema = KNOWN_SCHEMAS[e.tool]
    const tokens = estimateTokens(answer.description + (schema ? JSON.stringify(schema) : ''))
    await noteSeen($, 'tools', { id: `tool:${e.tool}`, label: e.tool, tokens })
    const off = (await read($, contextOff))[CLAUDE] ?? []
    return off.includes(`tool:${e.tool}`) ? { ...answer, isDeferred: true } : answer
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
    const tools = info.tools === false ? [] : await offeredTools($, (await read($, contextOff))[provider.id])

    const request = (fillReasoning: boolean) =>
      JSON.stringify({
        model: sel.model,
        messages: toOpenAiMessages(systemPrompt, history as unknown as AnthropicMessage[], {
          images: info.image !== false,
          reasoning: Object.fromEntries(reasonings),
          fillReasoning,
        }),
        ...(tools.length ? { tools } : {}),
        stream: false,
      })

    const payload = request(false)
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

    const send = (body: string) =>
      $.http.fetch(`${apiBase(provider.url)}/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...authHeaders(provider.key) },
        body,
      })

    let reply
    try {
      let res = await send(payload)
      // A thinking model wants reasoning back for a tool call made before it was kept
      // (an earlier session, another model): send those with an empty one, once.
      if (!res.ok && wantsReasoning(res.status, res.text)) res = await send(request(true))
      if (!res.ok) throw new Error(errorText(res.status, res.text))
      reply = parseReply(res.text, sel.model)
    } catch (error) {
      const why = `⚠ provider-switch: ${sel.model} (${provider.name}) failed: ${error instanceof Error ? error.message : String(error)}`
      yield { kind: 'text', index: 0, text: why }
      yield { kind: 'stop', stopReason: 'end_turn', usage: null }
      return fail(why)
    }

    await noteContext($, key, reply.usage.input_tokens || estimate, info.contextWindow)

    if (reply.reasoning) {
      for (const call of reply.toolCalls) reasonings.set(call.id, reply.reasoning)
      for (const id of reasonings.keys()) {
        if (reasonings.size <= REASONINGS_KEPT) break
        reasonings.delete(id)
      }
    }

    let block = 0
    // Shown live as thinking; the transcript does not keep it.
    if (reply.reasoning) yield { kind: 'thinking', index: block++, text: reply.reasoning }
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

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey) return next(e)

    const { Box, Text, Button } = $.ui.resolve(e)
    const bar = await barView($)
    return (
      <Box flexDirection="row" justifyContent="space-between" alignItems="center" columnGap={1}>
        <Box flexDirection="row" flexWrap="wrap" alignItems="center" columnGap={1} flexShrink={1} minWidth={0}>
          <Text>
            Provider Switch: <Text bold>{bar.name}</Text>
          </Text>
          {bar.labels.map((label, i) => (
            <Text
              size="caption"
              color={bar.isNearFull && i === bar.labels.length - 1 ? 'warning' : 'suggestion'}
              dimColor={i > 0 && !(bar.isNearFull && i === bar.labels.length - 1)}
            >
              {` ${label} `}
            </Text>
          ))}
        </Box>
        <Button
          key="open-pane"
          label="⚙"
          plain
          dimColor
          onPress={() => void $.ui.open({ id: PANE, title: 'Provider Switch', focus: true })}
        />
      </Box>
    )
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
    const open = await read($, editing)
    const confirming = await read($, removing)
    const target = targetOf(sel)
    const off = (await read($, contextOff))[target] ?? []
    const seen = await read($, contextSeen)
    const isOpenContext = await read($, isContextOpen)
    const shownServers = await read($, openServers)
    const proxyTools = sel.provider === 'proxy' ? await offeredTools($) : []

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
    const isLoading = list.some(p => all[p.id]?.isLoading)

    const summary = (
      <Box flexDirection="column">
        <Text bold>{sel.provider === 'proxy' ? `Using ${sel.model} via ${selName}` : 'Using Claude'}</Text>
        {sel.provider === 'proxy' && <Text dimColor>{describe(info)}</Text>}
        {usedLine && (
          <Text color={isNearFull ? 'warning' : undefined} dimColor={!isNearFull}>
            {isNearFull ? `⚠ ${usedLine}` : usedLine}
          </Text>
        )}
        {info.image === false && <Text dimColor>Images in the conversation are left out for this model.</Text>}
        {info.tools === false && <Text dimColor>This model gets no tools: it can only answer in text.</Text>}
      </Box>
    )

    /**
     * Context: what every request to the selected provider carries before the conversation
     * (tools, system prompt sections, injected reminders and context blocks), each switched
     * on or off with what it costs. Folded by default; its header shows the total.
     */
    const contextSection = () => {
      const targetName = sel.provider === 'proxy' ? selName : 'Claude'
      const tools: ContextItem[] =
        sel.provider === 'proxy'
          ? info.tools === false
            ? []
            : proxyTools.map(t => ({ id: `tool:${t.function.name}`, label: t.function.name, tokens: toolTokens(t) }))
          : seen.tools
      const groups = [
        {
          key: 'tools',
          title: 'Tools',
          items: tools,
          empty:
            sel.provider === 'proxy'
              ? info.tools === false
                ? 'This model gets no tools.'
                : 'No tool in this session has a schema the mod knows.'
              : 'Listed once Claude Code has sent its tools.',
        },
        { key: 'sections', title: 'System prompt', items: seen.sections, empty: 'Listed after the first request.' },
        { key: 'injected', title: 'Injected context', items: seen.injected, empty: 'Listed after the first request.' },
      ]
      const sum = (items: readonly ContextItem[]) => items.filter(i => !off.includes(i.id)).reduce((n, i) => n + i.tokens, 0)
      const total = groups.reduce((n, g) => n + sum(g.items), 0)
      const everything = groups.reduce((n, g) => n + g.items.reduce((m, i) => m + i.tokens, 0), 0)
      // One preset button: Lean while everything is on, Restore all once anything is off.
      const lean = leanOff(sel.provider !== 'proxy', tools, seen.injected)
      const leanSaves = groups.flatMap(g => g.items).filter(i => lean.includes(i.id)).reduce((n, i) => n + i.tokens, 0)
      const preset = off.length ? (
        <Button key="context-restore" label="Restore all" onPress={() => void setContextOff($, target, off, false)} />
      ) : lean.length ? (
        <Button key="context-lean" label={`Lean · saves ~${kTokens(leanSaves)}`} onPress={() => void setContextOff($, target, lean, true)} />
      ) : null
      const itemRow = (item: ContextItem) => {
        const isOn = !off.includes(item.id)
        return (
          <Button
            key={item.id}
            label={`${isOn ? '☑' : '☐'} ${item.label} · ~${kTokens(item.tokens)}`}
            plain
            dimColor={!isOn}
            onPress={() => void setContextOff($, target, [item.id], isOn)}
          />
        )
      }
      // An MCP server's tools as one row that switches them all, unfolded to switch each.
      const serverRow = (server: string, items: readonly ContextItem[]) => {
        const onCount = items.filter(i => !off.includes(i.id)).length
        const isOpenServer = shownServers.includes(server)
        const mark = onCount === items.length ? '☑' : onCount ? '◐' : '☐'
        return (
          <Box key={`server-row:${server}`} flexDirection="column">
            <Box flexDirection="row" alignItems="center" columnGap={1}>
              <Button
                key={`server-fold:${server}`}
                label={isOpenServer ? '▾' : '▸'}
                plain
                dimColor
                onPress={() => void update($, openServers, v => (v.includes(server) ? v.filter(x => x !== server) : [...v, server]))}
              />
              <Button
                key={`server:${server}`}
                label={`${mark} ${server} (MCP, ${onCount}/${items.length} tools) · ~${kTokens(sum(items))}`}
                plain
                dimColor={!onCount}
                onPress={() => void setContextOff($, target, items.map(i => i.id), onCount === items.length)}
              />
            </Box>
            {isOpenServer && (
              <Box flexDirection="column" paddingLeft={4}>
                {items.map(item => itemRow({ ...item, label: item.label.slice(`mcp__${server}__`.length) }))}
              </Box>
            )}
          </Box>
        )
      }
      const toggle = () => {
        void update($, isContextOpen, v => !v)
        void $.store.set(STORE_CONTEXT_OPEN, !isOpenContext)
      }
      return (
        <Box key="context" flexDirection="column" rowGap="xs" borderStyle="round" borderDimColor paddingX={1}>
          <Box flexDirection="row" flexWrap="wrap" justifyContent="space-between" alignItems="center" columnGap={1}>
            <Button key="context-toggle" plain onPress={toggle}>
              <Text bold>{`${isOpenContext ? '▾' : '▸'} Context for ${targetName}`}</Text>
            </Button>
            <Text size="caption" dimColor>
              {total === everything ? `~${kTokens(total)} per request` : `~${kTokens(total)} of ~${kTokens(everything)} per request`}
            </Text>
          </Box>
          {isOpenContext && (
            <Box flexDirection="column" rowGap="sm">
              <Text size="caption" dimColor>
                {sel.provider === 'proxy'
                  ? `Sent to ${targetName} with every request, before the conversation. Off items are left out.`
                  : 'Sent to Claude with every request, before the conversation. Off tools wait behind ToolSearch; other off items are left out.'}
              </Text>
              {preset && (
                <Box flexDirection="row" justifyContent="center">
                  {preset}
                </Box>
              )}
              {groups.map(g => {
                const ids = g.items.map(i => i.id)
                return (
                  <Box key={`group:${g.key}`} flexDirection="column" rowGap="xs">
                    <Box flexDirection="row" flexWrap="wrap" justifyContent="space-between" alignItems="center" columnGap={1}>
                      <Text size="caption" dimColor>
                        {`${g.title} · ~${kTokens(sum(g.items))}`}
                      </Text>
                      {g.items.length > 0 && (
                        <Box flexDirection="row" flexWrap="wrap" columnGap={1}>
                          <Button key={`${g.key}-all`} label="All" plain dimColor onPress={() => void setContextOff($, target, ids, false)} />
                          <Button key={`${g.key}-none`} label="None" plain dimColor onPress={() => void setContextOff($, target, ids, true)} />
                        </Box>
                      )}
                    </Box>
                    {g.items.length ? (
                      <Box flexDirection="column">
                        {byServer(g.items).map(([server, items]) => (server ? serverRow(server, items) : items.map(itemRow)))}
                      </Box>
                    ) : (
                      <Text size="caption" dimColor>
                        {g.empty}
                      </Text>
                    )}
                  </Box>
                )
              })}
              <Text size="caption" dimColor>
                {`~${kTokens(total)} tokens per request, plus the conversation itself`}
              </Text>
            </Box>
          )}
        </Box>
      )
    }

    /** One provider's status: `3 chat models`, `loading…` or its error. */
    const stateOf = (id: string) => {
      const cat = all[id]
      if (cat?.isLoading) return 'loading…'
      if (cat?.error) return `⚠ ${cat.error}`
      const n = cat?.models.length ?? 0
      return `${n} chat model${n === 1 ? '' : 's'}`
    }

    // Mobile draws no Input or Select: one button per provider, then one per model of the selected one.
    if (!('Select' in elements) || !('Input' in elements)) {
      const models = sel.provider === 'proxy' ? (all[sel.providerId]?.models ?? []) : []
      return (
        <Box flexDirection="column" gap="sm">
          {summary}
          <Button key={`pick:${CLAUDE}`} label={`${sel.provider === 'claude' ? '●' : '○'} Claude`} onPress={() => void pickProvider($, CLAUDE)} />
          {list.map(p => (
            <Button
              key={`pick:${p.id}`}
              label={`${sel.provider === 'proxy' && sel.providerId === p.id ? '●' : '○'} ${p.name} · ${stateOf(p.id)}`}
              onPress={() => void pickProvider($, p.id)}
            />
          ))}
          {sel.provider === 'proxy' &&
            models.map(model => (
              <Button
                key={`${PROXY_PREFIX}${sel.providerId}/${model}`}
                label={model === sel.model ? `● ${model}` : `  ${model}`}
                dimColor={model !== sel.model}
                onPress={() => void choose($, `${PROXY_PREFIX}${sel.providerId}/${model}`)}
              />
            ))}
          <Button key="reload" label={isLoading ? 'Reloading…' : 'Reload all'} onPress={() => void reloadAll($)} />
          {contextSection()}
        </Box>
      )
    }
    const { Input, Select } = elements

    /**
     * A small label beside its control, right-aligned in a column of one width so every field
     * lines up; the control starts from no width and grows, so a long Select stays on the
     * label's row. A pane too narrow for both puts the control under its label. `below` sits centred
     * under the control, a little apart (a field's own action).
     */
    const field = (label: string, control: RenderElement, below?: RenderElement) => (
      <Box flexDirection="row" flexWrap="wrap" alignItems={below ? 'flex-start' : 'center'} columnGap={1}>
        <Box width={LABEL_WIDTH} flexShrink={0} flexDirection="row" justifyContent="flex-end">
          <Text size="caption" dimColor>
            {label}
          </Text>
        </Box>
        <Box flexDirection="column" flexGrow={1} flexShrink={1} width={0} minWidth={CONTROL_MIN_WIDTH}>
          {control}
          {below && (
            <Box flexDirection="row" justifyContent="center" marginTop="sm">
              {below}
            </Box>
          )}
        </Box>
      </Box>
    )

    /** The settings of the provider `id` (or a new one): name, URL, key and its actions. */
    const form = (p: ProviderView | undefined) => {
      const isConfig = p?.source === 'config'
      return (
        <Box flexDirection="column" rowGap="xs" marginTop="xs">
          {isConfig ? (
            <Box flexDirection="column" rowGap="xs">
              {field('URL', <Text>{p!.url}</Text>)}
              {field('API key', <Text dimColor>{p!.keyHint ? `…${p!.keyHint}` : 'none'}</Text>)}
              <Text dimColor size="caption">
                Set in the plugin's options; change it there.
              </Text>
            </Box>
          ) : (
            <Box flexDirection="column" rowGap="xs">
              {field(
                'Name',
                <Input
                  key="form-name"
                  placeholder="e.g. DeepSeek"
                  value={d.name}
                  onInput={value => void editField($, 'name', value)}
                  onSubmit={value => void submitField($, 'name', value)}
                />,
              )}
              {field(
                'URL',
                <Input
                  key="form-url"
                  placeholder="https://api.deepseek.com"
                  value={d.url}
                  onInput={value => void editField($, 'url', value)}
                  onSubmit={value => void submitField($, 'url', value)}
                />,
              )}
              {field(
                'API key',
                <Input
                  key="form-key"
                  placeholder={p?.keyHint ? 'type to replace the saved key' : 'optional'}
                  value={d.key}
                  onInput={value => void editField($, 'key', value)}
                  onSubmit={value => void submitField($, 'key', value)}
                />,
                p?.keyHint ? (
                  <Box flexDirection="row" flexWrap="wrap" justifyContent="center" alignItems="center" columnGap={1}>
                    <Text size="caption" dimColor>
                      Saved key …{p.keyHint}
                    </Text>
                    <Button key={`forget:${p.id}`} variant="secondary" onPress={() => void forgetKey($, p.id)}>
                      <Text size="caption">Forget</Text>
                    </Button>
                  </Box>
                ) : undefined,
              )}
            </Box>
          )}
          {p && !isConfig && confirming === p.id ? (
            <Box flexDirection="column" rowGap="xs" marginTop="xs">
              <Text color="warning">Remove {p.name} and its key? This can't be undone.</Text>
              <Box flexDirection="row" flexWrap="wrap" columnGap={1}>
                <Button key={`remove-confirm:${p.id}`} label="Remove" variant="primary" onPress={() => void removeProvider($, p.id)} />
                <Button key={`remove-cancel:${p.id}`} label="Cancel" onPress={() => void askRemove($, null)} />
              </Box>
            </Box>
          ) : (
            <Box flexDirection="row" flexWrap="wrap" justifyContent="space-between" alignItems="center" columnGap={1} marginTop="xs">
              <Box flexDirection="row" flexWrap="wrap" alignItems="center" columnGap={1}>
                {!p && <Button key="save-provider" label="Add provider" variant="primary" onPress={() => void saveDraft($)} />}
                {p && (
                  <Button
                    key={`reload:${p.id}`}
                    label={all[p.id]?.isLoading ? 'Reloading…' : 'Reload'}
                    onPress={() => void reloadOne($, p.id)}
                  />
                )}
                {p && !isConfig && (
                  <Text size="caption" dimColor>
                    Saved as you type
                  </Text>
                )}
              </Box>
              {p && !isConfig && (
                <Button key={`remove:${p.id}`} label="🗑" plain dimColor onPress={() => void askRemove($, p.id)} />
              )}
            </Box>
          )}
        </Box>
      )
    }

    /** A provider's models as Select options, with each one's context window when known. */
    const modelOptions = (id: string, models: readonly string[]) =>
      models.map(model => {
        const w = effectiveInfo(all[id], set[modelKey(id, model)], model).contextWindow
        return { value: `${PROXY_PREFIX}${id}/${model}`, label: w ? `${model} · ${kTokens(w)}` : model }
      })

    /** The model picker of a provider not in use: picking one switches to it. */
    const modelPicker = (id: string) => {
      const models = all[id]?.models ?? []
      if (!models.length) return null
      return (
        <Box flexDirection="column" marginTop="xs">
          {field(
            'Model',
            <Select
              key={`model:${id}`}
              options={[{ value: '', label: 'not in use: pick a model' }, ...modelOptions(id, models)]}
              value=""
              onSelect={value => void (value && choose($, value))}
            />,
          )}
        </Box>
      )
    }

    /** The selected provider's model and the person's overrides for it. */
    const modelSettings = () => {
      if (sel.provider !== 'proxy') return null
      const models = all[sel.providerId]?.models ?? []
      const options = modelOptions(sel.providerId, models.includes(sel.model) ? models : [sel.model, ...models])
      const override = set[selKey]
      return (
        <Box flexDirection="column" rowGap="xs" marginTop="xs">
          {field(
            'Model',
            <Select
              key="model"
              options={options}
              value={`${PROXY_PREFIX}${sel.providerId}/${sel.model}`}
              onSelect={value => void choose($, value)}
            />,
          )}
          {field(
            'Vision',
            <Select
              key="vision"
              options={[
                { value: 'auto', label: `as reported (${reported.image === undefined ? 'unknown' : reported.image ? 'yes' : 'no'})` },
                { value: 'yes', label: 'yes' },
                { value: 'no', label: 'no' },
              ]}
              value={override?.image === undefined ? 'auto' : override.image ? 'yes' : 'no'}
              onSelect={value => void setVision($, selKey, value)}
            />,
          )}
          {field(
            'Context window',
            <Input
              key="context"
              placeholder={`reported: ${reported.contextWindow ? kTokens(reported.contextWindow) : 'unknown'}`}
              value={override?.contextWindow ? kTokens(override.contextWindow) : ''}
              onInput={value => void typeContextWindow($, selKey, value)}
              onSubmit={value => void setContextWindow($, selKey, value)}
            />,
          )}
        </Box>
      )
    }

    /** One selectable provider: a radio, its name and state, an Edit toggle; expanded below when open. */
    const row = (id: string, name: string, detail: string, p?: ProviderView) => {
      const isSelected = id === CLAUDE ? sel.provider === 'claude' : sel.provider === 'proxy' && sel.providerId === id
      const isError = Boolean(all[id]?.error)
      return (
        <Box
          key={`row:${id}`}
          flexDirection="column"
          borderStyle="round"
          borderColor={isSelected ? 'claude' : undefined}
          borderDimColor={!isSelected}
          paddingX={1}
        >
          <Box flexDirection="row" flexWrap="wrap" justifyContent="space-between" alignItems="center" columnGap={1}>
            <Box flexShrink={1} minWidth={0}>
              <Button key={`pick:${id}`} plain onPress={() => void pickProvider($, id)}>
                <Text color={isSelected ? 'claude' : undefined}>{isSelected ? '●' : '○'}</Text>{' '}
                <Text bold wrap="truncate-end">
                  {name}
                </Text>
              </Button>
            </Box>
            {p && <Button key={`edit:${id}`} label={open === id ? 'Close' : 'Edit'} dimColor={open !== id} onPress={() => void toggleForm($, id)} />}
          </Box>
          <Text size="caption" dimColor={!isError} color={isError ? 'error' : undefined} wrap="truncate-end">
            {detail}
          </Text>
          {isSelected ? modelSettings() : p && modelPicker(id)}
          {p && open === id && form(p)}
        </Box>
      )
    }

    return (
      <Box flexDirection="column" rowGap="sm">
        <Box flexDirection="row" flexWrap="wrap" justifyContent="space-between" alignItems="center" columnGap={1}>
          <Text size="heading" bold>
            Providers
          </Text>
          <Box flexDirection="row" flexWrap="wrap" columnGap={1}>
            <Button key="reload" label={isLoading ? 'Reloading…' : 'Reload all'} dimColor onPress={() => void reloadAll($)} />
            <Button
              key="add"
              label={open === NEW ? 'Cancel' : '+ Add'}
              variant={open === NEW ? undefined : 'primary'}
              onPress={() => void toggleForm($, NEW)}
            />
          </Box>
        </Box>
        {open === NEW && (
          <Box key="row:new" flexDirection="column" borderStyle="round" borderColor="suggestion" paddingX={1}>
            <Text bold>New provider</Text>
            {form(undefined)}
          </Box>
        )}
        {summary}
        {row(CLAUDE, 'Claude', 'Claude Code (Anthropic)')}
        {list.map(p =>
          row(
            p.id,
            p.name,
            `${stateOf(p.id)} · ${p.url} · ${p.keyHint ? `key …${p.keyHint}` : 'no key'}${p.source === 'config' ? ' · plugin config' : ''}`,
            p,
          ),
        )}
        {!list.length && open !== NEW && <Text dimColor>No providers yet: press + Add.</Text>}
        {contextSection()}
      </Box>
    )
  })
}

/** Enter in a field: adds the provider of the add form; an existing one is already saved. */
async function submitField($: EngineInterface, field: keyof Draft, value: string) {
  await editField($, field, value)
  if ((await read($, editing)) === NEW) await saveDraft($)
}

/** Reload one provider's models, then leave a model it no longer lists. */
async function reloadOne($: EngineInterface, id: string) {
  const p = (await syncProviders($)).find(x => x.id === id)
  if (p) await reloadProvider($, p)
  await checkSelection($)
}
