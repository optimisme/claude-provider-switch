import { expect, mock, test } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import type { On } from 'claude-code'

import {
  apiBase,
  authHeaders,
  isChatModel,
  parseCapabilities,
  parseModels,
  parseReply,
  parseTokens,
  toOpenAiMessages,
} from '../hooks/openai'

// The provider set in the plugin's options; a bare host, so `/v1` is added.
const OPTIONS = { proxyUrl: 'https://proxy.test', apiKey: 'ieti_sk_test' }

const PANE = {
  plugin: 'provider-switch',
  component: 'Pane',
  requestId: 'provider-switch',
  props: {
    title: 'Model provider',
    isFocused: false,
    bodyColumns: 60,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 60 },
    view: {},
  },
} as const

/** Every chunk a step's stream yields, and the result it returns. */
async function drain<C, R>(stream: AsyncGenerator<C, R>): Promise<{ chunks: C[]; result: R }> {
  const chunks: C[] = []
  for (;;) {
    const step = await stream.next()
    if (step.done) return { chunks, result: step.value }
    chunks.push(step.value)
  }
}

/** `/provider <args>` as the person types it. */
const provider = (args: string) => ({
  command: 'provider',
  args,
  origin: { kind: 'composer' } as const,
  presentation: { isFullscreen: false, columns: 80 },
})

const step = (index = 0) => ({ turnId: 't1', index, model: 'claude-opus-5-5', messageCount: 1 })

// The conversation in Messages API form, as `$.session.messages({ as: 'api' })` gives it.
type Message = { role: 'user' | 'assistant'; content: { type: string; [field: string]: unknown }[] }
const HISTORY: Message[] = [
  { role: 'user', content: [{ type: 'text', text: 'Read the README' }] },
  { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: '/r/README.md' } }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'hello' }] },
]
const IMAGE_HISTORY: Message[] = [
  {
    role: 'user',
    content: [
      { type: 'text', text: 'What is in this picture?' },
      { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } },
    ],
  },
]

// As IETI Agents Proxy answers `GET /v1/model-capabilities`.
const CAPABILITIES = {
  object: 'ieti.model_capabilities.list',
  data: [
    {
      id: 'seer',
      context_window: 90000,
      max_output_tokens: 8192,
      capabilities: { text: true, image: true, tools: true, reasoning: true },
      modalities: { input: ['text', 'image'], output: ['text'] },
    },
    {
      id: 'blind',
      context_window: 1000,
      max_output_tokens: 512,
      capabilities: { text: true, image: false, tools: false, reasoning: false },
    },
  ],
}

type Request = { url: string; body?: string; auth?: string }
type Answer = { status?: number; json?: unknown; text?: string }

const reply = (content: string, promptTokens = 10) => ({
  json: { choices: [{ finish_reason: 'stop', message: { content } }], usage: { prompt_tokens: promptTokens, completion_tokens: 3 } },
})

/**
 * The engine beneath the plugin for one test: store, clock, pane, the
 * conversation, the tools, and HTTP answered per URL (404 for any other).
 * Returns the requests the plugin made.
 */
function world(
  on: On,
  { routes, history = HISTORY, tools = ['Read'], sections = ['intro'], toasts, opened }: {
    routes: Record<string, Answer | ((r: Request) => Answer)>
    history?: Message[]
    tools?: string[]
    /** System prompt section ids; `intro` reads `You are Claude Code.`. */
    sections?: string[]
    toasts?: string[]
    opened?: unknown[]
  },
  store?: Record<string, unknown>,
): Request[] {
  const requests: Request[] = []
  mock.store(on, store)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', (_, e) => {
    toasts?.push(e.text)
    return { value: undefined }
  })
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', (_, e) => {
    opened?.push(e)
    return { value: { isPlaced: true } }
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('prompt.compose', () => ({
    sections: sections.map(id => ({ id, text: id === 'intro' ? 'You are Claude Code.' : `The ${id} section.`, scope: 'shared' as const })),
  }))
  on('tool.list', () => ({ value: tools.map(name => ({ name, description: `The ${name} tool`, mcp: false })) }))
  on('session.messages', () => ({ value: history as never }))
  on('http.fetch', (_, e) => {
    const request = { url: e.url, body: e.init?.body, auth: e.init?.headers?.authorization }
    requests.push(request)
    const route = routes[e.url]
    const answer = typeof route === 'function' ? route(request) : (route ?? { status: 404, text: 'Not found' })
    const status = answer.status ?? 200
    const text = answer.text ?? JSON.stringify(answer.json)
    return { value: { status, ok: status < 300, headers: {}, text } }
  })
  return requests
}

/** Start an interactive desktop session, as the app does. */
const start = ($: Engine) => $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })

test('the helpers read and write the OpenAI shapes', async () => {
  expect(apiBase('https://proxy.test/')).toBe('https://proxy.test/v1')
  expect(apiBase('https://proxy.test/v1/')).toBe('https://proxy.test/v1')
  expect(apiBase('https://generativelanguage.googleapis.com/v1beta/openai')).toBe(
    'https://generativelanguage.googleapis.com/v1beta/openai',
  )
  expect(authHeaders('')).toEqual({})
  expect(authHeaders('k')).toEqual({ authorization: 'Bearer k' })
  expect(parseTokens('128k')).toBe(128000)
  expect(parseTokens('1m')).toBe(1000000)
  expect(parseTokens('lots')).toBeUndefined()

  // OpenRouter-style /models entries carry their own capabilities.
  const listed = parseModels(
    JSON.stringify({
      data: [
        { id: 'b' },
        { id: 'a', context_length: 128000, architecture: { input_modalities: ['text', 'image'], output_modalities: ['text'] }, supported_parameters: ['tools'] },
        { id: 'a' },
      ],
    }),
  )
  expect(listed.ids).toEqual(['a', 'b'])
  expect(listed.capabilities).toEqual({ a: { contextWindow: 128000, image: true, tools: true, reasoning: false, text: true } })

  expect(isChatModel('gpt-4o', undefined)).toBe(true)
  expect(isChatModel('text-embedding-3-small', undefined)).toBe(false)
  expect(isChatModel('whisper-1', undefined)).toBe(false)
  expect(isChatModel('flux', { text: false })).toBe(false)

  expect(parseCapabilities(JSON.stringify(CAPABILITIES))).toEqual({
    seer: { contextWindow: 90000, maxOutputTokens: 8192, image: true, tools: true, reasoning: true, text: true },
    blind: { contextWindow: 1000, maxOutputTokens: 512, image: false, tools: false, reasoning: false, text: true },
  })

  const messages = toOpenAiMessages('SYS', HISTORY)
  expect(messages.map(m => m.role)).toEqual(['system', 'user', 'assistant', 'tool'])
  expect(messages[3]).toEqual({ role: 'tool', tool_call_id: 'toolu_1', content: 'hello' })
  expect(JSON.stringify(toOpenAiMessages('', IMAGE_HISTORY))).toContain('data:image/png;base64,iVBORw0KGgo=')
  expect(toOpenAiMessages('', IMAGE_HISTORY, { images: false })).toEqual([
    { role: 'user', content: 'What is in this picture?\n\n[image omitted: this model has no image input]' },
  ])

  const toolReply = parseReply(
    JSON.stringify({
      model: 'm',
      choices: [{ finish_reason: 'tool_calls', message: { content: null, tool_calls: [{ id: 'c1', function: { name: 'Bash', arguments: '{"command":"ls"}' } }] } }],
    }),
    'm',
  )
  expect(toolReply.stopReason).toBe('tool_use')
  expect(toolReply.toolCalls).toEqual([{ id: 'c1', name: 'Bash', arguments: '{"command":"ls"}' }])
})

test('picking a configured model sends the next step to it', { options: OPTIONS }, async ($, on) => {
  const requests = world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'deepseek-chat' }, { id: 'qwen' }] } },
      'https://proxy.test/v1/chat/completions': reply('The README says hello.'),
    },
    tools: ['Read', 'mcp__x__y'],
  })
  await start($)

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'reload' })
  expect((await ui.find({ type: 'Text', text: /chat models/ }))?.text).toBe(
    '2 chat models · https://proxy.test · key …test · plugin config',
  )
  expect(requests[0]).toEqual({ url: 'https://proxy.test/v1/models', body: undefined, auth: 'Bearer ieti_sk_test' })

  // Picking the provider selects its first model; the model list then shows under it.
  await ui.press({ key: 'pick:config' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using deepseek-chat via Configured')
  await ui.select({ key: 'model', value: 'proxy:config/qwen' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using qwen via Configured')
  await ui.select({ key: 'model', value: 'proxy:config/deepseek-chat' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using deepseek-chat via Configured')
  await ui.unmount()

  await $.prompt.compose({ model: 'claude-opus-5-5', promptModel: 'claude-opus-5-5', surfaces: ['desktop'], tools: ['Read'], outputStyle: null, traits: [] })
  const { chunks, result } = await drain($.turn.step(step()))
  expect(result.answer).toBe('The README says hello.')
  expect(chunks.map(c => c.kind)).toEqual(['text', 'stop'])

  const call = requests.find(r => r.url.endsWith('/chat/completions'))
  const sent = JSON.parse(call?.body ?? '{}')
  expect(call?.auth).toBe('Bearer ieti_sk_test')
  expect(sent.model).toBe('deepseek-chat')
  expect(sent.messages[0]).toEqual({ role: 'system', content: 'You are Claude Code.' })
  // Only tools with a known schema are offered.
  expect(sent.tools.map((t: { function: { name: string } }) => t.function.name)).toEqual(['Read'])
})

test('a provider tool call becomes a Claude Code tool use', { options: OPTIONS }, async ($, on) => {
  world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'qwen' }] } },
      'https://proxy.test/v1/chat/completions': {
        json: { choices: [{ finish_reason: 'tool_calls', message: { content: '', tool_calls: [{ id: 'c1', function: { name: 'Bash', arguments: '{"command":"ls"}' } }] } }] },
      },
    },
    tools: ['Bash'],
  })
  await start($)
  await $.command.run(provider('qwen'))

  const { chunks, result } = await drain($.turn.step(step()))
  expect(chunks.map(c => c.kind)).toEqual(['tool', 'input', 'stop'])
  expect(result.stopReason).toBe('tool_use')
  expect(result.toolUses).toEqual([{ name: 'Bash', input: { command: 'ls' } }])
})

test('with Claude selected the step goes to the engine untouched', { options: OPTIONS }, async ($, on) => {
  world(on, { routes: { 'https://proxy.test/v1/models': { json: { data: [] } } } })
  on('turn.step', async function* (_, e) {
    yield { kind: 'text', index: 0, text: `engine ${e.model}` }
    return { turnId: e.turnId, index: e.index, answer: `engine ${e.model}`, toolUses: [], stopReason: 'end_turn', usage: null }
  })
  await start($)

  const { result } = await drain($.turn.step(step()))
  expect(result.answer).toBe('engine claude-opus-5-5')
})

test('named providers: several at once, an empty key, a URL path kept, non-chat models hidden', async ($, on) => {
  const requests = world(on, {
    routes: {
      'https://api.deepseek.test/v1/models': { json: { data: [{ id: 'deepseek-chat' }] } },
      'https://gen.test/v1beta/openai/models': { json: { data: [{ id: 'gemini-x' }, { id: 'text-embedding-004' }] } },
      'https://gen.test/v1beta/openai/chat/completions': reply('from gemini'),
    },
  })
  await start($)

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect(await ui.find({ type: 'Text', text: /No providers yet/ })).toBeDefined()

  // + Add opens an empty form; typing fills it and the button saves it.
  await ui.press({ key: 'add' })
  await ui.input({ key: 'form-name', text: 'DeepSeek', kind: 'change' })
  await ui.input({ key: 'form-url', text: 'https://api.deepseek.test', kind: 'change' })
  await ui.input({ key: 'form-key', text: 'sk-deep', kind: 'change' })
  await ui.press({ key: 'save-provider' })
  expect(await ui.find({ key: 'form-name' })).toBeUndefined()
  // Enter in the key field saves too; this one has no key.
  await ui.press({ key: 'add' })
  await ui.input({ key: 'form-name', text: 'Gemini', kind: 'change' })
  await ui.input({ key: 'form-url', text: 'https://gen.test/v1beta/openai', kind: 'change' })
  await ui.input({ key: 'form-key', text: '' })

  expect((await ui.find({ type: 'Text', text: /^1 chat model/ }))?.text).toMatch(/^1 chat model · /)
  await ui.press({ key: 'pick:gemini' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using gemini-x via Gemini')
  await ui.unmount()

  const listing = (await $.command.run(provider('reload'))).text
  expect(listing).toBe('DeepSeek (deepseek): deepseek-chat\nGemini (gemini): gemini-x')

  await drain($.turn.step(step()))
  const deepseek = requests.find(r => r.url === 'https://api.deepseek.test/v1/models')
  const gemini = requests.find(r => r.url === 'https://gen.test/v1beta/openai/chat/completions')
  expect(deepseek?.auth).toBe('Bearer sk-deep')
  expect(gemini?.auth).toBeUndefined()

  // Removing the selected provider goes back to Claude.
  const again = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await again.press({ key: 'edit:gemini' })
  await again.press({ key: 'remove:gemini' })
  // Remove asks first; Cancel keeps the provider.
  await again.press({ key: 'remove-cancel:gemini' })
  expect(await again.find({ key: 'remove-confirm:gemini' })).toBeUndefined()
  await again.press({ key: 'remove:gemini' })
  expect((await again.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using gemini-x via Gemini')
  await again.press({ key: 'remove-confirm:gemini' })
  expect((await again.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using Claude')
  await again.unmount()
})

test('a model id with a slash is picked by provider and model', async ($, on) => {
  world(
    on,
    {
      routes: {
        'https://openrouter.test/api/v1/models': { json: { data: [{ id: 'openai/gpt-4o', context_length: 128000 }] } },
        'https://openrouter.test/api/v1/chat/completions': reply('hi'),
      },
    },
    { providers: [{ id: 'openrouter', name: 'OpenRouter', url: 'https://openrouter.test/api/v1' }], keys: { openrouter: 'sk-or' } },
  )
  await start($)

  expect((await $.command.run(provider('openrouter/openai/gpt-4o'))).text).toBe(
    'Model requests now go to openai/gpt-4o via OpenRouter.',
  )
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using openai/gpt-4o via OpenRouter')
  expect((await ui.find({ type: 'Text', text: /context/ }))?.text).toBe('128k context')
  await ui.unmount()
})

test('a model without vision or tools gets neither, and the pane says so', { options: OPTIONS }, async ($, on) => {
  const requests = world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'seer' }, { id: 'blind' }] } },
      'https://proxy.test/v1/model-capabilities': { json: CAPABILITIES },
      'https://proxy.test/v1/chat/completions': reply('A cat.', 100),
    },
    history: IMAGE_HISTORY,
  })
  await start($)
  await $.command.run(provider('blind'))

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ type: 'Text', text: /context ·/ }))?.text).toBe('1k context · 512 output · no vision · no tools')
  await ui.unmount()

  await drain($.turn.step(step()))
  const body = JSON.parse(requests.find(r => r.url.endsWith('/chat/completions'))?.body ?? '{}')
  expect(body.tools).toBeUndefined()
  expect(JSON.stringify(body.messages)).not.toContain('base64')
  expect(JSON.stringify(body.messages)).toContain('image omitted')
})

test('nearing the context window warns once and shows in the pane', { options: OPTIONS }, async ($, on) => {
  const toasts: string[] = []
  world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'blind' }] } },
      'https://proxy.test/v1/model-capabilities': { json: CAPABILITIES },
      'https://proxy.test/v1/chat/completions': reply('ok', 850),
    },
    history: HISTORY.slice(0, 1),
    tools: [],
    toasts,
  })
  await start($)
  await $.command.run(provider('blind'))
  toasts.length = 0

  await drain($.turn.step(step(0)))
  await drain($.turn.step(step(1)))
  expect(toasts.filter(t => t.includes('context'))).toEqual([
    'config/blind has used 850 of its 1k context. Run /compact or switch to Claude before it fills up.',
  ])

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ type: 'Text', text: /^⚠ Context/ }))?.text).toBe('⚠ Context: 850 of 1k (85%) at the last answer')
  await ui.unmount()
})

test('a conversation past the window is stopped before it is sent', { options: OPTIONS }, async ($, on) => {
  const requests = world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'blind' }] } },
      'https://proxy.test/v1/model-capabilities': { json: CAPABILITIES },
    },
    history: [{ role: 'user', content: [{ type: 'text', text: 'x'.repeat(8000) }] }],
    tools: [],
  })
  await start($)
  await $.command.run(provider('blind'))
  const { result } = await drain($.turn.step(step()))

  expect(requests.some(r => r.url.endsWith('/chat/completions'))).toBe(false)
  expect(result.answer).toContain("larger than blind's 1k context")
})

test('without a capabilities endpoint, vision and context can be set by hand', { options: OPTIONS }, async ($, on) => {
  const requests = world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'plain' }] } },
      'https://proxy.test/v1/chat/completions': reply('ok'),
    },
    history: IMAGE_HISTORY,
  })
  await start($)
  await $.command.run(provider('plain'))

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ type: 'Text', text: /capabilities/ }))?.text).toBe('capabilities unknown')

  // Unknown means try: the image and the tools are sent.
  await drain($.turn.step(step(0)))
  const first = JSON.parse(requests.filter(r => r.url.endsWith('/chat/completions'))[0]?.body ?? '{}')
  expect(JSON.stringify(first.messages)).toContain('base64')
  expect(first.tools.length).toBe(1)

  await ui.select({ key: 'vision', value: 'no' })
  await ui.input({ key: 'context', text: '64k' })
  expect((await ui.find({ type: 'Text', text: /context ·/ }))?.text).toBe('64k context · no vision')
  await ui.unmount()

  await drain($.turn.step(step(1)))
  const second = JSON.parse(requests.filter(r => r.url.endsWith('/chat/completions'))[1]?.body ?? '{}')
  expect(JSON.stringify(second.messages)).not.toContain('base64')
})

test('a URL and key from before named providers become a provider named Proxy', async ($, on) => {
  const requests = world(
    on,
    {
      routes: {
        'https://old.test/v1/models': { json: { data: [{ id: 'm' }] } },
        'https://old.test/v1/chat/completions': reply('still here'),
      },
    },
    { proxyUrl: 'https://old.test/v1', apiKey: 'sk-old', selection: { provider: 'proxy', model: 'm' } },
  )
  await start($)

  const { result } = await drain($.turn.step(step()))
  expect(result.answer).toBe('still here')
  expect(requests.find(r => r.url.endsWith('/chat/completions'))?.auth).toBe('Bearer sk-old')
  const listing = (await $.command.run(provider('reload'))).text
  expect(listing).toBe('Proxy (proxy): m')
})

test('the button above the prompt names the backend and opens the pane', { options: OPTIONS }, async ($, on) => {
  const opened: unknown[] = []
  world(on, { routes: { 'https://proxy.test/v1/models': { json: { data: [{ id: 'deepseek-chat' }] } } }, opened })
  await start($)

  const band = await $.ui.mount({
    plugin: 'provider-switch',
    component: 'AbovePrompt',
    surface: 'desktop',
    props: { hasSurvey: false, isWorking: false, maxRows: 10, bodyColumns: 60, scroll: { offset: 0, bodyRows: 10 }, view: {} },
  } as never)
  expect((await band.find({ type: 'Text', text: /^Provider Switch/ }))?.text).toBe('Provider Switch: Claude')

  opened.length = 0
  await band.press({ key: 'open-pane' })
  expect(opened).toEqual([expect.objectContaining({ id: 'provider-switch', focus: true })])

  await $.command.run({ command: 'provider', args: 'config/deepseek-chat' } as never)
  expect((await band.find({ type: 'Text', text: /^Provider Switch/ }))?.text).toBe('Provider Switch: Configured')
  expect(await band.find({ type: 'Text', text: ' deepseek-chat ' })).toBeDefined()
  await band.unmount()
})

test('Edit opens a provider form: renaming keeps its id and selection, + Add never replaces', async ($, on) => {
  world(
    on,
    { routes: { 'https://ds.test/v1/models': { json: { data: [{ id: 'deepseek-chat' }] } } } },
    { providers: [{ id: 'deepseek', name: 'DeepSeek', url: 'https://ds.test/v1' }], keys: { deepseek: 'sk-1234' } },
  )
  await start($)
  await $.command.run(provider('deepseek-chat'))

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'edit:deepseek' })
  expect((await ui.find({ key: 'form-name' }))?.props.value).toBe('DeepSeek')
  expect((await ui.find({ key: 'form-url' }))?.props.value).toBe('https://ds.test/v1')
  // Typing saves: no Save button for an existing provider.
  expect(await ui.find({ key: 'save-provider' })).toBeUndefined()
  await ui.input({ key: 'form-name', text: 'DeepSeek Work', kind: 'change' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using deepseek-chat via DeepSeek Work')
  await ui.input({ key: 'form-key', text: 'sk-new9', kind: 'change' })
  expect(await ui.find({ type: 'Text', text: 'Saved key …new9' })).toBeDefined()
  await ui.press({ key: 'edit:deepseek' })

  // The same name again is a second provider, not a replacement.
  await ui.press({ key: 'add' })
  await ui.input({ key: 'form-name', text: 'DeepSeek', kind: 'change' })
  await ui.input({ key: 'form-url', text: 'https://ds.test/v1', kind: 'change' })
  await ui.press({ key: 'save-provider' })
  expect(await ui.find({ key: 'edit:deepseek-2' })).toBeDefined()
  expect(await ui.find({ key: 'edit:deepseek' })).toBeDefined()

  await ui.press({ key: 'edit:deepseek' })
  await ui.press({ key: 'forget:deepseek' })
  expect(await ui.find({ key: 'forget:deepseek' })).toBeUndefined()
  await ui.press({ key: 'edit:deepseek' })
  expect(await ui.find({ key: 'form-name' })).toBeUndefined()
  await ui.unmount()

  const listing = (await $.command.run(provider('reload'))).text
  expect(listing).toBe('DeepSeek Work (deepseek): deepseek-chat\nDeepSeek (deepseek-2): deepseek-chat')
})

test('a thinking model gets its reasoning back with the tool calls it made', { options: OPTIONS }, async ($, on) => {
  // As DeepSeek's thinking mode answers: refused when a tool call comes back without its reasoning.
  const strict = (r: Request): Answer => {
    const sent = JSON.parse(r.body ?? '{}') as { messages: { role: string; tool_calls?: unknown[]; reasoning_content?: string }[] }
    if (sent.messages.some(m => m.tool_calls && m.reasoning_content === undefined)) {
      return { status: 400, json: { error: { message: 'The `reasoning_content` in the thinking mode must be passed back to the API.' } } }
    }
    const isFirst = !sent.messages.some(m => m.role === 'tool' && JSON.stringify(m).includes('call_new'))
    return isFirst
      ? {
          json: {
            choices: [
              {
                finish_reason: 'tool_calls',
                message: {
                  content: null,
                  reasoning_content: 'I should list the files.',
                  tool_calls: [{ id: 'call_new', function: { name: 'Read', arguments: '{"file_path":"/r/a"}' } }],
                },
              },
            ],
          },
        }
      : reply('Done.')
  }
  const history = [...HISTORY]
  const requests = world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'deepseek-flash' }] } },
      'https://proxy.test/v1/chat/completions': strict,
    },
    history,
  })
  await start($)
  await $.command.run(provider('deepseek-flash'))

  // HISTORY's toolu_1 predates the plugin keeping reasoning: refused, then sent again with an empty one.
  const first = await drain($.turn.step(step()))
  expect(first.result.toolUses).toEqual([{ name: 'Read', input: { file_path: '/r/a' } }])
  expect(first.chunks.map(c => c.kind)).toEqual(['thinking', 'tool', 'input', 'stop'])
  const calls = requests.filter(r => r.url.endsWith('/chat/completions'))
  expect(calls).toHaveLength(2)
  expect(JSON.parse(calls[1]!.body!).messages.find((m: { tool_calls?: { id: string }[] }) => m.tool_calls?.[0]?.id === 'toolu_1').reasoning_content).toBe('')

  history.push(
    { role: 'assistant', content: [{ type: 'tool_use', id: 'call_new', name: 'Read', input: { file_path: '/r/a' } }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call_new', content: 'a' }] },
  )
  const second = await drain($.turn.step(step(1)))
  expect(second.result.answer).toBe('Done.')
  const last = JSON.parse(requests.filter(r => r.url.endsWith('/chat/completions')).at(-1)!.body!)
  expect(last.messages.find((m: { tool_calls?: { id: string }[] }) => m.tool_calls?.[0]?.id === 'call_new').reasoning_content).toBe(
    'I should list the files.',
  )
})

test('context items switched off for a provider are not sent, and the pane counts their tokens', { options: OPTIONS }, async ($, on) => {
  const requests = world(on, {
    routes: {
      'https://proxy.test/v1/models': { json: { data: [{ id: 'qwen' }] } },
      'https://proxy.test/v1/chat/completions': reply('ok'),
    },
    tools: ['Read', 'Bash', 'mcp__x__y'],
    sections: ['intro', 'memory'],
  })
  await start($)
  await $.command.run(provider('qwen'))
  await $.prompt.compose({ model: 'claude-opus-5-5', promptModel: 'claude-opus-5-5', surfaces: ['desktop'], tools: ['Read'], outputStyle: null, traits: [] })

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  // Folded at first: the header shows the total alone.
  expect(await ui.find({ key: 'tool:Bash' })).toBeUndefined()
  expect((await ui.find({ type: 'Text', text: /per request$/ }))?.text).toMatch(/^~\d+ per request$/)
  await ui.press({ key: 'context-toggle' })

  // Only tools with a known schema are listed, each on and with its cost.
  expect(await ui.find({ key: 'tool:mcp__x__y' })).toBeUndefined()
  expect((await ui.find({ key: 'tool:Bash' }))?.text).toMatch(/^☑ Bash · ~\d+$/)
  expect((await ui.find({ key: 'section:memory' }))?.text).toMatch(/^☑ memory · ~\d+$/)
  const header = async () => (await ui.find({ type: 'Text', text: /per request$/ }))?.text

  await ui.press({ key: 'tool:Bash' })
  await ui.press({ key: 'section:memory' })
  expect((await ui.find({ key: 'tool:Bash' }))?.text).toMatch(/^☐ Bash/)
  expect(await header()).toMatch(/^~\d+ of ~\d+ per request$/)
  await ui.unmount()

  await $.prompt.compose({ model: 'claude-opus-5-5', promptModel: 'claude-opus-5-5', surfaces: ['desktop'], tools: ['Read'], outputStyle: null, traits: [] })
  await drain($.turn.step(step()))
  const sent = JSON.parse(requests.find(r => r.url.endsWith('/chat/completions'))?.body ?? '{}')
  expect(sent.tools.map((t: { function: { name: string } }) => t.function.name)).toEqual(['Read'])
  expect(sent.messages[0]).toEqual({ role: 'system', content: 'You are Claude Code.' })

  // None sends no tools at all; All brings them back.
  const again = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await again.press({ key: 'tools-none' })
  expect((await again.find({ type: 'Text', text: /^Tools · / }))?.text).toBe('Tools · ~0')
  await again.press({ key: 'tools-all' })
  expect((await again.find({ key: 'tool:Bash' }))?.text).toMatch(/^☑ Bash/)
})

test('Claude has its own context settings: an off tool is deferred, an off reminder is dropped', async ($, on) => {
  world(on, { routes: {} })
  on('tool.describe', (_, e) => ({ description: e.description }))
  on('prompt.attachment', (_, e) => ({ text: e.text }))
  await start($)
  const described = await $.tool.describe({ tool: 'Bash', description: 'Runs a command.', provider: { plugin: 'engine', tier: 'core' } } as never)
  expect(described.isDeferred).toBeUndefined()
  await $.prompt.attachment({ type: 'skill_listing', text: 'Skills: a, b, c', origin: { kind: 'engine' } } as never)

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'context-toggle' })
  expect((await ui.find({ type: 'Text', text: /^▾ Context for / }))?.text).toBe('▾ Context for Claude')
  await ui.press({ key: 'tool:Bash' })
  await ui.press({ key: 'inject:skill_listing' })

  expect((await $.tool.describe({ tool: 'Bash', description: 'Runs a command.', provider: { plugin: 'engine', tier: 'core' } } as never)).isDeferred).toBe(true)
  expect((await $.prompt.attachment({ type: 'skill_listing', text: 'Skills: a, b, c', origin: { kind: 'engine' } } as never)).text).toBeNull()
  // A mentioned file is the conversation itself: never offered, never dropped.
  expect((await $.prompt.attachment({ type: 'file', text: 'contents', origin: { kind: 'engine' } } as never)).text).toBe('contents')

  // An MCP server's tools fold into one row that switches them all.
  const mcp = { plugin: 'engine', tier: 'mcp' }
  await $.tool.describe({ tool: 'mcp__docs__read', description: 'Reads a doc.', provider: mcp } as never)
  await $.tool.describe({ tool: 'mcp__docs__write', description: 'Writes a doc.', provider: mcp } as never)
  expect((await ui.find({ key: 'server:docs' }))?.text).toMatch(/^☑ docs \(MCP, 2\/2 tools\)/)
  expect(await ui.find({ key: 'tool:mcp__docs__read' })).toBeUndefined()
  await ui.press({ key: 'server:docs' })
  expect((await $.tool.describe({ tool: 'mcp__docs__write', description: 'Writes a doc.', provider: mcp } as never)).isDeferred).toBe(true)
  await ui.press({ key: 'server-fold:docs' })
  await ui.press({ key: 'tool:mcp__docs__read' })
  expect((await ui.find({ key: 'tool:mcp__docs__read' }))?.text).toMatch(/^☑ read · ~\d+$/)
  expect((await ui.find({ key: 'server:docs' }))?.text).toMatch(/^◐ docs \(MCP, 1\/2 tools\)/)
})

test('Lean switches off what a coding turn can do without, and Restore all brings it back', async ($, on) => {
  world(on, { routes: {} })
  on('tool.describe', (_, e) => ({ description: e.description }))
  on('prompt.attachment', (_, e) => ({ text: e.text }))
  await start($)
  const core = { plugin: 'engine', tier: 'core' }
  await $.tool.describe({ tool: 'Bash', description: 'Runs a command.', provider: core } as never)
  await $.tool.describe({ tool: 'WebFetch', description: 'Fetches a page.', provider: core } as never)
  await $.prompt.attachment({ type: 'skill_listing', text: 'Skills: a, b, c', origin: { kind: 'engine' } } as never)
  const hook = { type: 'hook_additional_context', text: 'x'.repeat(4000), origin: { kind: 'hook', event: 'SessionStart' } }
  await $.prompt.attachment(hook as never)

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'context-toggle' })
  // Everything on: Lean is offered, Restore all is not.
  expect((await ui.find({ key: 'context-lean' }))?.text).toMatch(/^Lean · saves ~\dk$/)
  expect(await ui.find({ key: 'context-restore' })).toBeUndefined()

  await ui.press({ key: 'context-lean' })
  expect((await ui.find({ key: 'tool:Bash' }))?.text).toMatch(/^☑ Bash/)
  expect((await ui.find({ key: 'tool:WebFetch' }))?.text).toMatch(/^☐ WebFetch/)
  // Claude keeps the engine's listings (its Skill and ToolSearch need them), not hooks' context.
  expect((await ui.find({ key: 'inject:skill_listing' }))?.text).toMatch(/^☑ skill_listing/)
  expect((await ui.find({ key: 'inject:hook:SessionStart' }))?.text).toMatch(/^☐ SessionStart hook/)
  expect((await $.tool.describe({ tool: 'WebFetch', description: 'Fetches a page.', provider: core } as never)).isDeferred).toBe(true)
  expect((await $.prompt.attachment(hook as never)).text).toBeNull()
  expect(await ui.find({ key: 'context-lean' })).toBeUndefined()

  await ui.press({ key: 'context-restore' })
  expect((await ui.find({ key: 'tool:WebFetch' }))?.text).toMatch(/^☑ WebFetch/)
  expect((await ui.find({ key: 'inject:hook:SessionStart' }))?.text).toMatch(/^☑ SessionStart hook/)
  expect(await ui.find({ key: 'context-restore' })).toBeUndefined()
  expect(await ui.find({ key: 'context-lean' })).toBeDefined()
})

test("a provider's Lean also drops the engine's listings, which its model cannot use", { options: OPTIONS }, async ($, on) => {
  world(on, {
    routes: { 'https://proxy.test/v1/models': { json: { data: [{ id: 'qwen' }] } } },
    tools: ['Read', 'Bash', 'WebFetch', 'Agent'],
  })
  on('prompt.attachment', (_, e) => ({ text: e.text }))
  await start($)
  await $.command.run(provider('qwen'))
  await $.prompt.attachment({ type: 'skill_listing', text: 'Skills: a, b, c', origin: { kind: 'engine' } } as never)

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'context-toggle' })
  await ui.press({ key: 'context-lean' })
  expect((await ui.find({ key: 'tool:Read' }))?.text).toMatch(/^☑ Read/)
  expect((await ui.find({ key: 'tool:Agent' }))?.text).toMatch(/^☐ Agent/)
  expect((await ui.find({ key: 'inject:skill_listing' }))?.text).toMatch(/^☐ skill_listing/)
  expect(await ui.find({ key: 'context-restore' })).toBeDefined()
})

test('a provider not in use offers its models, and picking one switches to it', async ($, on) => {
  world(on, {
    routes: { 'https://api.deepseek.test/v1/models': { json: { data: [{ id: 'deepseek-chat' }, { id: 'deepseek-reasoner' }] } } },
  })
  await start($)
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'add' })
  await ui.input({ key: 'form-name', text: 'DeepSeek', kind: 'change' })
  await ui.input({ key: 'form-url', text: 'https://api.deepseek.test', kind: 'change' })
  await ui.press({ key: 'save-provider' })

  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using Claude')
  expect((await ui.find({ key: 'model:deepseek' }))?.props.options).toHaveLength(3)
  await ui.select({ key: 'model:deepseek', value: 'proxy:deepseek/deepseek-reasoner' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using deepseek-reasoner via DeepSeek')
  // Selected, the row shows the full model settings instead.
  expect(await ui.find({ key: 'model:deepseek' })).toBeUndefined()
  expect((await ui.find({ key: 'model' }))?.props.value).toBe('proxy:deepseek/deepseek-reasoner')
})
