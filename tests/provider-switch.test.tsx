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
  { routes, history = HISTORY, tools = ['Read'], toasts }: {
    routes: Record<string, Answer | ((r: Request) => Answer)>
    history?: Message[]
    tools?: string[]
    toasts?: string[]
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
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'You are Claude Code.', scope: 'shared' }] }))
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
  expect((await ui.find({ type: 'Text', text: /chat models/ }))?.text).toBe('2 chat models')
  expect(requests[0]).toEqual({ url: 'https://proxy.test/v1/models', body: undefined, auth: 'Bearer ieti_sk_test' })

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
  expect(await ui.find({ type: 'Text', text: /None yet/ })).toBeDefined()

  // Typing fills the form; the button saves it.
  await ui.input({ key: 'new-name', text: 'DeepSeek', kind: 'change' })
  await ui.input({ key: 'new-url', text: 'https://api.deepseek.test', kind: 'change' })
  await ui.input({ key: 'new-key', text: 'sk-deep', kind: 'change' })
  await ui.press({ key: 'save-provider' })
  // Enter in the key field saves too; this one has no key.
  await ui.input({ key: 'new-name', text: 'Gemini', kind: 'change' })
  await ui.input({ key: 'new-url', text: 'https://gen.test/v1beta/openai', kind: 'change' })
  await ui.input({ key: 'new-key', text: '' })

  expect((await ui.find({ type: 'Text', text: /^1 chat model/ }))?.text).toBe('1 chat model')
  await ui.select({ key: 'model', value: 'proxy:gemini/gemini-x' })
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
  await again.press({ key: 'remove:gemini' })
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
