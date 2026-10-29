import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { apiBase, parseCapabilities, parseModels, parseReply, toOpenAiMessages } from '../hooks/openai'

const OPTIONS = { proxyUrl: 'https://proxy.test/', apiKey: 'ieti_sk_test' }

const PANE = {
  plugin: 'provider-switch',
  component: 'Pane',
  requestId: 'provider-switch',
  props: {
    title: 'Model provider',
    isFocused: false,
    bodyColumns: 60,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 30 },
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

// The conversation in Messages API form, as `$.session.messages({ as: 'api' })` gives it.
const HISTORY: { role: 'user' | 'assistant'; content: { type: string; [field: string]: unknown }[] }[] = [
  { role: 'user', content: [{ type: 'text', text: 'Read the README' }] },
  { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: '/r/README.md' } }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'hello' }] },
]

test('the translation helpers read and write the OpenAI shapes', async () => {
  expect(apiBase('https://proxy.test/')).toBe('https://proxy.test/v1')
  expect(apiBase('https://proxy.test/v1/')).toBe('https://proxy.test/v1')
  expect(parseModels('{"data":[{"id":"b"},{"id":"a"},{"id":"a"}]}')).toEqual(['a', 'b'])

  const messages = toOpenAiMessages('SYS', HISTORY)
  expect(messages.map(m => m.role)).toEqual(['system', 'user', 'assistant', 'tool'])
  expect(messages[3]).toEqual({ role: 'tool', tool_call_id: 'toolu_1', content: 'hello' })

  const reply = parseReply(
    JSON.stringify({
      model: 'm',
      choices: [{ finish_reason: 'tool_calls', message: { content: null, tool_calls: [{ id: 'c1', function: { name: 'Bash', arguments: '{"command":"ls"}' } }] } }],
      usage: { prompt_tokens: 5, completion_tokens: 2 },
    }),
    'm',
  )
  expect(reply.stopReason).toBe('tool_use')
  expect(reply.toolCalls).toEqual([{ id: 'c1', name: 'Bash', arguments: '{"command":"ls"}' }])
})

test('picking a proxy model sends the next step to the proxy', { options: OPTIONS }, async ($, on) => {
  const requests: { url: string; body?: string; auth?: string }[] = []

  // Nothing sits beneath the plugin: each hook here stands for the engine.
  mock.store(on)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'You are Claude Code.', scope: 'shared' }] }))
  on('tool.list', () => ({ value: [{ name: 'Read', description: 'Reads a file', mcp: false }, { name: 'mcp__x__y', description: 'other', mcp: true }] }))
  on('session.messages', () => ({ value: HISTORY as never }))
  on('http.fetch', (_, e) => {
    requests.push({ url: e.url, body: e.init?.body, auth: e.init?.headers?.authorization })
    const text = e.url.endsWith('/models')
      ? JSON.stringify({ object: 'list', data: [{ id: 'deepseek-chat' }, { id: 'qwen' }] })
      : JSON.stringify({
          model: 'deepseek-chat',
          choices: [{ finish_reason: 'stop', message: { content: 'The README says hello.' } }],
          usage: { prompt_tokens: 10, completion_tokens: 4 },
        })
    return { value: { status: 200, ok: true, headers: {}, text } }
  })

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'reload' })
  expect((await ui.find({ type: 'Text', text: /2 proxy models/ }))?.text).toBe('2 proxy models available')
  expect(requests[0]).toEqual({ url: 'https://proxy.test/v1/models', body: undefined, auth: 'Bearer ieti_sk_test' })

  await ui.select({ key: 'model', value: 'proxy:deepseek-chat' })
  expect((await ui.find({ type: 'Text', text: /^Using / }))?.text).toBe('Using deepseek-chat via the proxy')
  await ui.unmount()

  await $.prompt.compose({ model: 'claude-opus-5-5', promptModel: 'claude-opus-5-5', surfaces: ['desktop'], tools: ['Read'], outputStyle: null, traits: [] })
  const stream = $.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 3 })
  const { chunks, result } = await drain(stream)

  expect(result.answer).toBe('The README says hello.')
  expect(chunks.map(c => c.kind)).toEqual(['text', 'stop'])

  const call = requests.find(r => r.url.endsWith('/chat/completions'))
  const sent = JSON.parse(call?.body ?? '{}')
  expect(sent.model).toBe('deepseek-chat')
  expect(sent.messages[0]).toEqual({ role: 'system', content: 'You are Claude Code.' })
  // Only tools with a known schema are offered.
  expect(sent.tools.map((t: { function: { name: string } }) => t.function.name)).toEqual(['Read'])
})

test('a proxy tool call becomes a Claude Code tool use', { options: OPTIONS }, async ($, on) => {
  mock.store(on)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('tool.list', () => ({ value: [{ name: 'Bash', description: 'Runs a command', mcp: false }] }))
  on('session.messages', () => ({ value: HISTORY.slice(0, 1) as never }))
  on('http.fetch', (_, e) => {
    const text = e.url.endsWith('/models')
      ? JSON.stringify({ data: [{ id: 'qwen' }] })
      : JSON.stringify({
          choices: [{ finish_reason: 'tool_calls', message: { content: '', tool_calls: [{ id: 'c1', function: { name: 'Bash', arguments: '{"command":"ls"}' } }] } }],
        })
    return { value: { status: 200, ok: true, headers: {}, text } }
  })

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'reload' })
  await ui.select({ key: 'model', value: 'proxy:qwen' })
  await ui.unmount()

  const stream = $.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 1 })
  const { chunks, result } = await drain(stream)

  expect(chunks.map(c => c.kind)).toEqual(['tool', 'input', 'stop'])
  expect(result.stopReason).toBe('tool_use')
  expect(result.toolUses).toEqual([{ name: 'Bash', input: { command: 'ls' } }])
})

test('with Claude selected the step goes to the engine untouched', { options: OPTIONS }, async ($, on) => {
  mock.store(on)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('http.fetch', () => ({ value: { status: 200, ok: true, headers: {}, text: '{"data":[]}' } }))
  on('turn.step', async function* (_, e) {
    yield { kind: 'text', index: 0, text: `engine ${e.model}` }
    return { turnId: e.turnId, index: e.index, answer: `engine ${e.model}`, toolUses: [], stopReason: 'end_turn', usage: null }
  })

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })
  const stream = $.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 1 })
  const { result } = await drain(stream)
  expect(result.answer).toBe('engine claude-opus-5-5')
})

// As IETI Agents Proxy answers `GET /v1/model-capabilities`.
const CAPABILITIES = {
  object: 'ieti.model_capabilities.list',
  schema_version: 1,
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

/** `/provider <args>` as the person types it. */
const provider = (args: string) => ({
  command: 'provider',
  args,
  origin: { kind: 'composer' } as const,
  presentation: { isFullscreen: false, columns: 80 },
})

const IMAGE_HISTORY = [
  {
    role: 'user',
    content: [
      { type: 'text', text: 'What is in this picture?' },
      { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } },
    ],
  },
] as const

test('capabilities are read from the IETI shape and images can be left out', async () => {
  expect(parseCapabilities(JSON.stringify(CAPABILITIES))).toEqual({
    seer: { contextWindow: 90000, maxOutputTokens: 8192, image: true, tools: true, reasoning: true },
    blind: { contextWindow: 1000, maxOutputTokens: 512, image: false, tools: false, reasoning: false },
  })

  const withImages = toOpenAiMessages('', IMAGE_HISTORY)
  expect(JSON.stringify(withImages)).toContain('data:image/png;base64,iVBORw0KGgo=')
  const without = toOpenAiMessages('', IMAGE_HISTORY, { images: false })
  expect(without).toEqual([
    { role: 'user', content: 'What is in this picture?\n\n[image omitted: this model has no image input]' },
  ])
})

/** A proxy with both models above; `promptTokens` is what each answer reports. */
function proxy(on: On, promptTokens: number, sent: string[]) {
  on('http.fetch', (_, e) => {
    if (e.url.endsWith('/models')) return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify({ data: [{ id: 'seer' }, { id: 'blind' }] }) } }
    if (e.url.endsWith('/model-capabilities')) return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify(CAPABILITIES) } }
    sent.push(e.init?.body ?? '')
    const text = JSON.stringify({
      choices: [{ finish_reason: 'stop', message: { content: 'A cat.' } }],
      usage: { prompt_tokens: promptTokens, completion_tokens: 3 },
    })
    return { value: { status: 200, ok: true, headers: {}, text } }
  })
}

test('a model without vision or tools gets neither, and the pane says so', { options: OPTIONS }, async ($, on) => {
  const sent: string[] = []
  mock.store(on)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('tool.list', () => ({ value: [{ name: 'Read', description: 'Reads a file', mcp: false }] }))
  on('session.messages', () => ({ value: IMAGE_HISTORY as never }))
  proxy(on, 100, sent)

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'reload' })
  await ui.select({ key: 'model', value: 'proxy:blind' })
  expect((await ui.find({ type: 'Text', text: /context ·/ }))?.text).toBe('1k context · 512 output · no vision · no tools')
  await ui.unmount()

  await drain($.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 1 }))
  const body = JSON.parse(sent[0] ?? '{}')
  expect(body.tools).toBeUndefined()
  expect(JSON.stringify(body.messages)).not.toContain('base64')
  expect(JSON.stringify(body.messages)).toContain('image omitted')
})

test('nearing the context window warns once and shows in the pane', { options: OPTIONS }, async ($, on) => {
  const toasts: string[] = []
  const sent: string[] = []
  mock.store(on)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', (_, e) => {
    toasts.push(e.text)
    return { value: undefined }
  })
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('tool.list', () => ({ value: [] }))
  on('session.messages', () => ({ value: HISTORY.slice(0, 1) as never }))
  proxy(on, 850, sent)

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })
  await $.command.run(provider('blind'))
  toasts.length = 0

  await drain($.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 1 }))
  await drain($.turn.step({ turnId: 't1', index: 1, model: 'claude-opus-5-5', messageCount: 1 }))
  expect(toasts.filter(t => t.includes('context'))).toEqual([
    'blind has used 850 of its 1k context. Run /compact or switch to Claude before it fills up.',
  ])

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ type: 'Text', text: /^⚠ Context/ }))?.text).toBe('⚠ Context: 850 of 1k (85%) at the last answer')
  await ui.unmount()
})

test('a conversation past the window is stopped before it is sent', { options: OPTIONS }, async ($, on) => {
  const sent: string[] = []
  mock.store(on)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('tool.list', () => ({ value: [] }))
  on('session.messages', () => ({ value: [{ role: 'user', content: [{ type: 'text', text: 'x'.repeat(8000) }] }] as never }))
  proxy(on, 100, sent)

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })
  await $.command.run(provider('blind'))
  const { result } = await drain($.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 1 }))

  expect(sent).toEqual([])
  expect(result.answer).toContain("larger than blind's 1k context")
})

test('a proxy without a capabilities endpoint still works', { options: OPTIONS }, async ($, on) => {
  const sent: string[] = []
  mock.store(on)
  mock.clock(on)
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'provider' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('tool.list', () => ({ value: [{ name: 'Read', description: 'Reads a file', mcp: false }] }))
  on('session.messages', () => ({ value: IMAGE_HISTORY as never }))
  on('http.fetch', (_, e) => {
    if (e.url.endsWith('/models')) return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify({ data: [{ id: 'plain' }] }) } }
    if (e.url.endsWith('/model-capabilities')) return { value: { status: 404, ok: false, headers: {}, text: 'Not found' } }
    sent.push(e.init?.body ?? '')
    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify({ choices: [{ message: { content: 'ok' } }] }) } }
  })

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })
  expect((await $.command.run(provider('plain'))).text).toContain('plain via the proxy')
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect((await ui.find({ type: 'Text', text: /capabilities/ }))?.text).toBe('capabilities unknown')
  await ui.unmount()

  await drain($.turn.step({ turnId: 't1', index: 0, model: 'claude-opus-5-5', messageCount: 1 }))
  const body = JSON.parse(sent[0] ?? '{}')
  // Unknown means try: the image and the tools are sent.
  expect(JSON.stringify(body.messages)).toContain('base64')
  expect(body.tools.length).toBe(1)
})
