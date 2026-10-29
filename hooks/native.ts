/**
 * Native Claude Code sessions for providers that serve the Anthropic Messages API
 * (IETI Agents Proxy's `/v1/messages`): Claude Code's own client talks to the provider,
 * so every tool, streaming and thinking work as with Claude. The mod API cannot point a
 * running session at another endpoint, so a launcher script starts a new one.
 */

/** The Anthropic base URL for an API base ending in `/v1` (`https://host/v1` → `https://host`); else undefined. */
export function anthropicBase(apiBase: string): string | undefined {
  return /\/v1$/.test(apiBase) ? apiBase.slice(0, -3) : undefined
}

/** Whether a `POST <base>/messages/count_tokens` reply is the Anthropic API's. */
export function isCountTokensReply(status: number, body: string): boolean {
  if (status < 200 || status >= 300) return false
  try {
    return typeof (JSON.parse(body) as { input_tokens?: unknown }).input_tokens === 'number'
  } catch {
    return false
  }
}

/** `value` as one POSIX shell word. */
export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

export type Launch = {
  providerName: string
  baseUrl: string
  key: string
  model: string
  contextWindow?: number
  cwd: string
}

/**
 * The launcher: Claude Code with every model alias pinned to `model`, so subagents,
 * compaction and helper calls also go to the provider and none of them to Anthropic.
 */
export function launcherScript(l: Launch): string {
  const q = shellQuote
  return [
    '#!/bin/sh',
    `# Written by provider-switch: Claude Code using ${l.model} through ${l.providerName}'s Anthropic endpoint.`,
    '# It holds the provider key: keep it private.',
    `export ANTHROPIC_BASE_URL=${q(l.baseUrl)}`,
    ...(l.key ? [`export ANTHROPIC_AUTH_TOKEN=${q(l.key)}`] : []),
    'unset ANTHROPIC_API_KEY',
    `export ANTHROPIC_MODEL=${q(l.model)}`,
    `export ANTHROPIC_DEFAULT_OPUS_MODEL=${q(l.model)}`,
    `export ANTHROPIC_DEFAULT_SONNET_MODEL=${q(l.model)}`,
    `export ANTHROPIC_DEFAULT_HAIKU_MODEL=${q(l.model)}`,
    `export CLAUDE_CODE_SUBAGENT_MODEL=${q(l.model)}`,
    ...(l.contextWindow ? [`export CLAUDE_CODE_MAX_CONTEXT_TOKENS=${l.contextWindow}`] : []),
    `cd ${q(l.cwd)} || exit 1`,
    'exec claude "$@"',
    '',
  ].join('\n')
}

/** Where the launcher of provider `id` is written, under the home directory. */
export function launcherPath(home: string, id: string): string {
  return `${home.replace(/\/+$/, '')}/.claude/provider-switch/launch-${id.replace(/[^a-z0-9_-]/gi, '-')}.sh`
}
