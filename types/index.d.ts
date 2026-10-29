/** Which backend answers the model requests: Claude Code's own, or a proxy model. */
export type Selection = { provider: 'claude' } | { provider: 'proxy'; model: string }

/**
 * What the proxy says a model can do, from `GET /v1/model-capabilities`.
 * A field is absent when the proxy does not say: treat it as unknown, not as false.
 */
export type ModelInfo = {
  contextWindow?: number
  maxOutputTokens?: number
  image?: boolean
  tools?: boolean
  reasoning?: boolean
}

/** The last attempt to list the proxy's models. */
export type Catalog = {
  models: string[]
  /** Per model id; empty when the proxy has no capabilities endpoint. */
  capabilities: Record<string, ModelInfo>
  isLoading: boolean
  /** Why the last reload failed; absent when it worked. */
  error?: string
  /** `$.clock.now()` of the last successful reload. */
  loadedAt?: number
}

/** How full the proxy model's context was at its last answer. */
export type ContextUse = {
  model: string
  /** The prompt tokens the proxy reported for the last request. */
  inputTokens: number
}

/** The proxy connection as the pane shows it; the key itself never lands in state. */
export type Connection = {
  url: string
  /** The key's last four characters, or '' when no key is set. */
  keyHint: string
  /** Where the key comes from: typed in the pane, the plugin's config, or nowhere. */
  keySource: 'pane' | 'config' | 'none'
}

declare module 'claude-code' {
  interface PluginState {
    'provider-switch': {
      selection: Selection
      catalog: Catalog
      connection: Connection
      context: ContextUse | null
    }
  }
}
