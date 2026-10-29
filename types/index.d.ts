/** Which backend answers the model requests: Claude Code's own, or a model of a named provider. */
export type Selection = { provider: 'claude' } | { provider: 'proxy'; providerId: string; model: string }

/**
 * An OpenAI-compatible provider as the pane shows it; its key never lands in state.
 * `config` is the one set in the plugin's options, which the pane cannot remove.
 */
export type ProviderView = {
  id: string
  name: string
  url: string
  /** The key's last four characters, or '' when the provider has no key. */
  keyHint: string
  source: 'pane' | 'config'
}

/**
 * What a provider says a model can do (`/models`, `/model-capabilities`).
 * A field is absent when the provider does not say: treat it as unknown, not as false.
 */
export type ModelInfo = {
  contextWindow?: number
  maxOutputTokens?: number
  image?: boolean
  tools?: boolean
  reasoning?: boolean
  text?: boolean
}

/** What the person set by hand for one model, over what its provider says. */
export type Override = { image?: boolean; contextWindow?: number }

/** The last attempt to list one provider's models. */
export type Catalog = {
  /** Chat models only, sorted. */
  models: string[]
  /** Per model id; empty when the provider says nothing about its models. */
  capabilities: Record<string, ModelInfo>
  isLoading: boolean
  /** Why the last reload failed; absent when it worked. */
  error?: string
  /** `$.clock.now()` of the last successful reload. */
  loadedAt?: number
}

/** How full a proxy model's context was at its last answer. */
export type ContextUse = {
  /** `<providerId>/<model>`. */
  model: string
  /** The prompt tokens the provider reported for the last request. */
  inputTokens: number
}

/** The add-provider form as typed so far. */
export type Draft = { name: string; url: string; key: string }

declare module 'claude-code' {
  interface PluginState {
    'provider-switch': {
      selection: Selection
      providers: ProviderView[]
      /** Per provider id. */
      catalogs: Record<string, Catalog>
      /** Per `<providerId>/<model>`. */
      overrides: Record<string, Override>
      context: ContextUse | null
      draft: Draft
      /** The provider whose form is open in the pane: its id, `+new` for the add form, or none. */
      editing: string | null
    }
  }
}
