import { type FormEvent, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  CheckCircle2,
  Copy,
  KeyRound,
  Link2,
  Play,
  RefreshCw,
  Send,
  ShieldCheck,
  Terminal,
  Trash2,
} from 'lucide-react'
import './App.css'

type Mode = 'generate' | 'chat'
type ConnectionState = 'idle' | 'checking' | 'connected' | 'error'

type AppConfig = {
  baseUrl: string
  username: string
  apiKey: string
  model: string
  mode: Mode
  stream: boolean
  remember: boolean
}

type OllamaModel = {
  name: string
  modified_at?: string
  size?: number
}

type TagsResponse = {
  models?: OllamaModel[]
}

type GenerateResponse = {
  response?: string
  error?: string
}

type ChatResponse = {
  message?: {
    content?: string
  }
  error?: string
}

const STORAGE_KEY = 'stylomex.ollama.tunnel.v1'

const DEFAULT_CONFIG: AppConfig = {
  baseUrl: '',
  username: 'ollama',
  apiKey: '',
  model: 'gemma3:1b',
  mode: 'generate',
  stream: true,
  remember: false,
}

const DEFAULT_PROMPT =
  'Explain how a zrok tunnel lets a deployed frontend call a local Ollama model.'

function normalizeBaseUrl(value: string) {
  return value.trim().replace(/\/+$/, '')
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong.'
}

function encodeBasicAuth(username: string, apiKey: string) {
  const bytes = new TextEncoder().encode(`${username}:${apiKey}`)
  let binary = ''
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte)
  })
  return `Basic ${window.btoa(binary)}`
}

function formatBytes(size?: number) {
  if (!size || Number.isNaN(size)) return 'size unknown'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = size
  let index = 0
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`
}

function loadStoredConfig(): AppConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_CONFIG
    const parsed = JSON.parse(raw) as Partial<AppConfig>
    return { ...DEFAULT_CONFIG, ...parsed, remember: true }
  } catch {
    return DEFAULT_CONFIG
  }
}

function buildCurlCommand(config: AppConfig, prompt: string) {
  const baseUrl = normalizeBaseUrl(config.baseUrl) || 'https://your-share.share.zrok.io'
  const payload =
    config.mode === 'chat'
      ? {
          model: config.model || 'gemma3:1b',
          messages: [{ role: 'user', content: prompt || DEFAULT_PROMPT }],
          stream: false,
        }
      : {
          model: config.model || 'gemma3:1b',
          prompt: prompt || DEFAULT_PROMPT,
          stream: false,
        }

  return [
    `curl -u "${config.username || 'ollama'}:${config.apiKey || 'API_KEY'}" "${baseUrl}/api/${config.mode}" \\`,
    '  -H "Content-Type: application/json" \\',
    `  -d '${JSON.stringify(payload)}'`,
  ].join('\n')
}

function App() {
  const [config, setConfig] = useState<AppConfig>(() => loadStoredConfig())
  const [models, setModels] = useState<OllamaModel[]>([])
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [output, setOutput] = useState('')
  const [status, setStatus] = useState<ConnectionState>('idle')
  const [statusText, setStatusText] = useState('Not checked')
  const [isRunning, setIsRunning] = useState(false)
  const [copyState, setCopyState] = useState('Copy')

  const baseUrl = normalizeBaseUrl(config.baseUrl)
  const canCallApi = Boolean(baseUrl && config.username.trim() && config.apiKey.trim())
  const selectedModel = config.model || models[0]?.name || DEFAULT_CONFIG.model
  const curlCommand = useMemo(() => buildCurlCommand(config, prompt), [config, prompt])

  useEffect(() => {
    if (config.remember) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
    } else {
      localStorage.removeItem(STORAGE_KEY)
    }
  }, [config])

  function updateConfig(patch: Partial<AppConfig>) {
    setConfig((current) => ({ ...current, ...patch }))
  }

  async function request(path: string, init: RequestInit = {}) {
    if (!canCallApi) {
      throw new Error('Enter the zrok URL, username, and API key first.')
    }

    try {
      const headers = new Headers(init.headers)
      headers.set('Authorization', encodeBasicAuth(config.username.trim(), config.apiKey.trim()))

      return await fetch(`${baseUrl}${path}`, {
        ...init,
        headers,
      })
    } catch {
      throw new Error(
        'The browser could not reach the endpoint. Check the zrok URL, tunnel process, and Ollama CORS origin.',
      )
    }
  }

  async function checkConnection() {
    setStatus('checking')
    setStatusText('Checking endpoint')
    try {
      const response = await request('/api/tags')
      if (response.status === 401) {
        throw new Error('Basic Auth rejected the username or API key.')
      }
      if (!response.ok) {
        throw new Error(`Ollama returned HTTP ${response.status}.`)
      }

      const text = await response.text()
      const payload = JSON.parse(text || '{}') as TagsResponse

      const nextModels = payload.models || []
      setModels(nextModels)
      if (nextModels.length > 0 && !nextModels.some((model) => model.name === config.model)) {
        updateConfig({ model: nextModels[0].name })
      }
      setStatus('connected')
      setStatusText(nextModels.length ? `${nextModels.length} model(s) available` : 'Connected, no models found')
    } catch (error) {
      setStatus('error')
      setStatusText(messageFromError(error))
    }
  }

  async function readStreamingResponse(response: Response) {
    if (!response.body) {
      throw new Error('The response body is not streamable in this browser.')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const rawLine of lines) {
        const line = rawLine.trim()
        if (!line) continue
        const chunk = JSON.parse(line) as GenerateResponse & ChatResponse
        if (chunk.error) throw new Error(chunk.error)
        const token = chunk.response || chunk.message?.content || ''
        if (token) {
          setOutput((current) => current + token)
        }
      }
    }
  }

  async function runPrompt(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const cleanPrompt = prompt.trim()
    if (!cleanPrompt) {
      setStatus('error')
      setStatusText('Prompt is empty.')
      return
    }

    setIsRunning(true)
    setOutput('')
    setStatus('checking')
    setStatusText('Calling model')

    const body =
      config.mode === 'chat'
        ? {
            model: selectedModel,
            messages: [{ role: 'user', content: cleanPrompt }],
            stream: config.stream,
          }
        : {
            model: selectedModel,
            prompt: cleanPrompt,
            stream: config.stream,
          }

    try {
      const response = await request(`/api/${config.mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (response.status === 401) {
        throw new Error('Basic Auth rejected the username or API key.')
      }
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Ollama returned HTTP ${response.status}.`)
      }

      if (config.stream) {
        await readStreamingResponse(response)
      } else {
        const payload = (await response.json()) as GenerateResponse & ChatResponse
        if (payload.error) throw new Error(payload.error)
        setOutput(payload.response || payload.message?.content || '')
      }

      setStatus('connected')
      setStatusText('Request complete')
    } catch (error) {
      setStatus('error')
      setStatusText(messageFromError(error))
    } finally {
      setIsRunning(false)
    }
  }

  async function copyCurl() {
    try {
      await navigator.clipboard.writeText(curlCommand)
      setCopyState('Copied')
    } catch {
      setCopyState('Copy failed')
    }
    window.setTimeout(() => setCopyState('Copy'), 1400)
  }

  function clearSession() {
    setOutput('')
    setPrompt(DEFAULT_PROMPT)
    setStatus('idle')
    setStatusText('Not checked')
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Stylomex local model gateway</p>
          <h1>Ollama Tunnel Console</h1>
        </div>
        <div className={`status-pill ${status}`}>
          {status === 'connected' ? <CheckCircle2 size={18} /> : <Activity size={18} />}
          <span>{statusText}</span>
        </div>
      </header>

      <section className="workspace">
        <form className="panel control-panel" onSubmit={runPrompt}>
          <div className="panel-title">
            <ShieldCheck size={20} />
            <h2>Connection</h2>
          </div>

          <label htmlFor="base-url">zrok URL</label>
          <div className="input-with-icon">
            <Link2 size={18} />
            <input
              id="base-url"
              value={config.baseUrl}
              placeholder="https://example.share.zrok.io"
              spellCheck={false}
              onChange={(event) => updateConfig({ baseUrl: event.target.value })}
            />
          </div>

          <div className="two-fields">
            <div>
              <label htmlFor="username">Username</label>
              <input
                id="username"
                value={config.username}
                spellCheck={false}
                onChange={(event) => updateConfig({ username: event.target.value })}
              />
            </div>
            <div>
              <label htmlFor="api-key">API key</label>
              <div className="input-with-icon">
                <KeyRound size={18} />
                <input
                  id="api-key"
                  type="password"
                  value={config.apiKey}
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(event) => updateConfig({ apiKey: event.target.value })}
                />
              </div>
            </div>
          </div>

          <div className="button-row">
            <button type="button" className="secondary-button" disabled={!canCallApi} onClick={checkConnection}>
              <RefreshCw size={18} />
              <span>Check</span>
            </button>
            <label className="toggle-row" htmlFor="remember-config">
              <span>Remember</span>
              <input
                id="remember-config"
                type="checkbox"
                checked={config.remember}
                onChange={(event) => updateConfig({ remember: event.target.checked })}
              />
            </label>
          </div>

          <div className="panel-title spaced">
            <Terminal size={20} />
            <h2>Request</h2>
          </div>

          <div className="two-fields">
            <div>
              <label htmlFor="model">Model</label>
              <input
                id="model"
                list="model-options"
                value={selectedModel}
                spellCheck={false}
                onChange={(event) => updateConfig({ model: event.target.value })}
              />
              <datalist id="model-options">
                {models.map((model) => (
                  <option key={model.name} value={model.name} />
                ))}
              </datalist>
            </div>
            <div>
              <label htmlFor="mode">Endpoint</label>
              <select
                id="mode"
                value={config.mode}
                onChange={(event) => updateConfig({ mode: event.target.value as Mode })}
              >
                <option value="generate">/api/generate</option>
                <option value="chat">/api/chat</option>
              </select>
            </div>
          </div>

          <label htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            value={prompt}
            rows={7}
            onChange={(event) => setPrompt(event.target.value)}
          />

          <div className="button-row">
            <button type="submit" className="primary-button" disabled={!canCallApi || isRunning}>
              {isRunning ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
              <span>{isRunning ? 'Running' : 'Run'}</span>
            </button>
            <label className="toggle-row" htmlFor="stream-response">
              <span>Stream</span>
              <input
                id="stream-response"
                type="checkbox"
                checked={config.stream}
                onChange={(event) => updateConfig({ stream: event.target.checked })}
              />
            </label>
          </div>
        </form>

        <section className="panel result-panel">
          <div className="result-head">
            <div className="panel-title">
              <Play size={20} />
              <h2>Response</h2>
            </div>
            <button type="button" className="icon-button" aria-label="Clear response" onClick={clearSession}>
              <Trash2 size={18} />
            </button>
          </div>

          <pre className="output-box">{output || 'Model output will appear here.'}</pre>

          <div className="lower-grid">
            <div className="models-box">
              <h3>Models</h3>
              {models.length ? (
                <ul>
                  {models.map((model) => (
                    <li key={model.name}>
                      <strong>{model.name}</strong>
                      <span>{formatBytes(model.size)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No models loaded.</p>
              )}
            </div>

            <div className="curl-box">
              <div className="curl-head">
                <h3>curl</h3>
                <button type="button" className="secondary-button compact" onClick={copyCurl}>
                  <Copy size={16} />
                  <span>{copyState}</span>
                </button>
              </div>
              <pre>{curlCommand}</pre>
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
