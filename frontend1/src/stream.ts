import { StreamVideoClient, type User } from '@stream-io/video-react-sdk'

export const DEFAULT_CALL_TYPE = 'default'
export const DEFAULT_CALL_ID = 'vision-session-1'

/**
 * If unset, frontend uses Vite dev proxy (/api -> token_server.py).
 * For deployed frontend, set VITE_TOKEN_API_URL to backend origin.
 */
export const TOKEN_API_BASE = (import.meta.env.VITE_TOKEN_API_URL as string | undefined)?.trim() || '/api'

export function tokenApiUrl(path: string): string {
  const trimmedBase = TOKEN_API_BASE.replace(/\/+$/, '')
  const trimmedPath = path.replace(/^\/+/, '')
  return `${trimmedBase}/${trimmedPath}`
}

export function novaApiUrl(path: string): string {
  const trimmedPath = path.replace(/^\/+/, '')
  return tokenApiUrl(`/api/${trimmedPath}`)
}

export function sonicWsUrl(sessionId: string): string {
  const httpLike = tokenApiUrl(`/ws/sonic/${encodeURIComponent(sessionId)}`)
  if (httpLike.startsWith('http://')) return httpLike.replace(/^http:\/\//, 'ws://')
  if (httpLike.startsWith('https://')) return httpLike.replace(/^https:\/\//, 'wss://')

  const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://'
  return `${protocol}${window.location.host}${httpLike}`
}

export function withTokenApiHeaders(init: RequestInit = {}): RequestInit {
  return init
}

export interface TokenResponse {
  token: string
  api_key: string
}

export async function fetchToken(userId: string): Promise<TokenResponse> {
  const url = `${tokenApiUrl('/token')}?user_id=${encodeURIComponent(userId)}`
  const response = await fetch(url, withTokenApiHeaders())
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`Token server error ${response.status}: ${body}`)
  }

  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    const body = await response.text()
    throw new Error(
      `Token server returned non-JSON response (${contentType || 'unknown'}): ${body.slice(0, 200)}`,
    )
  }

  return (await response.json()) as TokenResponse
}

export interface ClientBundle {
  client: StreamVideoClient
  apiKey: string
}

export async function createClient(userId: string, userName: string): Promise<ClientBundle> {
  const { token, api_key: apiKey } = await fetchToken(userId)

  const user: User = {
    id: userId,
    name: userName,
    type: 'authenticated',
  }

  const client = new StreamVideoClient({
    apiKey,
    user,
    token,
  })

  return { client, apiKey }
}
