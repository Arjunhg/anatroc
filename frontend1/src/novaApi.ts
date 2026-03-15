import { novaApiUrl, tokenApiUrl, withTokenApiHeaders } from './stream'

export interface ScreenResult {
  response_text: string
  context_hits: number
  context_ids: string[]
  stored_record_id: string | null
}

export interface FrameIngestResult {
  session_id: string
  processed: boolean
  skipped_reason: string | null
  ocr_text: string
  ocr_record_id: string | null
  ingested_at: string
  analysis: ScreenResult | null
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(
    url,
    withTokenApiHeaders({
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    }),
  )

  if (!response.ok) {
    const body = await response.text()
    throw new Error(`API error ${response.status}: ${body}`)
  }

  return (await response.json()) as T
}

export async function startNovaSession(sessionId: string): Promise<void> {
  await requestJson<{ status: string; session_id: string }>(tokenApiUrl('/api/session/start'), {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })
}

export async function stopNovaSession(sessionId: string): Promise<void> {
  await requestJson<{ status: string; session_id: string }>(tokenApiUrl('/api/session/stop'), {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })
}

export async function ingestScreenFrame(
  sessionId: string,
  imageBase64: string,
  analysisPrompt?: string,
): Promise<FrameIngestResult> {
  return requestJson<FrameIngestResult>(novaApiUrl('/frame/ingest'), {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      image_base64: imageBase64,
      source_type: 'screen_share',
      analysis_prompt: analysisPrompt?.trim() ? analysisPrompt.trim() : null,
    }),
  })
}

export async function queryNova(sessionId: string, userText: string): Promise<ScreenResult> {
  return requestJson<ScreenResult>(novaApiUrl('/voice/query'), {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      user_text: userText,
    }),
  })
}

export async function generateDiagram(prompt: string): Promise<string> {
  const response = await requestJson<{ mermaid: string }>(novaApiUrl('/diagram'), {
    method: 'POST',
    body: JSON.stringify({ prompt }),
  })
  return response.mermaid
}
