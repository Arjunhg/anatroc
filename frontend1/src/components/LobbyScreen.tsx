import { type FormEvent, useState } from 'react'
import { DEFAULT_CALL_ID } from '../stream'
import './LobbyScreen.css'

export interface LobbyValues {
  displayName: string
  userId: string
  callId: string
}

interface Props {
  loading: boolean
  error: string | null
  onJoin: (values: LobbyValues) => void
}

export default function LobbyScreen({ loading, error, onJoin }: Props) {
  const [name, setName] = useState('')
  const [callId, setCallId] = useState(DEFAULT_CALL_ID)

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    const userId = trimmed.toLowerCase().replace(/\s+/g, '-')
    onJoin({ displayName: trimmed, userId, callId })
  }

  return (
    <div className="lobby-root">
      <header className="lobby-header">
        <div className="lobby-logo">
          <span className="lobby-logo-icon">[A]</span>
          <span className="lobby-logo-text">Anatroc</span>
        </div>
        <p className="lobby-tagline">Stream call + pipeline session in one flow</p>
      </header>

      <div className="lobby-card">
        <h2 className="lobby-card-title">Start Session</h2>

        <form className="lobby-form" onSubmit={handleSubmit}>
          <label className="form-label">
            Your name
            <input
              className="form-input"
              type="text"
              placeholder="e.g. Alice"
              value={name}
              onChange={(event) => setName(event.target.value)}
              required
              disabled={loading}
              autoComplete="off"
            />
          </label>

          <label className="form-label">
            Session ID
            <input
              className="form-input"
              type="text"
              placeholder={DEFAULT_CALL_ID}
              value={callId}
              onChange={(event) => setCallId(event.target.value)}
              disabled={loading}
              autoComplete="off"
            />
            <span className="form-hint">Use the same ID for Stream call and backend session.</span>
          </label>

          <div className="mode-hint">
            <span className="mode-hint-icon">i</span>
            <span>
              Start with <strong>camera</strong>. Switch to <strong>Share Screen</strong> to trigger periodic
              frame ingestion, OCR extraction, and retrieval sync.
            </span>
          </div>

          {error && <p className="form-error">Error: {error}</p>}

          <button type="submit" className="btn-primary" disabled={loading || !name.trim()}>
            {loading ? (
              <>
                <span className="spinner" /> Joining...
              </>
            ) : (
              'Join Session'
            )}
          </button>
        </form>
      </div>
    </div>
  )
}
