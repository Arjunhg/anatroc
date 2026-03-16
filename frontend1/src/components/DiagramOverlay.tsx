import { useEffect, useMemo, useState } from 'react'
import mermaid from 'mermaid'

interface Props {
  mermaidText: string
  prompt: string
  updatedAt?: string | null
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center'
  minimized: boolean
  onClose: () => void
  onControl: (action: 'move' | 'minimize' | 'expand' | 'export', position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center') => void
}

let mermaidInitialized = false

function ensureMermaidInitialized(): void {
  if (mermaidInitialized) return
  mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    securityLevel: 'strict',
  })
  mermaidInitialized = true
}

export default function DiagramOverlay({ mermaidText, prompt, updatedAt, position, minimized, onClose, onControl }: Props) {
  const [svgMarkup, setSvgMarkup] = useState('')
  const [error, setError] = useState('')

  const title = useMemo(() => {
    const cleaned = prompt.trim()
    if (!cleaned) return 'Generated architecture diagram'
    return cleaned.length > 72 ? `${cleaned.slice(0, 69)}...` : cleaned
  }, [prompt])

  useEffect(() => {
    let cancelled = false

    const renderDiagram = async () => {
      ensureMermaidInitialized()
      setError('')
      setSvgMarkup('')
      try {
        const uniqueId = `nova-overlay-${Date.now()}`
        const { svg } = await mermaid.render(uniqueId, mermaidText)
        if (!cancelled) {
          setSvgMarkup(svg)
        }
      } catch (renderError) {
        if (!cancelled) {
          setError(renderError instanceof Error ? renderError.message : String(renderError))
        }
      }
    }

    void renderDiagram()
    return () => {
      cancelled = true
    }
  }, [mermaidText])

  return (
    <div className={`diagram-overlay-shell diagram-overlay-shell--${position}`} aria-live="polite">
      <section className="diagram-overlay-card">
        <header className="diagram-overlay-header">
          <div className="diagram-overlay-meta">
            <h4 className="diagram-overlay-title">Overlay Diagram</h4>
            <p className="diagram-overlay-subtitle">{title}</p>
            {updatedAt ? <p className="diagram-overlay-time">Updated {new Date(updatedAt).toLocaleTimeString()}</p> : null}
          </div>
          <div className="diagram-overlay-controls">
            {minimized ? (
              <button className="diagram-overlay-btn" onClick={() => onControl('expand')} type="button">
                Expand
              </button>
            ) : (
              <button className="diagram-overlay-btn" onClick={() => onControl('minimize')} type="button">
                Minimize
              </button>
            )}
            <button className="diagram-overlay-btn" onClick={() => onControl('export')} type="button">
              Export
            </button>
            <button className="diagram-overlay-close" onClick={onClose} type="button">
              Remove
            </button>
          </div>
        </header>

        <div className="diagram-overlay-position-bar">
          <button className="diagram-overlay-pos-btn" onClick={() => onControl('move', 'top-left')} type="button">TL</button>
          <button className="diagram-overlay-pos-btn" onClick={() => onControl('move', 'top-right')} type="button">TR</button>
          <button className="diagram-overlay-pos-btn" onClick={() => onControl('move', 'bottom-left')} type="button">BL</button>
          <button className="diagram-overlay-pos-btn" onClick={() => onControl('move', 'bottom-right')} type="button">BR</button>
          <button className="diagram-overlay-pos-btn" onClick={() => onControl('move', 'center')} type="button">C</button>
        </div>

        <div className="diagram-overlay-content" hidden={minimized}>
          {error ? (
            <div className="diagram-overlay-error">
              <p>Diagram render failed: {error}</p>
              <pre>{mermaidText}</pre>
            </div>
          ) : null}
          {!error && !svgMarkup ? <p className="diagram-overlay-loading">Rendering diagram...</p> : null}
          {!error && svgMarkup ? (
            <div
              className="diagram-overlay-svg"
              dangerouslySetInnerHTML={{ __html: svgMarkup }}
            />
          ) : null}
        </div>
      </section>
    </div>
  )
}
