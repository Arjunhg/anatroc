import { useEffect, useMemo, useRef, useState } from 'react'
import { ParticipantView, useCall, useCallStateHooks } from '@stream-io/video-react-sdk'
import { generateDiagram, ingestScreenFrame, queryNova } from '../novaApi'
import { sonicWsUrl } from '../stream'
import DiagramOverlay from './DiagramOverlay'
import Controls from './Controls'
import './CallScreen.css'

interface Props {
  sessionId: string
  onLeave: () => Promise<void>
}

interface PlaybackChunk {
  samples: Float32Array<ArrayBufferLike>
  sampleRateHz: number
}

interface SessionOverlayDiagram {
  mermaid: string
  prompt: string
  updatedAt: string | null
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center'
  minimized: boolean
}

const CAPTURE_INTERVAL_MS = Math.max(
  1000,
  Number(import.meta.env.VITE_SCREEN_CAPTURE_INTERVAL_SECONDS ?? '4') * 1000,
)
const CAPTURE_MAX_WIDTH = Math.max(640, Number(import.meta.env.VITE_SCREEN_CAPTURE_MAX_WIDTH ?? '1280'))
const CAPTURE_JPEG_QUALITY = Math.min(
  0.9,
  Math.max(0.45, Number(import.meta.env.VITE_SCREEN_CAPTURE_JPEG_QUALITY ?? '0.68')),
)
const SONIC_SPEECH_THRESHOLD = Math.max(10, Number(import.meta.env.VITE_SONIC_SPEECH_THRESHOLD ?? '120'))
const SONIC_TURN_SILENCE_MS = Math.max(300, Number(import.meta.env.VITE_SONIC_TURN_SILENCE_MS ?? '1200'))
const SONIC_PLAYBACK_MIC_SUPPRESSION_MS = Math.max(
  0,
  Number(import.meta.env.VITE_SONIC_PLAYBACK_MIC_SUPPRESSION_MS ?? '450'),
)

function downsampleFloat32(input: Float32Array, inputRate: number, outputRate: number): Float32Array {
  if (outputRate >= inputRate) return input
  const ratio = inputRate / outputRate
  const outputLength = Math.round(input.length / ratio)
  const output = new Float32Array(outputLength)
  let outputIndex = 0
  let inputIndex = 0
  while (outputIndex < outputLength) {
    const nextInputIndex = Math.round((outputIndex + 1) * ratio)
    let accumulator = 0
    let count = 0
    for (let i = inputIndex; i < nextInputIndex && i < input.length; i += 1) {
      accumulator += input[i]
      count += 1
    }
    output[outputIndex] = count > 0 ? accumulator / count : 0
    outputIndex += 1
    inputIndex = nextInputIndex
  }
  return output
}

function float32ToPcm16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length)
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]))
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
  }
  return output
}

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = ''
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize)
    binary += String.fromCharCode(...chunk)
  }
  return btoa(binary)
}

function pcm16Base64ToFloat32(pcmBase64: string): Float32Array {
  const binary = atob(pcmBase64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }
  const view = new DataView(bytes.buffer)
  const samples = new Float32Array(bytes.length / 2)
  for (let i = 0; i < samples.length; i += 1) {
    const sample = view.getInt16(i * 2, true)
    samples[i] = sample / 32768
  }
  return samples
}

export default function CallScreen({ sessionId, onLeave }: Props) {
  const call = useCall()
  const { useParticipants, useLocalParticipant, useScreenShareState } = useCallStateHooks()
  const participants = useParticipants()
  const localParticipant = useLocalParticipant()
  const { status: screenShareStatus, mediaStream: screenShareMediaStream } = useScreenShareState()
  const isSharing = screenShareStatus === 'enabled'

  const [elapsed, setElapsed] = useState(0)
  const [framesProcessed, setFramesProcessed] = useState(0)
  const [lastOcr, setLastOcr] = useState('')
  const [ingestStatus, setIngestStatus] = useState('Screen share is idle')
  const [analysisPrompt, setAnalysisPrompt] = useState('')
  const [manualQuestion, setManualQuestion] = useState('')
  const [userTranscript, setUserTranscript] = useState('')
  const [assistantReply, setAssistantReply] = useState('')
  const [contextHits, setContextHits] = useState(0)
  const [diagramPrompt, setDiagramPrompt] = useState('')
  const [diagramOutput, setDiagramOutput] = useState('')
  const [queryBusy, setQueryBusy] = useState(false)
  const [diagramBusy, setDiagramBusy] = useState(false)
  const [sonicConnected, setSonicConnected] = useState(false)
  const [sonicStatus, setSonicStatus] = useState('Disconnected')
  const [sonicUserTranscript, setSonicUserTranscript] = useState('')
  const [sonicAssistantTranscript, setSonicAssistantTranscript] = useState('')
  const [overlayDiagram, setOverlayDiagram] = useState<SessionOverlayDiagram | null>(null)

  const ingestBusyRef = useRef(false)
  const analysisPromptRef = useRef('')
  const sonicSocketRef = useRef<WebSocket | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const micContextRef = useRef<AudioContext | null>(null)
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const micProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const micSilenceGainRef = useRef<GainNode | null>(null)
  const playbackContextRef = useRef<AudioContext | null>(null)
  const playbackQueueRef = useRef<PlaybackChunk[]>([])
  const playbackLoopActiveRef = useRef(false)
  const sonicReadyRef = useRef(false)
  const sonicMicChunkCountRef = useRef(0)
  const sonicLastSpeechAtRef = useRef(0)
  const sonicTurnHasSpeechRef = useRef(false)
  const assistantPlaybackActiveRef = useRef(false)
  const assistantPlaybackMuteUntilRef = useRef(0)

  useEffect(() => {
    const timer = window.setInterval(() => setElapsed((sec) => sec + 1), 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    analysisPromptRef.current = analysisPrompt
  }, [analysisPrompt])

  const remoteParticipants = useMemo(
    () => participants.filter((p) => p.sessionId !== localParticipant?.sessionId),
    [localParticipant?.sessionId, participants],
  )

  const callId = call?.id ?? sessionId

  const stopSonic = async () => {
    const socket = sonicSocketRef.current
    sonicSocketRef.current = null
    sonicReadyRef.current = false
    sonicMicChunkCountRef.current = 0
    sonicLastSpeechAtRef.current = 0
    assistantPlaybackActiveRef.current = false
    assistantPlaybackMuteUntilRef.current = 0

    if (socket && socket.readyState === WebSocket.OPEN) {
      if (sonicTurnHasSpeechRef.current) {
        socket.send(JSON.stringify({ type: 'audio_turn_end' }))
      }
      socket.send(JSON.stringify({ type: 'stop' }))
    }
    if (socket && socket.readyState <= WebSocket.OPEN) {
      socket.close()
    }

    micProcessorRef.current?.disconnect()
    micSourceRef.current?.disconnect()
    micSilenceGainRef.current?.disconnect()
    if (micContextRef.current) {
      await micContextRef.current.close()
    }
    micProcessorRef.current = null
    micSourceRef.current = null
    micSilenceGainRef.current = null
    micContextRef.current = null

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop())
      micStreamRef.current = null
    }

    playbackQueueRef.current = []
    if (playbackContextRef.current) {
      await playbackContextRef.current.close()
      playbackContextRef.current = null
    }
    playbackLoopActiveRef.current = false

    setSonicConnected(false)
    setSonicStatus('Disconnected')
    sonicTurnHasSpeechRef.current = false
  }

  const playQueuedAssistantAudio = async () => {
    if (playbackLoopActiveRef.current) return
    playbackLoopActiveRef.current = true
    try {
      assistantPlaybackActiveRef.current = true
      while (playbackQueueRef.current.length > 0) {
        const next = playbackQueueRef.current.shift()
        if (!next) break

        if (!playbackContextRef.current) {
          playbackContextRef.current = new AudioContext()
        }
        const audioContext = playbackContextRef.current
        if (audioContext.state === 'suspended') {
          await audioContext.resume()
        }
        const audioBuffer = audioContext.createBuffer(1, next.samples.length, next.sampleRateHz)
        const channelData = audioBuffer.getChannelData(0)
        channelData.set(next.samples)

        const source = audioContext.createBufferSource()
        source.buffer = audioBuffer
        source.connect(audioContext.destination)
        await new Promise<void>((resolve) => {
          source.onended = () => resolve()
          source.start(0)
        })
      }
    } finally {
      assistantPlaybackActiveRef.current = false
      assistantPlaybackMuteUntilRef.current = Date.now() + SONIC_PLAYBACK_MIC_SUPPRESSION_MS
      playbackLoopActiveRef.current = false
    }
  }

  const startSonic = async () => {
    if (sonicSocketRef.current) return
    setSonicStatus('Connecting Sonic...')
    setSonicUserTranscript('')
    setSonicAssistantTranscript('')
    sonicTurnHasSpeechRef.current = false
    sonicLastSpeechAtRef.current = 0
    sonicMicChunkCountRef.current = 0

    try {
      const microphoneStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      micStreamRef.current = microphoneStream

      const socket = new WebSocket(sonicWsUrl(sessionId))
      sonicSocketRef.current = socket

      socket.onopen = () => {
        setSonicStatus('Sonic connected')
      }

      socket.onmessage = (event) => {
        let payload: Record<string, unknown> = {}
        try {
          payload = JSON.parse(String(event.data)) as Record<string, unknown>
        } catch {
          return
        }

        const eventType = String(payload.type ?? '')
        if (eventType === 'ready') {
          sonicReadyRef.current = true
          setSonicConnected(true)
          setSonicStatus('Sonic ready')
          return
        }
        if (eventType === 'transcript') {
          const role = String(payload.role ?? '')
          const text = String(payload.text ?? '')
          if (!text) return
          if (role === 'user') {
            setSonicUserTranscript((prev) => `${prev}\n${text}`.trim())
          } else if (role === 'assistant') {
            setSonicAssistantTranscript((prev) => `${prev}\n${text}`.trim())
          }
          return
        }
        if (eventType === 'assistant_audio') {
          const base64Audio = String(payload.audio_base64 ?? '')
          const sampleRate = Number(payload.sample_rate_hz ?? 24000)
          if (base64Audio) {
            playbackQueueRef.current.push({
              samples: pcm16Base64ToFloat32(base64Audio),
              sampleRateHz: sampleRate,
            })
            void playQueuedAssistantAudio()
          }
          return
        }
        if (eventType === 'assistant_interrupted') {
          playbackQueueRef.current = []
          return
        }
        if (eventType === 'overlay_diagram') {
          const mermaid = String(payload.mermaid ?? '').trim()
          if (!mermaid) return
          const prompt = String(payload.prompt ?? '').trim()
          const updatedAt = payload.updated_at ? String(payload.updated_at) : null
          const rawPosition = String(payload.position ?? 'top-left').trim().toLowerCase()
          const position =
            rawPosition === 'top-right' ||
              rawPosition === 'bottom-left' ||
              rawPosition === 'bottom-right' ||
              rawPosition === 'center'
              ? rawPosition
              : 'top-left'
          const minimized = Boolean(payload.minimized ?? false)
          setOverlayDiagram({
            mermaid,
            prompt,
            updatedAt,
            position,
            minimized,
          })
          return
        }
        if (eventType === 'overlay_control') {
          const action = String(payload.action ?? '').trim().toLowerCase()
          if (action === 'move') {
            const rawPosition = String(payload.position ?? '').trim().toLowerCase()
            setOverlayDiagram((prev) => {
              if (!prev) return prev
              const position =
                rawPosition === 'top-right' ||
                  rawPosition === 'bottom-left' ||
                  rawPosition === 'bottom-right' ||
                  rawPosition === 'center'
                  ? rawPosition
                  : 'top-left'
              return { ...prev, position }
            })
            return
          }
          if (action === 'minimize') {
            setOverlayDiagram((prev) => (prev ? { ...prev, minimized: true } : prev))
            return
          }
          if (action === 'expand') {
            setOverlayDiagram((prev) => (prev ? { ...prev, minimized: false } : prev))
            return
          }
          return
        }
        if (eventType === 'overlay_export') {
          const mermaid = String(payload.mermaid ?? '').trim()
          if (!mermaid) return
          const exportText = mermaid.endsWith('\n') ? mermaid : `${mermaid}\n`
          const nowStamp = new Date().toISOString().replace(/[:.]/g, '-')
          const fileName = `mermaid-overlay-${sessionId}-${nowStamp}.mmd`
          void navigator.clipboard
            .writeText(exportText)
            .then(() => {
              setSonicStatus('Overlay Mermaid copied to clipboard')
            })
            .catch(() => {
              const blob = new Blob([exportText], { type: 'text/plain;charset=utf-8' })
              const url = URL.createObjectURL(blob)
              const anchor = document.createElement('a')
              anchor.href = url
              anchor.download = fileName
              document.body.append(anchor)
              anchor.click()
              anchor.remove()
              URL.revokeObjectURL(url)
              setSonicStatus(`Overlay Mermaid downloaded as ${fileName}`)
            })
          return
        }
        if (eventType === 'overlay_clear') {
          setOverlayDiagram(null)
          return
        }
        if (eventType === 'overlay_error') {
          const message = String(payload.message ?? 'Unable to generate overlay diagram')
          setSonicStatus(`Overlay error: ${message}`)
          return
        }
        if (eventType === 'error') {
          const message = String(payload.message ?? 'Unknown Sonic error')
          setSonicStatus(`Sonic error: ${message}`)
        }
      }

      socket.onerror = () => {
        setSonicStatus('Sonic websocket error')
      }

      socket.onclose = () => {
        sonicReadyRef.current = false
        setSonicConnected(false)
        setSonicStatus('Disconnected')
      }

      const isFirefox = navigator.userAgent.toLowerCase().includes('firefox')
      const micContext = isFirefox ? new AudioContext() : new AudioContext({ sampleRate: 16000 })
      micContextRef.current = micContext
      if (micContext.state === 'suspended') {
        await micContext.resume()
      }
      const source = micContext.createMediaStreamSource(microphoneStream)
      micSourceRef.current = source
      const processor = micContext.createScriptProcessor(512, 1, 1)
      micProcessorRef.current = processor
      const silenceGain = micContext.createGain()
      silenceGain.gain.value = 0
      micSilenceGainRef.current = silenceGain

      processor.onaudioprocess = (audioProcessEvent: AudioProcessingEvent) => {
        const ws = sonicSocketRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN || !sonicReadyRef.current) return

        const now = Date.now()
        if (assistantPlaybackActiveRef.current || now < assistantPlaybackMuteUntilRef.current) {
          return
        }

        const channelData = audioProcessEvent.inputBuffer.getChannelData(0)
        const pcmInput =
          micContext.sampleRate === 16000 ? channelData : downsampleFloat32(channelData, micContext.sampleRate, 16000)
        const pcm16 = float32ToPcm16(pcmInput)
        let amplitudeTotal = 0
        for (let i = 0; i < pcm16.length; i += 1) {
          amplitudeTotal += Math.abs(pcm16[i])
        }
        const avgAbs = amplitudeTotal / Math.max(1, pcm16.length)
        if (avgAbs >= SONIC_SPEECH_THRESHOLD) {
          sonicTurnHasSpeechRef.current = true
          sonicLastSpeechAtRef.current = now
        } else if (
          sonicTurnHasSpeechRef.current &&
          sonicLastSpeechAtRef.current > 0 &&
          now - sonicLastSpeechAtRef.current >= SONIC_TURN_SILENCE_MS
        ) {
          ws.send(JSON.stringify({ type: 'audio_turn_end' }))
          sonicTurnHasSpeechRef.current = false
          setSonicStatus(`Sonic turn sent (${sonicMicChunkCountRef.current} mic chunks)`)
        }
        const audioBase64 = uint8ToBase64(new Uint8Array(pcm16.buffer))
        ws.send(JSON.stringify({ type: 'audio_chunk', audio_base64: audioBase64 }))
        sonicMicChunkCountRef.current += 1
        if (sonicMicChunkCountRef.current === 1 || sonicMicChunkCountRef.current % 100 === 0) {
          setSonicStatus(`Sonic streaming (${sonicMicChunkCountRef.current} mic chunks sent)`)
        }
      }

      source.connect(processor)
      processor.connect(silenceGain)
      silenceGain.connect(micContext.destination)
    } catch (error) {
      setSonicStatus(error instanceof Error ? error.message : String(error))
      await stopSonic()
    }
  }

  useEffect(() => {
    return () => {
      void stopSonic()
    }
  }, [])

  useEffect(() => {
    if (!isSharing || !screenShareMediaStream) {
      setIngestStatus('Screen share is idle')
      return
    }

    let isCancelled = false
    const captureVideo = document.createElement('video')
    captureVideo.autoplay = true
    captureVideo.muted = true
    captureVideo.playsInline = true
    captureVideo.srcObject = screenShareMediaStream

    const captureCanvas = document.createElement('canvas')
    const captureContext = captureCanvas.getContext('2d', { alpha: false })

    if (!captureContext) {
      setIngestStatus('Capture context unavailable')
      return
    }

    const ingestOnce = async () => {
      if (isCancelled || ingestBusyRef.current) return
      if (captureVideo.readyState < 2) return
      if (!captureVideo.videoWidth || !captureVideo.videoHeight) return

      ingestBusyRef.current = true
      try {
        const scaleRatio = Math.min(1, CAPTURE_MAX_WIDTH / captureVideo.videoWidth)
        captureCanvas.width = Math.max(1, Math.round(captureVideo.videoWidth * scaleRatio))
        captureCanvas.height = Math.max(1, Math.round(captureVideo.videoHeight * scaleRatio))
        captureContext.drawImage(captureVideo, 0, 0, captureCanvas.width, captureCanvas.height)

        const imageDataUrl = captureCanvas.toDataURL('image/jpeg', CAPTURE_JPEG_QUALITY)
        const result = await ingestScreenFrame(sessionId, imageDataUrl, analysisPromptRef.current)

        if (!result.processed) {
          setIngestStatus(`Frame skipped: ${result.skipped_reason ?? 'capture interval gate'}`)
          return
        }

        setFramesProcessed((count) => count + 1)
        setLastOcr(result.ocr_text ?? '')
        setIngestStatus(
          `Frame indexed at ${new Date(result.ingested_at).toLocaleTimeString()} (${captureCanvas.width}x${captureCanvas.height})`,
        )

        if (result.analysis) {
          setAssistantReply(result.analysis.response_text)
          setContextHits(result.analysis.context_hits)
        }

      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        setIngestStatus(`Ingest failed: ${message}`)
      } finally {
        ingestBusyRef.current = false
      }
    }

    void captureVideo.play().catch(() => {
      setIngestStatus('Unable to start local screen capture preview')
    })

    void ingestOnce()
    const timer = window.setInterval(() => {
      void ingestOnce()
    }, CAPTURE_INTERVAL_MS)

    return () => {
      isCancelled = true
      window.clearInterval(timer)
      captureVideo.pause()
      captureVideo.srcObject = null
    }
  }, [isSharing, screenShareMediaStream, sessionId, sonicConnected])

  const submitQuestion = async (question: string) => {
    const normalized = question.trim()
    if (!normalized || queryBusy) return

    setQueryBusy(true)
    setUserTranscript(normalized)

    try {
      const answer = await queryNova(sessionId, normalized)
      setAssistantReply(answer.response_text)
      setContextHits(answer.context_hits)
    } catch (error) {
      setAssistantReply(error instanceof Error ? error.message : String(error))
    } finally {
      setQueryBusy(false)
    }
  }

  const handleManualQuery = async () => {
    const current = manualQuestion
    setManualQuestion('')
    await submitQuestion(current)
  }

  const handleGenerateDiagram = async () => {
    const normalized = diagramPrompt.trim()
    if (!normalized || diagramBusy) return
    setDiagramBusy(true)
    try {
      const mermaid = await generateDiagram(normalized)
      setDiagramOutput(mermaid)
      setOverlayDiagram({
        mermaid,
        prompt: normalized,
        updatedAt: new Date().toISOString(),
        position: 'top-left',
        minimized: false,
      })
    } catch (error) {
      setDiagramOutput(error instanceof Error ? error.message : String(error))
    } finally {
      setDiagramBusy(false)
    }
  }

  const handleRemoveOverlay = () => {
    setOverlayDiagram(null)
    const socket = sonicSocketRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'overlay_clear' }))
    }
  }

  const handleOverlayControl = (action: 'move' | 'minimize' | 'expand' | 'export', position?: SessionOverlayDiagram['position']) => {
    const socket = sonicSocketRef.current
    if (action === 'move' && position) {
      setOverlayDiagram((prev) => (prev ? { ...prev, position } : prev))
    }
    if (action === 'minimize') {
      setOverlayDiagram((prev) => (prev ? { ...prev, minimized: true } : prev))
    }
    if (action === 'expand') {
      setOverlayDiagram((prev) => (prev ? { ...prev, minimized: false } : prev))
    }
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: 'overlay_control',
          action,
          position,
        }),
      )
    }
  }

  const formatTime = (secs: number) => {
    const m = String(Math.floor(secs / 60)).padStart(2, '0')
    const s = String(secs % 60).padStart(2, '0')
    return `${m}:${s}`
  }

  return (
    <div className="call-root">
      <header className="call-topbar">
        <div className="topbar-brand">
          <span className="topbar-logo">[N]</span>
          <span className="topbar-title">Nova Session</span>
        </div>

        <div className="topbar-center">
          <span className="topbar-session">#{callId}</span>
        </div>

        <div className="topbar-right">
          <ActiveModeBadge isSharing={isSharing} />
          <span className="topbar-timer">{formatTime(elapsed)}</span>
          <AgentStatusBadge hasAgent={remoteParticipants.length > 0} />
        </div>
      </header>

      <div className="call-body">
        <main className="call-stage">
          {localParticipant && (
            <div className="tile tile--local">
              <ParticipantView participant={localParticipant} className="participant-view" />
              <div className="tile-label">You</div>
            </div>
          )}

          {remoteParticipants.map((participant) => (
            <div key={participant.sessionId} className="tile tile--remote">
              <ParticipantView participant={participant} className="participant-view" />
              <div className="tile-label">{participant.name ?? participant.userId ?? 'Remote'}</div>
            </div>
          ))}

          {remoteParticipants.length === 0 && (
            <div className="tile tile--placeholder">
              <div className="agent-placeholder">
                <span className="agent-placeholder-icon">[]</span>
                <p className="agent-placeholder-text">No remote participant connected to this call.</p>
                <p className="agent-placeholder-hint">
                  Nova pipeline is active through backend session <code>{sessionId}</code>.
                </p>
              </div>
            </div>
          )}
        </main>

        <aside className="nova-panel">
          <h3 className="panel-title">Nova Sync</h3>

          <div className="panel-metrics">
            <span>Frames: {framesProcessed}</span>
            <span>Context hits: {contextHits}</span>
          </div>
          <p className="panel-status">{ingestStatus}</p>

          <label className="panel-label">
            Live analysis prompt (optional)
            <input
              className="panel-input"
              value={analysisPrompt}
              onChange={(event) => setAnalysisPrompt(event.target.value)}
              placeholder="Use while screen sharing for live guidance"
            />
          </label>

          <label className="panel-label">
            Latest OCR
            <textarea className="panel-textarea" value={lastOcr} readOnly rows={4} />
          </label>

          <div className="panel-actions">
            <button className="panel-btn" onClick={() => void (sonicConnected ? stopSonic() : startSonic())}>
              {sonicConnected ? 'Stop Sonic Voice' : 'Start Sonic Voice'}
            </button>
          </div>
          <p className="panel-status">Sonic: {sonicStatus}</p>

          <label className="panel-label">
            Sonic user transcript
            <textarea className="panel-textarea" value={sonicUserTranscript} readOnly rows={3} />
          </label>

          <label className="panel-label">
            Sonic assistant transcript
            <textarea className="panel-textarea" value={sonicAssistantTranscript} readOnly rows={4} />
          </label>

          <label className="panel-label">
            Ask Nova (text path)
            <input
              className="panel-input"
              value={manualQuestion}
              onChange={(event) => setManualQuestion(event.target.value)}
              placeholder="Where is auth flow in this screen?"
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void handleManualQuery()
                }
              }}
            />
          </label>
          <div className="panel-actions">
            <button className="panel-btn" disabled={queryBusy} onClick={() => void handleManualQuery()}>
              {queryBusy ? 'Asking...' : 'Send'}
            </button>
          </div>

          <label className="panel-label">
            Text path user transcript
            <textarea className="panel-textarea" value={userTranscript} readOnly rows={2} />
          </label>

          <label className="panel-label">
            Text path assistant response
            <textarea className="panel-textarea" value={assistantReply} readOnly rows={4} />
          </label>

          <label className="panel-label">
            Diagram prompt
            <input
              className="panel-input"
              value={diagramPrompt}
              onChange={(event) => setDiagramPrompt(event.target.value)}
              placeholder="User -> Sonic -> Lite -> Aurora -> Overlay"
            />
          </label>
          <button className="panel-btn panel-btn-wide" disabled={diagramBusy} onClick={() => void handleGenerateDiagram()}>
            {diagramBusy ? 'Generating...' : 'Generate Mermaid'}
          </button>

          <label className="panel-label">
            Mermaid output
            <textarea className="panel-textarea" value={diagramOutput} readOnly rows={6} />
          </label>
        </aside>
      </div>

      {overlayDiagram ? (
        <DiagramOverlay
          mermaidText={overlayDiagram.mermaid}
          prompt={overlayDiagram.prompt}
          updatedAt={overlayDiagram.updatedAt}
          position={overlayDiagram.position}
          minimized={overlayDiagram.minimized}
          onClose={handleRemoveOverlay}
          onControl={handleOverlayControl}
        />
      ) : null}

      <Controls onLeave={onLeave} />
    </div>
  )
}

function ActiveModeBadge({ isSharing }: { isSharing: boolean }) {
  return (
    <span className={`mode-badge ${isSharing ? 'mode-badge--screen' : 'mode-badge--camera'}`}>
      {isSharing ? 'Screen share' : 'Camera'}
    </span>
  )
}

function AgentStatusBadge({ hasAgent }: { hasAgent: boolean }) {
  return (
    <span className={`agent-badge ${hasAgent ? 'agent-badge--connected' : 'agent-badge--waiting'}`}>
      <span className="agent-badge-dot" />
      {hasAgent ? 'Remote connected' : 'Backend sync active'}
    </span>
  )
}
