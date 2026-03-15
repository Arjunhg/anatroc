/**
 * Controls — bottom action bar for the active call.
 *
 * Camera and screen share are mutually exclusive:
 *   • Clicking “Share Screen” → stops camera, starts screen capture
 *   • Clicking “Stop Sharing” → stops screen share, resumes camera
 * This is the single source of truth for mode switching — no lobby
 * config needed.
 *
 * Microphone and Leave are independent.
 */
import {
  useCallStateHooks,
} from '@stream-io/video-react-sdk'
import './Controls.css'

interface Props {
  onLeave: () => Promise<void>
}

export default function Controls({ onLeave }: Props) {
  const {
    useCameraState,
    useMicrophoneState,
    useScreenShareState,
  } = useCallStateHooks()

  const { camera,      isMute: camMuted }  = useCameraState()
  const { microphone,  isMute: micMuted }  = useMicrophoneState()
  const { screenShare, status: ssStatus }  = useScreenShareState()

  const isSharing = ssStatus === 'enabled'

  const toggleMic = () => { void microphone.toggle() }

  // ── Mutually exclusive camera ↔ screen share ────────────────────────────
  const startScreenShare = async () => {
    await camera.disable()       // turn off webcam first
    await screenShare.enable()   // then open the OS screen picker
  }

  const stopScreenShare = async () => {
    await screenShare.disable()  // close screen share
    await camera.enable()        // resume webcam
  }

  const toggleShare = isSharing ? stopScreenShare : startScreenShare

  // Camera toggle only works when not sharing screen
  const toggleCamera = async () => {
    if (isSharing) {
      // Don’t allow camera on while screen sharing — use Stop Sharing instead
      return
    }
    await camera.toggle()
  }

  return (
    <footer className="controls-bar">
      {/* Camera — disabled while screen sharing */}
      <ControlButton
        icon='📷'
        label={isSharing ? 'Camera off' : camMuted ? 'Camera off' : 'Camera on'}
        active={!camMuted && !isSharing}
        disabled={isSharing}
        onClick={() => void toggleCamera()}
        title={isSharing ? 'Stop sharing to use camera' : camMuted ? 'Turn camera on' : 'Turn camera off'}
      />

      {/* Microphone */}
      <ControlButton
        icon={micMuted ? '🔇' : '🎤'}
        label={micMuted ? 'Mic off' : 'Mic on'}
        active={!micMuted}
        onClick={() => void toggleMic()}
        title={micMuted ? 'Unmute microphone' : 'Mute microphone'}
      />

      {/* Screen share — mutually exclusive with camera */}
      <ControlButton
        icon={isSharing ? '⏹️' : '🖥️'}
        label={isSharing ? 'Stop sharing' : 'Share screen'}
        active={isSharing}
        highlight={isSharing}
        onClick={() => void toggleShare()}
        title={isSharing ? 'Stop screen sharing → resumes camera' : 'Share your screen → pauses camera'}
      />

      {/* Spacer */}
      <div className="controls-spacer" />

      {/* Leave */}
      <button
        className="btn-leave"
        onClick={() => void onLeave()}
        title="End session"
      >
        End session
      </button>
    </footer>
  )
}

// ── Generic control button ──────────────────────────────────────────────────

interface CtrlBtnProps {
  icon: string
  label: string
  active: boolean
  disabled?: boolean
  highlight?: boolean
  onClick: () => void
  title?: string
}

function ControlButton({ icon, label, active, disabled, highlight, onClick, title }: CtrlBtnProps) {
  return (
    <button
      className={`ctrl-btn ${active ? 'ctrl-btn--on' : 'ctrl-btn--off'} ${highlight ? 'ctrl-btn--highlight' : ''} ${disabled ? 'ctrl-btn--disabled' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      <span className="ctrl-btn-icon">{icon}</span>
      <span className="ctrl-btn-label">{label}</span>
    </button>
  )
}
