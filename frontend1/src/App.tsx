import { useCallback, useEffect, useRef, useState } from 'react'
import { Call, StreamCall, StreamVideo, StreamVideoClient } from '@stream-io/video-react-sdk'
import '@stream-io/video-react-sdk/dist/css/styles.css'

import { createClient, DEFAULT_CALL_ID, DEFAULT_CALL_TYPE } from './stream'
import { startNovaSession, stopNovaSession } from './novaApi'
import LobbyScreen, { type LobbyValues } from './components/LobbyScreen'
import CallScreen from './components/CallScreen'
import './App.css'

type AppState = 'lobby' | 'joining' | 'incall'

export default function App() {
  const [appState, setAppState] = useState<AppState>('lobby')
  const [error, setError] = useState<string | null>(null)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  const clientRef = useRef<StreamVideoClient | null>(null)
  const callRef = useRef<Call | null>(null)
  const sessionIdRef = useRef<string | null>(null)

  useEffect(() => {
    sessionIdRef.current = activeSessionId
  }, [activeSessionId])

  const teardown = useCallback(async () => {
    const sessionId = sessionIdRef.current
    if (sessionId) {
      try {
        await stopNovaSession(sessionId)
      } catch (error) {
        console.warn('Failed to stop Nova session:', error)
      }
      setActiveSessionId(null)
      sessionIdRef.current = null
    }

    try {
      await callRef.current?.leave()
    } catch {
      // ignore
    }
    try {
      await clientRef.current?.disconnectUser()
    } catch {
      // ignore
    }

    callRef.current = null
    clientRef.current = null
  }, [])

  useEffect(() => {
    return () => {
      void teardown()
    }
  }, [teardown])

  const handleJoin = useCallback(
    async (values: LobbyValues) => {
      setError(null)
      setAppState('joining')

      try {
        const { client } = await createClient(values.userId, values.displayName)
        clientRef.current = client

        const callId = values.callId.trim() || DEFAULT_CALL_ID
        const callType = DEFAULT_CALL_TYPE
        const call = client.call(callType, callId)
        await call.getOrCreate()
        callRef.current = call

        await call.join({ create: true })
        await call.camera.enable()
        await call.microphone.enable()

        await startNovaSession(callId)
        setActiveSessionId(callId)
        sessionIdRef.current = callId

        setAppState('incall')
      } catch (joinError) {
        console.error('Failed to join call:', joinError)
        setError(joinError instanceof Error ? joinError.message : String(joinError))
        await teardown()
        setAppState('lobby')
      }
    },
    [teardown],
  )

  const handleLeave = useCallback(async () => {
    await teardown()
    setAppState('lobby')
  }, [teardown])

  if (appState === 'lobby' || appState === 'joining') {
    return (
      <div className="app-root">
        <LobbyScreen loading={appState === 'joining'} error={error} onJoin={handleJoin} />
      </div>
    )
  }

  if (!clientRef.current || !callRef.current || !activeSessionId) {
    return (
      <div className="app-root">
        <LobbyScreen
          loading={false}
          error="Session state became invalid. Please join again."
          onJoin={handleJoin}
        />
      </div>
    )
  }

  return (
    <div className="app-root">
      <StreamVideo client={clientRef.current}>
        <StreamCall call={callRef.current}>
          <CallScreen sessionId={activeSessionId} onLeave={handleLeave} />
        </StreamCall>
      </StreamVideo>
    </div>
  )
}
