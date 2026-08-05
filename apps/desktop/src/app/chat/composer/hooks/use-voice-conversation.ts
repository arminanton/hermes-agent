import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import { notify, notifyError } from '@/store/notifications'
import { $syncTextAudio, setVoiceHold } from '@/store/voice-prefs'

import { useMicRecorder } from './use-mic-recorder'

export type ConversationStatus = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking'

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface VoiceConversationOptions {
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  onSubmit: (text: string) => Promise<void> | void
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
}

export function useVoiceConversation({
  busy,
  enabled,
  onFatalError,
  onSubmit,
  onTranscribeAudio,
  pendingResponse,
  consumePendingResponse
}: VoiceConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const { handle, level } = useMicRecorder(voiceCopy)
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const turnTimeoutRef = useRef<number | null>(null)
  const pendingStartRef = useRef(false)
  const turnClosingRef = useRef(false)
  const awaitingSpokenResponseRef = useRef(false)
  const responseIdRef = useRef<string | null>(null)
  const spokenSourceLengthRef = useRef(0)
  const speechBufferRef = useRef('')
  const enabledRef = useRef(enabled)
  const mutedRef = useRef(muted)
  const busyRef = useRef(busy)
  const statusRef = useRef<ConversationStatus>('idle')
  const wasEnabledRef = useRef(enabled)

  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    mutedRef.current = muted
  }, [muted])

  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  useEffect(() => {
    statusRef.current = status
  }, [status])

  const clearTurnTimeout = () => {
    if (turnTimeoutRef.current) {
      window.clearTimeout(turnTimeoutRef.current)
      turnTimeoutRef.current = null
    }
  }

  const resetSpeechBuffer = () => {
    responseIdRef.current = null
    spokenSourceLengthRef.current = 0
    speechBufferRef.current = ''
  }

  const handleTurn = useCallback(
    async (forceTranscribe = false) => {
      if (turnClosingRef.current) {
        return
      }

      turnClosingRef.current = true
      clearTurnTimeout()
      setStatus('transcribing')

      try {
        const result = await handle.stop()

        if (!result || (!result.heardSpeech && !forceTranscribe) || !onTranscribeAudio) {
          if (enabledRef.current && !mutedRef.current && !busyRef.current && statusRef.current !== 'speaking') {
            pendingStartRef.current = true
          }

          setStatus('idle')

          return
        }

        try {
          // Acknowledge capture so the user isn't guessing — a long utterance
          // + cold STT can take a while; the pill shows "transcribing" but a
          // toast makes it explicit that their audio landed and is processing.
          notify({ kind: 'info', title: 'Audio received — transcribing…', message: '' })
          const transcript = (await onTranscribeAudio(result.audio)).trim()

          if (!transcript) {
            if (enabledRef.current) {
              pendingStartRef.current = true
            }

            setStatus('idle')

            return
          }

          awaitingSpokenResponseRef.current = true
          resetSpeechBuffer()
          await onSubmit(transcript)
          setStatus('thinking')
        } catch (error) {
          notifyError(error, voiceCopy.transcriptionFailed)

          if (enabledRef.current && !mutedRef.current && !busyRef.current) {
            pendingStartRef.current = true
          }

          setStatus('idle')
        }
      } finally {
        turnClosingRef.current = false
      }
    },
    [handle, onSubmit, onTranscribeAudio, voiceCopy.transcriptionFailed]
  )

  const startListening = useCallback(async () => {
    pendingStartRef.current = false

    if (!enabledRef.current || mutedRef.current || busyRef.current) {
      return
    }

    if (statusRef.current !== 'idle') {
      return
    }

    try {
      // VAD tuning mirrors `tools.voice_mode` defaults so the browser loop matches the CLI.
      await handle.start({
        silenceLevel: 0.075,
        silenceMs: 5_000,
        idleSilenceMs: 25_000,
        onError: error => {
          notifyError(error, voiceCopy.microphoneFailed)
          pendingStartRef.current = false
          onFatalError?.()
        },
        onSilence: () => void handleTurn()
      })
      setStatus('listening')
      turnTimeoutRef.current = window.setTimeout(() => void handleTurn(), 300_000)
    } catch (error) {
      notifyError(error, voiceCopy.couldNotStartSession)
      pendingStartRef.current = false
      setStatus('idle')
      onFatalError?.()
    }
  }, [handle, handleTurn, onFatalError, voiceCopy.couldNotStartSession, voiceCopy.microphoneFailed])

  const speak = useCallback(async (text: string, holdId?: string) => {
    setStatus('speaking')

    // Safety net: never let a held reply stay hidden. If audio never starts
    // (TTS stalls/fails), reveal it anyway after a few seconds so the user
    // always sees text. onStart clears it sooner on the happy path.
    let revealTimer: number | null = null
    if (holdId) {
      revealTimer = window.setTimeout(() => setVoiceHold(null), 6_000)
    }

    try {
      await playSpeechText(text, {
        source: 'voice-conversation',
        onStart: () => setVoiceHold(null)
      })
    } catch (error) {
      notifyError(error, voiceCopy.playbackFailed)
    } finally {
      if (revealTimer) {
        window.clearTimeout(revealTimer)
      }
      // Whatever happened, never leave a reply hidden.
      setVoiceHold(null)
      if (enabledRef.current) {
        pendingStartRef.current = true
        setStatus('idle')
      } else {
        setStatus('idle')
      }
    }
  }, [voiceCopy.playbackFailed])

  const start = useCallback(async () => {
    if (!onTranscribeAudio) {
      notify({
        kind: 'warning',
        title: voiceCopy.unavailable,
        message: voiceCopy.configureSpeechToText
      })
      onFatalError?.()

      return
    }

    setMuted(false)
    awaitingSpokenResponseRef.current = false
    resetSpeechBuffer()
    consumePendingResponse()
    pendingStartRef.current = true
    await startListening()
  }, [consumePendingResponse, onFatalError, onTranscribeAudio, startListening, voiceCopy.configureSpeechToText, voiceCopy.unavailable])

  const end = useCallback(async () => {
    pendingStartRef.current = false
    clearTurnTimeout()
    stopVoicePlayback()
    setVoiceHold(null)
    handle.cancel()
    turnClosingRef.current = false
    awaitingSpokenResponseRef.current = false
    resetSpeechBuffer()
    consumePendingResponse()
    setMuted(false)
    setStatus('idle')
  }, [consumePendingResponse, handle])

  const stopTurn = useCallback(() => {
    if (statusRef.current === 'listening') {
      void handleTurn(true)
    }
  }, [handleTurn])

  // Barge-in: cut off the assistant mid-speech so the user can respond without
  // hearing the whole reply (the terminal's newest-wins behaviour). Stops the
  // current playback and immediately re-arms the mic, so the user isn't locked
  // out until the reply finishes (William's "won't let me record until idle").
  const interruptSpeech = useCallback(() => {
    if (statusRef.current !== 'speaking') {
      return
    }

    stopVoicePlayback()
    setVoiceHold(null)

    if (enabledRef.current && !mutedRef.current) {
      pendingStartRef.current = true
    }

    setStatus('idle')
  }, [])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value

      if (next) {
        clearTurnTimeout()
        handle.cancel()
        setStatus('idle')
      } else if (enabledRef.current && !busyRef.current && statusRef.current === 'idle') {
        pendingStartRef.current = true
      }

      return next
    })
  }, [handle])

  useEffect(() => {
    if (!enabled) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || event.repeat || event.metaKey || event.ctrlKey || event.altKey) {
        return
      }

      // While the assistant is speaking, Space barges in: stop playback and
      // re-arm the mic so the user can respond immediately, no need to wait.
      if (statusRef.current === 'speaking') {
        event.preventDefault()
        interruptSpeech()

        return
      }

      if (statusRef.current !== 'listening') {
        return
      }

      event.preventDefault()
      stopTurn()
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [enabled, interruptSpeech, stopTurn])

  // Drive the loop: after a voice-submitted turn, speak the WHOLE reply once it
  // is complete (matches the terminal's smooth one-shot TTS, which William much
  // prefers). The old design chopped the reply into per-sentence chunks and
  // `await speak(chunk)` serialized them — each sentence waited for the prior
  // one's full TTS round-trip + playback, ~20-30s/sentence. Speaking once at
  // the end removes that. Otherwise start listening when idle between turns.
  useEffect(() => {
    if (!enabled || muted) {
      return
    }

    if (awaitingSpokenResponseRef.current && status !== 'speaking') {
      const response = pendingResponse()

      if (response) {
        if (response.id !== responseIdRef.current) {
          resetSpeechBuffer()
          responseIdRef.current = response.id
        }

        // Sync text+audio: hold the in-flight reply out of the transcript as
        // soon as it appears, so its text doesn't render seconds before audio.
        // Released when playback begins (speak → onStart) or on completion below.
        if ($syncTextAudio.get()) {
          setVoiceHold(response.id)
        }

        // Only speak once the reply is COMPLETE (not pending, not busy). The
        // full text is read in a single TTS call, so playback is smooth and
        // fast — no inter-sentence gaps. While still streaming, just wait.
        if (!response.pending && !busy) {
          const fullReply = response.text.trim()
          const holdId = $syncTextAudio.get() ? response.id : undefined
          awaitingSpokenResponseRef.current = false
          consumePendingResponse()
          resetSpeechBuffer()

          if (fullReply) {
            void speak(fullReply, holdId)

            return
          }

          setVoiceHold(null)
          pendingStartRef.current = true
          setStatus('idle')

          return
        }
      }

      if (!busy && status === 'thinking') {
        awaitingSpokenResponseRef.current = false
        resetSpeechBuffer()
        pendingStartRef.current = true
        setStatus('idle')

        return
      }
    }

    if (busy || status !== 'idle') {
      return
    }

    if (pendingStartRef.current) {
      void startListening()
    }
  }, [busy, consumePendingResponse, enabled, muted, pendingResponse, speak, startListening, status])

  useEffect(() => {
    if (enabled && !wasEnabledRef.current) {
      void start()
    }

    if (!enabled && wasEnabledRef.current) {
      void end()
    }

    wasEnabledRef.current = enabled
  }, [enabled, end, start])

  return { end, interruptSpeech, level, muted, start, status, stopTurn, toggleMute }
}
