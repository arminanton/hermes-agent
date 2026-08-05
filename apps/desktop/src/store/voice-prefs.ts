import { atom } from 'nanostores'

// Local, persisted voice-conversation preferences. Kept client-side (plain
// localStorage, mirroring user-themes.ts) so toggles take effect instantly
// without a gateway config round-trip.

const SPEAK_THINKING_KEY = 'hermes.voice.speakThinking'
const SYNC_TEXT_AUDIO_KEY = 'hermes.voice.syncTextAudio'

function readBool(key: string, fallback: boolean): boolean {
  try {
    const raw = window.localStorage.getItem(key)

    return raw === null ? fallback : raw === '1'
  } catch {
    return fallback
  }
}

function writeBool(key: string, value: boolean) {
  try {
    window.localStorage.setItem(key, value ? '1' : '0')
  } catch {
    // best-effort: private mode / quota — toggle still works for the session
  }
}

// Opt-in: also speak the assistant's reasoning/thinking aloud (default OFF —
// thinking is long/raw and unpleasant spoken; only spoken when the user wants it).
export const $speakThinking = atom<boolean>(readBool(SPEAK_THINKING_KEY, false))

// Opt-in: in a voice conversation, hold the message text until its audio is ready
// so text + speech surface together (default ON — feels synced). Off = text shows
// immediately, audio follows.
export const $syncTextAudio = atom<boolean>(readBool(SYNC_TEXT_AUDIO_KEY, false))

export function setSpeakThinking(value: boolean) {
  $speakThinking.set(value)
  writeBool(SPEAK_THINKING_KEY, value)
}

export function setSyncTextAudio(value: boolean) {
  $syncTextAudio.set(value)
  writeBool(SYNC_TEXT_AUDIO_KEY, value)
}

// Sync-text-audio: while a completed voice reply's audio is being synthesized,
// hold its message id here so the transcript suppresses that reply until audio
// is ready, then clear it so text + speech surface together. Empty = nothing held.
export const $voiceHoldMessageId = atom<string | null>(null)

export function setVoiceHold(id: string | null) {
  $voiceHoldMessageId.set(id)
}

// Thinking blocks: when ON, reasoning disclosures render expanded by default and
// in a readable color. Default ON per William (he wants to read thinking as it
// appears, not hunt for a collapsed dim block). Tool calls stay collapsed regardless.
const EXPAND_THINKING_KEY = 'hermes.voice.expandThinking'
export const $expandThinking = atom<boolean>(readBool(EXPAND_THINKING_KEY, true))

export function setExpandThinking(value: boolean) {
  $expandThinking.set(value)
  writeBool(EXPAND_THINKING_KEY, value)
}
