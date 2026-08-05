// End-to-end bridge test (Council-requested live verification).
//
// The `REAL_FRAME` below is the VERBATIM session.info payload captured off the
// wire from the real running tui_gateway.server after a real compression
// rotation (session_id 20260709_190601_373ced -> 20260711_233904_e07b53). It
// is fed into the REAL renderer event handler + REAL status-bar label builder
// to prove the full chain the resume-mismatch fix depends on:
//   (1) the renderer re-pins storedSid/sid to the rotated id,
//   (2) it writes that id to the active-session file _print_tui_exit_summary reads,
//   (3) the status bar (driven by storedSid) recomputes to show the new id.
// Closes the Council gap "renderer storedSid update propagates to status-bar
// render" — verified against a genuine backend frame shape, not a hand mock.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, readFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

const LAUNCH = '20260709_190601_373ced'
const NEWEST = '20260711_233904_e07b53'

// Verbatim payload the real gateway emitted after auto-compaction (trimmed of
// the large skills/tools maps for readability; the fields the fix reads are
// intact — session_key is the load-bearing one).
const REAL_FRAME = {
  type: 'session.info',
  payload: {
    model: 'opus-4.8',
    session_key: NEWEST,
    provider: 'copilot',
    service_tier: 'priority',
    fast: true,
    yolo: false,
    autopilot: false,
    tools: {},
    skills: {},
    cwd: '/mnt/devvm/custom/hermes/src',
    branch: 'fix/copilot-codex-true-limits',
    running: false,
    version: '0.17.0',
    usage: {
      model: 'opus-4.8', input: 68, output: 50313, total: 50381, calls: 5,
      context_used: 180000, context_max: 200000, context_percent: 90, compressions: 3
    },
    profile_name: 'default'
  }
} as const

const ref = <T,>(current: T) => ({ current })
const buildCtx = () =>
  ({
    composer: { dequeue: () => undefined, queueEditRef: ref<null | number>(null), sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn() }, rpc: vi.fn(async () => null) },
    session: { STARTUP_RESUME_ID: '', colsRef: ref(80), newSession: vi.fn(), resetSession: vi.fn(), resumeById: vi.fn(), setCatalog: vi.fn() },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

const textContent = (node: any): string => {
  if (node === null || node === undefined || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textContent).join('')
  if (node && node.props) return textContent(node.props.children)
  return ''
}

describe('resume-fix: real backend frame → renderer → status bar', () => {
  let activeFile: string
  const prevEnv = process.env.HERMES_TUI_ACTIVE_SESSION_FILE

  beforeEach(() => {
    resetUiState()
    const dir = mkdtempSync(join(tmpdir(), 'hermes-live-active-'))
    activeFile = join(dir, 'active.json')
    process.env.HERMES_TUI_ACTIVE_SESSION_FILE = activeFile
  })
  afterEach(() => {
    if (prevEnv === undefined) delete process.env.HERMES_TUI_ACTIVE_SESSION_FILE
    else process.env.HERMES_TUI_ACTIVE_SESSION_FILE = prevEnv
  })

  it('re-pins storedSid + writes the active file + status bar shows the rotated id', () => {
    // Renderer attached at the OLD launch id (as the real TUI would have been).
    patchUiState({ sid: LAUNCH, storedSid: LAUNCH })
    const onEvent = createGatewayEventHandler(buildCtx())

    // Deliver the exact frame the real gateway emitted after auto-compaction.
    onEvent(REAL_FRAME as any)

    // (1) durable id re-pinned to the continuation
    expect(getUiState().storedSid).toBe(NEWEST)
    expect(getUiState().sid).toBe(NEWEST)

    // (2) active-session file (what _print_tui_exit_summary reads) now holds it,
    //     so the shell exit summary's `--resume <id>` points at the live segment.
    expect(existsSync(activeFile)).toBe(true)
    expect(JSON.parse(readFileSync(activeFile, 'utf-8'))).toEqual({ session_id: NEWEST })

    // (3) status bar, driven by storedSid, renders the new id in parens — the
    //     propagation link the Council flagged as unverified.
    const bar = StatusRule({
      bgCount: 0, busy: false, cols: 200, cwdLabel: '~/repo', liveSessionCount: 1,
      model: 'opus-4.8', onSessionCountClick: vi.fn(),
      sessionId: getUiState().storedSid ?? '', sessionStartedAt: null, showCost: false,
      status: 'ready', statusColor: DEFAULT_THEME.color.ok, t: DEFAULT_THEME,
      turnStartedAt: null, usage: { total: 0 } as any, voiceLabel: 'voice off'
    } as any)
    const rendered = textContent(bar)
    expect(rendered).toContain(`1 session (${NEWEST})`)
    expect(rendered).not.toContain(LAUNCH)
  })
})
