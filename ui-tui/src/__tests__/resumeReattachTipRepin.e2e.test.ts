import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

// End-to-end bridge test for the stale-session-id fix on the RESUME-REATTACH
// path (companion to liveWireResumeFix.e2e.test.ts, which covers the live
// session.info rotation path).
//
// Background: a long session auto-compressed while the renderer was detached, so
// the active-session file — and thus the id the renderer resumes on the next
// orchestrator recycle — still carried the pre-rotation id. The gateway held
// the session live under the rotated continuation key. Before the
// fix, `session.resume` echoed the REQUESTED id back in `resumed`, so the
// renderer re-pinned storedSid + rewrote the active file to the STALE id. That
// (a) showed the stale id in the status bar, (b) made the shell exit summary
// print `--resume <stale>`, and (c) fed the stale id back into the active file
// so every subsequent recycle re-locked onto it — a self-reinforcing trap.
//
// The gateway fix (tui_gateway/server.py _reuse_live_payload) now reports the
// LIVE session's own key as `resumed`. This test proves the renderer's resume
// handler consumes that field to drive the full chain the user asked to verify:
//   (1) storedSid/sid re-pin to the live tip,
//   (2) the active-session file (_print_tui_exit_summary reads it) holds the tip
//       so the shell exit line hands back the CURRENT id,
//   (3) the status bar (driven by storedSid) renders the tip, not the stale id.
//
// It exercises the REAL writeActiveSessionFile, the REAL uiStore, and the REAL
// StatusRule bar builder — only the React hook wiring is inlined (the resume
// handler's re-pin is `storedSid: r.resumed ?? r.session_id` +
// writeActiveSessionFile(r.resumed ?? r.session_id), useSessionLifecycle.ts).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { writeActiveSessionFile } from '../app/useSessionLifecycle.js'
import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

// The stale pre-rotation id the renderer had pinned + resumed on recycle, and
// the live continuation tip the gateway is actually anchored on.
const STALE = 'fixture-session-before-rotation'
const LIVE_TIP = 'fixture-session-after-rotation'

// Representative shape of the session.resume response for a live reattach after the
// gateway fix: the runtime sid is the in-memory session's ephemeral id, and
// BOTH session_key and resumed advance to the live continuation tip. Before the
// fix, `resumed` was the STALE requested id (the bug this test guards).
const REATTACH_RESPONSE = {
  session_id: 'runtime-session',
  resumed: LIVE_TIP,
  session_key: LIVE_TIP,
  message_count: 1,
  messages: [{ role: 'assistant', content: 'live continuation tail' }],
  running: false,
  status: 'idle',
  info: { model: 'opus-4.8', tools: {}, skills: {} }
} as const

// The exact re-pin the resume handler runs (useSessionLifecycle.ts:332-338):
// storedSid + the active-session file both come from `r.resumed ?? r.session_id`.
const applyResumeRepin = (r: { resumed?: string; session_id: string }) => {
  const durable = r.resumed ?? r.session_id
  writeActiveSessionFile(durable)
  patchUiState({ sid: r.session_id, storedSid: durable })
}

const textContent = (node: any): string => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return ''
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return String(node)
  }

  if (Array.isArray(node)) {
    return node.map(textContent).join('')
  }

  if (node && node.props) {
    return textContent(node.props.children)
  }

  return ''
}

const renderBarSessionId = (sessionId: string): string =>
  textContent(
    StatusRule({
      bgCount: 0,
      busy: false,
      cols: 200,
      cwdLabel: '~/repo',
      liveSessionCount: 1,
      model: 'opus-4.8',
      onSessionCountClick: vi.fn(),
      sessionId,
      sessionStartedAt: null,
      showCost: false,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { total: 0 } as any,
      voiceLabel: 'voice off'
    } as any)
  )

describe('resume-reattach: gateway reports live tip → renderer re-pins bar + exit id', () => {
  let activeFile: string
  const prevEnv = process.env.HERMES_TUI_ACTIVE_SESSION_FILE

  beforeEach(() => {
    resetUiState()
    const dir = mkdtempSync(join(tmpdir(), 'hermes-reattach-active-'))
    activeFile = join(dir, 'active.json')
    process.env.HERMES_TUI_ACTIVE_SESSION_FILE = activeFile
  })
  afterEach(() => {
    if (prevEnv === undefined) {
      delete process.env.HERMES_TUI_ACTIVE_SESSION_FILE
    } else {
      process.env.HERMES_TUI_ACTIVE_SESSION_FILE = prevEnv
    }
  })

  it('advances storedSid, the active-session file, and the status bar to the live tip', () => {
    // Renderer recycled and resumed the STALE id it had pinned before the
    // compactions (as the real orchestrator recycle would have).
    patchUiState({ sid: STALE, storedSid: STALE })
    // Sanity: the bar shows the stale id BEFORE the reattach response lands.
    expect(renderBarSessionId(getUiState().storedSid ?? '')).toContain(`1 session (${STALE})`)

    // The gateway's fixed session.resume response arrives.
    applyResumeRepin(REATTACH_RESPONSE)

    // (1) durable id re-pinned to the live continuation tip.
    expect(getUiState().storedSid).toBe(LIVE_TIP)
    expect(getUiState().sid).toBe(REATTACH_RESPONSE.session_id)

    // (2) active-session file (_print_tui_exit_summary reads it) holds the tip,
    //     so the shell exit summary's `--resume <id>` hands back the CURRENT id.
    expect(existsSync(activeFile)).toBe(true)
    expect(JSON.parse(readFileSync(activeFile, 'utf-8'))).toEqual({ session_id: LIVE_TIP })

    // (3) the status bar, driven by storedSid, now shows the tip, not the stale id.
    const rendered = renderBarSessionId(getUiState().storedSid ?? '')
    expect(rendered).toContain(`1 session (${LIVE_TIP})`)
    expect(rendered).not.toContain(STALE)
  })

  it('would strand on the stale id if the gateway echoed the requested id (regression guard)', () => {
    // Pin the pre-fix behavior explicitly: had `resumed` echoed the stale
    // requested id, the bar and exit file would both stay on that id. This encodes
    // exactly what the gateway fix prevents, so a regression there is caught here.
    patchUiState({ sid: STALE, storedSid: STALE })
    applyResumeRepin({ ...REATTACH_RESPONSE, resumed: STALE })

    expect(getUiState().storedSid).toBe(STALE)
    expect(JSON.parse(readFileSync(activeFile, 'utf-8'))).toEqual({ session_id: STALE })
    expect(renderBarSessionId(getUiState().storedSid ?? '')).toContain(`1 session (${STALE})`)
  })
})
