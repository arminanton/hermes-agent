import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

// Stub useInput so nothing tries to enter raw mode under renderSync.
vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal<typeof import('@hermes/ink')>()

  return { ...mod, useInput: () => {} }
})

import { StatusRule } from '../components/appChrome.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

// Exact-narrow-width screen-buffer regression for the YOLO safety badge.
//
// lucapohl-angel's narrow-width QA (an exact 28x44 PTY) found the reminder
// clipped to `⚠ YO` while idle and vanished entirely while busy, because the
// badge trailed model/context and its width was never reserved. This suite
// paints the REAL terminal frame (Yoga layout + truncation applied) at 44
// columns and asserts the FULL `⚠ YOLO` survives in BOTH idle and busy — the
// element-tree tests in appChromeStatusRule.test.tsx can't catch a mid-segment
// clip because clipping only happens when the frame is actually rendered to a
// fixed-width buffer.
//
// 44 columns is the clip-triggering width they reported. The cwd/branch label
// is intentionally long so it competes for the row and would, before the fix,
// shove the badge off-screen.

const NARROW_COLS = 44

// A realistic long cwd/branch so the right-hand label genuinely contends for
// columns at 44 wide (mirrors the existing narrow-terminal test's fixture).
const LONG_CWD = '~/src/hermes-agent/apps/desktop (bb/tui-statusbar-responsive)'

const baseProps = {
  bgCount: 0,
  busy: false,
  cols: NARROW_COLS,
  cwdLabel: LONG_CWD,
  liveSessionCount: 3,
  model: 'opus-4.8',
  sessionStartedAt: Date.now() - 60_000,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
  // Inactive voice renders zero-width now, so it must NOT appear or consume
  // columns. Passing '' models the app's own zero-width inactive-voice output.
  voiceLabel: ''
}

// Render the StatusRule to a real fixed-width terminal buffer and return the
// ANSI-stripped frame text, exactly as it would paint in a 44-column PTY.
function renderFrame(props: Parameters<typeof StatusRule>[0]): string {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  let output = ''

  Object.assign(stdout, { columns: NARROW_COLS, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(React.createElement(StatusRule, props), {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

describe('StatusRule YOLO badge — exact narrow-width (44 col) screen buffer', () => {
  it('paints the FULL "⚠ YOLO" while idle (never clips to "⚠ YO")', () => {
    const frame = renderFrame({
      ...baseProps,
      busy: false,
      yolo: true,
      yoloSource: 'session'
    })

    // The whole warning must be present in the painted buffer …
    expect(frame).toContain('⚠ YOLO')
    // … and it must not have been truncated mid-word to the dangerous stub.
    expect(frame).not.toMatch(/⚠ YO(?!LO)/)
  })

  it('paints the FULL "⚠ YOLO" while busy (never omits the warning)', () => {
    const frame = renderFrame({
      ...baseProps,
      busy: true,
      turnStartedAt: Date.now(),
      yolo: true,
      yoloSource: 'session'
    })

    // Under load the FaceTicker occupies the status slot, yet the safety badge
    // must still render in full — a bypass warning that vanishes while the
    // agent works is exactly when it matters most.
    expect(frame).toContain('⚠ YOLO')
    expect(frame).not.toMatch(/⚠ YO(?!LO)/)
  })

  it('paints the FULL "⚠ APPROVALS OFF" config-bypass label while idle', () => {
    const frame = renderFrame({
      ...baseProps,
      busy: false,
      yolo: true,
      yoloSource: 'config'
    })

    expect(frame).toContain('⚠ APPROVALS OFF')
  })

  it('keeps the badge ahead of model/context so the tail yields first', () => {
    const frame = renderFrame({
      ...baseProps,
      busy: false,
      yolo: true,
      yoloSource: 'session'
    })

    // The badge is pinned; the model may or may not survive at 44 cols, but the
    // safety warning always must. Assert the badge is positioned before the
    // context read-out in the painted row (higher render priority).
    const badgeAt = frame.indexOf('⚠ YOLO')
    const ctxAt = frame.indexOf(' tok')

    expect(badgeAt).toBeGreaterThanOrEqual(0)

    if (ctxAt >= 0) {
      expect(badgeAt).toBeLessThan(ctxAt)
    }
  })

  it('renders nothing for an inactive voice (zero-width) so it cannot crowd the badge', () => {
    const frame = renderFrame({
      ...baseProps,
      busy: false,
      voiceLabel: '',
      yolo: true,
      yoloSource: 'session'
    })

    // Inactive voice contributes no columns at all …
    expect(frame).not.toContain('voice off')
    // … while the badge it could have crowded out survives in full.
    expect(frame).toContain('⚠ YOLO')
  })
})
