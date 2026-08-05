import { renderSync } from '@hermes/ink'
import React from 'react'
import { PassThrough } from 'stream'
import { describe, expect, it } from 'vitest'

import { ToolTrail } from '../components/thinking.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

// Render ToolTrail through the real Ink reconciler. This is the guard that was
// MISSING when click-to-inspect first shipped: the tool row wrapped its content
// in a <Box> nested inside the <Text> that TreeTextRow already provides, which
// Ink rejects at reconcile time with:
//   Text string "Terminal(...)" must be rendered inside <Text> component
// The logic-only tests passed because they never mounted the component. Any
// future regression that puts a Box (or a raw string) where Ink wants a Text
// re-introduces that crash and fails HERE instead of in the user's terminal.
function renderToolTrail(props: Record<string, unknown>): string {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(React.createElement(ToolTrail, { t: DEFAULT_THEME, ...props }), {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

describe('ToolTrail rendering (Ink reconcile guard)', () => {
  it('renders a persisted tool trail row without throwing (regression: Box-in-Text crash)', () => {
    // A completed, truncated Terminal row — exactly the shape that sets
    // group.full and drove the click-to-inspect Box-in-Text crash.
    const trail = ['Terminal("cd /mnt/devvm/custom/gpt-native && echo ==== my local HEAD and it…") (1.2s) ✓']

    expect(() =>
      renderToolTrail({ detailsMode: 'expanded', trail })
    ).not.toThrow()

    const out = renderToolTrail({ detailsMode: 'expanded', trail })
    expect(out).toContain('Terminal')
    // The clickable affordance renders on inspectable rows.
    expect(out).toContain('/inspect')
  })

  it('keeps the inline row COMPACT even at a wide terminal (decoupled from the full popup content)', () => {
    // The whole command is retained in the stored trail line (so /inspect shows
    // it), but the inline row must stay short with a trailing … regardless of
    // terminal width — NOT rely on wrap-trim clamping. Render at a very wide
    // terminal so width can't be doing the truncation.
    const fullCmd =
      'cd /mnt/devvm/custom/gpt-native echo "==== verify remote HEAD via AUTHENTICATED ls-remote ====" TOK="$(gh auth token)" echo "my local HEAD: $(git rev-parse HEAD)" then mark checkpoint complete'
    const trail = [`Terminal("${fullCmd}") (1.2s) ✓`]

    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''
    Object.assign(stdout, { columns: 400, isTTY: false, rows: 40 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      React.createElement(ToolTrail, { detailsMode: 'expanded', t: DEFAULT_THEME, trail }),
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )
    instance.unmount()
    instance.cleanup()

    const row = stripAnsi(output)
      .split('\n')
      .find(l => l.includes('Terminal'))

    expect(row).toBeDefined()
    // Compacted: has the ellipsis and does NOT show the full-command tail,
    // even though the terminal is 400 cols wide.
    expect(row).toContain('…')
    expect(row).not.toContain('mark checkpoint complete')
  })

  it('renders a live in-flight tool row (full context) without throwing', () => {
    const tools = [
      {
        id: 't1',
        name: 'terminal',
        context: 'cd /very/long/path/that/is/truncated/in/the/inline/row/aaaa…',
        contextFull: 'cd /very/long/path/that/is/truncated/in/the/inline/row/aaaa && ls -la',
        startedAt: Date.now()
      }
    ]

    expect(() => renderToolTrail({ busy: true, detailsMode: 'expanded', tools })).not.toThrow()
  })

  it('renders a delegate-task row (no /inspect affordance, /agents hint instead)', () => {
    const trail = ['Delegate Task("split every java file…") (30.0s) ✓']
    const out = renderToolTrail({ detailsMode: 'expanded', trail })

    expect(out).toContain('Delegate Task')
    expect(out).toContain('/agents')
  })
})
