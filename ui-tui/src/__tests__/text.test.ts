import { describe, expect, it } from 'vitest'

import {
  boundedLiveRenderText,
  buildToolTrailLine,
  buildVerboseToolTrailLine,
  compactToolCallDisplay,
  edgePreview,
  estimateRows,
  estimateTokensRough,
  fmtK,
  hasAnsi,
  isToolTrailResultLine,
  lastCotTrailIndex,
  parseToolTrailResultLine,
  pasteTokenLabel,
  sameToolTrailGroup,
  sanitizeAnsiForRender,
  splitToolDuration,
  stripAnsi,
  thinkingPreview,
  toolCallInspectTitle
} from '../lib/text.js'

describe('isToolTrailResultLine', () => {
  it('detects completion markers', () => {
    expect(isToolTrailResultLine('foo ✓')).toBe(true)
    expect(isToolTrailResultLine('foo ✗')).toBe(true)
    expect(isToolTrailResultLine('foo …')).toBe(true)
    expect(isToolTrailResultLine('drafting x…')).toBe(false)
  })
})

describe('buildToolTrailLine', () => {
  it('puts completion duration inline before the result marker', () => {
    const line = buildToolTrailLine('read_file', 'x', false, '', 0.94)

    expect(line).toBe('Read File("x") (0.9s) ✓')
    expect(parseToolTrailResultLine(line)).toEqual({ call: 'Read File("x") (0.9s)', detail: '', mark: '✓' })
    expect(splitToolDuration('Read File("x") (0.9s)')).toEqual({ label: 'Read File("x")', duration: ' (0.9s)' })
  })
})

describe('toolCallInspectTitle', () => {
  it('returns a SIMPLE "<Label> tool call" title, NOT the full command (regression)', () => {
    // The inspect popup body already shows the full command verbatim; the
    // title must not duplicate it. A gnarly Terminal call with embedded quotes
    // + a `: ` must still collapse to just "Terminal tool call".
    const call =
      'Terminal("cd /mnt/devvm/custom/gpt-native && echo "==== fix the stale origin/master tracking ref: point it at HEAD ===="") (1.2s)'

    expect(toolCallInspectTitle(call)).toBe('Terminal tool call')
  })

  it('strips a trailing completion marker and duration', () => {
    expect(toolCallInspectTitle('Read File("/x/y.ts") (0.9s)')).toBe('Read File tool call')
    expect(toolCallInspectTitle('Patch("/a/b.py") ✓')).toBe('Patch tool call')
  })

  it('handles a bare label with no parenthesized context', () => {
    expect(toolCallInspectTitle('Delegate Task')).toBe('Delegate Task tool call')
  })

  it('falls back to "Tool call" for an empty label', () => {
    expect(toolCallInspectTitle('')).toBe('Tool call')
  })
})

describe('parseToolTrailResultLine', () => {
  it('keeps a live pending tool in the tool group without claiming success', () => {
    expect(parseToolTrailResultLine('Fixture Operation("fixture-live") …')).toEqual({
      call: 'Fixture Operation("fixture-live")',
      detail: '',
      mark: '…'
    })
  })

  it('parses an interrupted cold-resume tool as a failed operation with a reason', () => {
    expect(parseToolTrailResultLine('Fixture Operation("fixture-cold") :: interrupted ✗')).toEqual({
      call: 'Fixture Operation("fixture-cold")',
      detail: 'interrupted',
      mark: '✗'
    })
  })

  it('does NOT split on a `: ` embedded inside a modern Label("…") command (regression)', () => {
    // A completed shell call whose command contains an embedded `: ` — e.g.
    // `echo "the REAL question: is my commit on the remote?"`. The legacy
    // `Label: context` fallback used to mistake that embedded colon for the
    // call/detail separator, splitting the row into a bogus compact header +
    // a spilled full-command "detail" block below it. The whole command must
    // stay in `call` with an EMPTY detail so the inline row compacts cleanly.
    const ctx =
      'cd /mnt/devvm/custom/gpt-native && echo "==== the REAL question: is my commit currently on the remote? ask GitHub directly ===="'
    const line = buildToolTrailLine('terminal', ctx, false, undefined, 1.2)
    const parsed = parseToolTrailResultLine(line)

    expect(parsed?.detail).toBe('')
    expect(parsed?.call).toContain('the REAL question: is my commit')
    // And the inline compaction of that call is short + ellipsized.
    const compact = compactToolCallDisplay(parsed!.call)
    expect(compact).toContain('…")')
    expect(compact).not.toContain('ask GitHub directly')
  })

  it('still parses genuinely-legacy `Label: context` trail lines (pre-paren format)', () => {
    // Backward-compat: old persisted lines had no `("` and used a single
    // `: ` separator. Those must still split correctly.
    expect(parseToolTrailResultLine('Read File: /some/legacy/path ✓')).toEqual({
      call: 'Read File',
      detail: '/some/legacy/path',
      mark: '✓'
    })
  })

  it('still parses a modern `::` verbose-detail line', () => {
    const line = buildToolTrailLine('read_file', 'x', false, 'note', 0.5)
    const parsed = parseToolTrailResultLine(line)

    expect(parsed?.call).toBe('Read File("x") (0.5s)')
    expect(parsed?.detail).toBe('note')
  })
})

describe('compactToolCallDisplay', () => {
  const longCtx =
    'cd /mnt/devvm/custom/gpt-native && echo ==== fix the stale origin tracking ref and verify HEAD matches upstream main exactly, then mark checkpoint complete ===='

  it('shortens a long call to the compact inline form with a trailing ellipsis', () => {
    const compact = compactToolCallDisplay(`Terminal("${longCtx}")`)

    expect(compact.length).toBeLessThan(longCtx.length)
    expect(compact.startsWith('Terminal("')).toBe(true)
    expect(compact).toContain('…")')
    // The tail the popup keeps must NOT leak into the compact inline form.
    expect(compact).not.toContain('mark checkpoint complete')
  })

  it('preserves a trailing duration suffix while compacting the context', () => {
    const compact = compactToolCallDisplay(`Terminal("${longCtx}") (1.2s)`)

    expect(compact).toMatch(/…"\) \(1\.2s\)$/)
  })

  it('is a no-op for already-short calls (with or without duration)', () => {
    expect(compactToolCallDisplay('Read File("x") (0.9s)')).toBe('Read File("x") (0.9s)')
    expect(compactToolCallDisplay('Terminal("pwd")')).toBe('Terminal("pwd")')
  })

  it('is a no-op for a bare label with no parenthesized context', () => {
    expect(compactToolCallDisplay('Delegate Task')).toBe('Delegate Task')
  })

  it('compacts commands with EMBEDDED double-quotes (regression: regex anchored on end)', () => {
    // Real shell commands contain embedded quotes and heredocs, so the call
    // string does NOT end in a clean `")`. The old end-anchored regex silently
    // failed on exactly these, leaving the full command in the inline row.
    const gnarly =
      'cd /mnt/devvm/custom/gpt-native echo "==== verify remote HEAD via AUTHENTICATED ls-remote (same auth path) ====" cat > /tmp/x.sh <<\'SH\' TOK="$(gh auth token)" echo "my local HEAD: $(git rev-parse HEAD)"'
    const call = `Terminal("${gnarly}") (1.2s)`

    const compact = compactToolCallDisplay(call)

    expect(compact.length).toBeLessThan(call.length)
    expect(compact).toContain('…')
    expect(compact).toMatch(/\) \(1\.2s\)$/)
    expect(compact).not.toContain('git rev-parse HEAD')
  })

  it('compacts a truncation-clipped call that never closes its quote', () => {
    // parseToolTrailResultLine can hand us a call that was clipped mid-command
    // (no trailing `")`). Compaction must still shorten it, not pass it through.
    const clipped = 'Terminal("cd /very/long/path && echo ' + 'x'.repeat(200)

    const compact = compactToolCallDisplay(clipped)

    expect(compact.length).toBeLessThan(clipped.length)
    expect(compact).toContain('…')
  })
})

describe('buildVerboseToolTrailLine', () => {
  it('preserves multiline args and result details', () => {
    const line = buildVerboseToolTrailLine(
      'terminal',
      'npm test',
      false,
      1.25,
      '{\n  "cmd": "npm test"\n}',
      'first line\nsecond :: line'
    )

    expect(line).toContain('Args:\n{')
    expect(line).toContain('Result:\nfirst line\nsecond :: line')
    expect(parseToolTrailResultLine(line)).toEqual({
      call: 'Terminal("npm test") (1.3s)',
      detail: 'Args:\n{\n  "cmd": "npm test"\n}\nResult:\nfirst line\nsecond :: line',
      mark: '✓'
    })
  })

  it('labels verbose failures as errors', () => {
    const line = buildVerboseToolTrailLine('terminal', 'npm test', true, 0.5, undefined, 'command failed')

    expect(line).toContain('Error:\ncommand failed')
    expect(line).not.toContain('Result:\ncommand failed')
    expect(parseToolTrailResultLine(line)).toEqual({
      call: 'Terminal("npm test") (0.5s)',
      detail: 'Error:\ncommand failed',
      mark: '✗'
    })
  })

  it('caps a large result to a small persisted preview (#34095)', () => {
    // A 40KB browser-snapshot-sized result must NOT be embedded whole — the
    // persisted, expanded-by-default trail block is what blew up the Ink
    // render tree and silently OOM-killed the TUI. The block stays small.
    const huge = 'A'.repeat(40_000)
    const line = buildVerboseToolTrailLine('browser_snapshot', 'https://x.example', false, 2, undefined, huge)

    expect(line).toContain('Result:\n')
    // Far below the old 16KB live-render budget; the whole line (call + label +
    // omitted marker + preview) must stay on the order of ~1KB, not ~40KB.
    expect(line.length).toBeLessThan(2_000)
    expect(line).toContain('omitted')
    expect(line.endsWith(' ✓')).toBe(true)
  })

  it('does not truncate a result that already fits the preview budget', () => {
    const small = 'ok: 3 files changed'
    const line = buildVerboseToolTrailLine('patch', 'index.html', false, 0.1, undefined, small)

    expect(line).toContain(`Result:\n${small}`)
    expect(line).not.toContain('omitted')
  })
})

describe('lastCotTrailIndex', () => {
  it('finds last non-result line', () => {
    expect(lastCotTrailIndex(['a ✓', 'thinking…'])).toBe(1)
    expect(lastCotTrailIndex(['only result ✓'])).toBe(-1)
  })
})

describe('sameToolTrailGroup', () => {
  it('matches bare check lines', () => {
    expect(sameToolTrailGroup('searching', 'searching ✓')).toBe(true)
    expect(sameToolTrailGroup('searching', 'searching ✗')).toBe(true)
  })

  it('matches contextual lines', () => {
    expect(sameToolTrailGroup('searching', 'searching: * ✓')).toBe(true)
    expect(sameToolTrailGroup('searching', 'searching: foo ✓')).toBe(true)
  })

  it('rejects other tools', () => {
    expect(sameToolTrailGroup('searching', 'reading ✓')).toBe(false)
    expect(sameToolTrailGroup('searching', 'searching extra ✓')).toBe(false)
  })
})

describe('fmtK', () => {
  it('keeps small numbers plain', () => {
    expect(fmtK(999)).toBe('999')
  })

  it('formats thousands as lowercase k', () => {
    expect(fmtK(1000)).toBe('1k')
    expect(fmtK(1500)).toBe('1.5k')
  })

  it('formats millions and billions with lowercase suffixes', () => {
    expect(fmtK(1_000_000)).toBe('1m')
    expect(fmtK(1_000_000_000)).toBe('1b')
  })
})

describe('estimateTokensRough', () => {
  it('uses 4 chars per token rounding up', () => {
    expect(estimateTokensRough('')).toBe(0)
    expect(estimateTokensRough('a')).toBe(1)
    expect(estimateTokensRough('abcd')).toBe(1)
    expect(estimateTokensRough('abcde')).toBe(2)
  })
})

describe('ANSI sanitizers', () => {
  const ESC = String.fromCharCode(27)
  const BEL = String.fromCharCode(7)

  it('strips CSI/OSC/control bytes from plain previews', () => {
    const sample = `A${ESC}[31mB${ESC}[39m${ESC}[2J${ESC}]0;title${BEL}C${ESC}[?25lD`

    expect(stripAnsi(sample)).toBe('ABCD')
  })

  it('strips incomplete CSI prefixes and carriage returns', () => {
    const sample = `A${ESC}[31mB${ESC}[12;${ESC}[CD\rE`

    expect(stripAnsi(sample)).toBe('ABDE')
  })

  it('keeps SGR color spans but removes cursor controls for Ansi rendering', () => {
    const sample = `A${ESC}[31mB${ESC}[39m${ESC}[2J${ESC}]0;title${BEL}${ESC}[?25lC`

    expect(sanitizeAnsiForRender(sample)).toBe(`A${ESC}[31mB${ESC}[39mC`)
  })

  it('keeps valid SGR while removing dangling CSI and carriage returns', () => {
    const sample = `A${ESC}[31mB${ESC}[12;${ESC}[39mC\rD`

    expect(sanitizeAnsiForRender(sample)).toBe(`A${ESC}[31mB${ESC}[39mCD`)
  })

  it('strips multi-byte non-CSI ESC sequences without leaving trailing bytes', () => {
    const sample = `A${ESC}(0B${ESC}%GC${ESC})0D`

    expect(stripAnsi(sample)).toBe('ABCD')
    expect(sanitizeAnsiForRender(sample)).toBe('ABCD')
  })

  it('detects non-CSI escape prefixes too', () => {
    expect(hasAnsi(`ok${ESC}Ppayload${ESC}\\`)).toBe(true)
  })
})

describe('thinkingPreview', () => {
  it('strips markdown markers but keeps heading paragraph breaks', () => {
    const raw =
      '**Considering user instructions**\nI need to answer.**Planning tool execution**\nI can run tools.**Determining weather search parameters**\nUse SF.'

    // Thinking is rendered as plain text (no markdown renderer), so the `**`
    // markers are stripped to clean prose; headings still get a blank line
    // before them so they stand out as their own line.
    expect(thinkingPreview(raw, 'full')).toBe(
      'Considering user instructions\nI need to answer.\n\nPlanning tool execution\nI can run tools.\n\nDetermining weather search parameters\nUse SF.'
    )
  })

  it('strips inline bold sizing labels to plain letters', () => {
    // Reasoning models emit **S**/**M**/**L** inline. Since thinking is not
    // markdown-rendered, the `**` would show literally, so we strip them.
    const raw =
      "requires regional deployment, so that lands at **M**. The service fronting all four is **L**. Device is the cleanest play. That's **S**, three to four sprints."

    expect(thinkingPreview(raw, 'full')).toBe(
      "requires regional deployment, so that lands at M. The service fronting all four is L. Device is the cleanest play. That's S, three to four sprints."
    )
  })

  it('strips adjacent bold markers on one line to plain text', () => {
    const raw = 'Sizing now. **S** is 3-4 sprints, **M** is 5-6, **L** is 7-8, larger decomposes.'

    expect(thinkingPreview(raw, 'full')).toBe('Sizing now. S is 3-4 sprints, M is 5-6, L is 7-8, larger decomposes.')
  })
})

describe('boundedLiveRenderText', () => {
  it('preserves short live text verbatim', () => {
    expect(boundedLiveRenderText('one\ntwo', { maxChars: 100, maxLines: 10 })).toBe('one\ntwo')
  })

  it('keeps the live tail by character budget', () => {
    const out = boundedLiveRenderText('abcdefghij', { maxChars: 4, maxLines: 10 })

    expect(out).toContain('ghij')
    expect(out).toContain('omitted')
    expect(out).not.toContain('abcdef')
  })

  it('keeps the live tail by line budget', () => {
    const out = boundedLiveRenderText(['a', 'b', 'c', 'd'].join('\n'), { maxChars: 100, maxLines: 2 })

    expect(out).toContain('c\nd')
    expect(out).toContain('omitted 2 lines')
    expect(out).not.toContain('a\nb')
  })
})

describe('edgePreview', () => {
  it('keeps both ends for long text', () => {
    expect(edgePreview('Vampire Bondage ropes slipped from her neck, still stained with blood', 8, 18)).toBe(
      'Vampire.. stained with blood'
    )
  })
})

describe('pasteTokenLabel', () => {
  it('builds readable long-paste labels with counts', () => {
    const label = pasteTokenLabel('Vampire Bondage ropes slipped from her neck, still stained with blood', 250)
    expect(label.startsWith('[[ ')).toBe(true)
    expect(label).toContain('[250 lines]')
    expect(label.endsWith(' ]]')).toBe(true)
  })
})

describe('estimateRows', () => {
  it('handles tilde code fences', () => {
    const md = ['~~~markdown', '# heading', '~~~'].join('\n')

    expect(estimateRows(md, 40)).toBeGreaterThanOrEqual(2)
  })

  it('handles checklist bullets as list rows', () => {
    const md = ['- [x] done', '- [ ] todo'].join('\n')

    expect(estimateRows(md, 40)).toBe(2)
  })

  it('keeps intraword underscores when sizing snake_case identifiers', () => {
    const w = 80
    const snake = 'look at test_case_with_underscores now'
    const plain = 'look at test case with underscores now'

    expect(estimateRows(snake, w)).toBe(estimateRows(plain, w))
  })
})
