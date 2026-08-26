import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { attachedImageNotice, imageTokenMeta, toTranscriptMessages } from '../domain/messages.js'
import { tildePath } from '../domain/paths.js'
import { upsert } from '../lib/messages.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

describe('toTranscriptMessages', () => {
  it('preserves assistant tool-call rows so resume does not drop prior turns', () => {
    const rows = [
      { role: 'user', text: 'first prompt' },
      { role: 'tool', context: 'repo', name: 'search_files', text: 'ignored raw result' },
      { role: 'assistant', text: 'first answer' },
      { role: 'user', text: 'second prompt' }
    ]

    expect(toTranscriptMessages(rows).map(msg => [msg.role, msg.text])).toEqual([
      ['user', 'first prompt'],
      ['assistant', 'first answer'],
      ['user', 'second prompt']
    ])
    expect(toTranscriptMessages(rows)[1]?.tools?.[0]).toContain('Search Files')
  })

  it('attaches a live reconnect pending tool without claiming completion', () => {
    const rows = [
      { role: 'user', text: 'run the fixture operation' },
      { role: 'assistant', text: 'Starting the fixture operation.' },
      { role: 'tool', context: 'fixture-live', name: 'fixture_operation', pending: true }
    ]

    const messages = toTranscriptMessages(rows)

    expect(messages).toHaveLength(2)
    expect(messages[1]?.role).toBe('assistant')
    expect(messages[1]?.tools?.[0]).toContain('Fixture Operation')
    expect(messages[1]?.tools?.[0]).toContain('fixture-live')
    expect(messages[1]?.tools?.[0]).toContain('…')
    expect(messages[1]?.tools?.[0]).not.toContain('✓')
    expect(messages[1]?.tools?.[0]).not.toContain('interrupted')
  })

  it('renders a cold-resume unmatched tool as interrupted, not pending', () => {
    const rows = [
      { role: 'user', text: 'run the fixture operation' },
      { role: 'tool', context: 'fixture-cold', name: 'fixture_operation', status: 'interrupted' }
    ]

    const messages = toTranscriptMessages(rows)

    expect(messages).toHaveLength(2)
    expect(messages[1]).toMatchObject({ kind: 'trail', role: 'system', text: '' })
    expect(messages[1]?.tools?.[0]).toContain('Fixture Operation')
    expect(messages[1]?.tools?.[0]).toContain('interrupted')
    expect(messages[1]?.tools?.[0]).toContain('✗')
    expect(messages[1]?.tools?.[0]).not.toContain('…')
  })

  it('keeps completed and interrupted tools from one contentless batch', () => {
    const rows = [
      { role: 'user', text: 'run both fixture operations' },
      { role: 'tool', context: 'first', name: 'read_file' },
      { role: 'tool', context: 'second', name: 'fixture_operation', status: 'interrupted' }
    ]

    const messages = toTranscriptMessages(rows)
    const tools = messages[1]?.tools ?? []

    expect(tools).toHaveLength(2)
    expect(tools[0]).toContain('Read File')
    expect(tools[0]).toContain('✓')
    expect(tools[1]).toContain('Fixture Operation')
    expect(tools[1]).toContain('interrupted')
    expect(tools[1]).toContain('✗')
  })

  it('hydrates persisted assistant reasoning into the thinking transcript field', () => {
    const messages = toTranscriptMessages([
      {
        role: 'assistant',
        text: 'Final answer.',
        reasoning: '**Checking constraints**\n\nThe ordering works.'
      }
    ])

    expect(messages).toEqual([
      {
        role: 'assistant',
        text: 'Final answer.',
        thinking: '**Checking constraints**\n\nThe ordering works.'
      }
    ])
  })

  it('keeps a reasoning-only assistant row visible on resume', () => {
    const messages = toTranscriptMessages([{ role: 'assistant', text: '', reasoning_content: 'Still reasoning.' }])

    expect(messages).toEqual([{ role: 'assistant', text: '', thinking: 'Still reasoning.' }])
  })
})

describe('MessageLine', () => {
  it('preserves a separator after compound user prompt glyphs in transcript rows', () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const t = {
      ...DEFAULT_THEME,
      brand: { ...DEFAULT_THEME.brand, prompt: 'Ψ >' }
    }

    const instance = renderSync(
      React.createElement(MessageLine, {
        cols: 80,
        msg: { role: 'user', text: 'Okay' },
        t
      }),
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    instance.unmount()
    instance.cleanup()

    const renderedLine = stripAnsi(output)
      .split('\n')
      .find(line => line.includes('Okay'))

    expect(renderedLine).toContain('Ψ > Okay')
  })
})

describe('upsert', () => {
  it('appends when last role differs', () => {
    expect(upsert([{ role: 'user', text: 'hi' }], 'assistant', 'hello')).toHaveLength(2)
  })

  it('replaces when last role matches', () => {
    expect(upsert([{ role: 'assistant', text: 'partial' }], 'assistant', 'full')[0]!.text).toBe('full')
  })

  it('appends to empty', () => {
    expect(upsert([], 'user', 'first')).toEqual([{ role: 'user', text: 'first' }])
  })

  it('does not mutate', () => {
    const prev = [{ role: 'user' as const, text: 'hi' }]
    upsert(prev, 'assistant', 'yo')
    expect(prev).toHaveLength(1)
  })
})

describe('imageTokenMeta (path in attachment notice)', () => {
  it('renders dims · display_path · tok in that order', () => {
    const meta = imageTokenMeta({
      width: 3024,
      height: 1762,
      token_estimate: 2040,
      display_path: '~/.hermes/images/clip-1782671798.png'
    })

    expect(meta).toBe('3024x1762 · ~/.hermes/images/clip-1782671798.png · ~2k tok')
  })

  it('falls back to raw path when display_path is absent', () => {
    const meta = imageTokenMeta({ width: 10, height: 10, path: '/tmp/x.png' })
    expect(meta).toContain('/tmp/x.png')
  })

  it('omits the path segment entirely when no path is provided', () => {
    const meta = imageTokenMeta({ width: 800, height: 600, token_estimate: 170 })
    expect(meta).toBe('800x600 · ~170 tok')
  })

  it('handles a path-only meta (no dims/tokens yet)', () => {
    expect(imageTokenMeta({ display_path: '~/.hermes/images/a.png' })).toBe('~/.hermes/images/a.png')
  })
})

describe('attachedImageNotice includes the path', () => {
  it('shows name and the full meta with path', () => {
    const notice = attachedImageNotice({
      name: 'clip-1.png',
      width: 100,
      height: 100,
      display_path: '~/.hermes/images/clip-1.png'
    })

    expect(notice).toBe('📎 Attached image: clip-1.png · 100x100 · ~/.hermes/images/clip-1.png')
  })
})

describe('tildePath', () => {
  const origHome = process.env.HOME

  afterEach(() => {
    if (origHome === undefined) {
      delete process.env.HOME
    } else {
      process.env.HOME = origHome
    }
  })

  it('abbreviates a path under $HOME to ~ and never truncates', () => {
    process.env.HOME = '/home/ndsadmin'
    expect(tildePath('/home/ndsadmin/Downloads/really-long-file-name.pdf')).toBe(
      '~/Downloads/really-long-file-name.pdf'
    )
  })

  it('leaves a non-home path untouched', () => {
    process.env.HOME = '/home/ndsadmin'
    expect(tildePath('/mnt/data/x.png')).toBe('/mnt/data/x.png')
  })
})
