import { describe, expect, it } from 'vitest'

import type { Msg } from '../types.js'

import { appendToolShelfMessage, canHoldToolShelf, isTodoDone, mergeToolShelfInto } from './liveProgress.js'

describe('isTodoDone', () => {
  it('only treats non-empty all-completed/cancelled lists as done', () => {
    expect(isTodoDone([])).toBe(false)
    expect(isTodoDone([{ content: 'x', id: 'x', status: 'completed' }])).toBe(true)
    expect(isTodoDone([{ content: 'x', id: 'x', status: 'in_progress' }])).toBe(false)
    expect(
      isTodoDone([
        { content: 'x', id: 'x', status: 'completed' },
        { content: 'y', id: 'y', status: 'cancelled' }
      ])
    ).toBe(true)
  })
})

describe('tool shelf helpers', () => {
  it('recognizes contextual thinking shelves as holders', () => {
    expect(canHoldToolShelf({ kind: 'trail', role: 'system', text: '', thinking: 'plan' })).toBe(true)
    expect(canHoldToolShelf({ kind: 'trail', role: 'system', text: '', tools: ['one ✓'] })).toBe(true)
    expect(canHoldToolShelf({ role: 'assistant', text: 'done' })).toBe(false)
  })

  it('merges source rows into an existing shelf', () => {
    expect(
      mergeToolShelfInto(
        { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓'] },
        { kind: 'trail', role: 'system', text: '', tools: ['two ✓'] }
      )
    ).toEqual({ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓', 'two ✓'] })
  })
})

describe('appendToolShelfMessage', () => {
  it('merges adjacent tool shelves into one contextual shelf', () => {
    const merged = appendToolShelfMessage([{ kind: 'trail', role: 'system', text: '', tools: ['one ✓'] }], {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: ['two ✓']
    })

    expect(merged).toEqual([{ kind: 'trail', role: 'system', text: '', tools: ['one ✓', 'two ✓'] }])
  })

  it('adds tools to the nearest contextual thinking shelf', () => {
    const merged = appendToolShelfMessage(
      [{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓'] }],
      { kind: 'trail', role: 'system', text: '', tools: ['two ✓'] }
    )

    expect(merged).toEqual([{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓', 'two ✓'] }])
  })

  it('does not merge back across an intervening thinking row', () => {
    // REVISED (was: "merges through intervening thinking-only rows back into the
    // nearest holder"). The original encoded the "group across thinking"
    // behaviour from fork commit 113d4c7477, which hoisted a tool call ABOVE the
    // reasoning that produced it: the reader saw one dense block of tools and a
    // separate block of thoughts, with cause and effect inverted. That commit is
    // fork-local, not upstream, so this is a revision of our own earlier choice.
    // A reasoning row is now a barrier, exactly like assistant text.
    const prev: Msg[] = [
      { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓'] },
      { kind: 'trail', role: 'system', text: '', thinking: 'more plan' }
    ]

    const merged = appendToolShelfMessage(prev, {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: ['two ✓']
    })

    // The earlier shelf keeps only the tool that actually ran before the thinking.
    expect(merged[0]).toEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: 'plan',
      tools: ['one ✓']
    })

    // The new call lands on the reasoning row that caused it, never above it.
    expect(merged).toHaveLength(2)
    expect(merged[1]).toEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: 'more plan',
      tools: ['two ✓']
    })
  })

  it('keeps a thinking/tool/thinking/tool stream in chronological order', () => {
    // REVISED (was: "collapses a chronological thinking/tool/thinking/tool stream
    // into one shelf"). Collapsing that stream is precisely the defect: it moved
    // 'two ✓' and 'three ✓' above 'more plan', so the transcript claimed those
    // tools ran before the reasoning that decided to run them. Tools now group
    // under the reasoning they follow, and only consecutive tools merge.
    const events: Msg[] = [
      { kind: 'trail', role: 'system', text: '', thinking: 'plan' },
      { kind: 'trail', role: 'system', text: '', tools: ['one ✓'] },
      { kind: 'trail', role: 'system', text: '', thinking: 'more plan' },
      { kind: 'trail', role: 'system', text: '', tools: ['two ✓'] },
      { kind: 'trail', role: 'system', text: '', tools: ['three ✓'] }
    ]

    const reduced = events.reduce<Msg[]>((acc, msg) => appendToolShelfMessage(acc, msg), [])

    expect(reduced).toHaveLength(2)
    expect(reduced[0]).toEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: 'plan',
      tools: ['one ✓']
    })
    // 'two' and 'three' ran back-to-back after the second thought, so they still
    // group together - the grouping the shelf exists to provide is preserved.
    expect(reduced[1]).toEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: 'more plan',
      tools: ['two ✓', 'three ✓']
    })
  })

  it('starts a new shelf across assistant text boundaries', () => {
    const merged = appendToolShelfMessage(
      [
        { kind: 'trail', role: 'system', text: '', tools: ['one ✓'] },
        { role: 'assistant', text: 'done' }
      ],
      { kind: 'trail', role: 'system', text: '', tools: ['two ✓'] }
    )

    expect(merged).toHaveLength(3)
  })
})
