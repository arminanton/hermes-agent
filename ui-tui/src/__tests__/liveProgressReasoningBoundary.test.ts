import { describe, expect, it } from 'vitest'

import { appendToolShelfMessage } from '../lib/liveProgress.js'
import type { Msg } from '../types.js'

/**
 * Regression: tool calls were hoisted ABOVE the reasoning that produced them.
 *
 * Reported behaviour: "tool calls are always being pushed to before/above the
 * thinking/reasoning steps... as soon as the tool call completes, it is moved
 * out/above/before the whole thought process, then we have a huge list of tools
 * grouped together along with a huge list of thoughts."
 *
 * Cause: appendToolShelfMessage walks BACKWARD looking for an existing
 * tool-carrying trail to merge into, and only stops at a "barrier". A reasoning
 * segment was not a barrier, so a tool call made after thinking merged into the
 * shelf that preceded that thinking, inverting cause and effect in the
 * transcript.
 *
 * Fix: a segment carrying reasoning is a barrier, exactly like assistant text.
 */

const trail = (over: Partial<Msg> = {}): Msg =>
  ({ kind: 'trail', role: 'system', text: '', ...over }) as Msg

describe('appendToolShelfMessage reasoning boundary', () => {
  it('does not merge a tool call backward across a reasoning segment', () => {
    const prev: Msg[] = [
      trail({ tools: ['first_tool'] }),      // shelf from before the thinking
      trail({ thinking: 'deciding what to do next' })
    ]

    const next = appendToolShelfMessage(prev, trail({ tools: ['second_tool'] }))

    // The earlier shelf must be left alone...
    expect(next[0]!.tools).toEqual(['first_tool'])
    // ...and the new call must land at or after the reasoning, never before it.
    const reasoningIndex = next.findIndex(m => m.thinking?.includes('deciding'))
    const secondIndex = next.findIndex(m => m.tools?.includes('second_tool'))
    expect(secondIndex).toBeGreaterThanOrEqual(reasoningIndex)
  })

  it('still groups tools that ran together with no reasoning between them', () => {
    const prev: Msg[] = [trail({ tools: ['alpha'] })]

    const next = appendToolShelfMessage(prev, trail({ tools: ['beta'] }))

    // Parallel/consecutive calls should still collapse into one shelf; that
    // grouping is the whole point of the shelf and must survive the fix.
    expect(next).toHaveLength(1)
    expect(next[0]!.tools).toEqual(['alpha', 'beta'])
  })

  it('lets a reasoning segment hold the tools it caused', () => {
    const prev: Msg[] = [trail({ thinking: 'I should check the config' })]

    const next = appendToolShelfMessage(prev, trail({ tools: ['read_file'] }))

    // The tool belongs WITH the thinking that produced it, not appended as a
    // detached row above or below it.
    expect(next).toHaveLength(1)
    expect(next[0]!.thinking).toContain('I should check the config')
    expect(next[0]!.tools).toEqual(['read_file'])
  })

  it('does not reach past assistant text either (existing barrier intact)', () => {
    const prev: Msg[] = [
      trail({ tools: ['early'] }),
      { kind: 'trail', role: 'assistant', text: 'here is what I found' } as Msg
    ]

    const next = appendToolShelfMessage(prev, trail({ tools: ['late'] }))

    expect(next[0]!.tools).toEqual(['early'])
    expect(next.at(-1)!.tools).toEqual(['late'])
  })
})
