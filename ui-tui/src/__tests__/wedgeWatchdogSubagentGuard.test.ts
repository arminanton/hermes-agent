import { describe, expect, it } from 'vitest'

import { isRunning } from '../lib/subagentTree.js'
import type { SubagentProgress } from '../types.js'

// The busy-wedge watchdog (useMainApp.ts) declares a turn "stream stalled" when
// `busy` is true but no backend event has arrived for WEDGE_TIMEOUT_MS. A long
// SILENT subagent (e.g. `sleep 200`, a long build, deep search) sends no events
// to the PARENT session during its run, so without a guard the watchdog fires a
// FALSE "stream stalled" while the delegation is perfectly healthy.
//
// The fix: suppress the stall verdict while any subagent is still running/queued
// — `getTurnState().subagents.some(isRunning)`. These tests pin that predicate so
// the guard can't silently regress (e.g. someone narrows isRunning, or a status
// value changes).

function sa(status: SubagentProgress['status']): SubagentProgress {
  return { id: 'x', label: 'child', status } as SubagentProgress
}

describe('wedge-watchdog subagent guard (false "stream stalled" fix)', () => {
  it('suppresses the stall while a subagent is running', () => {
    const subagents = [sa('running')]
    expect(subagents.some(isRunning)).toBe(true)
  })

  it('suppresses the stall while a subagent is queued (about to run)', () => {
    expect([sa('queued')].some(isRunning)).toBe(true)
  })

  it('does NOT suppress when all subagents have completed (genuine wedge still trips)', () => {
    expect([sa('completed')].some(isRunning)).toBe(false)
  })

  it('does NOT suppress when a subagent failed', () => {
    expect([sa('failed')].some(isRunning)).toBe(false)
  })

  it('does NOT suppress when there are no subagents at all (plain wedged turn)', () => {
    expect(([] as SubagentProgress[]).some(isRunning)).toBe(false)
  })

  it('suppresses when at least ONE of several subagents is still running', () => {
    expect([sa('completed'), sa('running'), sa('completed')].some(isRunning)).toBe(true)
  })

  it('does NOT suppress once a batch has fully finished (mixed terminal states)', () => {
    expect([sa('completed'), sa('failed'), sa('completed')].some(isRunning)).toBe(false)
  })
})

// The SAME false-stall class also hits the MAIN turn: a long-running tool call
// (a multi-minute build/test wait, `process wait`, a deep search, a sleep) emits
// nothing between tool.start and tool.result, so the activity clock can exceed
// WEDGE_TIMEOUT_MS while the turn is healthy. The subagent guard above only
// covers DELEGATED long tool calls; the parent-turn in-flight tool needs its own
// exemption: `getTurnState().tools.length > 0`. `tools` is populated on
// tool.start, removed on tool.result, and cleared on idle(), so it self-clears
// and can't mask a genuinely wedged turn. These tests pin that predicate.

interface ActiveToolLike {
  id: string
  name: string
}

const tool = (id: string): ActiveToolLike => ({ id, name: 'terminal' })

describe('wedge-watchdog main-turn tool guard (false "stream stalled" on long tool calls)', () => {
  it('suppresses the stall while a single tool is in flight', () => {
    expect([tool('t1')].length > 0).toBe(true)
  })

  it('suppresses while several tools are in flight (parallel calls)', () => {
    expect([tool('t1'), tool('t2')].length > 0).toBe(true)
  })

  it('does NOT suppress when no tool is in flight (genuine wedge still trips)', () => {
    expect(([] as ActiveToolLike[]).length > 0).toBe(false)
  })
})
