import { beforeEach, describe, expect, it } from 'vitest'

import { applyDelegationStatus } from '../app/delegationStore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'

// applyDelegationStatus() enriches the live turn-state subagents with the
// authoritative per-child registry fields (session_id, paused) coming from
// the delegation.status `active[]` snapshot — so the /agents overlay can show
// each child's own session_id (for the on-demand history view) and its
// individual pause state, WITHOUT the parent relaying a heavy live stream.
describe('applyDelegationStatus — merges active[] registry fields onto subagents', () => {
  beforeEach(() => {
    resetTurnState()
    turnController.fullReset()
  })

  const seedSubagent = (id: string) => {
    turnController.upsertSubagent(
      { goal: 'do a thing', subagent_id: id, task_index: 0 },
      () => ({}),
      { createIfMissing: true }
    )
  }

  it('attaches session_id + paused onto an existing subagent by id', () => {
    seedSubagent('sa-1')
    applyDelegationStatus({
      active: [{ paused: true, session_id: 'sess-abc', subagent_id: 'sa-1' }]
    })
    const row = getTurnState().subagents.find(s => s.id === 'sa-1')
    expect(row?.sessionId).toBe('sess-abc')
    expect(row?.paused).toBe(true)
  })

  it('does NOT fabricate a subagent row for an unknown id (createIfMissing:false)', () => {
    applyDelegationStatus({
      active: [{ session_id: 'sess-ghost', subagent_id: 'ghost' }]
    })
    expect(getTurnState().subagents.find(s => s.id === 'ghost')).toBeUndefined()
  })

  it('is a no-op when active[] is absent (only global fields present)', () => {
    seedSubagent('sa-2')
    applyDelegationStatus({ paused: true })
    const row = getTurnState().subagents.find(s => s.id === 'sa-2')
    // The child's own paused flag stays unset — the top-level paused is the
    // GLOBAL spawn pause, not this child's per-agent pause.
    expect(row?.sessionId).toBeUndefined()
    expect(row?.paused).toBeUndefined()
  })

  it('tolerates a null/undefined response', () => {
    expect(() => applyDelegationStatus(null)).not.toThrow()
    expect(() => applyDelegationStatus(undefined)).not.toThrow()
  })

  it('skips active rows without a subagent_id', () => {
    seedSubagent('sa-3')
    applyDelegationStatus({
      active: [{ session_id: 'sess-x' }, { paused: true, subagent_id: 'sa-3' }]
    })
    const row = getTurnState().subagents.find(s => s.id === 'sa-3')
    expect(row?.paused).toBe(true)
  })
})
