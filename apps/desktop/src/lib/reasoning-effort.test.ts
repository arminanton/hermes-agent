import { describe, expect, it } from 'vitest'

import {
  isThinkingEnabled,
  REASONING_EFFORT_VALUES,
  REASONING_EFFORTS,
  reasoningEffortLabel,
  resolveReasoningEffort
} from './reasoning-effort'

describe('reasoning effort vocabulary', () => {
  it('exposes max but not the unsupported ultra mode', () => {
    expect(REASONING_EFFORTS).toEqual(['minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
    expect(REASONING_EFFORT_VALUES).toEqual(['none', ...REASONING_EFFORTS])
    expect(REASONING_EFFORT_VALUES).not.toContain('ultra')
  })

  it('keeps xhigh and max distinct in compact labels', () => {
    expect(reasoningEffortLabel('xhigh')).toBe('XHigh')
    expect(reasoningEffortLabel('max')).toBe('Max')
    expect(reasoningEffortLabel('ultra')).toBe('Med')
  })

  it('resolves inherited and disabled values', () => {
    expect(resolveReasoningEffort('')).toBe('medium')
    expect(resolveReasoningEffort('none')).toBe('')
    expect(resolveReasoningEffort('max')).toBe('max')
    expect(resolveReasoningEffort('ultra')).toBe('medium')
    expect(isThinkingEnabled('none')).toBe(false)
    expect(isThinkingEnabled('')).toBe(true)
  })
})
