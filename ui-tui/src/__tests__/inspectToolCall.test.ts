import { beforeEach, describe, expect, it } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { inspectToolCall } from '../lib/inspectToolCall.js'

describe('inspectToolCall', () => {
  beforeEach(() => {
    resetOverlayState()
  })

  it('opens the pager overlay with the title and content split into lines', () => {
    inspectToolCall('Terminal', 'line one\nline two\nline three')

    const pager = getOverlayState().pager
    expect(pager).not.toBeNull()
    expect(pager!.title).toBe('Terminal')
    expect(pager!.offset).toBe(0)
    expect(pager!.lines).toEqual(['line one', 'line two', 'line three'])
  })

  it('normalizes CRLF so Windows-style tool output paginates correctly', () => {
    inspectToolCall('Patch', 'a\r\nb\r\nc')

    expect(getOverlayState().pager!.lines).toEqual(['a', 'b', 'c'])
  })

  it('shows a placeholder when there is no captured content', () => {
    inspectToolCall('Empty', '')

    const pager = getOverlayState().pager
    expect(pager!.lines).toEqual(['(no content captured for this tool call)'])
  })

  it('resets the scroll offset each time it opens (fresh view per inspect)', () => {
    inspectToolCall('First', 'x\ny\nz')
    // Simulate the user having scrolled, then inspecting a different call.
    inspectToolCall('Second', 'only one line')

    const pager = getOverlayState().pager
    expect(pager!.title).toBe('Second')
    expect(pager!.offset).toBe(0)
  })
})
