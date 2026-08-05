import { describe, expect, it } from 'vitest'

import { buildSnipExpander, PASTE_SNIPPET_RE, type PasteSnip } from '../protocol/paste.js'

// Regression coverage for the bug where a large paste sent via /steer (or
// busy-mode steer) arrived as only the collapsed `[[ … [N lines] … ]]`
// placeholder instead of its full body. The normal Enter-submit path always
// expanded snips; the steer paths did not. buildSnipExpander is the shared
// helper both paths now use.
describe('buildSnipExpander (paste placeholder re-expansion)', () => {
  const label = '[[ @Library(.. [120 lines] .. } ]]'
  const fullBody = "@Library('ci-common') _\n\npipeline {\n  agent none\n}\n"

  const snips: PasteSnip[] = [{ label, text: fullBody }]

  it('expands a collapsed placeholder back to the full body', () => {
    const expand = buildSnipExpander(snips)
    expect(expand(label)).toBe(fullBody)
  })

  it('expands a placeholder embedded in surrounding steer text', () => {
    const expand = buildSnipExpander(snips)
    expect(expand(`please review this: ${label}`)).toBe(`please review this: ${fullBody}`)
  })

  it('leaves the result with no leftover placeholder token', () => {
    const expand = buildSnipExpander(snips)
    const out = expand(`steer: ${label}`)
    expect(PASTE_SNIPPET_RE.test(out)).toBe(false)
  })

  it('passes unknown tokens through unchanged', () => {
    const expand = buildSnipExpander(snips)
    const unknown = '[[ not a real snippet ]]'
    expect(expand(unknown)).toBe(unknown)
  })

  it('resolves repeated identical placeholders FIFO to their own bodies', () => {
    const dup: PasteSnip[] = [
      { label, text: 'FIRST' },
      { label, text: 'SECOND' }
    ]
    const expand = buildSnipExpander(dup)
    expect(expand(`${label} then ${label}`)).toBe('FIRST then SECOND')
  })

  it('is a no-op when there are no snips (plain steer text)', () => {
    const expand = buildSnipExpander([])
    expect(expand('just a short steer')).toBe('just a short steer')
  })

  // Regression for the queue-path leak (no /steer involved): a paste submitted
  // while busy is enqueued, and dispatchSubmission's clearIn() wipes pasteSnips
  // BEFORE the queue drains. The fix expands SYNCHRONOUSLY at enqueue time using
  // the still-populated snips, so the queued/sent value is already the full body
  // and a later snips-clear can't leak the `[[ … ]]` placeholder. This test
  // models that ordering: expand against live snips, THEN clear, and assert the
  // captured value is the full body, not the placeholder.
  it('queue path: value captured at enqueue time survives a later snips clear', () => {
    let liveSnips: PasteSnip[] = [{ label, text: fullBody }]

    // enqueue-time expansion (what handleBusyInput now does synchronously)
    const enqueuedValue = buildSnipExpander(liveSnips)(label)

    // clearIn() fires afterwards, emptying the store
    liveSnips = []

    // a later drain re-expanding against the now-empty store would NOT recover
    // the body — proving why expansion must happen BEFORE the clear:
    const lateExpand = buildSnipExpander(liveSnips)(label)

    expect(enqueuedValue).toBe(fullBody) // captured early → full content
    expect(lateExpand).toBe(label) // captured late → leaked placeholder (the bug)
  })
})
