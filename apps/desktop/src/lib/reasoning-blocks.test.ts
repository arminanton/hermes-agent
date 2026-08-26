import { describe, expect, it } from 'vitest'

import { separateGluedReasoningBlocks } from './reasoning-blocks'

describe('separateGluedReasoningBlocks', () => {
  it('splits heading-onto-heading parts', () => {
    const glued =
      '**Investigating likely culprit PRs****Inspecting message schema****Analyzing interrupted tool call impact**'

    expect(separateGluedReasoningBlocks(glued)).toBe(
      [
        '**Investigating likely culprit PRs**',
        '',
        '**Inspecting message schema**',
        '',
        '**Analyzing interrupted tool call impact**'
      ].join('\n')
    )
  })

  it('does not infer a summary boundary from prose followed by bold text', () => {
    const glued =
      '**Simulating a greeting stream**\n\nIt feels like a streaming interaction!**Simulating a greeting stream**\n\nI want to meet the request.'

    expect(separateGluedReasoningBlocks(glued)).toBe(glued)
  })

  it('is idempotent on already-separated text', () => {
    const separated = '**One**\n\n**Two**'
    expect(separateGluedReasoningBlocks(separated)).toBe(separated)
  })

  it('leaves emphasis inside prose alone', () => {
    const prose = 'Looking at the logs, the **signature** field is missing.'
    expect(separateGluedReasoningBlocks(prose)).toBe(prose)
  })

  it('leaves an unclosed emphasis run alone', () => {
    expect(separateGluedReasoningBlocks('weighing options **')).toBe('weighing options **')
  })

  it('does not split a heading that already opens the text', () => {
    expect(separateGluedReasoningBlocks('**Only one part**')).toBe('**Only one part**')
  })

  it('does not rewrite inline code', () => {
    const text = 'Use `foo**bar**` here.'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('does not rewrite fenced code', () => {
    const text = '```text\nfoo**bar**\n****\n```'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('does not treat an adjacent label value as a summary boundary', () => {
    const text = 'label:**value**'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('preserves adjacent prose and emphasis', () => {
    const text = 'Finished review**Next step**'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('leaves valid adjacent strong emphasis alone', () => {
    const text = '**First phrase****Second phrase** and word**suffix**'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('leaves links, images, escapes, and ordinary strong emphasis alone', () => {
    const text = [
      '[**label**](https://example.test/a**b**)',
      '![**alt**](image.png)',
      String.raw`\**literal****stars**`,
      'Finished review**next step**'
    ].join('\n')

    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('does not rewrite tilde fenced code', () => {
    const text = '~~~js\nfoo****bar\n~~~'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('does not close a backtick fence on embedded backticks', () => {
    const text = '```js\nconst marker = "```";\nfoo****bar\n```'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })

  it('does not rewrite indented code', () => {
    const text = 'Reasoning:\n\n    foo****bar\n\nFinished review**Next step**'
    expect(separateGluedReasoningBlocks(text)).toBe(text)
  })
})
