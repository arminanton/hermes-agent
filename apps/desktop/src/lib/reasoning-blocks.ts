/** Repair reasoning-summary blocks that were persisted or streamed without boundaries. */

const GLUED_HEADING_RUN = /(?<=[^\s*])\*{4}(?=[^\s*])/g
const SUMMARY_BLOCK_START = /^\*\*([^*\n]+?)\*\*(?:$|\r?\n\r?\n|(?=\*\*))/

function startsWithSummaryBlock(text: string): boolean {
  const match = SUMMARY_BLOCK_START.exec(text)

  return Boolean(match && /^[A-Z][A-Za-z]+ing\b/.test(match[1].trim()))
}

function blockStartBefore(text: string, offset: number): number {
  const paragraph = text.lastIndexOf('\n\n', offset - 1)

  return paragraph < 0 ? 0 : paragraph + 2
}

function repairPlainText(text: string): string {
  return text.replace(GLUED_HEADING_RUN, (stars, offset: number) => {
    const left = text.slice(blockStartBefore(text, offset), offset + 2)
    const right = text.slice(offset + 2)

    return startsWithSummaryBlock(left) && startsWithSummaryBlock(right)
      ? '**\n\n**'
      : stars
  })
}

function findClosingBacktickRun(text: string, start: number, delimiter: string): number {
  let candidate = text.indexOf(delimiter, start)

  while (candidate >= 0) {
    const before = candidate > 0 ? text[candidate - 1] : ''
    const after = text[candidate + delimiter.length] ?? ''

    if (before !== '`' && after !== '`') {
      return candidate
    }

    candidate = text.indexOf(delimiter, candidate + delimiter.length)
  }

  return -1
}

function repairOutsideCodeSpans(text: string): string {
  let output = ''
  let plainStart = 0
  let cursor = 0

  while (cursor < text.length) {
    if (text[cursor] !== '`') {
      cursor += 1

      continue
    }

    const delimiterStart = cursor

    while (cursor < text.length && text[cursor] === '`') {
      cursor += 1
    }

    const delimiter = text.slice(delimiterStart, cursor)
    const closingStart = findClosingBacktickRun(text, cursor, delimiter)

    if (closingStart < 0) {
      break
    }

    output += repairPlainText(text.slice(plainStart, delimiterStart))
    const closingEnd = closingStart + delimiter.length
    output += text.slice(delimiterStart, closingEnd)
    cursor = closingEnd
    plainStart = closingEnd
  }

  return output + repairPlainText(text.slice(plainStart))
}

function lineBody(line: string): string {
  const withoutLf = line.endsWith('\n') ? line.slice(0, -1) : line

  return withoutLf.endsWith('\r') ? withoutLf.slice(0, -1) : withoutLf
}

interface Fence {
  marker: '`' | '~'
  length: number
}

function openingFence(line: string): Fence | null {
  const match = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line)

  if (!match) {
    return null
  }

  const run = match[1]
  const marker = run[0] as Fence['marker']

  if (marker === '`' && match[2].includes('`')) {
    return null
  }

  return { marker, length: run.length }
}

function closesFence(line: string, fence: Fence): boolean {
  const match = /^ {0,3}(`+|~+)[ \t]*$/.exec(line)

  return Boolean(
    match && match[1][0] === fence.marker && match[1].length >= fence.length
  )
}

/** Repair prose while leaving Markdown code spans and blocks byte-for-byte intact. */
export function separateGluedReasoningBlocks(text: string): string {
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) ?? []
  let output = ''
  let plain = ''
  let fence: Fence | null = null

  const flushPlain = () => {
    output += repairOutsideCodeSpans(plain)
    plain = ''
  }

  for (const line of lines) {
    const body = lineBody(line)

    if (fence) {
      output += line

      if (closesFence(body, fence)) {
        fence = null
      }

      continue
    }

    const opened = openingFence(body)

    if (opened) {
      flushPlain()
      output += line
      fence = opened

      continue
    }

    if (/^(?: {4}|\t)/.test(body)) {
      flushPlain()
      output += line

      continue
    }

    plain += line
  }

  flushPlain()

  return output
}
