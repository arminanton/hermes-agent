import {
  LIVE_RENDER_MAX_CHARS,
  LIVE_RENDER_MAX_LINES,
  THINKING_COT_MAX,
  VERBOSE_TRAIL_MAX_CHARS,
  VERBOSE_TRAIL_MAX_LINES
} from '../config/limits.js'
import { stripInlineMarkup } from '../components/markdown.js'
import { VERBS } from '../content/verbs.js'
import type { ThinkingMode } from '../types.js'

const ESC = String.fromCharCode(27)
const BEL = String.fromCharCode(7)
const ANSI_CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*[@-~]`, 'g')
const ANSI_CSI_WITH_CMD_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*([@-~])`, 'g')
const ANSI_INCOMPLETE_CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*(?=${ESC}|\\n|$)`, 'g')
const ANSI_OSC_RE = new RegExp(`${ESC}\\][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')
const ANSI_STRING_RE = new RegExp(`${ESC}[PX^_][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')
const ANSI_NON_CSI_ESC_SEQ_RE = new RegExp(`${ESC}(?!\\[|\\]|P|X|\\^|_)[ -/]*[0-~]`, 'g')
const ANSI_STRAY_ESC_RE = new RegExp(`${ESC}(?!\\[)[\\s\\S]?`, 'g')
const CONTROL_RE = /[\x00-\x08\x0B\x0C\x0D\x0E-\x1A\x1C-\x1F\x7F]/g
const WS_RE = /\s+/g

export const stripAnsi = (s: string) =>
  s
    .replace(ANSI_OSC_RE, '')
    .replace(ANSI_STRING_RE, '')
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_CSI_RE, '')
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_NON_CSI_ESC_SEQ_RE, '')
    .replace(ANSI_STRAY_ESC_RE, '')
    .replace(CONTROL_RE, '')

export const sanitizeAnsiForRender = (s: string) =>
  s
    .replace(ANSI_OSC_RE, '')
    .replace(ANSI_STRING_RE, '')
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_CSI_WITH_CMD_RE, (seq, cmd: string) => (cmd === 'm' ? seq : ''))
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_NON_CSI_ESC_SEQ_RE, '')
    .replace(ANSI_STRAY_ESC_RE, '')
    .replace(CONTROL_RE, '')

export const hasAnsi = (s: string) => s.includes(ESC)

const renderEstimateLine = (line: string) => {
  const trimmed = line.trim()

  if (trimmed.startsWith('|')) {
    return trimmed
      .split('|')
      .filter(Boolean)
      .map(cell => cell.trim())
      .join('  ')
  }

  return line
    .replace(/!\[(.*?)\]\(([^)\s]+)\)/g, '[image: $1]')
    .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/(?<!\w)__(.+?)__(?!\w)/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/(?<!\w)_(.+?)_(?!\w)/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/==(.+?)==/g, '$1')
    .replace(/\[\^([^\]]+)\]/g, '[$1]')
    .replace(/^#{1,6}\s+/, '')
    .replace(/^\s*[-*+]\s+\[( |x|X)\]\s+/, (_m, checked: string) => `• [${checked.toLowerCase() === 'x' ? 'x' : ' '}] `)
    .replace(/^\s*[-*+]\s+/, '• ')
    .replace(/^\s*(\d+)\.\s+/, '$1. ')
    .replace(/^\s*(?:>\s*)+/, '│ ')
}

export const compactPreview = (s: string, max: number) => {
  const one = s.replace(WS_RE, ' ').trim()

  return !one ? '' : one.length > max ? one.slice(0, max - 1) + '…' : one
}

export const estimateTokensRough = (text: string) => (!text ? 0 : (text.length + 3) >> 2)

export const edgePreview = (s: string, head = 16, tail = 28) => {
  const one = s.replace(WS_RE, ' ').trim().replace(/\]\]/g, '] ]')

  return !one
    ? ''
    : one.length <= head + tail + 4
      ? one
      : `${one.slice(0, head).trimEnd()}.. ${one.slice(-tail).trimStart()}`
}

export const pasteTokenLabel = (text: string, lineCount: number) => {
  const preview = edgePreview(text)

  if (!preview) {
    return `[[ [${fmtK(lineCount)} lines] ]]`
  }

  const [head = preview, tail = ''] = preview.split('.. ', 2)

  return tail
    ? `[[ ${head.trimEnd()}.. [${fmtK(lineCount)} lines] .. ${tail.trimStart()} ]]`
    : `[[ ${preview} [${fmtK(lineCount)} lines] ]]`
}

const THINKING_STATUS_RE = new RegExp(`^(?:${VERBS.join('|')})\\.{0,3}$`, 'i')
const THINKING_STATUS_CHUNK_RE = new RegExp(`[^A-Za-z\n]+\\s*(?:${VERBS.join('|')})\\.{0,3}\\s*`, 'giu')

export const cleanThinkingText = (reasoning: string) =>
  reasoning
    // Thinking text is rendered as PLAIN text (no markdown renderer), so raw
    // `**`/`` ` ``/`*` markers would show as literal noise. Order matters:
    // (1) add a blank line before a bold **heading** (a `**…**` span glued to
    //     the end of the previous line) so it stands out as its own line;
    // (2) strip ALL inline markup to clean prose;
    // (3) THEN run the status-verb / face-ticker filter on the already-clean
    //     text — so a heading starting with a thinking-verb (e.g.
    //     `**Analyzing …**`) isn't half-eaten into an orphan `**` (the verb
    //     filter keys off a non-letter before the verb, and a leading `**`
    //     counts — the bug the old keep-the-`**` path masked). The filter keeps
    //     intentional blank lines (the heading spacing) while dropping
    //     verb-only ticker lines.
    .replace(/([^\n])(\*\*[^\n]{3,}\*\*)(\s*)$/gm, '$1\n\n$2$3')
    .split('\n')
    .map(line => stripInlineMarkup(line))
    .join('\n')
    .split('\n')
    .map(line => line.replace(THINKING_STATUS_CHUNK_RE, '').trim())
    .filter(line => !THINKING_STATUS_RE.test(line.replace(/\.\.\.$/, '').trim()))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

export const thinkingPreview = (reasoning: string, mode: ThinkingMode, max: number = THINKING_COT_MAX) => {
  const raw = cleanThinkingText(reasoning)

  return !raw || mode === 'collapsed' ? '' : mode === 'full' ? raw : compactPreview(raw.replace(WS_RE, ' '), max)
}

export const boundedLiveRenderText = (
  text: string,
  { maxChars = LIVE_RENDER_MAX_CHARS, maxLines = LIVE_RENDER_MAX_LINES } = {}
) => boundedRenderText(text, 'showing live tail', { maxChars, maxLines })

const boundedRenderText = (
  text: string,
  labelPrefix: string,
  { maxChars, maxLines }: { maxChars: number; maxLines: number }
) => {
  if (text.length <= maxChars && text.split('\n', maxLines + 1).length <= maxLines) {
    return text
  }

  let start = 0
  let idx = text.length

  for (let seen = 0; seen < maxLines && idx > 0; seen++) {
    idx = text.lastIndexOf('\n', idx - 1)
    start = idx < 0 ? 0 : idx + 1

    if (idx < 0) {
      break
    }
  }

  const lineStart = start
  start = Math.max(lineStart, text.length - maxChars)

  if (start > lineStart) {
    const nextBreak = text.indexOf('\n', start)

    if (nextBreak >= 0 && nextBreak < text.length - 1) {
      start = nextBreak + 1
    }
  }

  const tail = text.slice(start).trimStart()
  const omittedLines = countNewlines(text, start)
  const omittedChars = Math.max(0, text.length - tail.length)

  const label =
    omittedLines > 0
      ? `[${labelPrefix}; omitted ${fmtK(omittedLines)} lines / ${fmtK(omittedChars)} chars]\n`
      : `[${labelPrefix}; omitted ${fmtK(omittedChars)} chars]\n`

  return `${label}${tail}`
}

const countNewlines = (text: string, end: number) => {
  let count = 0

  for (let i = 0; i < end; i++) {
    if (text.charCodeAt(i) === 10) {
      count++
    }
  }

  return count
}

export const stripTrailingPasteNewlines = (text: string) => (/[^\n]/.test(text) ? text.replace(/\n+$/, '') : text)

export const toolTrailLabel = (name: string) =>
  name
    .split('_')
    .filter(Boolean)
    .map(p => p[0]!.toUpperCase() + p.slice(1))
    .join(' ') || name

export const formatToolCall = (name: string, context = '') => {
  const label = toolTrailLabel(name)
  // Keep a generous single-line cap: enough to retain a full command/path so
  // the click-to-inspect / `/inspect` popup shows the whole thing (the inline
  // trail row is width-clamped by Ink's wrap-trim regardless of this length,
  // so a long value never visually overflows). The old 64-char cap discarded
  // the tail of every long command at completion time, leaving the inspect
  // popup showing only the truncated preview. This is still bounded so a
  // pathological arg can't balloon the stored trail string.
  const preview = compactPreview(context, 2000)

  return preview ? `${label}("${preview}")` : label
}

export const buildToolTrailLine = (
  name: string,
  context: string,
  error?: boolean,
  note?: string,
  duration?: number
) => {
  const detail = compactPreview(note ?? '', 72)
  const took = duration !== undefined ? ` (${duration.toFixed(1)}s)` : ''

  return `${formatToolCall(name, context)}${took}${detail ? ` :: ${detail}` : ''} ${error ? '✗' : '✓'}`
}

const verboseToolBlock = (label: string, text?: string) => {
  const body = (text ?? '').trim()

  // Persisted trail blocks are kept all session and rendered expanded by
  // default — cap to a small readable preview (NOT the 16KB live-render
  // budget) so a large tool output can't balloon the Ink render tree and
  // silently OOM-kill the TUI. See VERBOSE_TRAIL_MAX_CHARS (#34095).
  return body
    ? `${label}:\n${boundedLiveRenderText(body, {
        maxChars: VERBOSE_TRAIL_MAX_CHARS,
        maxLines: VERBOSE_TRAIL_MAX_LINES
      })}`
    : ''
}

export const buildVerboseToolTrailLine = (
  name: string,
  context: string,
  error?: boolean,
  duration?: number,
  argsText?: string,
  resultText?: string
) => {
  const detail = [verboseToolBlock('Args', argsText), verboseToolBlock(error ? 'Error' : 'Result', resultText)]
    .filter(Boolean)
    .join('\n')
  const took = duration !== undefined ? ` (${duration.toFixed(1)}s)` : ''

  return `${formatToolCall(name, context)}${took}${detail ? ` :: ${detail}` : ''} ${error ? '✗' : '✓'}`
}

export const isToolTrailResultLine = (line: string) => line.endsWith(' ✓') || line.endsWith(' ✗') || line.endsWith(' …')

export const parseToolTrailResultLine = (line: string) => {
  if (!isToolTrailResultLine(line)) {
    return null
  }

  const mark = line.endsWith(' ✗') ? '✗' : line.endsWith(' …') ? '…' : '✓'
  const body = line.slice(0, -2)
  const sep = body.indexOf(' :: ')

  if (sep >= 0) {
    return { call: body.slice(0, sep), detail: body.slice(sep + 4), mark }
  }

  // Legacy fallback for pre-April-2026 trail lines that used `Label: context`
  // (single colon-space, no parens) before the `Label("…")` call shape existed.
  // Guard against misfiring on MODERN lines: a modern call is `Label("…")`,
  // whose context routinely contains an embedded `: ` (e.g. a shell command
  // like `echo "the question: answer"`). That embedded colon must NOT be
  // treated as the call/detail separator — doing so split the row into a bogus
  // compact header + a spilled full-command "detail" block (#regression from
  // storing full commands in the trail line). The legacy separator always
  // preceded the first `("`, so only honour a `: ` that comes before any `("`.
  const legacy = body.indexOf(': ')
  const paren = body.indexOf('("')

  if (legacy > 0 && (paren < 0 || legacy < paren)) {
    return { call: body.slice(0, legacy), detail: body.slice(legacy + 2), mark }
  }

  return { call: body, detail: '', mark }
}

export const splitToolDuration = (call: string) => {
  const match = call.match(/^(.*?)( \(\d+(?:\.\d)?s\))$/)

  return match ? { label: match[1]!, duration: match[2]! } : { label: call, duration: '' }
}

// Build a SIMPLE header title for the click / `/inspect` pager popup from a
// tool-call string like `Terminal("cd …")`, `Read File("x") (1.2s)`, or a bare
// `Delegate Task`. The popup BODY already shows the full command verbatim, so
// the title must NOT duplicate it (they were rendering identical full commands
// — one bold, one not). Extract just the tool label: strip a trailing ✓/✗
// completion marker + a duration suffix, then take everything before the first
// `("`, and render it as `<Label> tool call`.
export const toolCallInspectTitle = (call: string) => {
  const noMark = call.replace(/\s*[✓✗]\s*$/, '')
  const { label: withoutDuration } = splitToolDuration(noMark)
  const open = withoutDuration.match(/^(.*?)\("/)
  const label = (open ? open[1]! : withoutDuration).trim()

  return label ? `${label} tool call` : 'Tool call'
}

// Re-truncate a `Label("<context>")` call string to the COMPACT inline form for
// display in the tool-trail row. The stored trail line keeps the full context
// (so the click-to-inspect / `/inspect` popup shows everything), but the inline
// row should stay short with a trailing `…` like it always did — decoupling
// what's shown in the shell from what's available on inspect. No-op for calls
// with no parenthesized context (e.g. a bare `Delegate Task`) or already-short
// ones. Mirrors formatToolCall's `Label("preview")` shape.
const INLINE_TOOL_CALL_MAX = 64
export const compactToolCallDisplay = (call: string) => {
  // Peel off a trailing ` (1.2s)` duration first so it survives compaction.
  const { duration, label: withoutDuration } = splitToolDuration(call)

  // Match ONLY the opening `Label("` prefix, not a clean `Label("…")` shape:
  // real tool contexts (shell commands, heredocs) contain embedded double
  // quotes, so the string frequently does NOT end in a clean `")`. Anchoring
  // on the end made the old regex silently fail on exactly those commands,
  // leaving the full command in the inline row (the reported bug). Grab the
  // label + everything after the first `("`, then truncate that remainder.
  const open = withoutDuration.match(/^(.*?)\("/)

  if (!open) {
    return call
  }

  const label = open[1]!
  const rest = withoutDuration.slice(open[0].length) // everything after `Label("`
  // Drop a trailing closing `")` if present so we truncate the context body.
  const ctxRaw = rest.endsWith('")') ? rest.slice(0, -2) : rest
  const oneLine = ctxRaw.replace(WS_RE, ' ').trim()

  if (oneLine.length <= INLINE_TOOL_CALL_MAX) {
    return call
  }

  const preview = oneLine.slice(0, INLINE_TOOL_CALL_MAX - 1) + '…'

  return `${label}("${preview}")${duration}`
}

export const isTransientTrailLine = (line: string) => line.startsWith('drafting ') || line === 'analyzing tool output…'

export const sameToolTrailGroup = (label: string, entry: string) =>
  entry === `${label} ✓` ||
  entry === `${label} ✗` ||
  entry.startsWith(`${label}(`) ||
  entry.startsWith(`${label} ::`) ||
  entry.startsWith(`${label}:`)

export const lastCotTrailIndex = (trail: readonly string[]) => {
  for (let i = trail.length - 1; i >= 0; i--) {
    if (!isToolTrailResultLine(trail[i]!)) {
      return i
    }
  }

  return -1
}

export const estimateRows = (text: string, w: number, compact = false) => {
  let fence: { char: '`' | '~'; len: number } | null = null
  let rows = 0

  for (const raw of text.split('\n')) {
    const line = stripAnsi(raw)
    const maybeFence = line.match(/^\s*(`{3,}|~{3,})(.*)$/)

    if (maybeFence) {
      const marker = maybeFence[1]!
      const lang = maybeFence[2]!.trim()

      if (!fence) {
        fence = { char: marker[0] as '`' | '~', len: marker.length }

        if (lang) {
          rows += Math.ceil((`─ ${lang}`.length || 1) / w)
        }
      } else if (marker[0] === fence.char && marker.length >= fence.len) {
        fence = null
      }

      continue
    }

    const inCode = Boolean(fence)
    const trimmed = line.trim()

    if (!inCode && trimmed.startsWith('|') && /^[|\s:-]+$/.test(trimmed)) {
      continue
    }

    const rendered = inCode ? line : renderEstimateLine(line)

    if (compact && !rendered.trim()) {
      continue
    }

    rows += Math.ceil((rendered.length || 1) / w)
  }

  return Math.max(1, rows)
}

/**
 * Render an unanswered clarify prompt (timed out, or cancelled with Esc/Ctrl+C)
 * as a persistent transcript block.  The live `ClarifyPrompt` overlay is torn
 * down the moment the turn settles, so without this the question + options
 * vanish from the screen while the agent's follow-up still refers to "the
 * options above".  Mirrors the option formatting in ClarifyPrompt (the same
 * 1-based numbered list) so the persisted record reads identically to what was
 * on screen.  `reason` states why the prompt ended ("timed out", "cancelled").
 */
export const formatAbandonedClarify = (question: string, choices: string[] | null, reason: string) => {
  const head = `ask ${question.trim()}`
  const opts = (choices ?? []).map((c, i) => `  ${i + 1}. ${c}`)

  return [head, ...opts, `  (${reason} — no selection)`].join('\n')
}

export const flat = (r: Record<string, string[]>) => Object.values(r).flat()

const COMPACT_NUMBER = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1, notation: 'compact' })

export const fmtK = (n: number) => COMPACT_NUMBER.format(n).replace(/[KMBT]$/, s => s.toLowerCase())

export const pick = <T>(a: T[]) => a[Math.floor(Math.random() * a.length)]!

export const isPasteBackedText = (text: string) =>
  /\[\[paste:\d+(?:[^\n]*?)\]\]|\[paste #\d+ (?:attached|excerpt)(?:[^\n]*?)\]/.test(text)
