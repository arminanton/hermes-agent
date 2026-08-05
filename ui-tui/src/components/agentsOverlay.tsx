import { Box, NoSelect, ScrollBox, type ScrollBoxHandle, Text, useInput, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type ReactNode, type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import {
  $delegationState,
  $overlaySectionsOpen,
  applyDelegationStatus,
  toggleOverlaySection
} from '../app/delegationStore.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $spawnDiff, $spawnHistory, clearDiffPair, type SpawnSnapshot } from '../app/spawnHistoryStore.js'
import { useTurnSelector } from '../app/turnStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  DelegationPauseResponse,
  DelegationStatusResponse,
  SubagentControlResponse,
  SubagentInterruptResponse,
  SubagentKillAllResponse
} from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import {
  buildSubagentTree,
  descendantIds,
  flattenTree,
  fmtCost,
  fmtDuration,
  fmtTokens,
  formatSummary,
  hotnessBucket,
  peakHotness,
  sparkline,
  topLevelSubagents,
  treeTotals,
  widthByDepth
} from '../lib/subagentTree.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { SubagentNode, SubagentProgress } from '../types.js'

// ── Types + lookup tables ────────────────────────────────────────────

// One rendered history entry from the child's own session (session.history
// RPC). Kept loose: the gateway returns role + a text/preview per message.
interface HistoryMsg {
  content?: string
  preview?: string
  role?: string
  text?: string
}
interface SessionHistoryResponse {
  count?: number
  messages?: HistoryMsg[]
}

type SortMode = 'depth-first' | 'duration-desc' | 'status' | 'tools-desc'
type FilterMode = 'all' | 'failed' | 'leaf' | 'running'
type Status = SubagentProgress['status']

const SORT_ORDER: readonly SortMode[] = ['depth-first', 'tools-desc', 'duration-desc', 'status']
const FILTER_ORDER: readonly FilterMode[] = ['all', 'running', 'failed', 'leaf']

const SORT_LABEL: Record<SortMode, string> = {
  'depth-first': 'spawn order',
  'duration-desc': 'slowest',
  status: 'status',
  'tools-desc': 'busiest'
}

const FILTER_LABEL: Record<FilterMode, string> = {
  all: 'all',
  failed: 'failed',
  leaf: 'leaves',
  running: 'running'
}

const STATUS_RANK: Record<Status, number> = {
  error: 0,
  failed: 0,
  interrupted: 1,
  timeout: 1,
  running: 2,
  queued: 3,
  completed: 4
}

const statusRank = (status: string): number => STATUS_RANK[status as Status] ?? STATUS_RANK.error

const SORT_COMPARATORS: Record<SortMode, (a: SubagentNode, b: SubagentNode) => number> = {
  'depth-first': (a, b) => a.item.depth - b.item.depth || a.item.index - b.item.index,
  'tools-desc': (a, b) => b.aggregate.totalTools - a.aggregate.totalTools,
  'duration-desc': (a, b) => b.aggregate.totalDuration - a.aggregate.totalDuration,
  status: (a, b) => statusRank(a.item.status) - statusRank(b.item.status)
}

const FILTER_PREDICATES: Record<FilterMode, (n: SubagentNode) => boolean> = {
  all: () => true,
  leaf: n => n.children.length === 0,
  running: n => n.item.status === 'running' || n.item.status === 'queued',
  failed: n =>
    n.item.status === 'error' ||
    n.item.status === 'failed' ||
    n.item.status === 'interrupted' ||
    n.item.status === 'timeout'
}

const STATUS_GLYPH: Record<Status, { color: (t: Theme) => string; glyph: string }> = {
  running: { color: t => t.color.accent, glyph: '●' },
  queued: { color: t => t.color.muted, glyph: '○' },
  completed: { color: t => t.color.statusGood, glyph: '✓' },
  interrupted: { color: t => t.color.warn, glyph: '■' },
  failed: { color: t => t.color.error, glyph: '✗' },
  timeout: { color: t => t.color.warn, glyph: '⌛' },
  error: { color: t => t.color.error, glyph: '⚠' }
}

// Heatmap palette — cold → hot, resolved against the active theme.
const heatPalette = (t: Theme) => [t.color.border, t.color.accent, t.color.primary, t.color.warn, t.color.error]

// ── Pure helpers ─────────────────────────────────────────────────────

const fmtDur = (seconds?: number) => (seconds == null || seconds <= 0 ? '' : fmtDuration(seconds))
const fmtElapsedLabel = (seconds: number) => (seconds < 0 ? '' : fmtDuration(seconds))

const displayElapsedSeconds = (item: SubagentProgress, nowMs: number): number | null => {
  if (item.durationSeconds != null) {
    return item.durationSeconds
  }

  if (item.startedAt != null && (item.status === 'running' || item.status === 'queued')) {
    return Math.max(0, (nowMs - item.startedAt) / 1000)
  }

  return null
}

const indentFor = (depth: number): string => '  '.repeat(Math.max(0, depth))
const formatRowId = (n: number): string => String(n + 1).padStart(2, ' ')
const cycle = <T,>(order: readonly T[], current: T): T => order[(order.indexOf(current) + 1) % order.length]!

const statusGlyph = (item: SubagentProgress, t: Theme) => {
  // Defensive fallback for cross-version snapshots with unknown statuses.
  const g = STATUS_GLYPH[item.status] ?? STATUS_GLYPH.error

  return { color: g.color(t), glyph: g.glyph }
}

const prepareRows = (tree: SubagentNode[], sort: SortMode, filter: FilterMode): SubagentNode[] =>
  tree.length === 0 ? [] : flattenTree([...tree].sort(SORT_COMPARATORS[sort])).filter(FILTER_PREDICATES[filter])

const diffMetricLine = (name: string, a: number, b: number, fmt: (n: number) => string) => {
  const d = b - a
  const sign = d === 0 ? '' : d > 0 ? '+' : '-'

  return `${name}: ${fmt(a)} → ${fmt(b)}  (${sign}${fmt(Math.abs(d)) || '0'})`
}

// ── Sub-components ───────────────────────────────────────────────────

/** Polled on parent `tick` so accordions can resize the thumb without a scroll event. */
function OverlayScrollbar({
  scrollRef,
  t,
  tick
}: {
  scrollRef: RefObject<null | ScrollBoxHandle>
  t: Theme
  tick: number
}) {
  void tick // ensures re-render when the parent clock advances

  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<null | number>(null)

  const s = scrollRef.current
  const vp = Math.max(0, s?.getViewportHeight() ?? 0)

  if (!vp) {
    return <Box width={1} />
  }

  const total = Math.max(vp, s?.getScrollHeight() ?? vp)
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const pos = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0
  const below = Math.max(0, vp - thumbTop - thumb)

  const vBar = (n: number) => (n > 0 ? `${'│\n'.repeat(n - 1)}│` : '')
  const thumbBody = `${'┃\n'.repeat(Math.max(0, thumb - 1))}┃`
  const thumbColor = grab !== null ? t.color.primary : t.color.accent
  const trackColor = hover ? t.color.border : t.color.muted

  const jump = (row: number, offset: number) => {
    if (!s || !scrollable) {
      return
    }

    s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)))
  }

  return (
    <Box
      flexDirection="column"
      onMouseDown={(e: { localRow?: number }) => {
        const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0))
        const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2)
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grab ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => setGrab(null)}
      width={1}
    >
      {!scrollable ? (
        <Text color={trackColor} dim>
          {vBar(vp)}
        </Text>
      ) : (
        <>
          {thumbTop > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {vBar(thumbTop)}
            </Text>
          ) : null}

          <Text color={thumbColor}>{thumbBody}</Text>

          {below > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {vBar(below)}
            </Text>
          ) : null}
        </>
      )}
    </Box>
  )
}

function GanttStrip({
  cols,
  cursor,
  flatNodes,
  maxRows,
  now,
  t
}: {
  cols: number
  cursor: number
  flatNodes: SubagentNode[]
  maxRows: number
  now: number
  t: Theme
}) {
  const spans = flatNodes
    .map((node, idx) => {
      const started = node.item.startedAt ?? now

      const ended =
        node.item.durationSeconds != null && node.item.startedAt != null
          ? node.item.startedAt + node.item.durationSeconds * 1000
          : now

      return { endAt: ended, idx, node, startAt: started }
    })
    .filter(s => s.endAt >= s.startAt)

  if (!spans.length) {
    return null
  }

  const globalStart = Math.min(...spans.map(s => s.startAt))
  const globalEnd = Math.max(...spans.map(s => s.endAt))
  const totalSpan = Math.max(1, globalEnd - globalStart)
  const totalSeconds = (globalEnd - globalStart) / 1000

  // 5-col id gutter ("  12  ") so the bar doesn't press against the id.
  // 10-col right reserve: pad + up to `12m 30s`-style label without
  // truncate-end against a full-width bar.
  const idGutter = 5
  const labelReserve = 10
  const barWidth = Math.max(10, cols - idGutter - labelReserve)
  const startIdx = Math.max(0, Math.min(Math.max(0, spans.length - maxRows), cursor - Math.floor(maxRows / 2)))
  const shown = spans.slice(startIdx, startIdx + maxRows)

  const bar = (startAt: number, endAt: number) => {
    const s = Math.floor(((startAt - globalStart) / totalSpan) * barWidth)
    const e = Math.min(barWidth, Math.ceil(((endAt - globalStart) / totalSpan) * barWidth))
    const fill = Math.max(1, e - s)

    return ' '.repeat(s) + '█'.repeat(fill) + ' '.repeat(Math.max(0, barWidth - s - fill))
  }

  const charStep = totalSeconds < 20 && barWidth > 20 ? 5 : 10

  const ruler = Array.from({ length: barWidth }, (_, i) => {
    if (i > 0 && i % 10 === 0) {
      return '┼'
    }

    if (i > 0 && i % 5 === 0) {
      return '·'
    }

    return '─'
  }).join('')

  const rulerLabels = (() => {
    const chars = new Array(barWidth).fill(' ')

    for (let pos = 0; pos < barWidth; pos += charStep) {
      const secs = (pos / barWidth) * totalSeconds
      const label = pos === 0 ? '0' : secs >= 1 ? `${Math.round(secs)}s` : `${secs.toFixed(1)}s`

      for (let j = 0; j < label.length && pos + j < barWidth; j++) {
        chars[pos + j] = label[j]!
      }
    }

    return chars.join('')
  })()

  const windowLabel =
    spans.length > maxRows ? `  (${startIdx + 1}-${Math.min(spans.length, startIdx + maxRows)}/${spans.length})` : ''

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color={t.color.muted}>
        Timeline · {fmtElapsedLabel(Math.max(0, totalSeconds))}
        {windowLabel}
      </Text>

      {shown.map(({ endAt, idx, node, startAt }) => {
        const active = idx === cursor
        const { color } = statusGlyph(node.item, t)
        const accent = active ? t.color.accent : t.color.muted

        const elSec = displayElapsedSeconds(node.item, now)
        const elLabel = elSec != null ? fmtElapsedLabel(elSec) : ''

        return (
          <Text key={node.item.id} wrap="truncate-end">
            <Text bold={active} color={accent}>
              {formatRowId(idx)}
              {'  '}
            </Text>

            <Text color={active ? t.color.accent : color}>{bar(startAt, endAt)}</Text>

            {elLabel ? (
              <Text color={accent}>
                {'   '}
                {elLabel}
              </Text>
            ) : null}
          </Text>
        )
      })}

      <Text color={t.color.muted} dim>
        {'    '}
        {ruler}
      </Text>

      {totalSeconds > 0 ? (
        <Text color={t.color.muted} dim>
          {'    '}
          {rulerLabels}
        </Text>
      ) : null}
    </Box>
  )
}

function OverlaySection({
  children,
  count,
  defaultOpen = false,
  title,
  t
}: {
  children: ReactNode
  count?: number
  defaultOpen?: boolean
  title: string
  t: Theme
}) {
  const openMap = useStore($overlaySectionsOpen)
  const open = title in openMap ? openMap[title]! : defaultOpen

  return (
    <Box flexDirection="column" marginTop={1}>
      <Box onClick={() => toggleOverlaySection(title, defaultOpen)}>
        <Text color={t.color.label}>
          <Text color={t.color.accent}>{open ? '▾ ' : '▸ '}</Text>
          {title}
          {typeof count === 'number' ? ` (${count})` : ''}
        </Text>
      </Box>

      {open ? <Box flexDirection="column">{children}</Box> : null}
    </Box>
  )
}

function Field({ name, t, value }: { name: string; t: Theme; value: ReactNode }) {
  return (
    <Text wrap="truncate-end">
      <Text color={t.color.label}>{name} · </Text>
      <Text color={t.color.text}>{value}</Text>
    </Text>
  )
}

function Detail({ id, node, t }: { id?: string; node: SubagentNode; t: Theme }) {
  const { aggregate: agg, item } = node
  const { color, glyph } = statusGlyph(item, t)

  const inputTokens = item.inputTokens ?? 0
  const outputTokens = item.outputTokens ?? 0
  const localTokens = inputTokens + outputTokens
  const subtreeTokens = agg.inputTokens + agg.outputTokens - localTokens
  const localCost = item.costUsd ?? 0
  const subtreeCost = agg.costUsd - localCost

  const filesRead = item.filesRead ?? []
  const filesWritten = item.filesWritten ?? []
  const outputTail = item.outputTail ?? []
  // Tool calls: prefer the live stream; for archived / post-turn views
  // that stream is often empty even when tool_count > 0, so fall back to
  // the tool names captured in outputTail at subagent.complete time.
  const toolLines = item.tools.length > 0 ? item.tools : outputTail.map(e => e.tool).filter(Boolean)

  const filesOverflow = Math.max(0, filesRead.length - 8) + Math.max(0, filesWritten.length - 8)

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.text} wrap="wrap">
        {id ? <Text color={t.color.accent}>#{id} </Text> : null}
        <Text color={color}>{glyph}</Text> {item.goal}
      </Text>

      <Box flexDirection="column" marginTop={1}>
        <Field name="depth" t={t} value={`${item.depth} · ${item.status}${item.paused ? ' · ⏸ paused' : ''}`} />
        {item.model ? <Field name="model" t={t} value={item.model} /> : null}
        {item.sessionId ? <Field name="session" t={t} value={item.sessionId} /> : null}
        {item.toolsets?.length ? <Field name="toolsets" t={t} value={item.toolsets.join(', ')} /> : null}
        <Field name="tools" t={t} value={`${item.toolCount ?? 0} (subtree ${agg.totalTools})`} />
        <Field
          name="subtree"
          t={t}
          value={`${agg.descendantCount} agent${agg.descendantCount === 1 ? '' : 's'} · d${agg.maxDepthFromHere} · ⚡${agg.activeCount}`}
        />
        {item.durationSeconds ? <Field name="elapsed" t={t} value={fmtDur(item.durationSeconds)} /> : null}
        {item.iteration != null ? <Field name="iteration" t={t} value={String(item.iteration)} /> : null}
        {item.apiCalls ? <Field name="api calls" t={t} value={String(item.apiCalls)} /> : null}
      </Box>

      {localTokens > 0 || localCost > 0 ? (
        <OverlaySection defaultOpen t={t} title="Budget">
          {localTokens > 0 ? (
            <Field
              name="tokens"
              t={t}
              value={
                <>
                  {fmtTokens(inputTokens)} in · {fmtTokens(outputTokens)} out
                  {item.reasoningTokens ? ` · ${fmtTokens(item.reasoningTokens)} reasoning` : ''}
                </>
              }
            />
          ) : null}

          {localCost > 0 ? (
            <Field
              name="cost"
              t={t}
              value={
                <>
                  {fmtCost(localCost)}
                  {subtreeCost >= 0.01 ? ` · subtree +${fmtCost(subtreeCost)}` : ''}
                </>
              }
            />
          ) : null}

          {subtreeTokens > 0 ? <Field name="subtree tokens" t={t} value={`+${fmtTokens(subtreeTokens)}`} /> : null}
        </OverlaySection>
      ) : null}

      {filesRead.length > 0 || filesWritten.length > 0 ? (
        <OverlaySection count={filesRead.length + filesWritten.length} t={t} title="Files">
          {filesWritten.slice(0, 8).map((p, i) => (
            <Text color={t.color.statusGood} key={`w-${i}`} wrap="truncate-end">
              +{p}
            </Text>
          ))}

          {filesRead.slice(0, 8).map((p, i) => (
            <Text color={t.color.text} key={`r-${i}`} wrap="truncate-end">
              <Text color={t.color.muted}>·</Text> {p}
            </Text>
          ))}

          {filesOverflow > 0 ? <Text color={t.color.muted}>…+{filesOverflow} more</Text> : null}
        </OverlaySection>
      ) : null}

      {toolLines.length > 0 ? (
        <OverlaySection count={toolLines.length} defaultOpen t={t} title="Tool calls">
          {toolLines.map((line, i) => (
            <Text color={t.color.text} key={i} wrap="wrap">
              <Text color={t.color.muted}>·</Text> {line}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {outputTail.length > 0 ? (
        <OverlaySection count={outputTail.length} defaultOpen t={t} title="Output">
          {outputTail.map((entry, i) => (
            <Text color={entry.isError ? t.color.error : t.color.text} key={i} wrap="wrap">
              <Text bold color={entry.isError ? t.color.error : t.color.accent}>
                {entry.tool}
              </Text>{' '}
              {entry.preview}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {item.notes.length ? (
        <OverlaySection count={item.notes.length} t={t} title="Progress">
          {item.notes.slice(-6).map((line, i) => (
            <Text color={t.color.text} key={i} wrap="wrap">
              <Text color={t.color.label}>·</Text> {line}
            </Text>
          ))}
        </OverlaySection>
      ) : null}

      {item.summary ? (
        <OverlaySection defaultOpen t={t} title="Summary">
          <Text color={t.color.text} wrap="wrap">
            {item.summary}
          </Text>
        </OverlaySection>
      ) : null}
    </Box>
  )
}

function ListRow({
  active,
  index,
  node,
  peak,
  t,
  width
}: {
  active: boolean
  index: number
  node: SubagentNode
  peak: number
  t: Theme
  width: number
}) {
  const { color, glyph } = statusGlyph(node.item, t)
  const palette = heatPalette(t)
  const heatIdx = hotnessBucket(node.aggregate.hotness, peak, palette.length)
  const heatMarker = heatIdx >= 2 ? palette[heatIdx]! : null

  const goal = compactPreview(node.item.goal || 'subagent', width - 28 - node.item.depth * 2)
  const toolsCount = node.aggregate.totalTools > 0 ? ` ·${node.aggregate.totalTools}t` : ''
  const kids = node.children.length ? ` ·${node.children.length}↓` : ''
  const line = node.item.status === 'running' ? node.item.tools.at(-1) : undefined
  const paren = line ? line.indexOf('(') : -1
  const toolShort = line ? (paren > 0 ? line.slice(0, paren) : line).trim() : ''
  const trailing = toolShort ? ` · ${compactPreview(toolShort, 14)}` : ''
  const fg = active ? t.color.accent : t.color.text

  return (
    <Text bold={active} color={fg} inverse={active} wrap="truncate-end">
      {' '}
      <Text color={active ? fg : t.color.muted}>{formatRowId(index)} </Text>
      {indentFor(node.item.depth)}
      {heatMarker ? <Text color={heatMarker}>▍</Text> : null}
      <Text color={active ? fg : color}>{glyph}</Text> {goal}
      <Text color={active ? fg : t.color.muted}>
        {toolsCount}
        {kids}
        {trailing}
      </Text>
    </Text>
  )
}

function DiffPane({
  label,
  snapshot,
  t,
  totals,
  width
}: {
  label: string
  snapshot: SpawnSnapshot
  t: Theme
  totals: ReturnType<typeof treeTotals>
  width: number
}) {
  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.text}>
        {label}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {snapshot.label}
      </Text>

      <Box marginTop={1}>
        <Text color={t.color.muted} wrap="truncate-end">
          {formatSummary(totals)}
        </Text>
      </Box>

      <Box flexDirection="column" marginTop={1}>
        {topLevelSubagents(snapshot.subagents)
          .slice(0, 8)
          .map(s => {
            const { color, glyph } = statusGlyph(s, t)

            return (
              <Text color={t.color.muted} key={s.id} wrap="truncate-end">
                <Text color={color}>{glyph}</Text> {s.goal || 'subagent'}
              </Text>
            )
          })}
      </Box>
    </Box>
  )
}

function DiffView({
  cols,
  onClose,
  pair,
  t
}: {
  cols: number
  onClose: () => void
  pair: { baseline: SpawnSnapshot; candidate: SpawnSnapshot }
  t: Theme
}) {
  const aTotals = useMemo(() => treeTotals(buildSubagentTree(pair.baseline.subagents)), [pair.baseline])
  const bTotals = useMemo(() => treeTotals(buildSubagentTree(pair.candidate.subagents)), [pair.candidate])
  const paneWidth = Math.floor((cols - 4) / 2)

  useInput((ch, key) => {
    if (key.escape || ch === 'q') {
      onClose()
    }
  })

  const round = (n: number) => String(Math.round(n))
  const sumTokens = (x: typeof aTotals) => x.inputTokens + x.outputTokens
  const dollars = (n: number) => fmtCost(n) || '$0.00'

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
      <Box flexDirection="column" marginBottom={1}>
        <Text bold color={t.color.border}>
          Replay diff
        </Text>
        <Text color={t.color.muted}>baseline vs candidate · esc/q close</Text>
      </Box>

      <Box flexDirection="row" marginBottom={1}>
        <DiffPane label="A · baseline" snapshot={pair.baseline} t={t} totals={aTotals} width={paneWidth} />
        <Box width={2} />
        <DiffPane label="B · candidate" snapshot={pair.candidate} t={t} totals={bTotals} width={paneWidth} />
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color={t.color.accent}>
          Δ
        </Text>

        <Text color={t.color.text}>
          {diffMetricLine('agents', aTotals.descendantCount, bTotals.descendantCount, round)}
        </Text>
        <Text color={t.color.text}>{diffMetricLine('tools', aTotals.totalTools, bTotals.totalTools, round)}</Text>
        <Text color={t.color.text}>
          {diffMetricLine('depth', aTotals.maxDepthFromHere, bTotals.maxDepthFromHere, round)}
        </Text>
        <Text color={t.color.text}>
          {diffMetricLine('duration', aTotals.totalDuration, bTotals.totalDuration, n => `${n.toFixed(1)}s`)}
        </Text>
        <Text color={t.color.text}>{diffMetricLine('tokens', sumTokens(aTotals), sumTokens(bTotals), fmtTokens)}</Text>
        <Text color={t.color.text}>{diffMetricLine('cost', aTotals.costUsd, bTotals.costUsd, dollars)}</Text>
      </Box>
    </Box>
  )
}

// ── Main overlay ─────────────────────────────────────────────────────

export function AgentsOverlay({ gw, initialHistoryIndex = 0, onClose, t }: AgentsOverlayProps) {
  const liveSubagents = useTurnSelector(state => state.subagents)
  const delegation = useStore($delegationState)
  const history = useStore($spawnHistory)
  const diffPair = useStore($spawnDiff)
  const { stdout } = useStdout()

  // historyIndex === 0: live turn.  1..N pulls the Nth-most-recent archived
  // snapshot.  /replay passes N on open.
  const [historyIndex, setHistoryIndex] = useState(() =>
    Math.max(0, Math.min(history.length, Math.floor(initialHistoryIndex)))
  )

  const [sort, setSort] = useState<SortMode>('depth-first')
  const [filter, setFilter] = useState<FilterMode>('all')
  const [cursor, setCursor] = useState(0)
  const [flash, setFlash] = useState<string>('')
  const [now, setNow] = useState(() => Date.now())
  // cc-style view switching: list = full-width row picker, detail = full-width
  // scrollable pane.  Two panes side-by-side in Ink fought Yoga flex.
  const [mode, setMode] = useState<'detail' | 'list'>('list')
  // Steer/interrupt-message compose box. When set, keystrokes build the text
  // and Enter sends it to the selected subagent via the matching RPC.
  const [compose, setCompose] = useState<null | { kind: 'interrupt' | 'steer'; text: string }>(null)
  // On-demand history for the selected subagent, fetched from its OWN session
  // (session.history RPC by session_id) rather than the parent relaying a heavy
  // live stream. Keyed by session_id so switching agents refetches.
  const [history2, setHistory2] = useState<null | { messages: HistoryMsg[]; sessionId: string }>(null)

  const detailScrollRef = useRef<null | ScrollBoxHandle>(null)
  const prevLiveCountRef = useRef(liveSubagents.length)

  // ── Derived state ──────────────────────────────────────────────────

  const activeSnapshot = historyIndex > 0 ? history[historyIndex - 1] : null
  // Instant fallback to history[0] the moment the live list clears — avoids
  // a one-frame "no subagents" flash while the auto-follow effect fires.
  const justFinishedSnapshot = historyIndex === 0 && liveSubagents.length === 0 ? (history[0] ?? null) : null
  const effectiveSnapshot = activeSnapshot ?? justFinishedSnapshot
  const replayMode = effectiveSnapshot != null
  const subagents = replayMode ? effectiveSnapshot.subagents : liveSubagents

  const tree = useMemo(() => buildSubagentTree(subagents), [subagents])
  const totals = useMemo(() => treeTotals(tree), [tree])
  const widths = useMemo(() => widthByDepth(tree), [tree])
  const spark = useMemo(() => sparkline(widths), [widths])
  const peak = useMemo(() => peakHotness(tree), [tree])
  const rows = useMemo(() => prepareRows(tree, sort, filter), [tree, sort, filter])

  const selected = rows[cursor] ?? null

  const cols = stdout?.columns ?? 80
  const rowsH = Math.max(8, (stdout?.rows ?? 24) - 10)
  const listWindowStart = Math.max(0, cursor - Math.floor(rowsH / 2))

  // ── Effects ────────────────────────────────────────────────────────

  useEffect(() => {
    // Ticker drives both the live gantt and OverlayScrollbar content-reflow
    // detection.  Slower in replay (nothing's growing) but not stopped
    // because accordions still expand.
    const id = setInterval(() => setNow(Date.now()), replayMode ? 300 : 500)

    return () => clearInterval(id)
  }, [replayMode])

  useEffect(() => {
    // Clamp stale index when history grows/shrinks beneath us.
    if (historyIndex > history.length) {
      setHistoryIndex(history.length)
    }
  }, [history.length, historyIndex])

  useEffect(() => {
    // Auto-follow the just-finished turn onto history[1] so the user isn't
    // dropped into an empty live view.  Fires only when transitioning from
    // "had live subagents" → "live empty" while in live mode.
    const prev = prevLiveCountRef.current
    prevLiveCountRef.current = liveSubagents.length

    if (historyIndex === 0 && prev > 0 && liveSubagents.length === 0 && history.length > 0) {
      setHistoryIndex(1)
      setCursor(0)
      setFlash('turn finished · inspect freely · q to close')
    }
  }, [history.length, historyIndex, liveSubagents.length])

  useEffect(() => {
    // Reset detail scroll on navigation so the top of the new node shows.
    detailScrollRef.current?.scrollTo(0)
  }, [cursor, historyIndex, mode])

  useEffect(() => {
    // Warm caps + paused flag on open.
    gw.request<DelegationStatusResponse>('delegation.status', {})
      .then(r => applyDelegationStatus(asRpcResult<DelegationStatusResponse>(r)))
      .catch(() => {})
  }, [gw])

  useEffect(() => {
    if (cursor >= rows.length) {
      setCursor(Math.max(0, rows.length - 1))
    }
  }, [cursor, rows.length])

  // ── Actions ────────────────────────────────────────────────────────

  const guardLive = (action: () => void) => {
    if (replayMode) {
      setFlash('replay mode — controls disabled')
    } else {
      action()
    }
  }

  const isRunning = (item: SubagentProgress) => item.status === 'running'

  // ── Soft kill (graceful stop at next boundary) ─────────────────────
  const softKillOne = (item: SubagentProgress) =>
    guardLive(() => {
      if (!isRunning(item)) {
        return setFlash(`${item.id} already ${item.status} — nothing to kill`)
      }
      gw.request<SubagentControlResponse>('subagent.soft_kill', { subagent_id: item.id })
        .then(raw => {
          const r = asRpcResult<SubagentControlResponse>(raw)
          setFlash(r?.found ? `soft-kill sent · ${item.id} (stops at next step)` : `not running: ${item.id}`)
        })
        .catch(() => setFlash(`soft-kill failed: ${item.id}`))
    })

  const softKillAll = () =>
    guardLive(() => {
      gw.request<SubagentKillAllResponse>('subagent.soft_kill_all', {})
        .then(raw => {
          const r = asRpcResult<SubagentKillAllResponse>(raw)
          setFlash(`soft-kill sent to ${r?.count ?? 0} running agent${r?.count === 1 ? '' : 's'}`)
        })
        .catch(() => setFlash('soft-kill-all failed'))
    })

  // ── Hard kill (interrupt + terminate the child's own tool procs NOW) ─
  const hardKillOne = (item: SubagentProgress) =>
    guardLive(() => {
      if (!isRunning(item)) {
        return setFlash(`${item.id} already ${item.status} — nothing to kill`)
      }
      gw.request<SubagentControlResponse>('subagent.hard_kill', { subagent_id: item.id })
        .then(raw => {
          const r = asRpcResult<SubagentControlResponse>(raw)
          if (!r?.found) {
            return setFlash(`not running: ${item.id}`)
          }
          setFlash(`HARD-kill ${item.id} · ${r.procs_killed ?? 0} proc${r.procs_killed === 1 ? '' : 's'} terminated`)
        })
        .catch(() => setFlash(`hard-kill failed: ${item.id}`))
    })

  const hardKillAll = () =>
    guardLive(() => {
      gw.request<SubagentKillAllResponse>('subagent.hard_kill_all', {})
        .then(raw => {
          const r = asRpcResult<SubagentKillAllResponse>(raw)
          setFlash(`HARD-kill ${r?.count ?? 0} agent${r?.count === 1 ? '' : 's'} · ${r?.procs_killed ?? 0} procs`)
        })
        .catch(() => setFlash('hard-kill-all failed'))
    })

  // legacy subtree kill (kept for the X keybind) — cooperative interrupt only
  const interrupt = (id: string) => gw.request<SubagentInterruptResponse>('subagent.interrupt', { subagent_id: id })
  const killSubtree = (node: SubagentNode) =>
    guardLive(() => {
      const ids = [node.item.id, ...descendantIds(node)]
      ids.forEach(id => interrupt(id).catch(() => {}))
      setFlash(`interrupting subtree · ${ids.length} node${ids.length === 1 ? '' : 's'}`)
    })

  // ── Steer / interrupt-with-message (compose box drives these) ──────
  const sendCompose = (kind: 'interrupt' | 'steer', item: SubagentProgress, text: string) =>
    guardLive(() => {
      if (!text.trim()) {
        return setFlash('empty message — nothing sent')
      }
      if (!isRunning(item)) {
        return setFlash(`${item.id} not running — can't ${kind}`)
      }
      const method = kind === 'steer' ? 'subagent.steer' : 'subagent.interrupt_message'
      gw.request<SubagentControlResponse>(method, { subagent_id: item.id, text })
        .then(raw => {
          const r = asRpcResult<SubagentControlResponse>(raw)
          if (!r?.delivered) {
            return setFlash(`${kind} not delivered: ${item.id}`)
          }
          setFlash(kind === 'steer' ? `steer sent → ${item.id} (before next tool)` : `interrupt+msg → ${item.id} (refocus & continue)`)
        })
        .catch(() => setFlash(`${kind} failed: ${item.id}`))
    })

  // ── Per-subagent pause / resume (distinct from global spawn pause) ──
  const togglePauseOne = (item: SubagentProgress) =>
    guardLive(() => {
      if (!isRunning(item)) {
        return setFlash(`${item.id} not running`)
      }
      const method = item.paused ? 'subagent.resume' : 'subagent.pause'
      gw.request<SubagentControlResponse>(method, { subagent_id: item.id })
        .then(raw => {
          const r = asRpcResult<SubagentControlResponse>(raw)
          const ok = item.paused ? r?.resumed : r?.paused
          setFlash(ok ? (item.paused ? `resumed ${item.id}` : `paused ${item.id} (holds at next step)`) : `pause toggle failed: ${item.id}`)
        })
        .catch(() => setFlash(`pause toggle failed: ${item.id}`))
    })

  // ── On-demand history from the child's OWN session ─────────────────
  const loadHistory = (item: SubagentProgress) => {
    const sid = item.sessionId
    if (!sid) {
      return setFlash('no session id for this agent yet')
    }
    setFlash(`loading history · ${sid}`)
    gw.request<SessionHistoryResponse>('session.history', { session_id: sid })
      .then(raw => {
        const r = asRpcResult<SessionHistoryResponse>(raw)
        const msgs = r?.messages ?? []
        setHistory2({ messages: msgs, sessionId: sid })
        setFlash(`history · ${msgs.length} message${msgs.length === 1 ? '' : 's'}`)
      })
      .catch(() => setFlash(`history load failed: ${sid}`))
  }

  // ── Reload the selected child's live view + its history snapshot ────
  // A subagent has no renderer of its own to recycle (it's a worker under the
  // durable gateway), so the subagent-scope "reload" just REFRESHES the
  // snapshot data the overlay shows: re-fetch delegation.status (updates every
  // row's status/session_id/paused) and re-pull this child's history if the
  // history pane is open. It never touches the child's actual run.
  const reloadSelected = (item: SubagentProgress) => {
    gw.request<DelegationStatusResponse>('delegation.status', {})
      .then(raw => applyDelegationStatus(asRpcResult<DelegationStatusResponse>(raw)))
      .catch(() => {})
    if (history2 && item.sessionId && history2.sessionId === item.sessionId) {
      loadHistory(item) // refresh the open history pane from the live session
    } else {
      setFlash(`reloaded ${item.id} view`)
    }
  }

  const togglePause = () =>
    guardLive(() => {
      gw.request<DelegationPauseResponse>('delegation.pause', { paused: !delegation.paused })
        .then(raw => {
          const r = asRpcResult<DelegationPauseResponse>(raw)
          applyDelegationStatus({ paused: r?.paused })
          setFlash(r?.paused ? 'spawning paused' : 'spawning resumed')
        })
        .catch(() => setFlash('pause failed'))
    })

  const stepHistory = (delta: -1 | 1) =>
    setHistoryIndex(idx => {
      const next = Math.max(0, Math.min(history.length, idx + delta))

      if (next !== idx) {
        setCursor(0)
        setFlash(next === 0 ? 'live turn' : `replay · ${next}/${history.length}`)
      }

      return next
    })

  const closeWithCleanup = () => {
    clearDiffPair()
    onClose()
  }

  // ── Input ──────────────────────────────────────────────────────────

  const detailPageSize = Math.max(4, rowsH - 2)
  const wheelDetailDy = 3
  const scrollDetail = (dy: number) => detailScrollRef.current?.scrollBy(dy)

  useInput((ch, key) => {
    // ── Compose mode: building a steer / interrupt message ───────────
    // While composing, keystrokes edit the buffer; Enter sends, Esc cancels.
    // Intercept FIRST so control keys below don't fire mid-typing.
    if (compose) {
      if (key.escape) {
        setCompose(null)
        return setFlash('cancelled')
      }
      if (key.return) {
        const target = selected?.item
        const { kind, text } = compose
        setCompose(null)
        if (target) {
          sendCompose(kind, target, text)
        }
        return
      }
      if (key.backspace || key.delete) {
        return setCompose(c => (c ? { ...c, text: c.text.slice(0, -1) } : c))
      }
      if (ch && !key.ctrl && !key.meta) {
        return setCompose(c => (c ? { ...c, text: c.text + ch } : c))
      }
      return
    }

    if (ch === 'q') {
      return closeWithCleanup()
    }

    if (key.escape) {
      if (history2) {
        setHistory2(null)
        return setFlash('history closed')
      }
      return mode === 'detail' ? setMode('list') : closeWithCleanup()
    }

    // Shared actions (both modes).
    if (ch === '<' || ch === '[') {
      return stepHistory(1)
    }

    if (ch === '>' || ch === ']') {
      return stepHistory(-1)
    }

    // ── Per-subagent controls (selected, running-only enforced in fns) ─
    if (ch === 'x' && selected) {
      return softKillOne(selected.item) // soft kill (graceful)
    }

    if (ch === 'K' && selected) {
      return hardKillOne(selected.item) // hard kill NOW (interrupt + procs)
    }

    if (ch === 'X' && selected) {
      return killSubtree(selected) // cooperative interrupt of the subtree
    }

    // Kill-ALL running (completed untouched).
    if (ch === 'a') {
      return softKillAll()
    }
    if (ch === 'A') {
      return hardKillAll()
    }

    // Steer (before next tool) / interrupt-with-message (refocus & continue).
    if (ch === 't' && selected) {
      guardLive(() => setCompose({ kind: 'steer', text: '' }))
      return
    }
    if (ch === 'i' && selected) {
      guardLive(() => setCompose({ kind: 'interrupt', text: '' }))
      return
    }

    // Per-subagent pause/resume (distinct from global spawn pause = 'p').
    if (ch === 'P' && selected) {
      return togglePauseOne(selected.item)
    }

    // Load the selected child's own-session history (scrollable).
    if (ch === 'H' && selected) {
      return loadHistory(selected.item)
    }

    // Reload the selected child's view (refresh its snapshot data; the
    // subagent's actual run is untouched). Subagent-scope analogue of the
    // main session's Ctrl+R renderer reload.
    if (ch === 'R' && selected) {
      return reloadSelected(selected.item)
    }

    if (ch === 'p') {
      return togglePause() // global: pause NEW spawns
    }

    if (mode === 'detail') {
      if (key.leftArrow || ch === 'h') {
        return setMode('list')
      }

      if (key.pageUp || (key.ctrl && ch === 'u')) {
        return scrollDetail(-detailPageSize)
      }

      if (key.pageDown || (key.ctrl && ch === 'd')) {
        return scrollDetail(detailPageSize)
      }

      if (key.wheelUp) {
        return scrollDetail(-wheelDetailDy)
      }

      if (key.wheelDown) {
        return scrollDetail(wheelDetailDy)
      }

      if (key.upArrow || ch === 'k') {
        return scrollDetail(-2)
      }

      if (key.downArrow || ch === 'j') {
        return scrollDetail(2)
      }

      if (ch === 'g') {
        return detailScrollRef.current?.scrollTo(0)
      }

      if (ch === 'G') {
        return detailScrollRef.current?.scrollToBottom?.()
      }

      return
    }

    // List mode.
    if ((key.return || key.rightArrow || ch === 'l') && selected) {
      return setMode('detail')
    }

    if (key.upArrow || ch === 'k' || key.wheelUp) {
      return setCursor(c => Math.max(0, c - 1))
    }

    if (key.downArrow || ch === 'j' || key.wheelDown) {
      return setCursor(c => Math.min(Math.max(0, rows.length - 1), c + 1))
    }

    if (ch === 'g') {
      return setCursor(0)
    }

    if (ch === 'G') {
      return setCursor(Math.max(0, rows.length - 1))
    }

    if (ch === 's') {
      return setSort(m => cycle(SORT_ORDER, m))
    }

    if (ch === 'f') {
      return setFilter(m => cycle(FILTER_ORDER, m))
    }
  })

  // ── Header assembly ────────────────────────────────────────────────

  const mix = Object.entries(
    subagents.reduce<Record<string, number>>((acc, it) => {
      const key = it.model ? it.model.split('/').pop()! : 'inherit'
      acc[key] = (acc[key] ?? 0) + 1

      return acc
    }, {})
  )
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k, v]) => `${k}×${v}`)
    .join(' · ')

  const capsLabel = delegation.maxSpawnDepth
    ? `caps d${delegation.maxSpawnDepth}/${delegation.maxConcurrentChildren ?? '?'}`
    : ''

  const title =
    replayMode && effectiveSnapshot
      ? `${historyIndex > 0 ? `Replay ${historyIndex}/${history.length}` : 'Last turn'} · finished ${new Date(
          effectiveSnapshot.finishedAt
        ).toLocaleTimeString()}`
      : `Spawn tree${delegation.paused ? ' · ⏸ paused' : ''}`

  const metaLine = [formatSummary(totals), spark, capsLabel, mix ? `· ${mix}` : ''].filter(Boolean).join('  ')

  const controlsHint = replayMode
    ? ' · controls locked'
    : ` · x soft-kill · K hard-kill · a/A kill-all · t steer · i intr+msg · P ${selected?.item.paused ? 'resume' : 'pause'} · H history · R reload · X subtree · p ${delegation.paused ? 'resume' : 'pause'}-spawns`

  // ── Rendering ──────────────────────────────────────────────────────

  if (diffPair) {
    return <DiffView cols={cols} onClose={closeWithCleanup} pair={diffPair} t={t} />
  }

  return (
    <Box alignItems="stretch" flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
      <Box flexDirection="column" marginBottom={1}>
        <Text wrap="truncate-end">
          <Text bold color={replayMode ? t.color.border : t.color.primary}>
            {title}
          </Text>
          {metaLine ? (
            <Text color={t.color.muted}>
              {'   '}
              {metaLine}
            </Text>
          ) : null}
        </Text>
      </Box>

      {rows.length === 0 ? (
        <Box flexDirection="column" flexGrow={1}>
          <Text color={t.color.muted}>No subagents this turn. Trigger delegate_task to populate the tree.</Text>
        </Box>
      ) : mode === 'list' ? (
        <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0}>
          <GanttStrip cols={cols} cursor={cursor} flatNodes={rows} maxRows={6} now={now} t={t} />

          <Box flexDirection="column" flexGrow={0} flexShrink={0} overflow="hidden">
            {rows.slice(listWindowStart, listWindowStart + rowsH).map((node, i) => (
              <ListRow
                active={listWindowStart + i === cursor}
                index={listWindowStart + i}
                key={node.item.id}
                node={node}
                peak={peak}
                t={t}
                width={cols}
              />
            ))}
          </Box>
        </Box>
      ) : (
        <Box flexDirection="row" flexGrow={1} flexShrink={1} minHeight={0}>
          <ScrollBox flexDirection="column" flexGrow={1} flexShrink={1} ref={detailScrollRef}>
            <Box flexDirection="column" paddingBottom={4} paddingRight={1}>
              {selected ? <Detail id={formatRowId(cursor).trim()} node={selected} t={t} /> : null}
            </Box>
          </ScrollBox>

          <NoSelect flexShrink={0} marginLeft={1}>
            <OverlayScrollbar scrollRef={detailScrollRef} t={t} tick={now} />
          </NoSelect>
        </Box>
      )}

      {history2 ? (
        <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
          <Text bold color={t.color.primary}>
            History · {history2.sessionId}{' '}
            <Text color={t.color.muted}>({history2.messages.length} msg · Esc to close)</Text>
          </Text>
          {history2.messages.slice(-40).map((m, i) => {
            const body = (m.text ?? m.content ?? m.preview ?? '').toString().replace(/\s+/g, ' ').slice(0, 200)
            const roleColor =
              m.role === 'user' ? t.color.accent : m.role === 'assistant' ? t.color.primary : t.color.muted
            return (
              <Text key={i} wrap="truncate-end">
                <Text bold color={roleColor}>
                  {(m.role ?? '?').padEnd(9)}
                </Text>
                <Text color={t.color.text}>{body}</Text>
              </Text>
            )
          })}
        </Box>
      ) : null}

      {compose ? (
        <Box borderColor={t.color.accent} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
          <Text color={t.color.muted}>
            {compose.kind === 'steer'
              ? `steer → ${selected?.item.id ?? '?'} (delivered before its next tool run · Enter send · Esc cancel)`
              : `interrupt+msg → ${selected?.item.id ?? '?'} (aborts current tool, refocuses, continues · Enter send · Esc cancel)`}
          </Text>
          <Text color={t.color.text}>
            {'> '}
            {compose.text}
            <Text color={t.color.accent}>▏</Text>
          </Text>
        </Box>
      ) : null}

      <Box flexDirection="column" marginTop={1}>
        {flash ? <Text color={t.color.accent}>{flash}</Text> : null}

        {mode === 'list' ? (
          <Text color={t.color.muted}>
            ↑↓/jk move · g/G top/bottom · Enter/→ open detail{controlsHint} · s sort:{SORT_LABEL[sort]} · f filter:
            {FILTER_LABEL[filter]}
            {history.length > 0 ? ` · [ / ] history ${historyIndex}/${history.length}` : ''}
            {' · q close'}
          </Text>
        ) : (
          <Text color={t.color.muted}>
            ↑↓/jk scroll · PgUp/PgDn page · g/G top/bottom · Esc/← back to list{controlsHint} · q close
          </Text>
        )}
      </Box>
    </Box>
  )
}

interface AgentsOverlayProps {
  gw: GatewayClient
  initialHistoryIndex?: number
  onClose: () => void
  t: Theme
}

export const closeAgentsOverlay = () => patchOverlayState({ agents: false })
export const openAgentsOverlay = () => patchOverlayState({ agents: true })
