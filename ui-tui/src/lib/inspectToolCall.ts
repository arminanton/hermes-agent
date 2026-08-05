import { patchOverlayState } from '../app/overlayStore.js'

// Opening a tool call's full content in the shared pager overlay. The pager
// (appOverlays.tsx) is already a full-screen scrollable popup with title +
// nav footer (↑↓/jk/PgDn/g/G/Esc), so "inspect" just feeds it the untruncated
// content. Called from a tool-row click (thinking.tsx) and from the /inspect
// slash command (mouse-off fallback), so it lives in a standalone module that
// both can import without a dependency cycle.
//
// NOTE on completeness: for a LIVE (in-flight) tool the full raw args/context
// are shown. For a PERSISTED trail row only the captured trail line survives
// (args/result are folded in, capped at VERBOSE_TRAIL_MAX_CHARS=800 /12 lines
// by the OOM guard in text.ts, #34095) — we surface whatever was retained.
export function inspectToolCall(title: string, body: string): void {
  const text = (body ?? '').replace(/\r\n/g, '\n')
  const lines = text.length ? text.split('\n') : ['(no content captured for this tool call)']

  patchOverlayState({ pager: { lines, offset: 0, title } })
}
