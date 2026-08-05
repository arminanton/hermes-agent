export const PASTE_SNIPPET_RE = /\[\[[^\n]*?\]\]/g

/** A collapsed large-paste snippet: the `[[ … [N lines] … ]]` placeholder
 *  token shown in the composer, plus the full original text (and the optional
 *  backing file written by the `paste.collapse` RPC). */
export interface PasteSnip {
  label: string
  path?: string
  text: string
}

/**
 * Build a function that re-expands collapsed paste placeholders back to their
 * full original text. A large paste is shown in the composer as a short
 * `[[ … [N lines] … ]]` token (see pasteTokenLabel) while the full body is held
 * in `snips`. Every submission path that delivers composer text to the agent
 * MUST run it through this expander first, otherwise the agent receives only
 * the placeholder, not the content. The normal Enter-submit path does this; the
 * steer paths historically did NOT, so a large paste sent via /steer or
 * busy-mode steer arrived truncated to its placeholder.
 *
 * Matching is by exact label, FIFO per label, so repeated identical pastes each
 * resolve to their own body in order. Unknown tokens pass through unchanged.
 */
export const buildSnipExpander = (snips: readonly PasteSnip[]) => {
  const byLabel = new Map<string, string[]>()

  for (const { label, text } of snips) {
    const hit = byLabel.get(label)
    hit ? hit.push(text) : byLabel.set(label, [text])
  }

  return (value: string) => value.replace(PASTE_SNIPPET_RE, tok => byLabel.get(tok)?.shift() ?? tok)
}
