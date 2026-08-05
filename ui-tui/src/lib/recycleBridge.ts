/**
 * Recycle bridge (Stage 1).
 *
 * The memory monitor lives in entry.tsx (module scope, before React mounts);
 * the live sid + scroll handle live inside the React app. To let a sustained-
 * pressure signal trigger a SEAMLESS renderer recycle, the app registers a
 * recycle callback here and entry.tsx invokes it.
 *
 * A recycle is ONLY safe when the renderer is a disposable client of a durable
 * gateway — i.e. running in attach mode under the orchestrator
 * (HERMES_TUI_GATEWAY_URL set). In spawned-gateway mode the renderer OWNS the
 * gateway, so exiting would kill the session — there we must NOT recycle, only
 * warn (today's behaviour). `canRecycle()` encodes that guard.
 */

let _recycle: (() => void) | null = null

/**
 * Exit code the renderer uses for a DELIBERATE recycle (Ctrl+R or the
 * memory-pressure auto-recycle). MUST match RECYCLE_EXIT_CODE in the Python
 * orchestrator (tui_gateway/orchestrator.py). It must NOT be 0: the normal
 * `die` quit path exits 0 and the orchestrator tears the session down on 0.
 * On this code the supervisor respawns a fresh renderer that re-attaches to
 * the still-live gateway and resumes the session — the seamless recycle.
 */
export const RECYCLE_EXIT_CODE = 75

/** The app registers its recycle implementation once on mount. */
export function registerRecycleHandler(fn: () => void): () => void {
  _recycle = fn
  return () => {
    if (_recycle === fn) {
      _recycle = null
    }
  }
}

/**
 * True only when a recycle is safe: the renderer is an attached client of a
 * durable gateway (orchestrator/attach mode). In spawned-gateway mode the
 * renderer exiting would take the session with it, so recycle is unsafe.
 */
export function canRecycle(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.HERMES_TUI_GATEWAY_URL?.trim())
}

/**
 * Trigger a recycle if one is registered AND safe. Returns true when a recycle
 * was actually initiated; false when it was skipped (no handler, or unsafe
 * spawned-gateway mode) so the caller can fall back to a warning.
 */
export function triggerRecycle(env: NodeJS.ProcessEnv = process.env): boolean {
  if (!_recycle || !canRecycle(env)) {
    return false
  }
  _recycle()
  return true
}
