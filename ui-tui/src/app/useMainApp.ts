import { type ScrollBoxHandle, useApp, useHasSelection, useSelection, useStdout, useTerminalTitle } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { AUTO_RESYNC_AUTOPILOT, STARTUP_RESUME_ID } from '../config/env.js'
import { MAX_HISTORY, WHEEL_SCROLL_STEP } from '../config/limits.js'
import { hasLeadGap, prevRenderedMsg } from '../domain/blockLayout.js'
import { SECTION_NAMES, sectionMode } from '../domain/details.js'
import { attachedImageNotice, imageTokenMeta } from '../domain/messages.js'
import { composeTabTitle, fmtCwdBranch, shortCwd, tildePath } from '../domain/paths.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  ClarifyRespondResponse,
  ClipboardPasteResponse,
  ConfigSetResponse,
  GatewayEvent,
  SessionActiveListResponse,
  SessionCloseResponse,
  TerminalResizeResponse
} from '../gatewayTypes.js'
import { useGitBranch } from '../hooks/useGitBranch.js'
import { useVirtualHistory } from '../hooks/useVirtualHistory.js'
import { composerPromptWidth } from '../lib/inputMetrics.js'
import { appendTranscriptMessage } from '../lib/messages.js'
import { DEFAULT_VOICE_RECORD_KEY, isMac, type ParsedVoiceRecordKey } from '../lib/platform.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { RECYCLE_EXIT_CODE, registerRecycleHandler } from '../lib/recycleBridge.js'
import { persistScrollState } from '../lib/scrollPersistence.js'
import { isRunning } from '../lib/subagentTree.js'
import { terminalParityHints } from '../lib/terminalParity.js'
import { getViewportSnapshot } from '../lib/viewportStore.js'
import { buildToolTrailLine, formatAbandonedClarify, sameToolTrailGroup, toolTrailLabel } from '../lib/text.js'
import { estimatedMsgHeight, messageHeightKey } from '../lib/virtualHeights.js'
import type { Msg, PanelSection, SlashCatalog } from '../types.js'

import { createGatewayEventHandler } from './createGatewayEventHandler.js'
import { createSlashHandler } from './createSlashHandler.js'
import { planGatewayRecovery } from './gatewayRecovery.js'
import { getInputSelection } from './inputSelectionStore.js'
import { type GatewayRpc, type TranscriptRow } from './interfaces.js'
import { $overlayState, getOverlayState, patchOverlayState } from './overlayStore.js'
import { buildSnipExpander } from '../protocol/paste.js'
import { forceRedraw, hardResetScreen } from '@hermes/ink'
import { scrollWithSelectionBy } from './scroll.js'
import { turnController } from './turnController.js'
import { patchTurnState, getTurnState, useTurnSelector } from './turnStore.js'
import { $uiState, getUiState, patchUiState } from './uiStore.js'
import { useComposerState } from './useComposerState.js'
import { useConfigSync } from './useConfigSync.js'
import { useInputHandlers } from './useInputHandlers.js'
import { useLongRunToolCharms } from './useLongRunToolCharms.js'
import { useSessionLifecycle } from './useSessionLifecycle.js'
import { useSubmission } from './useSubmission.js'

const GOOD_VIBES_RE = /\b(good bot|thanks|thank you|thx|ty|ily|love you)\b/i
const BRACKET_PASTE_ON = '\x1b[?2004h'
const BRACKET_PASTE_OFF = '\x1b[?2004l'
const MAX_HEIGHT_CACHE_BUCKETS = 12

const capHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0]!, ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

const statusColorOf = (status: string, t: { error: string; muted: string; ok: string; warn: string }) => {
  if (status === 'ready') {
    return t.ok
  }

  if (status.startsWith('error')) {
    return t.error
  }

  if (status === 'interrupted') {
    return t.warn
  }

  return t.muted
}

export interface PromptLiveSessionOptions {
  dispatchSubmission: (full: string) => void
  maybeWarn: (value: unknown) => void
  modelArg?: string
  newLiveSession: (msg?: string, title?: string) => Promise<null | string> | null | string | void
  onModelSwitched?: (value: string, result: ConfigSetResponse) => void
  prompt: string
  rpc: GatewayRpc
  sys: (text: string) => void
}

export async function startPromptLiveSession({
  dispatchSubmission,
  maybeWarn,
  modelArg,
  newLiveSession,
  onModelSwitched,
  prompt,
  rpc,
  sys
}: PromptLiveSessionOptions) {
  const trimmed = prompt.trim()

  if (!trimmed) {
    return null
  }

  // Let the backend-created session key (YYYYMMDD_HHMMSS_xxxxxx) remain
  // the initial title. Auto-title generation can rename it after the first
  // response; pre-queuing prompt text here causes duplicate-title errors when
  // users dispatch common prompts like "Hello, what model are you?".
  const sid = (await newLiveSession('new live session started')) ?? null

  if (!sid) {
    sys('error: failed to start new live session')

    return null
  }

  const requestedModel = modelArg?.trim()

  if (requestedModel) {
    const result = await rpc<ConfigSetResponse>('config.set', { key: 'model', session_id: sid, value: requestedModel })

    if (!result?.value) {
      sys('error: invalid response: model switch')

      return sid
    }

    sys(`model → ${result.value}`)
    maybeWarn(result)
    onModelSwitched?.(result.value, result)
  }

  dispatchSubmission(trimmed)

  return sid
}

export function useMainApp(gw: GatewayClient) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const [cols, setCols] = useState(stdout?.columns ?? 80)

  useEffect(() => {
    if (!stdout) {
      return
    }

    const sync = () => setCols(stdout.columns ?? 80)

    stdout.on('resize', sync)

    if (stdout.isTTY) {
      stdout.write(BRACKET_PASTE_ON)
    }

    return () => {
      stdout.off('resize', sync)

      if (stdout.isTTY) {
        stdout.write(BRACKET_PASTE_OFF)
      }
    }
  }, [stdout])

  const [historyItems, setHistoryItems] = useState<Msg[]>(() => [{ kind: 'intro', role: 'system', text: '' }])
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [stickyPrompt, setStickyPrompt] = useState('')
  const [catalog, setCatalog] = useState<null | SlashCatalog>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [voiceTts, setVoiceTts] = useState(false)
  const [voiceRecording, setVoiceRecording] = useState(false)
  const [voiceProcessing, setVoiceProcessing] = useState(false)
  const [voiceSynthesizing, setVoiceSynthesizing] = useState(false)
  const [voiceRecordKey, setVoiceRecordKey] = useState<ParsedVoiceRecordKey>(DEFAULT_VOICE_RECORD_KEY)
  const [sessionStartedAt, setSessionStartedAt] = useState(() => Date.now())
  const [turnStartedAt, setTurnStartedAt] = useState<null | number>(null)
  const [lastTurnEndedAt, setLastTurnEndedAt] = useState<null | number>(null)
  const [goodVibesTick, setGoodVibesTick] = useState(0)
  const [bellOnComplete, setBellOnComplete] = useState(false)

  const ui = useStore($uiState)
  const overlay = useStore($overlayState)

  const turnLiveTailActive = useTurnSelector(state =>
    Boolean(
      state.streaming ||
      state.streamPendingTools.length ||
      state.streamSegments.length ||
      state.reasoning.trim() ||
      state.reasoningActive ||
      state.tools.length ||
      state.subagents.length ||
      state.todos.length
    )
  )

  const slashFlightRef = useRef(0)
  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const colsRef = useRef(cols)
  const scrollRef = useRef<null | ScrollBoxHandle>(null)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  // Timestamp of the last backend event, for the busy-state wedge watchdog
  // installed in the gw.on('event') effect below. Initialised to mount time.
  const lastActivityRef = useRef<number>(Date.now())
  const clipboardPasteRef = useRef<(quiet?: boolean) => Promise<void> | void>(() => {})
  const submitRef = useRef<(value: string) => void>(() => {})
  const terminalHintsShownRef = useRef(new Set<string>())
  const historyItemsRef = useRef(historyItems)
  const lastUserMsgRef = useRef(lastUserMsg)
  const recoverSidRef = useRef<null | string>(null)
  const recoveryAtRef = useRef<number[]>([])
  const msgIdsRef = useRef(new WeakMap<Msg, string>())
  const msgIdSeqRef = useRef(0)
  const heightCachesRef = useRef(new Map<string, Map<string, number>>())

  colsRef.current = cols
  historyItemsRef.current = historyItems
  lastUserMsgRef.current = lastUserMsg

  const hasSelection = useHasSelection()
  const selection = useSelection()
  const lastCopiedVersionRef = useRef(-1)

  useEffect(() => {
    selection.setSelectionBgColor(ui.theme.color.selectionBg)
  }, [selection, ui.theme.color.selectionBg])

  // macOS Terminal.app does not forward Cmd+C to fullscreen TUIs that enable
  // mouse tracking, so the only reliable native-feeling path is iTerm-style
  // copy-on-select: once a drag creates a stable TUI selection, write it to
  // the system clipboard while keeping the highlight visible.
  //
  // Subscribe directly via the ink selection bus (not useSyncExternalStore)
  // so React doesn't re-render MainApp on every drag-move tick. The version
  // ref de-dupes against re-entrant notifications.
  useEffect(() => {
    if (!isMac) {
      return
    }

    return selection.subscribe(() => {
      if (!selection.hasSelection()) {
        return
      }

      const state = selection.getState() as { isDragging?: boolean } | null

      if (state?.isDragging) {
        return
      }

      const version = selection.version()

      if (version === lastCopiedVersionRef.current) {
        return
      }

      lastCopiedVersionRef.current = version
      void selection.copySelectionNoClear()
    })
  }, [selection])

  const clearSelection = useCallback(() => {
    selection.clearSelection()
    getInputSelection()?.collapseToEnd()
  }, [selection])

  const composer = useComposerState({
    gw,
    onClipboardPaste: quiet => clipboardPasteRef.current(quiet),
    onImageAttached: info => {
      sys(attachedImageNotice(info))
    },
    submitRef
  })

  const { actions: composerActions, refs: composerRefs, state: composerState } = composer
  const empty = !historyItems.some(msg => msg.kind !== 'intro')

  useEffect(() => {
    void terminalParityHints()
      .then(hints => {
        for (const hint of hints) {
          if (terminalHintsShownRef.current.has(hint.key)) {
            continue
          }

          terminalHintsShownRef.current.add(hint.key)
          turnController.pushActivity(hint.message, hint.tone)
        }
      })
      .catch(() => {})
  }, [])

  const messageId = useCallback((msg: Msg) => {
    const hit = msgIdsRef.current.get(msg)

    if (hit) {
      return hit
    }

    const next = `${messageHeightKey(msg)}:${++msgIdSeqRef.current}`

    msgIdsRef.current.set(msg, next)

    return next
  }, [])

  // Wrapped row heights are width-dependent. Cached layout outlives a resize
  // and lands sticky-scroll at the stale max, cutting off the tail. The
  // hook's "scale heights by oldCols/newCols" path is too approximate for
  // mixed markdown — we deliberately remount every row so yoga re-measures
  // off live geometry. Cost: per-row local state (e.g. systemOpen toggles)
  // resets on resize; small UX hit for a hard correctness win.
  const virtualRows = useMemo<TranscriptRow[]>(
    () => historyItems.map((msg, index) => ({ index, key: `${messageId(msg)}:c${cols}`, msg })),
    [cols, historyItems, messageId]
  )

  const detailsLayoutKey = useMemo(() => {
    const thinking = sectionMode('thinking', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride)
    const tools = sectionMode('tools', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride)

    return `${thinking}:${tools}`
  }, [ui.detailsMode, ui.detailsModeCommandOverride, ui.sections])

  const [thinkingDetailsMode, toolsDetailsMode] = detailsLayoutKey.split(':')
  const thinkingDetailsVisible = thinkingDetailsMode !== 'hidden'
  const toolsDetailsVisible = toolsDetailsMode !== 'hidden'
  const detailsVisible = thinkingDetailsVisible || toolsDetailsVisible
  const userPromptWidth = composerPromptWidth(ui.theme.brand.prompt)
  const heightCacheKey = `${ui.sid ?? 'draft'}:${cols}:${userPromptWidth}:${ui.compact ? '1' : '0'}:${detailsLayoutKey}`

  const heightCache = useMemo(() => {
    let cache = heightCachesRef.current.get(heightCacheKey)

    if (!cache) {
      cache = new Map()
      heightCachesRef.current.set(heightCacheKey, cache)

      if (heightCachesRef.current.size > MAX_HEIGHT_CACHE_BUCKETS) {
        heightCachesRef.current.delete(heightCachesRef.current.keys().next().value!)
      }
    }

    return cache
  }, [heightCacheKey])

  // Index of the first user-role message — separator-rendering in
  // appLayout.tsx skips this row, so the height estimator must skip it
  // too. -1 when no user message exists yet (no row will gate true).
  const firstUserIdx = useMemo(() => virtualRows.findIndex(r => r.msg.role === 'user'), [virtualRows])

  const estimateRowHeight = useCallback(
    (index: number) =>
      estimatedMsgHeight(virtualRows[index]!.msg, cols, {
        compact: ui.compact,
        details: detailsVisible,
        leadGap: hasLeadGap(
          prevRenderedMsg(i => virtualRows[i]?.msg, index, {
            commandOverride: ui.detailsModeCommandOverride,
            detailsMode: ui.detailsMode,
            sections: ui.sections
          }),
          virtualRows[index]!.msg
        ),
        thinkingVisible: thinkingDetailsVisible,
        toolsVisible: toolsDetailsVisible,
        userPrompt: ui.theme.brand.prompt,
        withSeparator: virtualRows[index]!.msg.role === 'user' && firstUserIdx >= 0 && index > firstUserIdx
      }),
    [
      cols,
      detailsVisible,
      firstUserIdx,
      thinkingDetailsVisible,
      toolsDetailsVisible,
      ui.compact,
      ui.detailsMode,
      ui.detailsModeCommandOverride,
      ui.sections,
      ui.theme.brand.prompt,
      virtualRows
    ]
  )

  const syncHeightCache = useCallback(
    (heights: ReadonlyMap<string, number>) => {
      for (const row of virtualRows) {
        const h = heights.get(row.key)

        if (h) {
          heightCache.set(row.key, h)
        }
      }
    },
    [heightCache, virtualRows]
  )

  const virtualHistory = useVirtualHistory(scrollRef, virtualRows, cols, {
    estimateHeight: estimateRowHeight,
    initialHeights: heightCache,
    liveTailActive: turnLiveTailActive,
    onHeightsChange: syncHeightCache
  })

  const scrollWithSelection = useCallback(
    (delta: number) => scrollWithSelectionBy(delta, { scrollRef, selection }),
    [selection]
  )

  const appendMessage = useCallback(
    (msg: Msg) => setHistoryItems(prev => capHistory(appendTranscriptMessage(prev, msg))),
    []
  )

  const sys = useCallback((text: string) => appendMessage({ role: 'system', text }), [appendMessage])

  const page = useCallback(
    (text: string, title?: string) => patchOverlayState({ pager: { lines: text.split('\n'), offset: 0, title } }),
    []
  )

  const panel = useCallback(
    (title: string, sections: PanelSection[]) =>
      appendMessage({ kind: 'panel', panelData: { sections, title }, role: 'system', text: '' }),
    [appendMessage]
  )

  const maybeWarn = useCallback(
    (value: unknown) => {
      const warning = (value as { warning?: unknown } | null)?.warning

      if (typeof warning === 'string' && warning) {
        sys(`warning: ${warning}`)
      }
    },
    [sys]
  )

  const maybeGoodVibes = useCallback((text: string) => {
    if (GOOD_VIBES_RE.test(text)) {
      setGoodVibesTick(v => v + 1)
    }
  }, [])

  const rpc: GatewayRpc = useCallback(
    async <T extends Record<string, any> = Record<string, any>>(
      method: string,
      params: Record<string, unknown> = {}
    ) => {
      try {
        const result = asRpcResult<T>(await gw.request<T>(method, params))

        if (result) {
          return result
        }

        sys(`error: invalid response: ${method}`)
      } catch (e) {
        sys(`error: ${rpcErrorMessage(e)}`)
      }

      return null
    },
    [gw, sys]
  )

  const gateway = useMemo(() => ({ gw, rpc }), [gw, rpc])

  const die = useCallback(() => {
    gw.kill('app.die')
    exit()
    // Ink's exit() calls unmount() which resets terminal modes but does NOT
    // call process.exit().  Without an explicit exit the Node process stays
    // alive (stdin listener keeps the event loop open), so the process.on('exit')
    // handler in entry.tsx — which sends the final resetTerminalModes() — never
    // fires.  This leaves kitty keyboard protocol, mouse modes, etc. enabled
    // in the parent shell.  See issue #19194.
    process.exit(0)
  }, [exit, gw])

  const dieWithCode = useCallback((code: number) => {
    gw.kill(`app.dieWithCode:${code}`)
    exit()
    process.exit(code)
  }, [exit, gw])

  // Stage 1 SEAMLESS RECYCLE: persist the current scroll position keyed by the
  // live sid, then exit 0. In attach mode `gw.kill()` only closes THIS
  // renderer's ws (this.proc is null — no spawned gateway child), so the
  // durable gateway + in-flight turn survive. The orchestrator respawns a fresh
  // renderer with HERMES_TUI_RESUME=<sid>; it resumes the live session (adopting
  // the running turn via the gateway's _live_session_payload) and restores the
  // scroll position — so the recycle is invisible to the user. Guarded by
  // canRecycle(): never fires in spawned-gateway mode where exiting would kill
  // the session.
  const recycle = useCallback(() => {
    try {
      const sid = getUiState().sid
      const handle = scrollRef.current
      if (sid && handle) {
        const snap = getViewportSnapshot(handle)
        persistScrollState(sid, { top: snap.top, atBottom: snap.atBottom })
      }
    } catch {
      // best-effort: a failed persist just means the fresh renderer falls back
      // to scrollToBottom — never blocks the recycle.
    }
    gw.kill('app.recycle')
    exit()
    // Exit with the distinct RECYCLE code (not 0): the orchestrator treats 0
    // as a voluntary /quit and tears the session down, but a recycle must make
    // the supervisor respawn a fresh renderer that re-attaches to the still-live
    // gateway and resumes the session. See RECYCLE_EXIT_CODE (mirrored in the
    // Python orchestrator).
    process.exit(RECYCLE_EXIT_CODE)
  }, [exit, gw])

  useEffect(() => registerRecycleHandler(recycle), [recycle])

  const session = useSessionLifecycle({
    colsRef,
    composerActions,
    gw,
    panel,
    rpc,
    scrollRef,
    setHistoryItems,
    setLastUserMsg,
    setSessionStartedAt,
    setStickyPrompt,
    setVoiceProcessing,
    setVoiceRecording,
    sys
  })

  useEffect(() => {
    if (ui.busy) {
      setTurnStartedAt(prev => prev ?? Date.now())
    } else if (turnStartedAt != null) {
      // Only stamp the idle marker when a turn was actually live — busy is
      // also false on mount and we don't want a phantom "done" timestamp
      // before the first turn has completed.
      setLastTurnEndedAt(Date.now())
      setTurnStartedAt(null)
    }
  }, [ui.busy, turnStartedAt])

  useConfigSync({ gw, setBellOnComplete, setVoiceEnabled, setVoiceRecordKey, sid: ui.sid })

  useEffect(() => {
    if (!ui.sid) {
      patchUiState({ liveSessionCount: 0 })

      return
    }

    let stopped = false

    const refresh = () => {
      gw.request<SessionActiveListResponse>('session.active_list', { current_session_id: getUiState().sid })
        .then(raw => {
          const result = asRpcResult<SessionActiveListResponse>(raw)

          if (!stopped && result?.sessions) {
            const liveSessionCount = result.sessions.length

            // Surface the current session's (auto-)title for the terminal
            // titlebar. The active_list poll already carries it, so no extra
            // round-trip is needed.
            const currentSid = getUiState().sid

            const sessionTitle =
              result.sessions.find(s => s.current || s.id === currentSid)?.title?.trim() ?? ''

            // Only patch when something actually changed. patchUiState always
            // produces a new state object, which notifies every $uiState
            // subscriber; patching unconditionally on each 1.5s poll re-renders
            // the whole TUI and causes idle flicker.
            const prev = getUiState()

            if (prev.liveSessionCount !== liveSessionCount || prev.sessionTitle !== sessionTitle) {
              patchUiState({ liveSessionCount, sessionTitle })
            }
          }
        })
        .catch(() => {})
    }

    refresh()
    const timer = setInterval(refresh, 1500)

    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [gw, ui.sid])

  // Tab title: `⚠` waiting on approval/sudo/secret/clarify, `⏳` busy, `✓` idle.
  // Format: `<marker> <session name> · <model> · <cwd>` — name/cwd omitted when absent.
  const model = ui.info?.model?.replace(/^.*\//, '') ?? ''

  const marker = overlay.approval || overlay.sudo || overlay.secret || overlay.clarify ? '⚠' : ui.busy ? '⏳' : '✓'

  const tabCwd = ui.info?.cwd

  useTerminalTitle(
    model ? composeTabTitle(marker, ui.sessionTitle, model, tabCwd ? shortCwd(tabCwd, 24) : '') : 'Hermes'
  )

  useEffect(() => {
    if (!ui.sid || !stdout) {
      return
    }

    let timer: ReturnType<typeof setTimeout> | undefined

    // Resize reflows wrapped lines; if the user is still pinned to the tail
    // we need to re-snap once React has remeasured. virtualRows is keyed on
    // cols so every column change forces a fresh measurement pass before
    // this timer fires. Re-check isSticky() inside the timeout — a manual
    // scroll during the 100ms window otherwise yanks the user back to tail.
    const onResize = () => {
      clearTimeout(timer)
      timer = setTimeout(() => {
        timer = undefined

        if (scrollRef.current?.isSticky()) {
          scrollRef.current.scrollToBottom()
        }

        void rpc<TerminalResizeResponse>('terminal.resize', { cols: stdout.columns ?? 80, session_id: ui.sid })
      }, 100)
    }

    stdout.on('resize', onResize)

    return () => {
      clearTimeout(timer)
      stdout.off('resize', onResize)
    }
  }, [rpc, stdout, ui.sid])

  const answerClarify = useCallback(
    (answer: string) => {
      const clarify = overlay.clarify

      if (!clarify) {
        return
      }

      const label = toolTrailLabel('clarify')

      turnController.turnTools = turnController.turnTools.filter(line => !sameToolTrailGroup(label, line))
      patchTurnState({ turnTrail: turnController.turnTools })

      rpc<ClarifyRespondResponse>('clarify.respond', { answer, request_id: clarify.requestId }).then(r => {
        if (!r) {
          return
        }

        if (answer) {
          turnController.persistedToolLabels.add(label)
          appendMessage({
            kind: 'trail',
            role: 'system',
            text: '',
            tools: [buildToolTrailLine('clarify', clarify.question)]
          })
          appendMessage({ role: 'user', text: answer })
          patchUiState({ status: 'running…' })
        } else {
          // Esc / Ctrl+C cancel: persist the question + options as a system
          // line (not a transient "prompt cancelled" flash) so the prompt
          // survives on screen as standard output, matching the timeout path.
          appendMessage({
            role: 'system',
            text: formatAbandonedClarify(clarify.question, clarify.choices, 'cancelled')
          })
        }

        patchOverlayState({ clarify: null })
      })
    },
    [appendMessage, overlay.clarify, rpc]
  )

  const paste = useCallback(
    (quiet = false) =>
      rpc<ClipboardPasteResponse>('clipboard.paste', { session_id: getUiState().sid }).then(r => {
        if (!r) {
          return
        }

        if (r.attached) {
          // imageTokenMeta now renders "dims · ~/path · ~N tok" (path included),
          // so the user can locate/copy the exact file AND the model has an
          // explicit handle to it (the same on-host path is also threaded into
          // the prompt content the model receives, so it never has to hunt for a
          // freshly pasted image).
          const meta = imageTokenMeta(r)

          if (r.kind === 'file') {
            // A pasted non-image file (pdf, exe, zip, ...) transferred verbatim.
            const pathStr = r.display_path || (r.path ? tildePath(r.path) : '')
            const label = r.name ? `File: ${r.name}` : 'File'
            return sys(`📎 ${label} attached from clipboard${pathStr ? ` · ${pathStr}` : ''}`)
          }

          return sys(`📎 Image #${r.count} attached from clipboard${meta ? ` · ${meta}` : ''}`)
        }

        if (!quiet) {
          sys(r.message || 'No image found in clipboard')
        }
      }),
    [rpc, sys]
  )

  clipboardPasteRef.current = paste

  const { dispatchSubmission, send, sendQueued, submit } = useSubmission({
    appendMessage,
    composerActions,
    composerRefs,
    composerState,
    gw,
    maybeGoodVibes,
    setLastUserMsg,
    slashRef,
    submitRef,
    sys
  })

  // Drain one queued message whenever the session settles (busy → false):
  // agent turn ends, interrupt, shell.exec finishes, error recovered, or the
  // session first comes up with pre-queued messages. Without this, shell.exec
  // and error paths never emit message.complete, so anything enqueued while
  // `!sleep` / a failed turn was running would stay stuck forever.
  useEffect(() => {
    if (
      !ui.sid ||
      ui.busy ||
      composerRefs.queueEditRef.current !== null ||
      composerRefs.queueRef.current.length === 0
    ) {
      return
    }

    const next = composerActions.dequeue()

    if (next) {
      patchUiState({ busy: true, status: 'running…' })
      sendQueued(next)
    }
  }, [ui.sid, ui.busy, composerActions, composerRefs, sendQueued])

  const { pagerPageSize } = useInputHandlers({
    actions: {
      answerClarify,
      appendMessage,
      die,
      dispatchSubmission,
      guardBusySessionSwitch: session.guardBusySessionSwitch,
      newSession: session.newSession,
      sys
    },
    composer: { actions: composerActions, refs: composerRefs, state: composerState },
    gateway,
    terminal: { hasSelection, scrollRef, scrollWithSelection, selection, stdout },
    voice: {
      enabled: voiceEnabled,
      recordKey: voiceRecordKey,
      recording: voiceRecording,
      setProcessing: setVoiceProcessing,
      setRecording: setVoiceRecording,
      setSynthesizing: setVoiceSynthesizing,
      setVoiceEnabled,
      setVoiceTts,
      synthesizing: voiceSynthesizing
    },
    wheelStep: WHEEL_SCROLL_STEP
  })

  const onEvent = useMemo(
    () =>
      createGatewayEventHandler({
        composer: { setInput: composerActions.setInput },
        gateway,
        session: {
          STARTUP_RESUME_ID,
          colsRef,
          newSession: session.newSession,
          recoverSidRef,
          resetSession: session.resetSession,
          resumeById: session.resumeById,
          setCatalog
        },
        submission: { submitRef },
        system: { bellOnComplete, stdout, sys },
        transcript: { appendMessage, panel, setHistoryItems },
        voice: {
          setProcessing: setVoiceProcessing,
          setRecording: setVoiceRecording,
          setSynthesizing: setVoiceSynthesizing,
          setVoiceEnabled,
          setVoiceTts
        }
      }),
    [
      appendMessage,
      bellOnComplete,
      clearSelection,
      composerActions.setInput,
      gateway,
      panel,
      session.newSession,
      session.resetSession,
      session.resumeById,
      setVoiceEnabled,
      setVoiceProcessing,
      setVoiceRecording,
      stdout,
      submitRef,
      sys
    ]
  )

  onEventRef.current = onEvent

  // ── Auto-healing full repaint (render-corruption self-recovery) ──────────
  // In alt-screen mode every frame is an INCREMENTAL diff; a full repaint only
  // ever fires on a terminal resize (renderer.ts: "every frame is incremental;
  // no fullResetSequence"). So if Ink's screen model desyncs even once — a cell
  // the diff loop misses, a scroll the relative-cursor math mistracks (the known
  // log-update cursor-desync class, worse under tmux), or any foreign write to
  // the shared TTY — the corruption ACCUMULATES with nothing to clear it until
  // the user manually resizes the terminal. That manual resize is exactly what
  // William has been doing every ~30s to un-garble the screen.
  //
  // This timer does that resize-grade recovery AUTOMATICALLY, on EVERY tick,
  // idle OR busy (garble accumulates on a screen left mid-turn or scrolled while
  // idle, not only during active render). William's steer evolved in two parts:
  //   • (2026-07-09 part 1) run the heal every 20s regardless of busy state.
  //   • (2026-07-09 part 2) use the SOFT heal — forceRedraw() (resize-grade,
  //     repaints IN PLACE) — as the automatic default, NOT the strong ?1049h
  //     hardResetScreen(). The strong swap holds garble off longer (~5min vs the
  //     soft redraw's ~20s decay) but it DISCARDS+reallocates the terminal's
  //     alt-screen buffer, which kicks tmux out of copy-mode and BROKE his
  //     scrollback. Running the soft redraw every 20s keeps pace with the decay
  //     without ever touching the alt buffer, so scrolling stays intact.
  // Cheap: an erase+home+repaint of changed cells; a no-op when not mounted/TTY
  // (forceRedraw guards internally). This auto heal is KEYLESS — it repaints on
  // a timer and occupies NO keyboard shortcut (Ctrl+O is voice; the manual hard
  // reset is Ctrl+T / /hardreset; everyday redraw is Ctrl+L / /redraw).
  //
  // Nothing is hardcoded — two env knobs tune it:
  //   HERMES_TUI_HEAL_INTERVAL_MS  cadence in ms (default 20000, floor 2000)
  //   HERMES_TUI_HEAL_MODE         'redraw' (default) | 'hardreset' | 'off'
  // MODE=hardreset promotes the automatic heal to the strong ?1049h buffer swap
  // (for hosts where only a full swap clears the garble, at the cost of tmux
  // scrollback); MODE=off disables the automatic paint heal entirely (Ctrl+L,
  // Ctrl+T, and the /redraw + /hardreset slash commands still work by hand).
  // The legacy HERMES_TUI_HARD_RESET_AFTER_N_HEALS escalation counter is retired.
  //
  // STATE drift (separate from PAINT drift): a long AUTOPILOT run re-enters
  // _run_prompt_submit each continuation (server.py:6843), emitting repeated
  // message.start + status.update(goal) sys() rows. The append-only TUI
  // projection accumulates these and, on any event mis-order, the visible
  // transcript DRIFTS from the gateway's durable history — the "looped old
  // message / stuck stall, detached from the live run" symptom. forceRedraw
  // can't fix that (it re-paints the same wrong rows). So during autopilot ONLY,
  // and ONLY on the busy→idle settle edge (a continuation boundary — never
  // mid-stream), we also re-attach to the gateway via resumeById(storedSid) at a
  // slow cadence. This is the proven-safe /resync path (live-tested: subagents
  // survive, in-flight turn preserved); it self-heals unattended autopilot runs
  // the user isn't watching. Off entirely for normal interactive sessions.
  useEffect(() => {
    const intervalEnv = process.env.HERMES_TUI_HEAL_INTERVAL_MS
    const HEAL_INTERVAL_MS = Math.max(2_000, parseInt(intervalEnv ?? '20000', 10) || 20_000)
    const RESYNC_EVERY_N_TICKS = 3 // ≈60s between autopilot state re-syncs
    // Automatic paint-heal mode. 'redraw' (DEFAULT) = the no-flash resize-grade
    // forceRedraw that repaints IN PLACE — it does NOT touch the terminal's
    // alt-screen buffer, so it leaves tmux scrollback / copy-mode intact.
    // William's steer (2026-07-09, part 2): the previous 'hardreset' default
    // (?1049h alt-screen buffer swap) was BREAKING his tmux scrolling because
    // re-entering the alt screen discards+reallocates the buffer, kicking tmux
    // out of copy-mode. forceRedraw decays back in ~20s on its own but now RUNS
    // every 20s, so it keeps pace without disturbing scroll. 'hardreset' = the
    // strong buffer swap (still available on demand via Ctrl+T / /hardreset, and
    // selectable here for hosts where only a full swap clears the garble);
    // 'off' = no automatic paint heal at all.
    const healMode = (process.env.HERMES_TUI_HEAL_MODE ?? 'redraw').trim().toLowerCase()
    const paintHeal =
      healMode === 'off'
        ? null
        : healMode === 'hardreset'
          ? hardResetScreen
          : forceRedraw
    let lastBusy = false
    let ticksSinceResync = 0

    const healer = setInterval(() => {
      const ui = getUiState()
      const busy = ui.busy

      // Paint heal on EVERY tick, idle OR busy (garble accumulates on a screen
      // left mid-turn or scrolled while idle, not only during active render).
      // Soft forceRedraw is the default (tmux-scroll-safe); see the block
      // comment above for the mode/interval env knobs.
      if (paintHeal) {
        paintHeal(stdout ?? process.stdout)
      }

      // State heal (autopilot only): on the busy→idle settle edge — a
      // continuation boundary, safe (no live stream to clobber) — re-attach to
      // gateway truth at a slow cadence so projection drift can't accumulate
      // across an unattended multi-hour run. storedSid (durable key) is what
      // session.resume needs; sid is the ephemeral renderer id.
      const settledEdge = !busy && lastBusy
      const inAutopilot = Boolean(ui.info?.autopilot)
      const key = ui.storedSid

      if (AUTO_RESYNC_AUTOPILOT && settledEdge && inAutopilot && key) {
        ticksSinceResync += 1

        if (ticksSinceResync >= RESYNC_EVERY_N_TICKS) {
          ticksSinceResync = 0
          // Same proven-safe path as /resync: fast-path reuse of the live
          // session (no teardown, subagents/inflight preserved).
          session.resumeById(key)
        }
      }

      lastBusy = busy
    }, HEAL_INTERVAL_MS)

    healer.unref?.()

    return () => clearInterval(healer)
  }, [stdout, session])

  useEffect(() => {
    // Busy-state watchdog. The status indicator (FaceTicker) repaints the Ink
    // tree at SPINNER_TICK_MS (100ms = 10Hz) while `busy` is true. That's
    // correct during a live turn, but if the gateway *wedges* — stream stalls
    // with no clean `exit` event — `busy` never resets and the 10Hz repaint
    // runs for hours, pegging a CPU core (the "idle TUI burning 70% CPU" bug).
    // `exitHandler` only covers a clean child death; this covers the silent
    // stall. Every backend event bumps `lastActivityRef`; if `busy` is true but
    // nothing has arrived for WEDGE_TIMEOUT_MS, the turn is dead, so drop
    // `busy` and surface it. This makes the runaway repaint structurally
    // impossible regardless of how the backend fails.
    const WEDGE_TIMEOUT_MS = 180_000
    const handler = (ev: GatewayEvent) => {
      lastActivityRef.current = Date.now()
      onEventRef.current(ev)
    }

    const wedgeWatchdog = setInterval(() => {
      if (!getUiState().busy) {
        return
      }

      if (Date.now() - lastActivityRef.current < WEDGE_TIMEOUT_MS) {
        return
      }

      // A delegated subagent can legitimately run SILENT for minutes — a long
      // tool call (build, sleep, deep search) or pure-token generation. The
      // parent session only receives subagent.start then nothing until
      // subagent.tool/complete (subagent.text is intentionally not relayed to
      // the parent), so the activity clock can exceed WEDGE_TIMEOUT_MS while the
      // turn is perfectly healthy. Firing the stall here wrongly drops `busy`,
      // hides the live transcript, and surfaces a false "stream stalled". While
      // a subagent is still running/queued the turn is NOT wedged — defer the
      // verdict and keep the activity window fresh so we re-evaluate once they
      // finish (completed subagents don't suppress, so a genuinely wedged turn
      // after delegation still trips the watchdog).
      if (getTurnState().subagents.some(isRunning)) {
        lastActivityRef.current = Date.now()
        return
      }

      // SAME legitimate-silence case, but in the MAIN turn: a long-running tool
      // call in flight (a multi-minute build/test wait, `process wait`, a deep
      // search, a sleep) emits nothing between tool.start and tool.result, so
      // the activity clock can blow past WEDGE_TIMEOUT_MS while the turn is
      // perfectly healthy. The subagent guard above only covers DELEGATED long
      // tool calls; an in-flight tool on the parent turn hit neither exemption
      // and tripped a false "stream stalled" (transcript clears, last message
      // loops). `tools` is the published in-flight set: populated on tool.start,
      // removed on tool.result, cleared on idle() — so it self-clears and can't
      // mask a genuinely wedged turn any longer than the running subagent guard.
      // While a tool is still in flight the turn is NOT wedged; keep the window
      // fresh and re-evaluate once it resolves.
      if (getTurnState().tools.length > 0) {
        lastActivityRef.current = Date.now()
        return
      }

      // A blocking prompt (clarify / sudo / secret / confirm / approval) is the
      // OTHER legitimately-busy-but-silent state: the turn is parked waiting on
      // a HUMAN, so no backend events arrive while the user reads the question,
      // picks "Other", and types a custom answer. Without this guard the 3m
      // watchdog tears down the live overlay (turnController.reset drops `busy`
      // and resetFlowOverlays clears `clarify`), so a slow answer makes the
      // prompt vanish, surfaces a false "stream stalled", and the user's typed
      // response lands on a clarify that no longer exists — it's lost. While a
      // prompt is open the turn is NOT wedged; keep the activity window fresh
      // and re-evaluate once the user responds and the overlay clears.
      const ov = getOverlayState()
      if (ov.clarify || ov.sudo || ov.secret || ov.confirm || ov.approval) {
        lastActivityRef.current = Date.now()
        return
      }

      // No backend activity for WEDGE_TIMEOUT_MS while busy: the turn is wedged.
      // Reset so the 10Hz indicator stops and the session is usable again.
      turnController.reset()

      // Self-heal instead of stranding the user on a dead sid. The durable
      // session usually survives the drop (it lives in SQLite and, until the
      // gateway's orphan reaper fires, still in gateway memory), so re-attach to
      // it via the same proven-safe path /resync and the autopilot healer use.
      // resumeById re-registers a live session, adopts a fresh sid, and re-renders
      // the transcript from gateway truth — turning the old "stream stalled" dead
      // end into a transparent reconnect. If there's no durable key we fall back
      // to the old visible-stall notice so a genuinely unrecoverable wedge still
      // surfaces. Keep `busy` reset either way so the 10Hz repaint stops (the
      // CPU-peg bug this watchdog originally fixed stays fixed).
      const storedSid = getUiState().storedSid

      if (storedSid) {
        patchUiState({ busy: false, status: 'stream stalled · reconnecting…' })
        turnController.pushActivity(
          'no backend activity for 3m — reconnecting to your session (any in-flight reply may be lost)',
          'warn'
        )

        try {
          session.resumeById(storedSid)
        } catch {
          patchUiState({ busy: false, status: 'stream stalled · /resync to reconnect' })
        }
      } else {
        patchUiState({ busy: false, status: 'stream stalled · /logs to inspect' })
        turnController.pushActivity(
          'no backend activity for 3m — turn marked stalled (any in-flight reply may be lost)',
          'warn'
        )
      }

      lastActivityRef.current = Date.now()
    }, 15_000)
    wedgeWatchdog.unref?.()

    const exitHandler = () => {
      turnController.reset()

      // A still-owned child dying while the TUI is alive is an *unexpected*
      // death — a user /quit exits Node before this fires, and a replaced child
      // is identity-skipped in GatewayClient. Rather than stranding a long
      // session (the user's complaint), respawn the gateway and resume the
      // persisted session via the next gateway.ready, so a single crash / OOM /
      // signal doesn't lose their work. planGatewayRecovery bounds the attempts
      // so a gateway that crash-loops on startup can't spawn-storm, and falls
      // back to recoverSidRef when sid was already cleared by a prior exit.
      const plan = planGatewayRecovery(getUiState().sid, recoverSidRef.current, recoveryAtRef.current, Date.now())

      // Clear sid immediately: while the gateway is down, sid-guarded effects
      // (session.active_list poll, queue drain) would otherwise fire RPCs at a
      // dead/respawning gateway. recoverSidRef carries the session forward, and
      // resumeById restores sid once the fresh gateway is ready.
      recoveryAtRef.current = plan.attempts
      patchUiState({ busy: false, sid: null, status: 'gateway exited' })

      if (plan.recover && plan.sid) {
        recoverSidRef.current = plan.sid
        turnController.pushActivity('gateway exited · recovering session…', 'warn')
        sys('gateway exited — recovering your session (any in-flight reply was lost)')
        gw.start()

        return
      }

      recoverSidRef.current = null
      turnController.pushActivity('gateway exited · /logs to inspect', 'error')
      sys('error: gateway exited')
    }

    gw.on('event', handler)
    gw.on('exit', exitHandler)
    gw.drain()

    // entry.tsx's setupGracefulExit handles process cleanup on real exit.
    return () => {
      clearInterval(wedgeWatchdog)
      gw.off('event', handler)
      gw.off('exit', exitHandler)
    }
  }, [gw, sys, session])

  useLongRunToolCharms()

  const slash = useMemo(
    () =>
      createSlashHandler({
        composer: {
          enqueue: composerActions.enqueue,
          expandPaste: (text: string) => buildSnipExpander(composerState.pasteSnips)(text),
          hasSelection,
          paste,
          queueRef: composerRefs.queueRef,
          selection,
          setInput: composerActions.setInput
        },
        gateway,
        local: {
          catalog,
          getHistoryItems: () => historyItemsRef.current,
          getLastUserMsg: () => lastUserMsgRef.current,
          maybeWarn,
          setCatalog
        },
        session: {
          closeSession: session.closeSession,
          die,
          dieWithCode,
          guardBusySessionSwitch: session.guardBusySessionSwitch,
          newLiveSession: session.newLiveSession,
          newSession: session.newSession,
          resetVisibleHistory: session.resetVisibleHistory,
          resumeById: session.resumeById,
          setSessionStartedAt
        },
        slashFlightRef,
        transcript: { page, panel, send, setHistoryItems, sys, trimLastExchange: session.trimLastExchange },
        voice: { setVoiceEnabled, setVoiceRecordKey, setVoiceTts }
      }),
    [
      catalog,
      composerActions,
      composerRefs,
      composerState.pasteSnips,
      die,
      gateway,
      hasSelection,
      maybeWarn,
      page,
      panel,
      paste,
      selection,
      send,
      session,
      sys
    ]
  )

  slashRef.current = slash

  const respondWith = useCallback(
    (method: string, params: Record<string, unknown>, done: () => void) => rpc(method, params).then(r => r && done()),
    [rpc]
  )

  const answerApproval = useCallback(
    (choice: string) =>
      respondWith('approval.respond', { choice, session_id: ui.sid }, () => {
        patchOverlayState({ approval: null })
        patchTurnState({ outcome: choice === 'deny' ? 'denied' : `approved (${choice})` })
        patchUiState({ status: 'running…' })
      }),
    [respondWith, ui.sid]
  )

  const answerSudo = useCallback(
    (pw: string) => {
      if (!overlay.sudo) {
        return
      }

      return respondWith('sudo.respond', { password: pw, request_id: overlay.sudo.requestId }, () => {
        patchOverlayState({ sudo: null })
        patchUiState({ status: 'running…' })
      })
    },
    [overlay.sudo, respondWith]
  )

  const answerSecret = useCallback(
    (value: string) => {
      if (!overlay.secret) {
        return
      }

      return respondWith('secret.respond', { request_id: overlay.secret.requestId, value }, () => {
        patchOverlayState({ secret: null })
        patchUiState({ status: 'running…' })
      })
    },
    [overlay.secret, respondWith]
  )

  const onModelSelect = useCallback((value: string) => {
    patchOverlayState({ modelPicker: false })
    slashRef.current(`/model ${value}`)
  }, [])

  const closeLiveSession = useCallback(
    async (id: string) => {
      patchUiState({ status: 'closing session…' })

      try {
        const result = (await session.closeSession(id)) as null | SessionCloseResponse
        patchUiState({ status: 'ready' })

        return result
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e)
        sys(`error: ${message}`)
        patchUiState({ status: 'ready' })

        throw e
      }
    },
    [session, sys]
  )

  const newPromptSession = useCallback(
    (prompt: string, modelArg?: string) => {
      void startPromptLiveSession({
        dispatchSubmission,
        maybeWarn,
        modelArg,
        newLiveSession: session.newLiveSession,
        onModelSwitched: value =>
          patchUiState(state => ({
            ...state,
            info: state.info ? { ...state.info, model: value } : { model: value, skills: {}, tools: {} }
          })),
        prompt,
        rpc,
        sys
      })
    },
    [dispatchSubmission, maybeWarn, rpc, session.newLiveSession, sys]
  )

  const hasReasoning = useTurnSelector(state => Boolean(state.reasoning.trim()))

  // Per-section overrides win over the global mode — when every section is
  // resolved to hidden, the only thing ToolTrail will surface is the
  // floating-alert backstop (errors/warnings).  Mirror that so we don't
  // render an empty wrapper Box above the streaming area in quiet mode.
  const anyPanelVisible = SECTION_NAMES.some(
    s => sectionMode(s, ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'
  )

  const thinkingPanelVisible =
    sectionMode('thinking', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'

  const toolsPanelVisible =
    sectionMode('tools', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'

  const activityPanelVisible =
    sectionMode('activity', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'

  const showProgressArea = useTurnSelector(state =>
    anyPanelVisible
      ? Boolean(
          ui.busy ||
          state.outcome ||
          state.streamPendingTools.length ||
          state.streamSegments.some(segment => {
            const hasThinking = Boolean(segment.thinking?.trim())
            const hasTrailTools = Boolean(segment.tools?.length)

            if (segment.kind === 'trail' && !segment.text) {
              return (
                (thinkingPanelVisible && hasThinking) || ((toolsPanelVisible || activityPanelVisible) && hasTrailTools)
              )
            }

            return (
              Boolean(segment.text?.trim()) ||
              (thinkingPanelVisible && hasThinking) ||
              ((toolsPanelVisible || activityPanelVisible) && hasTrailTools)
            )
          }) ||
          state.subagents.length ||
          state.tools.length ||
          state.todos.length ||
          state.turnTrail.length ||
          (thinkingPanelVisible && hasReasoning) ||
          state.activity.length
        )
      : state.activity.some(item => item.tone !== 'info')
  )

  const appActions = useMemo(
    () => ({
      activateLiveSession: session.activateLiveSession,
      closeLiveSession,
      answerApproval,
      answerClarify,
      answerSecret,
      answerSudo,
      clearSelection,
      newLiveSession: () => session.newLiveSession(),
      newPromptSession,
      onModelSelect,
      // Resuming a cold session from the overlay CLOSES the current one, so it
      // must respect the busy guard just like the `/resume` slash path.
      // (Switching between live sessions and `+ new` keep the current session
      // running, so those stay unguarded — that's the orchestrator's purpose.)
      resumeById: (id: string) => {
        if (session.guardBusySessionSwitch('switch sessions')) {
          return
        }

        session.resumeById(id)
      },
      setStickyPrompt
    }),
    [
      answerApproval,
      answerClarify,
      answerSecret,
      answerSudo,
      clearSelection,
      closeLiveSession,
      newPromptSession,
      onModelSelect,
      session.activateLiveSession,
      session.guardBusySessionSwitch,
      session.newLiveSession,
      session.resumeById
    ]
  )

  const appComposer = useMemo(
    () => ({
      cols,
      compIdx: composerState.compIdx,
      completions: composerState.completions,
      empty,
      handleTextPaste: composerActions.handleTextPaste,
      input: composerState.input,
      inputBuf: composerState.inputBuf,
      pagerPageSize,
      queueEditIdx: composerState.queueEditIdx,
      queuedDisplay: composerState.queuedDisplay,
      submit,
      updateInput: composerActions.setInput,
      voiceRecordKey
    }),
    [cols, composerActions, composerState, empty, pagerPageSize, submit, voiceRecordKey]
  )

  // Pass current progress through unfrozen — streaming update throttling
  // handles interaction load; progress must stay truthful so panels don't
  // randomly disappear when the live tail scrolls offscreen.
  const appProgress = useMemo(() => ({ showProgressArea }), [showProgressArea])

  const cwd = ui.info?.cwd || process.env.HERMES_CWD || process.cwd()
  const gitBranch = useGitBranch(cwd)

  const appStatus = useMemo(
    () => ({
      // Cap the status-bar cwd/branch label tighter than the shared default so
      // it doesn't dominate the bar; the status rule reserves the left-side
      // essentials and truncates this further on narrow terminals.
      cwdLabel: fmtCwdBranch(cwd, gitBranch, 28),
      goodVibesTick,
      lastTurnEndedAt: ui.sid ? lastTurnEndedAt : null,
      sessionStartedAt: ui.sid ? sessionStartedAt : null,
      showStickyPrompt: !!stickyPrompt,
      statusColor: statusColorOf(ui.status, ui.theme.color),
      stickyPrompt,
      turnStartedAt: ui.sid ? turnStartedAt : null,
      // CLI parity: the classic prompt_toolkit status bar shows a red dot
      // on REC (cli.py:_get_voice_status_fragments line 2344).
      voiceLabel: voiceRecording ? '● REC' : voiceProcessing ? '◉ STT' : voiceSynthesizing ? '♪ generating audio response…' : `voice ${voiceEnabled ? 'on' : 'off'}${voiceTts ? ' [tts]' : ''}`
    }),
    [
      cwd,
      gitBranch,
      goodVibesTick,
      lastTurnEndedAt,
      sessionStartedAt,
      stickyPrompt,
      turnStartedAt,
      ui,
      voiceEnabled,
      voiceProcessing,
      voiceRecording,
      voiceSynthesizing,
      voiceTts
    ]
  )

  const appTranscript = useMemo(
    () => ({ historyItems, scrollRef, virtualHistory, virtualRows }),
    [historyItems, virtualHistory, virtualRows]
  )

  return { appActions, appComposer, appProgress, appStatus, appTranscript, gateway }
}
