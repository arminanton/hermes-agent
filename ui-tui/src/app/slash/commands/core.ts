import { forceRedraw, hardResetScreen, type MouseTrackingMode } from '@hermes/ink'

import { DASHBOARD_TUI_MODE, NO_CONFIRM_DESTRUCTIVE } from '../../../config/env.js'
import { dailyFortune, randomFortune } from '../../../content/fortunes.js'
import { HOTKEYS } from '../../../content/hotkeys.js'
import { isSectionName, nextDetailsMode, parseDetailsMode, SECTION_NAMES } from '../../../domain/details.js'
import type {
  ConfigGetValueResponse,
  ConfigSetResponse,
  SessionSaveResponse,
  SessionStatusResponse,
  SessionSteerResponse,
  SessionTitleResponse,
  SessionUndoResponse
} from '../../../gatewayTypes.js'
import { writeClipboardText } from '../../../lib/clipboard.js'
import { writeOsc52Clipboard } from '../../../lib/osc52.js'
import { configureDetectedTerminalKeybindings, configureTerminalKeybindings } from '../../../lib/terminalSetup.js'
import type { Msg, PanelSection } from '../../../types.js'
import type { StatusBarMode } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

const flagFromArg = (arg: string, current: boolean): boolean | null => {
  if (!arg) {
    return !current
  }

  const mode = arg.trim().toLowerCase()

  if (mode === 'on') {
    return true
  }

  if (mode === 'off') {
    return false
  }

  if (mode === 'toggle') {
    return !current
  }

  return null
}

// `/mouse` toggles between full tracking and off when called bare so the
// old binary muscle-memory still works. Explicit presets (wheel / buttons /
// all) target the tmux-friendly hover-free subsets.
const MOUSE_MODE_ALIASES: Record<string, MouseTrackingMode> = {
  all: 'all',
  any: 'all',
  button: 'buttons',
  buttons: 'buttons',
  click: 'buttons',
  full: 'all',
  off: 'off',
  on: 'all',
  scroll: 'wheel',
  wheel: 'wheel'
}

const mouseModeFromArg = (arg: string, current: MouseTrackingMode): MouseTrackingMode | null => {
  if (!arg || arg.trim().toLowerCase() === 'toggle') {
    return current === 'off' ? 'all' : 'off'
  }

  return MOUSE_MODE_ALIASES[arg.trim().toLowerCase()] ?? null
}

const RESET_WORDS = new Set(['reset', 'clear', 'default'])
const CYCLE_WORDS = new Set(['cycle', 'toggle'])

const DETAILS_USAGE =
  'usage: /details [hidden|collapsed|expanded|cycle]  or  /details <section> [hidden|collapsed|expanded|reset]'

const DETAILS_SECTION_USAGE = 'usage: /details <section> [hidden|collapsed|expanded|reset]'

// Shown when /exit or /quit is refused in the hosted dashboard chat. Kept as a
// constant so the test asserts against the same source of truth as production.
export const DASHBOARD_EXIT_DISABLED_MESSAGE =
  'exit is disabled in hosted dashboard chat — use /new to start a fresh session'

export const DASHBOARD_UPDATE_DISABLED_MESSAGE =
  'update is disabled in hosted dashboard chat — the hosted environment is managed separately'

export const coreCommands: SlashCommand[] = [
  {
    help: 'list commands + hotkeys',
    name: 'help',
    run: (_arg, ctx) => {
      const sections: PanelSection[] = (ctx.local.catalog?.categories ?? []).map(cat => ({
        rows: cat.pairs,
        title: cat.name
      }))

      if (ctx.local.catalog?.skillCount) {
        sections.push({ text: `${ctx.local.catalog.skillCount} skill commands available — /skills to browse` })
      }

      sections.push(
        {
          rows: [
            ['/details [hidden|collapsed|expanded|cycle]', 'set global agent detail visibility mode'],
            [
              '/details <section> [hidden|collapsed|expanded|reset]',
              'override one section (thinking/tools/subagents/activity)'
            ],
            ['/fortune [random|daily]', 'show a random or daily local fortune']
          ],
          title: 'TUI'
        },
        { rows: HOTKEYS, title: 'Hotkeys' }
      )

      ctx.transcript.panel(ctx.ui.theme.brand.helpHeader, sections)
    }
  },

  {
    aliases: ['exit'],
    help: 'exit hermes',
    name: 'quit',
    run: (_arg, ctx) => {
      // In the hosted dashboard chat there is no in-page restart path after
      // the PTY child exits, so quitting bricks the tab until a refresh. The
      // keyboard idle-exit (Ctrl+C / Ctrl+D) and SIGINT handling already refuse
      // to die in this mode (see useInputHandlers + entry.tsx); gate /exit and
      // /quit on the same DASHBOARD_TUI_MODE flag. Unlike the keyboard path
      // (which auto-starts a fresh chat), the explicit quit command refuses and
      // instructs the user to run /new themselves.
      if (DASHBOARD_TUI_MODE) {
        ctx.transcript.sys(DASHBOARD_EXIT_DISABLED_MESSAGE)

        return
      }

      ctx.session.die()
    }
  },

  {
    help: 'update Hermes Agent to the latest version (exits TUI)',
    name: 'update',
    run: (_arg, ctx) => {
      if (DASHBOARD_TUI_MODE) {
        ctx.transcript.sys(DASHBOARD_UPDATE_DISABLED_MESSAGE)

        return
      }

      ctx.transcript.sys('exiting TUI to run update...')
      // Exit code 42 signals the Python wrapper to exec `hermes update`.
      // Use dieWithCode for proper cleanup (gateway kill + Ink unmount).
      setTimeout(() => ctx.session.dieWithCode(42), 100)
    }
  },

  {
    aliases: ['scroll'],
    help: 'set mouse tracking preset [on|off|toggle|wheel|buttons|all]',
    name: 'mouse',
    run: (arg, ctx) => {
      const current = ctx.ui.mouseTracking
      const next = mouseModeFromArg(arg, current)

      if (next === null) {
        return ctx.transcript.sys('usage: /mouse [on|off|toggle|wheel|buttons|all]')
      }

      patchUiState({ mouseTracking: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'mouse', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`mouse tracking ${next}`))
    }
  },

  {
    aliases: ['new'],
    help: 'start a new session',
    name: 'clear',
    run: (arg, ctx, cmd) => {
      if (ctx.session.guardBusySessionSwitch('switch sessions')) {
        return
      }

      const isNew = cmd.startsWith('/new')
      const requestedTitle = isNew ? arg.trim() : ''

      const commit = () => {
        patchUiState({ status: 'forging session…' })
        ctx.session.newSession(isNew ? 'new session started' : undefined, requestedTitle || undefined)
      }

      if (NO_CONFIRM_DESTRUCTIVE) {
        return commit()
      }

      patchOverlayState({
        confirm: {
          cancelLabel: 'No, keep going',
          confirmLabel: isNew ? 'Yes, start a new session' : 'Yes, clear the session',
          danger: true,
          detail: 'This ends the current conversation and clears the transcript.',
          onConfirm: commit,
          title: isNew ? 'Start a new session?' : 'Clear the current session?'
        }
      })
    }
  },

  {
    help: 'force a full UI repaint',
    name: 'redraw',
    run: (_arg, ctx) => {
      forceRedraw(process.stdout)
      ctx.transcript.sys('ui redrawn')
    }
  },

  {
    // The strongest PAINT recovery: /redraw repaints in place (resize-grade
    // erase now), but when the terminal's own alt-screen buffer is polluted —
    // glyphs migrated to wrong columns, two frames composited, corruption that
    // recurs every few seconds and only a manual window resize clears — an
    // in-buffer erase can't win. /hardreset re-enters the alternate screen
    // (?1049h) so the terminal DISCARDS and reallocates that buffer, the one
    // thing a manual resize does that /redraw didn't. The React tree and the
    // gateway session are untouched (no turn teardown); only the terminal
    // screen buffer and the renderer's diff model reset.
    help: 'hard-reset the screen buffer (fixes garble a plain redraw cannot)',
    name: 'hardreset',
    run: (_arg, ctx) => {
      hardResetScreen(process.stdout)
      ctx.transcript.sys('screen hard-reset')
    }
  },

  {
    // /redraw only re-PAINTS the current (possibly drifted) rows. /resync goes
    // deeper: it re-fetches the session from the gateway's durable state.db
    // (history + the live in-flight turn) and rebuilds the transcript from that
    // ground truth — the same restore /resume does, but for the CURRENT session
    // so the running turn is NOT torn down (resumeById only closes the previous
    // session when switching to a DIFFERENT sid; same-sid re-attaches in place).
    // Use this when the transcript has detached from the real run: stale/looped
    // rows, a stuck stall notice, or messages that vanished while the agent kept
    // working. forceRedraw fixes corrupted PAINT; /resync fixes corrupted STATE.
    help: 're-sync transcript from the gateway (fix a drifted/stuck view)',
    name: 'resync',
    run: (_arg, ctx) => {
      // session.resume looks up by the DURABLE session key (stored_session_id /
      // session_key), NOT the ephemeral renderer sid — passing ctx.sid 404s
      // ("session not found"). Use storedSid; fall back to sid only if unset.
      const resumeKey = ctx.ui.storedSid ?? ctx.sid

      if (!resumeKey) {
        return ctx.transcript.sys('no active session to re-sync')
      }

      ctx.transcript.sys('re-syncing transcript from gateway…')
      ctx.session.resumeById(resumeKey)
    }
  },

  {
    help: 'show live session info',
    name: 'status',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('no active session')
      }

      ctx.gateway
        .rpc<SessionStatusResponse>('session.status', { session_id: ctx.sid })
        .then(ctx.guarded<SessionStatusResponse>(r => ctx.transcript.page(r.output || '(no status)', 'Status')))
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'set or show current session title',
    name: 'title',
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('no active session')
      }

      const title = arg.trim()

      if (!arg) {
        ctx.gateway
          .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid })
          .then(
            ctx.guarded<SessionTitleResponse>(r => {
              const current = (r?.title ?? '').trim()
              ctx.transcript.sys(current ? `title: ${current}` : 'no title set')
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (!title) {
        return ctx.transcript.sys('usage: /title <your session title>')
      }

      ctx.gateway
        .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid, title })
        .then(
          ctx.guarded<SessionTitleResponse>(r => {
            const next = (r?.title ?? title).trim()
            const suffix = r?.pending ? ' (queued while session initializes)' : ''
            ctx.transcript.sys(`session title set: ${next}${suffix}`)
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'toggle compact transcript',
    name: 'compact',
    run: (arg, ctx) => {
      const next = flagFromArg(arg, ctx.ui.compact)

      if (next === null) {
        return ctx.transcript.sys('usage: /compact [on|off|toggle]')
      }

      patchUiState({ compact: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'compact', value: next ? 'on' : 'off' }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`compact ${next ? 'on' : 'off'}`))
    }
  },

  {
    aliases: ['detail'],
    help: 'control agent detail visibility (global or per-section)',
    name: 'details',
    run: (arg, ctx) => {
      const { gateway, transcript, ui } = ctx

      if (!arg) {
        gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'details_mode' })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            const mode = parseDetailsMode(r?.value) ?? ui.detailsMode
            patchUiState({ detailsMode: mode, detailsModeCommandOverride: false })

            const overrides = SECTION_NAMES.filter(s => ui.sections[s])
              .map(s => `${s}=${ui.sections[s]}`)
              .join(' ')

            transcript.sys(`details: ${mode}${overrides ? `  (${overrides})` : ''}`)
          })
          .catch(() => !ctx.stale() && transcript.sys(`details: ${ui.detailsMode}`))

        return
      }

      const [first, second] = arg.trim().toLowerCase().split(/\s+/)

      if (second && isSectionName(first)) {
        const reset = RESET_WORDS.has(second)
        const mode = reset ? null : parseDetailsMode(second)

        if (!reset && !mode) {
          return transcript.sys(DETAILS_SECTION_USAGE)
        }

        const { [first]: _drop, ...rest } = ui.sections

        patchUiState({ sections: mode ? { ...rest, [first]: mode } : rest })
        gateway
          .rpc<ConfigSetResponse>('config.set', { key: `details_mode.${first}`, value: mode ?? '' })
          .catch(() => {})
        transcript.sys(`details ${first}: ${mode ?? 'reset'}`)

        return
      }

      const next = CYCLE_WORDS.has(first ?? '') ? nextDetailsMode(ui.detailsMode) : parseDetailsMode(first)

      if (!next) {
        return transcript.sys(DETAILS_USAGE)
      }

      const sections = Object.fromEntries(SECTION_NAMES.map(section => [section, next]))

      patchUiState({ detailsMode: next, detailsModeCommandOverride: true, sections })
      gateway.rpc<ConfigSetResponse>('config.set', { key: 'details_mode', value: next }).catch(() => {})
      transcript.sys(`details: ${next}`)
    }
  },

  {
    help: 'local fortune',
    name: 'fortune',
    run: (arg, ctx) => {
      const key = arg.trim().toLowerCase()

      if (!arg || key === 'random') {
        return ctx.transcript.sys(randomFortune())
      }

      if (['daily', 'stable', 'today'].includes(key)) {
        return ctx.transcript.sys(dailyFortune(ctx.sid))
      }

      ctx.transcript.sys('usage: /fortune [random|daily]')
    }
  },

  {
    help: 'copy selection or assistant message',
    name: 'copy',
    run: async (arg, ctx) => {
      const { sys } = ctx.transcript

      if (!arg && ctx.composer.hasSelection) {
        const text = await ctx.composer.selection.copySelection()

        if (text) {
          return sys(`copied ${text.length} characters`)
        } else {
          return sys(
            'clipboard copy failed — try HERMES_TUI_FORCE_OSC52=1 to force the escape sequence'
          )
        }
      }

      if (arg && Number.isNaN(parseInt(arg, 10))) {
        return sys('usage: /copy [number]')
      }

      const all = ctx.local.getHistoryItems().filter(m => m.role === 'assistant')
      const target = all[arg ? Math.min(parseInt(arg, 10), all.length) - 1 : all.length - 1]

      if (!target) {
        return sys('nothing to copy — start a conversation first')
      }

      void writeClipboardText(target.text)
        .then(nativeOk => {
          if (ctx.stale()) {
            return
          }

          if (nativeOk) {
            sys('copied to clipboard')
          } else {
            writeOsc52Clipboard(target.text)
            sys('sent OSC52 copy sequence (terminal support required)')
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            sys(`copy failed: ${String(error)}`)
          }
        })
    }
  },

  {
    help: 'attach clipboard image',
    name: 'paste',
    run: (arg, ctx) => (arg ? ctx.transcript.sys('usage: /paste') : ctx.composer.paste())
  },

  {
    help: 'configure IDE terminal keybindings for multiline + undo/redo',
    name: 'terminal-setup',
    run: (arg, ctx) => {
      const target = arg.trim().toLowerCase()

      if (target && !['auto', 'cursor', 'vscode', 'windsurf'].includes(target)) {
        return ctx.transcript.sys('usage: /terminal-setup [auto|vscode|cursor|windsurf]')
      }

      const runner =
        !target || target === 'auto'
          ? configureDetectedTerminalKeybindings()
          : configureTerminalKeybindings(target as 'cursor' | 'vscode' | 'windsurf')

      void runner
        .then(result => {
          if (ctx.stale()) {
            return
          }

          ctx.transcript.sys(result.message)

          if (result.success && result.requiresRestart) {
            ctx.transcript.sys('restart the IDE terminal for the new keybindings to take effect')
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            ctx.transcript.sys(`terminal setup failed: ${String(error)}`)
          }
        })
    }
  },

  {
    help: 'view gateway logs',
    name: 'logs',
    run: (arg, ctx) => {
      const text = ctx.gateway.gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))

      text ? ctx.transcript.page(text, 'Logs') : ctx.transcript.sys('no gateway logs')
    }
  },

  {
    help: 'view current transcript (user + assistant messages)',
    name: 'history',
    run: (arg, ctx) => {
      // The CLI-side `/history` runs in a detached slash-worker subprocess
      // that never sees the TUI's turns — it only surfaces whatever was
      // persisted before this process started.  Render the TUI's own
      // transcript so `/history` actually reflects what the user just did.
      const items = ctx.local.getHistoryItems().filter(m => m.role === 'user' || m.role === 'assistant')

      if (!items.length) {
        return ctx.transcript.sys('no conversation yet')
      }

      const preview = Math.max(80, parseInt(arg, 10) || 400)

      const lines = items.map((m, i) => {
        const tag = m.role === 'user' ? `You #${i + 1}` : `Hermes #${i + 1}`
        const body = m.text.trim() || (m.tools?.length ? `(${m.tools.length} tool calls)` : '(empty)')
        const clipped = body.length > preview ? `${body.slice(0, preview).trimEnd()}…` : body

        return `[${tag}]\n${clipped}`
      })

      ctx.transcript.page(lines.join('\n\n'), 'History')
    }
  },

  {
    help: 'save the current transcript to JSON',
    name: 'save',
    run: (_arg, ctx) => {
      const hasConversation = ctx.local
        .getHistoryItems()
        .some(m => m.role === 'user' || m.role === 'assistant' || m.role === 'tool')

      if (!hasConversation) {
        return ctx.transcript.sys('no conversation yet')
      }

      if (!ctx.sid) {
        return ctx.transcript.sys('no active session — nothing to save')
      }

      ctx.gateway
        .rpc<SessionSaveResponse>('session.save', { session_id: ctx.sid })
        .then(
          ctx.guarded<SessionSaveResponse>(r => {
            const file = r?.file

            if (file) {
              ctx.transcript.sys(`conversation saved to: ${file}`)
            } else {
              ctx.transcript.sys('failed to save')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['sb'],
    help: 'status bar position (on|off|top|bottom)',
    name: 'statusbar',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const toggle: StatusBarMode = ctx.ui.statusBar === 'off' ? 'top' : 'off'

      const next: null | StatusBarMode =
        !mode || mode === 'toggle'
          ? toggle
          : mode === 'on' || mode === 'top'
            ? 'top'
            : mode === 'off' || mode === 'bottom'
              ? mode
              : null

      if (!next) {
        return ctx.transcript.sys('usage: /statusbar [on|off|top|bottom|toggle]')
      }

      patchUiState({ statusBar: next })
      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'statusbar', value: next }).catch(() => {})

      queueMicrotask(() => ctx.transcript.sys(`status bar ${next}`))
    }
  },

  {
    aliases: ['q'],
    help: 'inspect or enqueue a message',
    name: 'queue',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys(`${ctx.composer.queueRef.current.length} queued message(s)`)
      }

      ctx.composer.enqueue(arg)
      ctx.transcript.sys(`queued: "${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}"`)
    }
  },

  {
    help: 'inject a message after the next tool call (no interrupt)',
    name: 'steer',
    run: (arg, ctx) => {
      const payload = ctx.composer.expandPaste(arg?.trim() ?? '')

      if (!payload) {
        return ctx.transcript.sys('usage: /steer <prompt>')
      }

      // If the agent isn't running, fall back to the queue so the user's
      // message isn't lost — identical semantics to the gateway handler. But
      // when the renderer thinks it's idle only because a stream/WS drop marked
      // the turn stalled (the wedge-watchdog path), the durable session may
      // still be live in the gateway. Re-attach to it first so the queued
      // message actually lands on a real turn instead of a client-side queue
      // that never drains — the "steer queued / no active turn" dead end.
      if (!ctx.ui.busy || !ctx.sid) {
        ctx.composer.enqueue(payload)
        ctx.transcript.sys(
          `no active turn — queued for next: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
        )

        const resumeKey = ctx.ui.storedSid ?? ctx.sid

        if (resumeKey) {
          ctx.transcript.sys('re-attaching to your session so the queued message can land…')
          ctx.session.resumeById(resumeKey)
        }

        return
      }

      ctx.gateway
        .rpc<SessionSteerResponse>('session.steer', { session_id: ctx.sid, text: payload })
        .then(
          ctx.guarded<SessionSteerResponse>(r => {
            if (r?.status === 'queued') {
              ctx.transcript.sys(
                `steer queued — arrives after next tool call: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
              )
            } else {
              ctx.transcript.sys('steer rejected')
            }
          })
        )
        .catch((e: unknown) => {
          // The gateway reaped the in-memory session on a transient drop while
          // the renderer still shows it busy. Recover the durable session and
          // queue the steer so it lands on the reconnected turn instead of
          // surfacing a bare "session not found".
          const message = e instanceof Error ? e.message : String(e)

          if (/session not found/i.test(message)) {
            const resumeKey = ctx.ui.storedSid ?? ctx.sid

            if (resumeKey) {
              ctx.composer.enqueue(payload)
              ctx.transcript.sys('reconnecting to your session — your steer is queued to land on it')
              ctx.session.resumeById(resumeKey)

              return
            }
          }

          ctx.guardedErr(e)
        })
    }
  },

  {
    help: 'open a tool call\'s full content in the pager (default: latest)',
    name: 'inspect',
    usage: '/inspect [n]',
    run: (arg, ctx) => {
      // Mouse-off / Option-drag fallback for the click-to-inspect affordance
      // on tool-call rows. Collects the tool-call trail lines currently on
      // screen — the live turn first (turnStore: completed trail + any
      // in-flight tools), then the most recent persisted trail message from
      // the transcript — and opens the Nth (1-based, default = latest) in the
      // shared pager. Each line already carries the captured Args/Result block
      // (capped at VERBOSE_TRAIL_MAX_CHARS by the OOM guard); the pager shows
      // it in full, scrollable.
      const turn = getTurnState()
      const liveTools = turn.tools.map(tool => {
        const fullCtx = tool.contextFull || tool.context || ''
        const ctxPart = fullCtx ? `(${fullCtx})` : ''
        const argsPart = tool.verboseArgs ? `\nArgs:\n${tool.verboseArgs}` : ''

        return `${tool.name}${ctxPart}${argsPart}`.trim()
      })

      // Completed tool lines this turn live in streamSegments[*].tools and
      // streamPendingTools (not turnTrail, which only holds active/transient
      // rows). Gather every source the renderer shows, in visual order:
      // finished segments → pending → active-turn trail → in-flight tools.
      const segmentTools = turn.streamSegments.flatMap(seg => seg.tools ?? [])
      let lines: string[] = [
        ...segmentTools,
        ...turn.streamPendingTools,
        ...turn.turnTrail,
        ...liveTools
      ].filter(line => line && line !== 'analyzing tool output…' && !line.startsWith('drafting '))

      // Idle between turns: turnStore is reset, so fall back to the newest
      // transcript message that carries a tool trail.
      if (!lines.length) {
        const history = ctx.local.getHistoryItems()

        for (let i = history.length - 1; i >= 0; i--) {
          const msgTools = history[i]?.tools

          if (msgTools?.length) {
            lines = [...msgTools]
            break
          }
        }
      }

      if (!lines.length) {
        return ctx.transcript.sys('no tool calls to inspect yet')
      }

      const raw = arg.trim()
      const n = raw ? parseInt(raw, 10) : lines.length

      if (raw && (!Number.isInteger(n) || n < 1 || n > lines.length)) {
        return ctx.transcript.sys(`/inspect: pick 1–${lines.length} (found ${lines.length} tool call(s))`)
      }

      const chosen = lines[n - 1] ?? lines[lines.length - 1]!
      // Title = a SIMPLE tool label (e.g. "Terminal tool call") so the pager
      // header does NOT duplicate the full command that the body already shows.
      const firstLine = chosen.split('\n')[0] ?? 'Tool call'
      const title = toolCallInspectTitle(firstLine)

      inspectToolCall(`${title}  (${n}/${lines.length})`, chosen)
    }
  },

  {
    help: 'undo last exchange',
    name: 'undo',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('nothing to undo')
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) > 0) {
            ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
            ctx.transcript.sys(`undid ${r.removed} messages`)
          } else {
            ctx.transcript.sys('nothing to undo')
          }
        })
      )
    }
  },

  {
    help: 'retry last user message',
    name: 'retry',
    run: (_arg, ctx) => {
      const last = ctx.local.getLastUserMsg()

      if (!last) {
        return ctx.transcript.sys('nothing to retry')
      }

      if (!ctx.sid) {
        return ctx.transcript.send(last)
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) <= 0) {
            return ctx.transcript.sys('nothing to retry')
          }

          ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
          ctx.transcript.send(last)
        })
      )
    }
  }
]
