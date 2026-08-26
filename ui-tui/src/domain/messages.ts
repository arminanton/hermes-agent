import { LONG_MSG } from '../config/limits.js'
import { buildToolTrailLine, fmtK, formatToolCall } from '../lib/text.js'
import type { Msg, SessionInfo } from '../types.js'

export const introMsg = (info: SessionInfo): Msg => ({ info, kind: 'intro', role: 'system', text: '' })

export const imageTokenMeta = (info?: ImageMeta | null) => {
  const { width, height, token_estimate: t } = info ?? {}
  const dims = width && height ? `${width}x${height}` : ''
  const path = (info?.display_path || info?.path || '').trim()
  const tok = (t ?? 0) > 0 ? `~${fmtK(t!)} tok` : ''

  // Order: dimensions · path · token-estimate. The path sits in the middle so
  // the user can locate/copy the exact file and the model has an explicit handle
  // to it (the same on-host path is also threaded into the prompt content).
  return [dims, path, tok].filter(Boolean).join(' · ')
}

export const attachedImageNotice = (info?: ({ name?: string } & ImageMeta) | null) => {
  const meta = imageTokenMeta(info)
  const label = info?.name ? `📎 Attached image: ${info.name}` : '📎 Attached image'

  return `${label}${meta ? ` · ${meta}` : ''}`
}

export const userDisplay = (text: string) => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return `${prefix || '(message)'} [long message]`
}

export const toTranscriptMessages = (rows: unknown): Msg[] => {
  if (!Array.isArray(rows)) {
    return []
  }

  const out: Msg[] = []
  let pending: string[] = []

  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const {
      context,
      name,
      pending: isPending,
      reasoning,
      reasoning_content: reasoningContent,
      role,
      status,
      text
    } = row as TranscriptRow

    if (role === 'tool') {
      pending.push(
        status === 'interrupted'
          ? `${formatToolCall(name ?? 'tool', context ?? '')} :: interrupted ✗`
          : isPending
            ? `${formatToolCall(name ?? 'tool', context ?? '')} …`
            : buildToolTrailLine(name ?? 'tool', context ?? '')
      )

      continue
    }

    if (role === 'assistant') {
      const visibleText = typeof text === 'string' ? text : ''

      const thinking = [reasoning, reasoningContent].find(value => typeof value === 'string' && value.trim())

      if (!visibleText.trim() && !thinking) {
        continue
      }

      out.push({
        role,
        text: visibleText,
        ...(thinking && { thinking }),
        ...(pending.length && { tools: pending })
      })
      pending = []
    } else if (role === 'user' || role === 'system') {
      if (typeof text !== 'string' || !text.trim()) {
        continue
      }

      out.push({ role, text })
      pending = []
    }
  }

  if (pending.length) {
    const last = out.at(-1)

    if (last?.role === 'assistant') {
      out[out.length - 1] = { ...last, tools: [...(last.tools ?? []), ...pending] }
    } else {
      out.push({ kind: 'trail', role: 'system', text: '', tools: pending })
    }
  }

  return out
}

export const fmtDuration = (ms: number) => {
  const t = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60

  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`
}

interface ImageMeta {
  display_path?: string
  height?: number
  path?: string
  token_estimate?: number
  width?: number
}

interface TranscriptRow {
  context?: string
  name?: string
  pending?: boolean
  reasoning?: string
  reasoning_content?: string
  role?: string
  status?: string
  text?: string
}
