/** Hermes reasoning-effort vocabulary for Desktop controls. */

export const REASONING_EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const
export type ReasoningEffort = (typeof REASONING_EFFORTS)[number]
export const REASONING_EFFORT_VALUES = ['none', ...REASONING_EFFORTS] as const
export const DEFAULT_REASONING_EFFORT: ReasoningEffort = 'medium'

const SHORT_LABELS: Record<string, string> = {
  none: 'Off',
  minimal: 'Min',
  low: 'Low',
  medium: 'Med',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max'
}

const normalize = (value: string): string => value.trim().toLowerCase()

export function reasoningEffortLabel(effort: string): string {
  const key = normalize(effort)

  return key ? (SHORT_LABELS[key] ?? SHORT_LABELS[DEFAULT_REASONING_EFFORT]) : ''
}

export const isReasoningEffort = (value: string): value is ReasoningEffort =>
  REASONING_EFFORTS.includes(normalize(value) as ReasoningEffort)

export function isThinkingEnabled(effort: string, fallback: string = DEFAULT_REASONING_EFFORT): boolean {
  return normalize(effort || fallback) !== 'none'
}

export function resolveReasoningEffort(effort: string, fallback: string = DEFAULT_REASONING_EFFORT): string {
  const value = normalize(effort || fallback)

  if (value === 'none') {
    return ''
  }

  return isReasoningEffort(value) ? value : DEFAULT_REASONING_EFFORT
}
