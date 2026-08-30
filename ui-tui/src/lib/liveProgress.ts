import type { Msg, TodoItem } from '../types.js'

export const countPendingTodos = (todos: readonly TodoItem[]) =>
  todos.filter(todo => todo.status === 'in_progress' || todo.status === 'pending').length

export const isTodoDone = (todos: readonly TodoItem[]) =>
  todos.length > 0 && todos.every(todo => todo.status === 'completed' || todo.status === 'cancelled')

export const isToolShelfMessage = (msg: Msg | undefined) =>
  Boolean(msg?.kind === 'trail' && !msg.text && !msg.thinking?.trim() && msg.tools?.length)

export const canHoldToolShelf = (msg: Msg | undefined) =>
  Boolean(msg?.kind === 'trail' && !msg.text && (msg.thinking?.trim() || msg.tools?.length))

export const mergeToolShelfInto = (target: Msg, source: Msg): Msg => ({
  ...target,
  tools: [...(target.tools ?? []), ...(source.tools ?? [])]
})

const isBarrierMessage = (msg: Msg | undefined) => {
  if (!msg) {
    return true
  }

  // Assistant text, user input, intro/panel rows all terminate the shelf.
  if (msg.kind === 'intro' || msg.kind === 'panel' || msg.kind === 'diff') {
    return true
  }

  if (msg.role && msg.role !== 'system') {
    return true
  }

  if (msg.text) {
    return true
  }

  // A reasoning segment terminates the shelf too. Without this, tool calls made
  // AFTER a block of thinking merge backward into a shelf that sits BEFORE it,
  // so the transcript shows one large group of tools followed by one large group
  // of thoughts and the causal order is lost: the reader sees a tool call above
  // the reasoning that actually decided to make it. Reasoning is a real event in
  // the turn, so it bounds a shelf exactly like assistant text does. Tools that
  // genuinely ran together with no thinking between them still merge, which is
  // the grouping this shelf exists to provide.
  if (msg.thinking?.trim()) {
    return true
  }

  return false
}

const isToolCarryingTrail = (msg: Msg | undefined) => Boolean(msg?.kind === 'trail' && !msg.text && msg.tools?.length)

export const appendToolShelfMessage = (prev: readonly Msg[], msg: Msg): Msg[] => {
  if (!isToolShelfMessage(msg)) {
    return [...prev, msg]
  }

  let fallbackHolder: number | null = null

  for (let index = prev.length - 1; index >= 0; index--) {
    const candidate = prev[index]

    if (isToolCarryingTrail(candidate)) {
      const next = [...prev]

      next[index] = mergeToolShelfInto(candidate!, msg)

      return next
    }

    if (fallbackHolder === null && canHoldToolShelf(candidate)) {
      fallbackHolder = index
    }

    if (isBarrierMessage(candidate)) {
      break
    }
  }

  if (fallbackHolder !== null) {
    const next = [...prev]

    next[fallbackHolder] = mergeToolShelfInto(prev[fallbackHolder]!, msg)

    return next
  }

  return [...prev, msg]
}
