import { EventEmitter } from 'events'
import React from 'react'
import { describe, expect, it } from 'vitest'

import Text from './components/Text.js'
import Ink from './ink.js'
import { CURSOR_HOME, ERASE_SCREEN } from './termio/csi.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 20
  rows = 5
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()
    return true
  }
}

const tick = () => new Promise<void>(resolve => queueMicrotask(resolve))

describe('Ink resize healing', () => {
  it('heals same-dimension alt-screen resize events with an erase before repaint', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const ink = new Ink({
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()
    stdout.chunks = []

    stdout.emit('resize')
    ink.onRender()
    await tick()

    expect(stdout.chunks.join('')).toContain(ERASE_SCREEN + CURSOR_HOME)

    ink.unmount()
  })

  it('forceRedraw in alt-screen emits an erase-before-repaint (resize-grade heal)', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const ink = new Ink({
      exitOnCtrlC: false,
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()
    stdout.chunks = []

    // The auto-healer / Ctrl+L / /redraw path. Previously this emitted only a
    // bare ERASE_SCREEN with no needsEraseBeforePaint, so the atomic
    // erase-in-preamble (the part that actually un-garbles a desynced screen)
    // never ran. Now it must route through the resize-grade recovery.
    ink.forceRedraw()
    await tick()

    expect(stdout.chunks.join('')).toContain(ERASE_SCREEN + CURSOR_HOME)

    ink.unmount()
  })
})
