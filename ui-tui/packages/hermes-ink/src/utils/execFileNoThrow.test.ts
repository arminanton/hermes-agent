import { chmodSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { execFileNoThrow } from './execFileNoThrow.js'

// These tests shell out to /bin/sh, use chmodSync(0o755), and rely on
// POSIX sleep/job control. They will not work on Windows.
const onWindows = process.platform === 'win32'

// We simulate `wl-copy`'s daemonization behavior with a tiny shell script:
//   1. Fork a short-lived background sleeper that inherits stdio (so the
//      parent process's pipes can never close).
//   2. Record the sleeper PID to a file so afterEach can clean it up.
//   3. Exit immediately with status 0.
//
// Without resolveOnExit, the await on `'close'` hangs until SIGTERM at
// timeout — exactly the production wl-copy bug. With resolveOnExit, the
// promise settles on `'exit'` regardless of the inherited pipes.

let scriptDir: string
let daemonScript: string
let sleeperPids: number[]

/** Read the PID file the daemon script writes, and track it for afterEach cleanup. */
function trackSleeperPid(pidFile: string): void {
  try {
    const pid = parseInt(readFileSync(pidFile, 'utf8').trim(), 10)
    if (pid > 0) {
      sleeperPids.push(pid)
    }
  } catch {
    // PID file not written or unreadable — sleeper may have already exited.
  }
}

beforeEach(() => {
  sleeperPids = []
  scriptDir = join(tmpdir(), `hermes-execfile-test-${process.pid}-${Date.now()}`)
  mkdirSync(scriptDir, { recursive: true })
  daemonScript = join(scriptDir, 'fake-daemonizer.sh')
  // Posix sh: the `sleep 3 &` child inherits stdin/stdout/stderr from the
  // shell, which inherited them from `spawn(stdio: 'pipe')`. The shell
  // exits but its child (the sleeper) keeps the pipes open. Mirrors how
  // wl-copy double-forks then exits while the daemon holds the selection.
  // The sleeper writes its PID to $1 so we can clean it up reliably.
  writeFileSync(daemonScript, '#!/bin/sh\nsleep 3 &\necho $! > "$1"\nexit 0\n')
  chmodSync(daemonScript, 0o755)
})

afterEach(() => {
  // Kill orphaned sleepers so they don't accumulate across watch runs.
  for (const pid of sleeperPids) {
    try {
      process.kill(pid, 'SIGKILL')
    } catch {
      // Already exited — fine.
    }
  }
  rmSync(scriptDir, { recursive: true, force: true })
})

describe.skipIf(onWindows)('execFileNoThrow with daemon-style children', () => {
  // Regression guard for the wl-copy clipboard hang that motivated the
  // resolveOnExit option. A daemon-style child forks a background process that
  // inherits stdio, then exits 0 immediately. WITHOUT resolveOnExit the promise
  // waits on 'close', which cannot fire until the orphaned daemon releases the
  // inherited pipes — so the call is held hostage to the daemon's whole
  // lifetime even though a timeout was set. WITH resolveOnExit it settles on the
  // child's own 'exit', returning promptly regardless of the daemon. This pins
  // that observable difference so a refactor can't silently revert osc.ts's
  // clipboard spawns back to the hanging path.
  //
  // Previously this contrast lived in a permanently `it.skip`'d "documented
  // hang" test on the false premise that the no-resolveOnExit path hangs
  // forever. Measured (2026-07-13, Linux): it does NOT hang — it resolves with
  // code 124 once the daemon's own `sleep` ends (resolve time tracks the daemon
  // lifetime: 1s→~1005ms, 2s→~2004ms). Because the daemon self-terminates, the
  // case is bounded and fully runnable, so it is now a real test.
  it('without resolveOnExit, the call is held until the inherited-stdio daemon releases the pipes', async () => {
    // Own 1s daemon (the shared daemonScript sleeps 3s — needlessly slow here).
    // The call sets timeout:300, but the non-resolveOnExit path settles on
    // 'close', which only fires after the daemon exits. So the promise resolves
    // at ~the daemon lifetime, flagged timed-out (code 124), NOT at 300ms. This
    // is exactly the latency wl-copy exhibited.
    const holdScript = join(scriptDir, 'hold-1s.sh')
    const pidFile = join(scriptDir, 'sleeper-hold.pid')
    writeFileSync(holdScript, '#!/bin/sh\nsleep 1 &\necho $! > "$1"\nexit 0\n')
    chmodSync(holdScript, 0o755)
    const start = Date.now()

    const result = await execFileNoThrow(holdScript, [pidFile], { timeout: 300 })
    trackSleeperPid(pidFile)

    const elapsed = Date.now() - start

    expect(result.code).toBe(124)
    // Held past the 300ms timeout, up toward the 1s daemon lifetime — proof the
    // timeout could NOT settle it early on the 'close' path. Upper-bounded too
    // so a genuine forever-hang regression (the original fear) still fails loud
    // instead of silently blowing the suite timeout.
    expect(elapsed).toBeGreaterThan(700)
    expect(elapsed).toBeLessThan(2500)
  })

  it("settles immediately on 'exit' when resolveOnExit is true, regardless of daemon stdio", async () => {
    const pidFile = join(scriptDir, 'sleeper-exit.pid')
    const start = Date.now()

    const result = await execFileNoThrow(daemonScript, [pidFile], {
      timeout: 2000,
      resolveOnExit: true
    })
    trackSleeperPid(pidFile)

    const elapsed = Date.now() - start

    // The shell exits in a few ms. resolveOnExit lets us return on exit
    // (code 0) instead of waiting for the orphaned sleeper to release
    // stdio. Should be well under 200ms even on slow CI.
    expect(result.code).toBe(0)
    expect(elapsed).toBeLessThan(500)
  })

  it("still surfaces the right code when resolveOnExit'd child exits non-zero", async () => {
    const pidFile = join(scriptDir, 'sleeper-fail.pid')
    const failScript = join(scriptDir, 'fail.sh')
    writeFileSync(failScript, `#!/bin/sh\nsleep 3 &\necho $! > "${pidFile}"\nexit 7\n`)
    chmodSync(failScript, 0o755)

    const result = await execFileNoThrow(failScript, [], {
      timeout: 2000,
      resolveOnExit: true
    })
    trackSleeperPid(pidFile)

    expect(result.code).toBe(7)
  })

  it('settles on timeout=124 when the child itself never exits, even with resolveOnExit', async () => {
    const slowScript = join(scriptDir, 'slow.sh')
    writeFileSync(slowScript, '#!/bin/sh\nsleep 30\n')
    chmodSync(slowScript, 0o755)

    const result = await execFileNoThrow(slowScript, [], {
      timeout: 200,
      resolveOnExit: true
    })

    // Child process never exits on its own → timer fires → SIGTERM →
    // child exits → 'exit' fires with non-null signal. The settle()
    // call from the timer registers code=124 first. Either way: 124.
    expect(result.code).toBe(124)
  })

  it('does not double-resolve when both timer and exit fire', async () => {
    const pidFile = join(scriptDir, 'sleeper-race.pid')
    // Race: child happens to exit right around the timeout. The settled
    // guard ensures only the first resolution wins.
    const result = await execFileNoThrow(daemonScript, [pidFile], {
      timeout: 50, // very tight
      resolveOnExit: true
    })
    trackSleeperPid(pidFile)

    // Either code=0 (exit beat timer) or code=124 (timer beat exit).
    // Both are valid outcomes; the contract is that the promise settles
    // exactly once and doesn't throw.
    expect([0, 124]).toContain(result.code)
  })
})
