#!/usr/bin/env tsx
// Phase 49 — regression harness for the trust-boundary + helper fixes.
//
// Each FIXED site gets at least one assertion here. The tests are pure
// function-level checks; the ratchet guard (verify:ts-safety) covers the
// AST side and the silent-catch audit (verify:silent-catch) covers any new
// swallowing handlers. Most of these tests would have FAILED against the
// pre-Phase-49 code (their bodies describe what was wrong).

import {
  parseRunConfig,
} from '../src/training/parseRunConfig'
import { normalizeManifest } from '../src/datasets/manifestNormalize'
import { errMessage, isFiniteNumber } from '../src/errors/report'
import { isEvalSummary, latestEval } from '../src/training/charts/evaluation'
import { execFileSync } from 'node:child_process'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

console.log('phase 49: ts-safety fixes')

// 1. parseRunConfig — trust boundary on run.json
//
//   Was: `JSON.parse(raw) as RunConfig` — unvalidated cast, a torn file would
//   have made `cfg.dataset.relpath` throw later in a confusing place.
{
  check('parseRunConfig(null) throws "kein Objekt"',
    (() => { try { parseRunConfig(null); return false } catch (e) { return errMessage(e).includes('kein Objekt') } })())
  check('parseRunConfig("x") throws "kein Objekt"',
    (() => { try { parseRunConfig('x'); return false } catch (e) { return errMessage(e).includes('kein Objekt') } })())
  check('parseRunConfig(42) throws "kein Objekt"',
    (() => { try { parseRunConfig(42); return false } catch (e) { return errMessage(e).includes('kein Objekt') } })())
  check('parseRunConfig([1,2,3]) throws "kein Objekt"',
    (() => { try { parseRunConfig([1, 2, 3]); return false } catch (e) { return errMessage(e).includes('kein Objekt') } })())
  check('parseRunConfig({}) throws "dataset.relpath fehlt"',
    (() => { try { parseRunConfig({}); return false } catch (e) { return errMessage(e).includes('relpath fehlt') } })())
  check('parseRunConfig({dataset: null}) throws "dataset.relpath fehlt"',
    (() => { try { parseRunConfig({ dataset: null }); return false } catch (e) { return errMessage(e).includes('relpath fehlt') } })())
  check('parseRunConfig({dataset: {}}) throws "dataset.relpath fehlt"',
    (() => { try { parseRunConfig({ dataset: {} }); return false } catch (e) { return errMessage(e).includes('relpath fehlt') } })())
  check('parseRunConfig({dataset: {relpath: "x"}}) returns the value',
    (() => { try { return JSON.stringify(parseRunConfig({ dataset: { relpath: 'x' } })) === '{"dataset":{"relpath":"x"}}' } catch { return false } })())
  // hostile payload — __proto__ keys must not bleed into the result
  check('parseRunConfig hostile __proto__ survives as plain object', (() => {
    try {
      const hostile = JSON.parse('{"dataset":{"relpath":"x"},"__proto__":{"polluted":true}}')
      const cfg = parseRunConfig(hostile)
      // hostile.__proto__.polluted does not pollute Object.prototype because
      // JSON.parse does not parse __proto__ as a setter, and parseRunConfig
      // returns the literal object the user passed in.
      return cfg && (cfg as { dataset: { relpath: string } }).dataset.relpath === 'x'
    } catch { return false }
  })())
}

// 2. normalizeManifest — trust boundary on the manifest editor
//
//   Was: `function normalizeManifest(raw: any)` — explicit `any` parameter.
{
  check('normalizeManifest(null) returns the documented default', (() => {
    const cfg = normalizeManifest(null)
    return cfg.table === '' && cfg.branches.length === 1 && cfg.target.column === '' && cfg.target.type === 'regression' && cfg.cache === true
  })())
  check('normalizeManifest(undefined) returns the documented default', (() => {
    const cfg = normalizeManifest(undefined)
    return cfg.table === '' && cfg.branches.length === 1
  })())
  check('normalizeManifest("garbage") returns the documented default', (() => {
    const cfg = normalizeManifest('garbage')
    return cfg.table === ''
  })())
  check('normalizeManifest([1,2,3]) returns the documented default', (() => {
    const cfg = normalizeManifest([1, 2, 3])
    return cfg.table === ''
  })())
  check('normalizeManifest({pairs: {mol: {kind: "molecule"}}}) maps correctly', (() => {
    const cfg = normalizeManifest({ pairs: { mol: { kind: 'molecule' } } })
    return cfg.branches.length === 1 && cfg.branches[0].mode === 'molecule' && cfg.branches[0].name === 'mol'
  })())
  check('normalizeManifest hostile __proto__ does not throw', (() => {
    try { normalizeManifest(JSON.parse('{"__proto__":{"polluted":true}}')); return true } catch { return false }
  })())
  check('normalizeManifest({cache: false}) preserves cache=false', (() => {
    const cfg = normalizeManifest({ cache: false })
    return cfg.cache === false
  })())
  check('normalizeManifest({target: {type: "classification"}}) preserves type', (() => {
    const cfg = normalizeManifest({ target: { type: 'classification' } })
    return cfg.target.type === 'classification'
  })())
}

// 3. isEvalSummary / latestEval — trust boundary on the SSE eval.summary
//
//   Was: `return e as unknown as EvalSummary` — unvalidated cast on the
//   TrainingEvent (which itself was trusted upstream).
{
  check('isEvalSummary({}) false (missing task)', !isEvalSummary({}))
  check('isEvalSummary({task: 123}) false (task not string)', !isEvalSummary({ task: 123 }))
  check('isEvalSummary(null) false', !isEvalSummary(null))
  check('isEvalSummary("x") false', !isEvalSummary('x'))
  check('isEvalSummary({task: "classification"}) true', isEvalSummary({ task: 'classification' }))
  check('isEvalSummary({task: "regression"}) true', isEvalSummary({ task: 'regression' }))
  check('latestEval([]) === null', latestEval([]) === null)
  check('latestEval([{kind: "metric.epoch", epoch: 1}]) === null (no eval.summary)', latestEval([{ kind: 'metric.epoch', epoch: 1 } as never]) === null)
  check('latestEval([{kind: "eval.summary", task: "classification"}]) === summary', (() => {
    const r = latestEval([{ kind: 'eval.summary', task: 'classification' } as never])
    return r !== null && r.task === 'classification'
  })())
  check('latestEval rejects an eval.summary with wrong type', (() => {
    const r = latestEval([{ kind: 'eval.summary', task: 123 } as never])
    return r === null
  })())
  check('latestEval multitask payload (heads array) returns null', (() => {
    const r = latestEval([{ kind: 'eval.summary', heads: [{ output: 'a', task: 'classification' }] } as never])
    return r === null
  })())
}

// 4. errMessage — Phase 50's `catch (e)` bodies must not crash on
//    non-Error throws (the previous pattern `e.message` would throw
//    `Cannot read properties of undefined (reading 'message')`).
{
  check('errMessage(Error("x")) === "x"', errMessage(new Error('x')) === 'x')
  check('errMessage("plain string") === "plain string"', errMessage('plain string') === 'plain string')
  check('errMessage(null) coerces without throwing', (() => { try { errMessage(null); return true } catch { return false } })())
  check('errMessage(undefined) coerces without throwing', (() => { try { errMessage(undefined); return true } catch { return false } })())
  check('errMessage(42) coerces to a string', typeof errMessage(42) === 'string')
  check('errMessage({code: "ENOENT"}) coerces without throwing', (() => { try { errMessage({ code: 'ENOENT' }); return true } catch { return false } })())
}

// 5. isFiniteNumber — NaN/Infinity must not silently propagate into shapes,
//    ids, epochs.
{
  check('isFiniteNumber(1) true', isFiniteNumber(1))
  check('isFiniteNumber(0) true', isFiniteNumber(0))
  check('isFiniteNumber(-1) true', isFiniteNumber(-1))
  check('isFiniteNumber(NaN) false', !isFiniteNumber(NaN))
  check('isFiniteNumber(Infinity) false', !isFiniteNumber(Infinity))
  check('isFiniteNumber(-Infinity) false', !isFiniteNumber(-Infinity))
  check('isFiniteNumber("1") false (string is not a number)', !isFiniteNumber('1'))
  check('isFiniteNumber(undefined) false', !isFiniteNumber(undefined))
  check('isFiniteNumber(null) false', !isFiniteNumber(null))
}

// 6. ratchet guard itself — the AST scan must detect the same sites we
//    inventory here. This catches a future change to the detector that
//    silently drops a category (e.g. someone removing `non-null` checks).
{
  const out = execFileSync('npx', ['tsx', 'scripts/verify-ts-safety.ts'], { encoding: 'utf8' })
  check('verify:ts-safety exits 0', /\u2713 all sites are commented and documented/.test(out))
  check('verify:ts-safety reports the expected non-null count', /non-null: 30/.test(out))
  check('verify:ts-safety reports the expected as-unknown-as count', /as-unknown-as: 3/.test(out))
}

// 7. global error handlers — unhandled rejections/errors reach the diagnostics store,
//    benign noise (intentional aborts, ResizeObserver loop) does not raise a banner.
//    Was: nothing listened — an unhandled rejection vanished into the console.
{
  const target = new EventTarget()
  ;(globalThis as unknown as { window: EventTarget }).window = target
  const { useDiagnostics } = await import('../src/errors/report')
  await import('../src/errors/globalHandlers')
  const fire = (type: string, props: Record<string, unknown>): Event => {
    const ev = new Event(type, { cancelable: true })
    Object.assign(ev, props)
    target.dispatchEvent(ev)
    return ev
  }
  useDiagnostics.getState().clear()
  const rej = fire('unhandledrejection', { reason: new Error('boom in a promise') })
  const entries1 = useDiagnostics.getState().entries
  check('unhandledrejection is recorded', entries1.length === 1 && entries1[0].message === 'boom in a promise' && entries1[0].context === 'window.unhandledrejection')
  check('unhandledrejection default is prevented (no duplicate console noise)', rej.defaultPrevented)
  fire('unhandledrejection', { reason: new Error('boom in a promise') })
  check('duplicates collapse with a count', useDiagnostics.getState().entries.length === 1 && useDiagnostics.getState().entries[0].count === 2)
  fire('error', { error: new Error('sync throw'), message: 'sync throw', filename: 'x.js', lineno: 1, colno: 2 })
  check('window error is recorded', useDiagnostics.getState().entries.some((e) => e.message === 'sync throw'))
  useDiagnostics.getState().clear()
  fire('unhandledrejection', { reason: new DOMException('aborted by the user', 'AbortError') })
  check('an intentional AbortError raises NO banner', useDiagnostics.getState().entries.length === 0)
  fire('error', { error: undefined, message: 'ResizeObserver loop completed with undelivered notifications.', filename: '', lineno: 0, colno: 0 })
  check('ResizeObserver loop noise raises NO banner', useDiagnostics.getState().entries.length === 0)
  fire('error', { error: undefined, message: 'Uncaught TypeError: x is undefined', filename: 'a.js', lineno: 3, colno: 4 })
  check('a message-only error (no Error object) is still recorded', useDiagnostics.getState().entries.length === 1)
}

if (failures > 0) {
  console.error(`\n${failures} ts-safety fix failure(s)`)
  process.exit(1)
}
console.log('\n✓ all ts-safety fixes verified')
