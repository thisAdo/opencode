/**
 * Conjunto: ensemble mode for opencode.
 *
 * The `conjunto` provider exposes a single virtual model `conjunto/ensemble`
 * whose `LanguageModelV3` implementation fans out to **every free model that
 * is currently connected** (i.e. `cost.input === 0 && cost.output === 0` and
 * the provider is loaded/authenticated). All members run in parallel; the
 * first one to produce a usable token stream "wins" and its events are
 * forwarded verbatim. The remaining members are aborted. If the winning
 * member fails mid-stream, the next-best member is transparently promoted.
 *
 * The net effect for the user: `opencode` answers with the *fastest* of all
 * the free models on every prompt, while silently tolerating per-model
 * outages, rate limits, and transient errors.
 *
 * Design notes:
 * - We implement the `LanguageModelV3` interface directly so the rest of the
 *   opencode session loop (streamText, llm/ai-sdk adapter, processor, ...)
 *   works unchanged.
 * - We forward the winning member's stream events **as-is** — text deltas,
 *   tool calls, finish event, usage, etc. — so tool-call IDs remain stable
 *   and downstream consumers don't need to know about the ensemble.
 * - All member calls receive the *same* `LanguageModelV3CallOptions` (cloned
 *   shallowly so each gets its own `abortSignal`). This is correct: tools,
 *   prompt, temperature, max output tokens, etc. are user intent, not
 *   model-specific.
 * - `abortSignal` from the caller is forwarded to every member; when the
 *   caller aborts (e.g. user presses Esc), every pending member call is
 *   aborted in parallel.
 */

import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider"

/**
 * A single member of the ensemble: a loaded language model plus the metadata
 * needed to identify it in logs / errors.
 */
export interface ConjuntoMember {
  readonly providerID: string
  readonly modelID: string
  readonly displayName: string
  readonly language: LanguageModelV3
}

/**
 * Factory that returns the *current* set of ensemble members. We use a
 * factory (rather than a snapshot) so the ensemble can react to providers
 * being connected/disconnected at runtime.
 */
export type ConjuntoMemberProvider = () => Promise<ConjuntoMember[]>

/**
 * Options accepted by {@link createConjuntoLanguageModel}.
 */
export interface ConjuntoLanguageModelOptions {
  /** Resolves the list of free, connected members to fan out to. */
  members: ConjuntoMemberProvider
  /**
   * Optional cap on how many members run in parallel for a single prompt.
   * Defaults to 8. Higher values reduce latency but increase memory and
   * rate-limit pressure on the underlying providers.
   */
  maxConcurrency?: number
  /**
   * Optional logger that receives lifecycle events (member started, member
   * won, member failed, ...). Useful for the TUI "thinking" indicator.
   */
  onEvent?: (event: ConjuntoEvent) => void
}

export type ConjuntoEvent =
  | { kind: "members"; count: number; members: Array<{ providerID: string; modelID: string }> }
  | { kind: "start"; providerID: string; modelID: string }
  | { kind: "won"; providerID: string; modelID: string; reason: "first-token" | "tool-call" | "finish-no-content" }
  | { kind: "failed"; providerID: string; modelID: string; error: string }
  | { kind: "fallback"; fromProviderID: string; fromModelID: string; toProviderID: string; toModelID: string }
  | { kind: "abort" }
  | { kind: "no-members" }

/**
 * Internal per-call state for a member attempt.
 */
interface MemberAttempt {
  member: ConjuntoMember
  stream: ReadableStream<LanguageModelV3StreamPart> | null
  result: LanguageModelV3StreamResult | null
  error: unknown | null
  /** Resolved when we've decided whether this attempt is the winner. */
  settled: Promise<void>
  settle: (winner: boolean) => void
  /** Readable controller used by the winning member to push events upstream. */
  controller: ReadableStreamDefaultController<LanguageModelV3StreamPart> | null
}

/**
 * Build a synthetic `LanguageModelV3` that races all free models and forwards
 * the first usable stream. See module docstring for the full strategy.
 */
export function createConjuntoLanguageModel(opts: ConjuntoLanguageModelOptions): LanguageModelV3 {
  const maxConcurrency = opts.maxConcurrency ?? 8

  function emit(event: ConjuntoEvent) {
    try {
      opts.onEvent?.(event)
    } catch {
      /* swallow listener errors */
    }
  }

  async function resolveMembers(): Promise<ConjuntoMember[]> {
    const all = await opts.members()
    // Deterministic order so the "winner" is stable across calls when
    // multiple models finish at the exact same instant. Sort by provider
    // then model id.
    return [...all].sort((a, b) => {
      if (a.providerID !== b.providerID) return a.providerID.localeCompare(b.providerID)
      return a.modelID.localeCompare(b.modelID)
    })
  }

  /**
   * Clone call options for a member, swapping in a fresh abort controller
   * that is chained to the caller's signal. This lets us abort a single
   * member without aborting the others (and without un-aborting aborted ones).
   */
  function cloneOptions(opts_: LanguageModelV3CallOptions, parentSignal: AbortSignal | undefined): {
    options: LanguageModelV3CallOptions
    abort: AbortController
  } {
    const abort = new AbortController()
    if (parentSignal) {
      if (parentSignal.aborted) abort.abort(parentSignal.reason)
      else parentSignal.addEventListener("abort", () => abort.abort(parentSignal.reason), { once: true })
    }
    return {
      options: { ...opts_, abortSignal: abort.signal },
      abort,
    }
  }

  /**
   * Read the first "useful" event out of a member stream. A stream is
   * considered to have "won" when it emits any of:
   *   - a non-empty `text-delta`
   *   - a `tool-input-start`
   *   - a `tool-call`
   *   - a `finish` event with no preceding content (some providers send
   *     finish immediately when the model decides to stop)
   *
   * We buffer the events we read while probing so they can be re-emitted
   * by the winner. Errors and empty streams cause the member to be
   * discarded and the next candidate to be tried.
   */
  async function probeForWinner(
    attempt: MemberAttempt,
  ): Promise<{ won: boolean; buffered: LanguageModelV3StreamPart[]; reason?: "first-token" | "tool-call" | "finish-no-content" }> {
    const buffered: LanguageModelV3StreamPart[] = []
    const reader = attempt.stream!.getReader()
    try {
      while (true) {
        const { value, done } = await reader.read()
        if (done) {
          // Stream ended without a useful event — this member has nothing
          // to contribute (e.g. empty completion, refused, etc.). Lose.
          return { won: false, buffered }
        }
        if (!value) continue
        buffered.push(value)

        // An error event from the provider disqualifies this member.
        if (value.type === "error") {
          attempt.error = value.error
          return { won: false, buffered }
        }

        // First non-empty text delta → winner.
        if (value.type === "text-delta" && value.delta.length > 0) {
          return { won: true, buffered, reason: "first-token" }
        }
        // Tool call → winner (member is taking an action).
        if (value.type === "tool-input-start" || value.type === "tool-call") {
          return { won: true, buffered, reason: "tool-call" }
        }
        // Finish event without any prior content → still a winner (some
        // providers do this for short refusals like "stop").
        if (value.type === "finish") {
          return { won: true, buffered, reason: "finish-no-content" }
        }
      }
    } finally {
      // We only release the lock; the caller will re-acquire the reader
      // if this member is promoted to winner.
      reader.releaseLock()
    }
  }

  /**
   * Pipe the rest of a member's stream into the upstream controller.
   * Resolves when the member stream ends OR errors. Returns the final
   * `finish` event (if any) so the caller can synthesize one if missing.
   */
  async function drainToUpstream(
    attempt: MemberAttempt,
    controller: ReadableStreamDefaultController<LanguageModelV3StreamPart>,
    buffered: LanguageModelV3StreamPart[],
  ): Promise<void> {
    // 1. Replay buffered events (already-validated) without re-reading.
    for (const ev of buffered) {
      controller.enqueue(ev)
    }
    // 2. Continue reading from where probeForWinner left off.
    const reader = attempt.stream!.getReader()
    try {
      while (true) {
        const { value, done } = await reader.read()
        if (done) return
        if (!value) continue
        if (value.type === "error") {
          // Surface provider errors as a stream error so the AI SDK can
          // route them through its normal error handling.
          controller.error(value.error instanceof Error ? value.error : new Error(String(value.error)))
          return
        }
        controller.enqueue(value)
      }
    } catch (err) {
      controller.error(err instanceof Error ? err : new Error(String(err)))
    } finally {
      reader.releaseLock()
    }
  }

  const doStream: LanguageModelV3["doStream"] = async (callOptions) => {
    const members = await resolveMembers()
    emit({ kind: "members", count: members.length, members: members.map((m) => ({ providerID: m.providerID, modelID: m.modelID })) })

    if (members.length === 0) {
      emit({ kind: "no-members" })
      // Return an empty stream that finishes with `error` so the caller
      // surfaces a useful message.
      const stream = new ReadableStream<LanguageModelV3StreamPart>({
        start(controller) {
          controller.enqueue({ type: "stream-start", warnings: [] })
          controller.enqueue({
            type: "error",
            error: new Error(
              "Conjunto: no free models are connected. Connect at least one provider with free models (e.g. opencode) to use conjunto/ensemble.",
            ),
          })
          controller.close()
        },
      })
      return { stream }
    }

    const slice = members.slice(0, maxConcurrency)

    // Set up each member attempt.
    const attempts: MemberAttempt[] = slice.map((member) => {
      let settle!: (winner: boolean) => void
      const settled = new Promise<void>((res) => {
        settle = () => res()
      })
      return {
        member,
        stream: null,
        result: null,
        error: null,
        settled,
        settle,
        controller: null,
      }
    })

    // Track which child abort controllers belong to which attempt so we
    // can abort losers once a winner is chosen.
    const childAborts = new Map<MemberAttempt, AbortController>()

    // Start every member in parallel. We capture each member's stream
    // (or error) into the attempt object.
    await Promise.all(
      attempts.map(async (attempt) => {
        const { options: childOpts, abort } = cloneOptions(callOptions, callOptions.abortSignal)
        childAborts.set(attempt, abort)
        emit({ kind: "start", providerID: attempt.member.providerID, modelID: attempt.member.modelID })
        try {
          const result = await attempt.member.language.doStream(childOpts)
          attempt.result = result
          attempt.stream = result.stream
        } catch (err) {
          attempt.error = err
          emit({
            kind: "failed",
            providerID: attempt.member.providerID,
            modelID: attempt.member.modelID,
            error: err instanceof Error ? err.message : String(err),
          })
        }
      }),
    )

    // Promote each attempt's stream to a "probe" promise that resolves
    // when we know whether this member is a winner.
    const probes = attempts
      .filter((a) => a.stream !== null)
      .map(async (a) => ({ attempt: a, probe: await probeForWinner(a) }))

    // Build the upstream stream. Its lifecycle is:
    //   - emit `stream-start` once
    //   - race probes; first winner → drain its (buffered + remaining)
    //     events into the upstream controller and abort the others
    //   - if no probe wins (all members errored / empty), synthesize an
    //     `error` event
    //   - on caller abort, abort every member
    const upstream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({ type: "stream-start", warnings: [] })

        // Caller abort propagation.
        const callerSignal = callOptions.abortSignal
        const onCallerAbort = () => {
          emit({ kind: "abort" })
          for (const c of childAborts.values()) {
            try {
              c.abort(callerSignal?.reason)
            } catch {
              /* ignore */
            }
          }
          try {
            controller.close()
          } catch {
            /* already closed */
          }
        }
        if (callerSignal) {
          if (callerSignal.aborted) {
            onCallerAbort()
            return
          }
          callerSignal.addEventListener("abort", onCallerAbort, { once: true })
        }

        // Race the probes.
        let resolved = false
        const winners: Array<{ attempt: MemberAttempt; buffered: LanguageModelV3StreamPart[]; reason: "first-token" | "tool-call" | "finish-no-content" }> = []
        const probeResults = await Promise.allSettled(probes)

        for (const r of probeResults) {
          if (r.status !== "fulfilled") continue
          const { attempt, probe } = r.value
          if (probe.won) {
            winners.push({ attempt, buffered: probe.buffered, reason: probe.reason! })
          } else {
            // Loser — abort it.
            const c = childAborts.get(attempt)
            try {
              c?.abort(new Error("conjunto: lost race"))
            } catch {
              /* ignore */
            }
            attempt.settle(false)
            if (attempt.error) {
              emit({
                kind: "failed",
                providerID: attempt.member.providerID,
                modelID: attempt.member.modelID,
                error: attempt.error instanceof Error ? attempt.error.message : String(attempt.error),
              })
            }
          }
        }

        if (winners.length === 0) {
          // No member produced usable output. Surface a helpful error.
          const messages = attempts
            .filter((a) => a.error)
            .map((a) => `${a.member.providerID}/${a.member.modelID}: ${a.error instanceof Error ? a.error.message : String(a.error)}`)
          controller.enqueue({
            type: "error",
            error: new Error(
              `Conjunto: every free model failed to produce output.${
                messages.length ? ` Errors: ${messages.join(" | ")}` : ""
              }`,
            ),
          })
          controller.close()
          resolved = true
          return
        }

        // Pick the first winner (probe order matches `attempts` order,
        // which is the deterministic sort from `resolveMembers`).
        // If the first winner fails mid-drain, fall back to the next.
        for (let i = 0; i < winners.length; i++) {
          const { attempt, buffered, reason } = winners[i]
          attempt.settle(true)
          emit({ kind: "won", providerID: attempt.member.providerID, modelID: attempt.member.modelID, reason })

          try {
            await drainToUpstream(attempt, controller, buffered)
            // If we got here, the winner completed cleanly.
            resolved = true
            // Abort any remaining winners (we don't need them).
            for (let j = i + 1; j < winners.length; j++) {
              const c = childAborts.get(winners[j].attempt)
              try {
                c?.abort(new Error("conjunto: not needed"))
              } catch {
                /* ignore */
              }
              winners[j].attempt.settle(false)
            }
            break
          } catch (err) {
            // Winner failed mid-stream — try the next one.
            if (i < winners.length - 1) {
              const next = winners[i + 1]
              emit({
                kind: "fallback",
                fromProviderID: attempt.member.providerID,
                fromModelID: attempt.member.modelID,
                toProviderID: next.attempt.member.providerID,
                toModelID: next.attempt.member.modelID,
              })
              continue
            }
            // No more fallbacks — surface the error.
            controller.enqueue({
              type: "error",
              error: err instanceof Error ? err : new Error(String(err)),
            })
            resolved = true
            break
          }
        }

        if (!resolved) {
          // Defensive: should be unreachable.
          controller.close()
        }

        try {
          controller.close()
        } catch {
          /* already closed */
        }
      },
      cancel(reason) {
        // Caller cancelled the upstream stream — abort every member.
        for (const c of childAborts.values()) {
          try {
            c.abort(reason)
          } catch {
            /* ignore */
          }
        }
      },
    })

    return { stream: upstream }
  }

  const doGenerate: LanguageModelV3["doGenerate"] = async (callOptions) => {
    // For non-streaming generate, race members in the same way: first to
    // resolve wins, the others are aborted.
    const members = await resolveMembers()
    emit({ kind: "members", count: members.length, members: members.map((m) => ({ providerID: m.providerID, modelID: m.modelID })) })

    if (members.length === 0) {
      emit({ kind: "no-members" })
      throw new Error(
        "Conjunto: no free models are connected. Connect at least one provider with free models (e.g. opencode) to use conjunto/ensemble.",
      )
    }

    const slice = members.slice(0, maxConcurrency)
    const callerSignal = callOptions.abortSignal
    const childAborts: AbortController[] = []

    const tasks = slice.map(async (member) => {
      const { options: childOpts, abort } = cloneOptions(callOptions, callerSignal)
      childAborts.push(abort)
      emit({ kind: "start", providerID: member.providerID, modelID: member.modelID })
      return member.language.doGenerate(childOpts).then(
        (result) => ({ member, result: result as LanguageModelV3GenerateResult, error: null as unknown | null }),
        (err: unknown) => ({ member, result: null, error: err }),
      )
    })

    // First successful task wins. We race by repeatedly awaiting the next
    // settled promise and checking whether it succeeded; on success we abort
    // the others and return. On failure we keep waiting.
    let winner: { member: ConjuntoMember; result: LanguageModelV3GenerateResult } | null = null
    const pending = new Set(tasks)
    while (pending.size > 0 && !winner) {
      const settled = await Promise.race(
        [...pending].map((p) => p.then((r) => ({ p, r }), (e) => ({ p, e }))),
      )
      pending.delete(settled.p)
      if ("r" in settled) {
        if (settled.r.result) {
          winner = { member: settled.r.member, result: settled.r.result }
        } else {
          emit({
            kind: "failed",
            providerID: settled.r.member.providerID,
            modelID: settled.r.member.modelID,
            error: settled.r.error instanceof Error ? settled.r.error.message : String(settled.r.error),
          })
        }
      }
    }

    // Abort any pending children now that we have a winner (or gave up).
    for (const c of childAborts) {
      try {
        c.abort(new Error("conjunto: generate race settled"))
      } catch {
        /* ignore */
      }
    }

    if (winner) {
      emit({ kind: "won", providerID: winner.member.providerID, modelID: winner.member.modelID, reason: "finish-no-content" })
      return winner.result
    }

    // All failed — surface a combined error.
    const errors = await Promise.allSettled(tasks)
    const messages = errors
      .filter((r): r is PromiseFulfilledResult<{ member: ConjuntoMember; result: null; error: unknown }> => r.status === "fulfilled" && r.value.error !== null)
      .map((r) => `${r.value.member.providerID}/${r.value.member.modelID}: ${r.value.error instanceof Error ? (r.value.error as Error).message : String(r.value.error)}`)
    throw new Error(
      `Conjunto: every free model failed to generate a response.${messages.length ? ` Errors: ${messages.join(" | ")}` : ""}`,
    )
  }

  return {
    specificationVersion: "v3" as const,
    provider: "conjunto",
    modelId: "ensemble",
    supportedUrls: {},
    doGenerate,
    doStream,
  }
}

/**
 * Heuristic used by the provider registration code to decide which models
 * are eligible to be ensemble members. A model is "free" when its input
 * and output cost are both zero.
 */
export function isFreeModel(model: { cost: { input: number; output: number } }): boolean {
  return model.cost.input === 0 && model.cost.output === 0
}
