/**
 * Conjunto: ensemble mode for opencode.
 *
 * The `conjunto` provider exposes a single virtual model `conjunto/ensemble`
 * whose `LanguageModelV3` implementation runs **every connected free model
 * in parallel** and combines their outputs into one unified response.
 *
 * ## Modes
 *
 * - **ensemble** (default): All N free models run in parallel, each at its
 *   maximum reasoning capability. Each model's contribution is streamed as
 *   a separate labeled text block (`### provider/model`) so the user can
 *   see exactly which model produced what. After all members complete, a
 *   designated synthesizer model produces a final unified answer that
 *   combines the best of all contributions. Tool calls from any member are
 *   forwarded as-is (the processor executes them independently).
 *
 * - **race**: First model to produce a usable token wins; others are
 *   aborted. More efficient but only one model's answer is used. Useful
 *   when latency matters more than answer quality.
 *
 * ## Max reasoning
 *
 * When `forceMaxReasoning` is true (default), the ensemble injects
 * per-provider reasoning options into each member's callOptions so that
 * models supporting extended thinking use their highest reasoning
 * capability. Supported providers:
 *   - Anthropic: `thinking: { type: "enabled", budget_tokens: 16000 }`
 *   - OpenAI / GitHub Copilot: `reasoningEffort: "high"`
 *   - Google / Vertex: `thinkingConfig: { thinkingBudget: 24576 }`
 *   - xAI: `reasoning: { effort: "high" }`
 *   - Groq: `reasoning_format: "parsed"`
 *   - OpenRouter: `reasoning: { enabled: true, effort: "high" }`
 *
 * ## Transparency
 *
 * Every contribution is labeled with the model's `provider/model`
 * identifier. The synthesizer's output is labeled
 * `### Conjunto Synthesis`. Lifecycle events (`onEvent`) let the TUI
 * surface real-time progress (which models started, which completed,
 * synthesis in progress, etc.).
 *
 * ## Abort handling
 *
 * The caller's `abortSignal` is forwarded to every member and to the
 * synthesizer. When the caller aborts (e.g. user presses Esc), every
 * pending call is aborted in parallel.
 */

import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3Message,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
  LanguageModelV3FinishReason,
} from "@ai-sdk/provider"

/**
 * Minimal subset of opencode's `Model` schema that the ensemble needs to
 * decide how to call each member (reasoning capability, SDK npm package,
 * existing options). Keeping this minimal avoids a hard dependency on the
 * full Model type and makes the ensemble testable in isolation.
 */
export interface ConjuntoMemberModel {
  api: { id: string; url: string; npm: string }
  capabilities: {
    reasoning: boolean
    toolcall: boolean
    temperature: boolean
    attachment: boolean
  }
  options: Record<string, any>
}

/**
 * A single member of the ensemble: a loaded language model plus the
 * metadata needed to identify it in logs / errors and to decide how to
 * invoke it.
 */
export interface ConjuntoMember {
  readonly providerID: string
  readonly modelID: string
  readonly displayName: string
  readonly language: LanguageModelV3
  /** Model metadata for reasoning injection. Optional — when absent, the
   * member is called with the original options unchanged. */
  readonly model?: ConjuntoMemberModel
}

/**
 * Factory that returns the *current* set of ensemble members. We use a
 * factory (rather than a snapshot) so the ensemble can react to providers
 * being connected/disconnected at runtime.
 */
export type ConjuntoMemberProvider = () => Promise<ConjuntoMember[]>

export interface ConjuntoLanguageModelOptions {
  /** Resolves the list of free, connected members to fan out to. */
  members: ConjuntoMemberProvider
  /**
   * Cap on how many members run in parallel for a single prompt.
   * Defaults to 8. Higher values reduce latency (more models racing)
   * but increase memory and rate-limit pressure on the underlying
   * providers.
   */
  maxConcurrency?: number
  /**
   * "ensemble" (default): all members contribute, then a synthesizer
   *   combines their responses into one final answer.
   * "race": first member to produce a usable token wins; others are
   *   aborted (more efficient but only one model is used).
   */
  mode?: "ensemble" | "race"
  /**
   * When true (default), injects per-provider max-reasoning options into
   * each member's callOptions so models that support extended thinking
   * use their highest reasoning capability.
   */
  forceMaxReasoning?: boolean
  /**
   * Optional logger that receives lifecycle events (members resolved,
   * member started, contribution complete, synthesis in progress, ...).
   * Useful for the TUI "thinking" indicator.
   */
  onEvent?: (event: ConjuntoEvent) => void
}

export type ConjuntoEvent =
  | { kind: "members"; count: number; members: Array<{ providerID: string; modelID: string }> }
  | { kind: "phase"; phase: "fanout" | "synthesis" | "done" }
  | { kind: "start"; providerID: string; modelID: string }
  | { kind: "contribution"; providerID: string; modelID: string; textLength: number; hadToolCall: boolean }
  | { kind: "failed"; providerID: string; modelID: string; error: string }
  | { kind: "synthesis-start"; providerID: string; modelID: string }
  | { kind: "synthesis-end"; providerID: string; modelID: string }
  | { kind: "no-members" }
  | { kind: "abort" }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let idCounter = 0
function randomId(prefix: string): string {
  idCounter += 1
  return `${prefix}-${Date.now().toString(36)}-${idCounter.toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

function isNonNullable<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined
}

/**
 * Per-provider reasoning options that ask the model to use its highest
 * reasoning capability. Returns `{}` for providers we don't know how to
 * configure (the model's defaults will apply).
 */
export function maxReasoningProviderOptions(model: ConjuntoMemberModel): Record<string, any> {
  if (!model.capabilities?.reasoning) return {}

  const npm = model.api?.npm ?? ""

  // Anthropic: extended thinking with a generous token budget.
  if (npm === "@ai-sdk/anthropic" || npm === "@ai-sdk/google-vertex/anthropic") {
    return { anthropic: { thinking: { type: "enabled", budget_tokens: 16000 } } }
  }

  // OpenAI / GitHub Copilot: reasoning effort (only o-series and gpt-5
  // honor this; other models silently ignore it).
  if (npm === "@ai-sdk/openai" || npm === "@ai-sdk/github-copilot") {
    return { openai: { reasoningEffort: "high" } }
  }

  // Google / Vertex: thinking budget (max for gemini-2.5-pro is 24576).
  if (npm === "@ai-sdk/google" || npm === "@ai-sdk/google-vertex") {
    return { google: { thinkingConfig: { thinkingBudget: 24576 } } }
  }

  // xAI (grok): reasoning effort.
  if (npm === "@ai-sdk/xai") {
    return { xai: { reasoning: { effort: "high" } } }
  }

  // Groq: parsed reasoning format for DeepSeek R1 / Qwen QwQ.
  if (npm === "@ai-sdk/groq") {
    return { groq: { reasoning_format: "parsed" } }
  }

  // OpenRouter: passthrough reasoning config.
  if (npm === "@openrouter/ai-sdk-provider") {
    return { openrouter: { reasoning: { enabled: true, effort: "high" } } }
  }

  // OpenAI-compatible providers (DeepInfra, Together, etc.): most accept
  // the generic `reasoning.effort` field.
  if (
    npm === "@ai-sdk/openai-compatible" ||
    npm === "@ai-sdk/deepinfra" ||
    npm === "@ai-sdk/togetherai" ||
    npm === "@ai-sdk/cerebras" ||
    npm === "@ai-sdk/perplexity"
  ) {
    return { reasoning: { effort: "high" } }
  }

  return {}
}

/**
 * Deep-merge two providerOptions objects. Arrays and primitives from `b`
 * override `a`; nested objects are merged recursively.
 */
function deepMergeProviderOptions(
  a: Record<string, any> | undefined,
  b: Record<string, any> | undefined,
): Record<string, any> {
  if (!a && !b) return {}
  if (!a) return { ...b! }
  if (!b) return { ...a }
  const out: Record<string, any> = { ...a }
  for (const [key, value] of Object.entries(b)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      out[key] = deepMergeProviderOptions(out[key], value)
    } else {
      out[key] = value
    }
  }
  return out
}

/**
 * Clone call options for a member, injecting max-reasoning options when
 * available and swapping in a fresh abort controller chained to the
 * caller's signal.
 */
function buildMemberOptions(
  callOptions: LanguageModelV3CallOptions,
  member: ConjuntoMember,
  parentSignal: AbortSignal | undefined,
  forceMaxReasoning: boolean,
): { options: LanguageModelV3CallOptions; abort: AbortController } {
  const abort = new AbortController()
  if (parentSignal) {
    if (parentSignal.aborted) abort.abort(parentSignal.reason)
    else parentSignal.addEventListener("abort", () => abort.abort(parentSignal.reason), { once: true })
  }

  const reasoningOpts =
    forceMaxReasoning && member.model ? maxReasoningProviderOptions(member.model) : {}

  const mergedProviderOptions = deepMergeProviderOptions(
    callOptions.providerOptions as Record<string, any> | undefined,
    reasoningOpts,
  )

  return {
    options: {
      ...callOptions,
      abortSignal: abort.signal,
      providerOptions: mergedProviderOptions,
    },
    abort,
  }
}

/**
 * Build the synthesis prompt: the original prompt + a new user message
 * containing all member contributions and an instruction to synthesize.
 */
function buildSynthesisPrompt(
  originalPrompt: LanguageModelV3Prompt,
  contributions: Array<{ member: ConjuntoMember; text: string }>,
): LanguageModelV3Prompt {
  const lines: string[] = [
    "You are the synthesis layer of an AI model ensemble. Multiple AI models independently answered the user's previous message. Your task is to synthesize their responses into a single, comprehensive, and coherent answer that:",
    "",
    "1. Combines the best insights and information from all responses",
    "2. Resolves any contradictions by favoring the most accurate and well-reasoned position",
    "3. Eliminates redundancy and presents a unified voice",
    "4. Preserves important technical details, code snippets, and factual claims",
    "5. Does NOT mention the individual models or that synthesis occurred — present the final answer as if it came from you directly",
    "",
    `Here are the ${contributions.length} responses from the ensemble members:`,
    "",
  ]

  for (let i = 0; i < contributions.length; i++) {
    const c = contributions[i]
    lines.push(`## Response ${i + 1} (${c.member.providerID}/${c.member.modelID})`)
    lines.push("")
    lines.push(c.text || "(empty response)")
    lines.push("")
    if (i < contributions.length - 1) lines.push("---")
    lines.push("")
  }

  lines.push("## Your synthesized response:")

  const synthesisMessage: LanguageModelV3Message = {
    role: "user",
    content: [{ type: "text", text: lines.join("\n") }],
  }

  return [...originalPrompt, synthesisMessage]
}

function emptyUsage(): LanguageModelV3Usage {
  return {
    inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 0, text: 0, reasoning: 0 },
  }
}

/**
 * Aggregate a member's usage into the running total. Handles both the v3
 * schema (inputTokens/outputTokens as objects) and the legacy v2 schema
 * (inputTokens/outputTokens as numbers) for safety.
 */
function addUsage(into: LanguageModelV3Usage, from: LanguageModelV3Usage | undefined): void {
  if (!from) return
  // inputTokens
  const intoInput = into.inputTokens as any
  const fromInput = (from as any).inputTokens
  if (typeof fromInput === "number") {
    intoInput.total = ((intoInput.total ?? 0) as number) + fromInput
  } else if (fromInput && typeof fromInput === "object") {
    intoInput.total = ((intoInput.total ?? 0) as number) + ((fromInput.total as number) ?? 0)
    intoInput.noCache = ((intoInput.noCache ?? 0) as number) + ((fromInput.noCache as number) ?? 0)
    intoInput.cacheRead = ((intoInput.cacheRead ?? 0) as number) + ((fromInput.cacheRead as number) ?? 0)
    intoInput.cacheWrite = ((intoInput.cacheWrite ?? 0) as number) + ((fromInput.cacheWrite as number) ?? 0)
  }
  // outputTokens
  const intoOutput = into.outputTokens as any
  const fromOutput = (from as any).outputTokens
  if (typeof fromOutput === "number") {
    intoOutput.total = ((intoOutput.total ?? 0) as number) + fromOutput
  } else if (fromOutput && typeof fromOutput === "object") {
    intoOutput.total = ((intoOutput.total ?? 0) as number) + ((fromOutput.total as number) ?? 0)
    intoOutput.text = ((intoOutput.text ?? 0) as number) + ((fromOutput.text as number) ?? 0)
    intoOutput.reasoning = ((intoOutput.reasoning ?? 0) as number) + ((fromOutput.reasoning as number) ?? 0)
  }
}

// ---------------------------------------------------------------------------
// Main factory
// ---------------------------------------------------------------------------

export function createConjuntoLanguageModel(opts: ConjuntoLanguageModelOptions): LanguageModelV3 {
  const maxConcurrency = opts.maxConcurrency ?? 8
  const mode: "ensemble" | "race" = opts.mode ?? "ensemble"
  const forceMaxReasoning = opts.forceMaxReasoning ?? true

  function emit(event: ConjuntoEvent) {
    try {
      opts.onEvent?.(event)
    } catch {
      /* swallow listener errors */
    }
  }

  async function resolveMembers(): Promise<ConjuntoMember[]> {
    const all = await opts.members()
    // Deterministic order so the synthesizer pick is stable across calls.
    return [...all].sort((a, b) => {
      if (a.providerID !== b.providerID) return a.providerID.localeCompare(b.providerID)
      return a.modelID.localeCompare(b.modelID)
    })
  }

  // -------------------------------------------------------------------------
  // ENSEMBLE MODE (default): all members contribute, then synthesis.
  // -------------------------------------------------------------------------

  async function doStreamEnsemble(
    callOptions: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const members = await resolveMembers()
    emit({
      kind: "members",
      count: members.length,
      members: members.map((m) => ({ providerID: m.providerID, modelID: m.modelID })),
    })

    if (members.length === 0) {
      emit({ kind: "no-members" })
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
    const childAborts = new Set<AbortController>()
    const memberUsages = new Map<string, LanguageModelV3Usage>()
    const totalUsage = emptyUsage()
    let finishReason: LanguageModelV3FinishReason = { unified: "stop", raw: undefined }

    const callerSignal = callOptions.abortSignal

    const upstream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({ type: "stream-start", warnings: [] })
        emit({ kind: "phase", phase: "fanout" })

        // ---- Caller abort propagation ----
        const onCallerAbort = () => {
          emit({ kind: "abort" })
          for (const c of childAborts) {
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

        // ---- Phase 1: fan out to all members in parallel ----
        const contributions: Array<{
          member: ConjuntoMember
          text: string
          hadToolCall: boolean
          error: string | null
        }> = []

        const fanoutPromises = slice.map(async (member, index) => {
          emit({ kind: "start", providerID: member.providerID, modelID: member.modelID })
          const blockId = randomId(`conjunto-m${index}`)
          const { options: memberOpts, abort } = buildMemberOptions(
            callOptions,
            member,
            callerSignal,
            forceMaxReasoning,
          )
          childAborts.add(abort)

          let textBuffer = ""
          let hadToolCall = false
          let textStarted = false
          let memberFinishReason: LanguageModelV3FinishReason | null = null

          try {
            const result = await member.language.doStream(memberOpts)
            const reader = result.stream.getReader()
            try {
              while (true) {
                const { value, done } = await reader.read()
                if (done) break
                if (!value) continue

                switch (value.type) {
                  case "text-start": {
                    // Replace the id with our block id and emit a header
                    // line so the user can see which model produced this
                    // block.
                    controller.enqueue({ type: "text-start", id: blockId })
                    controller.enqueue({
                      type: "text-delta",
                      id: blockId,
                      delta: `### ${member.displayName}\n\n`,
                    })
                    textStarted = true
                    break
                  }
                  case "text-delta": {
                    controller.enqueue({ type: "text-delta", id: blockId, delta: value.delta })
                    textBuffer += value.delta
                    break
                  }
                  case "text-end": {
                    controller.enqueue({ type: "text-end", id: blockId })
                    break
                  }
                  case "reasoning-start":
                  case "reasoning-delta":
                  case "reasoning-end": {
                    // Forward reasoning events as-is (they have their own
                    // id and the TUI can render them as "thinking").
                    controller.enqueue(value)
                    break
                  }
                  case "tool-input-start":
                  case "tool-input-delta":
                  case "tool-input-end":
                  case "tool-call": {
                    // Forward tool calls as-is. Each model can
                    // independently call tools; the processor will
                    // execute them all.
                    controller.enqueue(value)
                    if (value.type === "tool-call") hadToolCall = true
                    break
                  }
                  case "tool-result":
                  case "tool-approval-request":
                  case "file":
                  case "source": {
                    controller.enqueue(value)
                    break
                  }
                  case "finish": {
                    // Don't forward; we'll emit our own aggregated finish.
                    memberUsages.set(`${member.providerID}/${member.modelID}`, value.usage)
                    addUsage(totalUsage, value.usage)
                    memberFinishReason = value.finishReason
                    break
                  }
                  case "response-metadata":
                  case "raw": {
                    // Drop — we synthesize our own.
                    break
                  }
                  case "error": {
                    throw value.error
                  }
                  case "stream-start": {
                    // Drop — we already emitted our own.
                    break
                  }
                  default: {
                    // Forward unknown event types as-is.
                    controller.enqueue(value)
                  }
                }
              }
            } finally {
              reader.releaseLock()
            }

            contributions.push({ member, text: textBuffer, hadToolCall, error: null })
            emit({
              kind: "contribution",
              providerID: member.providerID,
              modelID: member.modelID,
              textLength: textBuffer.length,
              hadToolCall,
            })
            if (memberFinishReason && (memberFinishReason as { unified?: string }).unified !== "stop") {
              finishReason = memberFinishReason
            }
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err)
            contributions.push({ member, text: textBuffer, hadToolCall, error: msg })
            emit({
              kind: "failed",
              providerID: member.providerID,
              modelID: member.modelID,
              error: msg,
            })
            // If the member had started a text block, close it cleanly so
            // the upstream doesn't end up with an unclosed block.
            if (textStarted) {
              try {
                controller.enqueue({ type: "text-end", id: blockId })
              } catch {
                /* already closed */
              }
            }
          } finally {
            childAborts.delete(abort)
          }
        })

        await Promise.all(fanoutPromises)

        // ---- Phase 2: synthesis ----
        const successfulContributions = contributions.filter((c) => c.text.length > 0 && !c.error)
        const anyToolCall = contributions.some((c) => c.hadToolCall)

        // We only synthesize when:
        //  - no member emitted a tool call (otherwise the processor is
        //    already executing tools and the next turn will continue the
        //    conversation — synthesis would be confusing)
        //  - at least 2 members produced text (1 member = nothing to
        //    synthesize)
        //  - the caller hasn't aborted
        const shouldSynthesize =
          !anyToolCall &&
          successfulContributions.length >= 2 &&
          !(callerSignal?.aborted ?? false)

        if (shouldSynthesize) {
          // Pick the synthesizer: prefer the member with the longest text
          // contribution (it's likely the most thorough). Fall back to the
          // first successful contribution.
          const synthesizer = [...successfulContributions].sort(
            (a, b) => b.text.length - a.text.length,
          )[0].member

          emit({ kind: "phase", phase: "synthesis" })
          emit({
            kind: "synthesis-start",
            providerID: synthesizer.providerID,
            modelID: synthesizer.modelID,
          })

          const synthesisBlockId = randomId("conjunto-synthesis")
          const synthesisPrompt = buildSynthesisPrompt(
            callOptions.prompt,
            successfulContributions.map((c) => ({ member: c.member, text: c.text })),
          )

          // Strip tools/toolChoice from the synthesizer's options — its
          // job is to combine text, not to take new actions.
          const synthesisCallOptions: LanguageModelV3CallOptions = {
            ...callOptions,
            prompt: synthesisPrompt,
            tools: undefined,
            toolChoice: undefined,
          }

          const { options: synthOpts, abort: synthAbort } = buildMemberOptions(
            synthesisCallOptions,
            synthesizer,
            callerSignal,
            forceMaxReasoning,
          )
          childAborts.add(synthAbort)

          let synthesisStarted = false
          try {
            const result = await synthesizer.language.doStream(synthOpts)
            const reader = result.stream.getReader()
            try {
              while (true) {
                const { value, done } = await reader.read()
                if (done) break
                if (!value) continue

                switch (value.type) {
                  case "text-start": {
                    controller.enqueue({ type: "text-start", id: synthesisBlockId })
                    controller.enqueue({
                      type: "text-delta",
                      id: synthesisBlockId,
                      delta: `### Conjunto Synthesis\n\n`,
                    })
                    synthesisStarted = true
                    break
                  }
                  case "text-delta": {
                    controller.enqueue({
                      type: "text-delta",
                      id: synthesisBlockId,
                      delta: value.delta,
                    })
                    break
                  }
                  case "text-end": {
                    controller.enqueue({ type: "text-end", id: synthesisBlockId })
                    break
                  }
                  case "finish": {
                    addUsage(totalUsage, value.usage)
                    if (
                      value.finishReason &&
                      (value.finishReason as { unified?: string }).unified !== "stop"
                    ) {
                      finishReason = value.finishReason
                    }
                    break
                  }
                  case "error": {
                    throw value.error
                  }
                  // Drop reasoning/tool/response-metadata/raw events from
                  // the synthesizer — the synthesis is text-only.
                  default:
                    break
                }
              }
            } finally {
              reader.releaseLock()
            }
            if (synthesisStarted) {
              try {
                controller.enqueue({ type: "text-end", id: synthesisBlockId })
              } catch {
                /* already closed */
              }
            }
            emit({
              kind: "synthesis-end",
              providerID: synthesizer.providerID,
              modelID: synthesizer.modelID,
            })
          } catch (err) {
            // Synthesis failed — log and continue with member
            // contributions only. The user still gets all member outputs.
            emit({
              kind: "failed",
              providerID: synthesizer.providerID,
              modelID: synthesizer.modelID,
              error: err instanceof Error ? err.message : String(err),
            })
            if (synthesisStarted) {
              try {
                controller.enqueue({ type: "text-end", id: synthesisBlockId })
              } catch {
                /* already closed */
              }
            }
          } finally {
            childAborts.delete(synthAbort)
          }
        }

        // ---- Phase 3: finish ----
        emit({ kind: "phase", phase: "done" })

        // If every member failed and there's no synthesis, emit an error.
        const allFailed = contributions.every((c) => c.error !== null && c.text.length === 0)
        if (allFailed && !shouldSynthesize) {
          const messages = contributions
            .filter((c) => c.error)
            .map((c) => `${c.member.providerID}/${c.member.modelID}: ${c.error}`)
          controller.enqueue({
            type: "error",
            error: new Error(
              `Conjunto: every free model failed to produce output.${
                messages.length ? ` Errors: ${messages.join(" | ")}` : ""
              }`,
            ),
          })
        }

        controller.enqueue({
          type: "finish",
          usage: totalUsage,
          finishReason,
        })

        try {
          controller.close()
        } catch {
          /* already closed */
        }
      },
      cancel(reason) {
        for (const c of childAborts) {
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

  // -------------------------------------------------------------------------
  // RACE MODE: first usable token wins, others aborted.
  // -------------------------------------------------------------------------

  async function doStreamRace(
    callOptions: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const members = await resolveMembers()
    emit({
      kind: "members",
      count: members.length,
      members: members.map((m) => ({ providerID: m.providerID, modelID: m.modelID })),
    })

    if (members.length === 0) {
      emit({ kind: "no-members" })
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
    const childAborts = new Map<ConjuntoMember, AbortController>()
    const callerSignal = callOptions.abortSignal

    interface MemberAttempt {
      member: ConjuntoMember
      stream: ReadableStream<LanguageModelV3StreamPart> | null
      error: unknown | null
    }

    const attempts: MemberAttempt[] = slice.map((member) => ({
      member,
      stream: null,
      error: null,
    }))

    // Start every member in parallel.
    await Promise.all(
      attempts.map(async (attempt) => {
        emit({ kind: "start", providerID: attempt.member.providerID, modelID: attempt.member.modelID })
        const { options: memberOpts, abort } = buildMemberOptions(
          callOptions,
          attempt.member,
          callerSignal,
          forceMaxReasoning,
        )
        childAborts.set(attempt.member, abort)
        try {
          const result = await attempt.member.language.doStream(memberOpts)
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

    // Probe each member for the first useful event.
    async function probeForWinner(attempt: MemberAttempt): Promise<{
      won: boolean
      buffered: LanguageModelV3StreamPart[]
      reason?: "first-token" | "tool-call" | "finish-no-content"
    }> {
      const buffered: LanguageModelV3StreamPart[] = []
      const reader = attempt.stream!.getReader()
      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) return { won: false, buffered }
          if (!value) continue
          buffered.push(value)

          if (value.type === "error") {
            attempt.error = value.error
            return { won: false, buffered }
          }
          if (value.type === "text-delta" && value.delta.length > 0) {
            return { won: true, buffered, reason: "first-token" }
          }
          if (value.type === "tool-input-start" || value.type === "tool-call") {
            return { won: true, buffered, reason: "tool-call" }
          }
          if (value.type === "finish") {
            return { won: true, buffered, reason: "finish-no-content" }
          }
        }
      } finally {
        reader.releaseLock()
      }
    }

    async function drainToUpstream(
      attempt: MemberAttempt,
      controller: ReadableStreamDefaultController<LanguageModelV3StreamPart>,
      buffered: LanguageModelV3StreamPart[],
    ): Promise<void> {
      for (const ev of buffered) controller.enqueue(ev)
      const reader = attempt.stream!.getReader()
      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) return
          if (!value) continue
          if (value.type === "error") {
            controller.error(
              value.error instanceof Error ? value.error : new Error(String(value.error)),
            )
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

    const upstream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({ type: "stream-start", warnings: [] })

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

        const probes = attempts
          .filter((a) => a.stream !== null)
          .map(async (a) => ({ attempt: a, probe: await probeForWinner(a) }))

        let resolved = false
        const probeResults = await Promise.allSettled(probes)

        const winners: Array<{
          attempt: MemberAttempt
          buffered: LanguageModelV3StreamPart[]
          reason: "first-token" | "tool-call" | "finish-no-content"
        }> = []

        for (const r of probeResults) {
          if (r.status !== "fulfilled") continue
          const { attempt, probe } = r.value
          if (probe.won) {
            winners.push({ attempt, buffered: probe.buffered, reason: probe.reason! })
          } else {
            const c = childAborts.get(attempt.member)
            try {
              c?.abort(new Error("conjunto: lost race"))
            } catch {
              /* ignore */
            }
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
          const messages = attempts
            .filter((a) => a.error)
            .map(
              (a) =>
                `${a.member.providerID}/${a.member.modelID}: ${a.error instanceof Error ? a.error.message : String(a.error)}`,
            )
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

        for (let i = 0; i < winners.length; i++) {
          const { attempt, buffered } = winners[i]
          emit({
            kind: "contribution",
            providerID: attempt.member.providerID,
            modelID: attempt.member.modelID,
            textLength: 0,
            hadToolCall: false,
          })
          try {
            await drainToUpstream(attempt, controller, buffered)
            resolved = true
            for (let j = i + 1; j < winners.length; j++) {
              const c = childAborts.get(winners[j].attempt.member)
              try {
                c?.abort(new Error("conjunto: not needed"))
              } catch {
                /* ignore */
              }
            }
            break
          } catch (err) {
            if (i < winners.length - 1) {
              const next = winners[i + 1]
              emit({
                kind: "failed",
                providerID: attempt.member.providerID,
                modelID: attempt.member.modelID,
                error: err instanceof Error ? err.message : String(err),
              })
              continue
            }
            controller.enqueue({
              type: "error",
              error: err instanceof Error ? err : new Error(String(err)),
            })
            resolved = true
            break
          }
        }

        if (!resolved) controller.close()
        try {
          controller.close()
        } catch {
          /* already closed */
        }
      },
      cancel(reason) {
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

  // -------------------------------------------------------------------------
  // doStream / doGenerate dispatchers
  // -------------------------------------------------------------------------

  const doStream: LanguageModelV3["doStream"] = async (callOptions) => {
    if (mode === "race") return doStreamRace(callOptions)
    return doStreamEnsemble(callOptions)
  }

  const doGenerate: LanguageModelV3["doGenerate"] = async (callOptions) => {
    // For non-streaming generate, we run all members in parallel and
    // return the result from the first to complete successfully (race
    // semantics). This is the most efficient non-streaming behavior and
    // avoids the complexity of synthesizing non-streaming results.
    const members = await resolveMembers()
    emit({
      kind: "members",
      count: members.length,
      members: members.map((m) => ({ providerID: m.providerID, modelID: m.modelID })),
    })

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
      emit({ kind: "start", providerID: member.providerID, modelID: member.modelID })
      const { options: memberOpts, abort } = buildMemberOptions(
        callOptions,
        member,
        callerSignal,
        forceMaxReasoning,
      )
      childAborts.push(abort)
      return member.language.doGenerate(memberOpts).then(
        (result) => ({ member, result: result as LanguageModelV3GenerateResult, error: null as unknown | null }),
        (err: unknown) => ({ member, result: null, error: err }),
      )
    })

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

    for (const c of childAborts) {
      try {
        c.abort(new Error("conjunto: generate race settled"))
      } catch {
        /* ignore */
      }
    }

    if (winner) {
      emit({
        kind: "contribution",
        providerID: winner.member.providerID,
        modelID: winner.member.modelID,
        textLength: 0,
        hadToolCall: false,
      })
      return winner.result
    }

    const errors = await Promise.allSettled(tasks)
    const messages = errors
      .filter(
        (r): r is PromiseFulfilledResult<{ member: ConjuntoMember; result: null; error: unknown }> =>
          r.status === "fulfilled" && r.value.error !== null,
      )
      .map(
        (r) =>
          `${r.value.member.providerID}/${r.value.member.modelID}: ${r.value.error instanceof Error ? (r.value.error as Error).message : String(r.value.error)}`,
      )
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

// Re-export types that consumers (provider.ts, tests) need.
export type { LanguageModelV3CallOptions, LanguageModelV3StreamPart, LanguageModelV3Usage }
