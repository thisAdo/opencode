import { describe, test, expect } from "bun:test"
import {
  createConjuntoLanguageModel,
  isFreeModel,
  maxReasoningProviderOptions,
  type ConjuntoMember,
  type ConjuntoMemberModel,
  type ConjuntoEvent,
} from "../../src/provider/conjunto"
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider"

/**
 * Build a fake LanguageModelV3 that streams a fixed sequence of events.
 */
function fakeLanguageModel(
  events:
    | LanguageModelV3StreamPart[]
    | ((opts: LanguageModelV3CallOptions) => LanguageModelV3StreamPart[]),
  opts: { id: string; delay?: number; errorOnStream?: Error; errorOnGenerate?: Error } = { id: "fake" },
): LanguageModelV3 {
  const delay = opts.delay ?? 0
  return {
    specificationVersion: "v3",
    provider: "fake",
    modelId: opts.id,
    supportedUrls: {},
    async doGenerate(_options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
      if (opts.errorOnGenerate) throw opts.errorOnGenerate
      if (delay > 0) await new Promise((r) => setTimeout(r, delay))
      return {
        content: [{ type: "text", text: `result from ${opts.id}` }],
        usage: {
          inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 1, text: 1, reasoning: 0 },
        } as any,
        finishReason: { unified: "stop", raw: undefined } as any,
        warnings: [],
        request: { body: {} },
        response: { headers: {}, id: opts.id, modelId: opts.id, timestamp: new Date() },
      } as any
    },
    async doStream(_options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
      if (opts.errorOnStream) throw opts.errorOnStream
      const resolvedEvents = typeof events === "function" ? events(_options) : events
      const stream = new ReadableStream<LanguageModelV3StreamPart>({
        async start(controller) {
          if (delay > 0) await new Promise((r) => setTimeout(r, delay))
          for (const ev of resolvedEvents) {
            controller.enqueue(ev)
          }
          controller.close()
        },
      })
      return { stream }
    },
  }
}

function makeMember(
  id: string,
  language: LanguageModelV3,
  model?: ConjuntoMemberModel,
): ConjuntoMember {
  const [providerID, modelID] = id.split("/")
  return { providerID, modelID, displayName: id, language, model }
}

function fakeModelMeta(npm: string, reasoning = false): ConjuntoMemberModel {
  return {
    api: { id: "test", url: "http://test", npm },
    capabilities: { reasoning, toolcall: true, temperature: false, attachment: false },
    options: {},
  }
}

const FINISH_EVENT: LanguageModelV3StreamPart = {
  type: "finish",
  finishReason: { unified: "stop", raw: undefined } as any,
  usage: {
    inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 20, reasoning: 0 },
  } as any,
}

async function readStream(stream: ReadableStream<LanguageModelV3StreamPart>): Promise<{
  events: LanguageModelV3StreamPart[]
  error: unknown
}> {
  const reader = stream.getReader()
  const events: LanguageModelV3StreamPart[] = []
  let streamErrored: unknown = null
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      if (value) events.push(value)
    }
  } catch (err) {
    streamErrored = err
  }
  return { events, error: streamErrored }
}

describe("conjunto", () => {
  describe("isFreeModel", () => {
    test("returns true when both input and output cost are zero", () => {
      expect(isFreeModel({ cost: { input: 0, output: 0 } })).toBe(true)
    })
    test("returns false when input cost is positive", () => {
      expect(isFreeModel({ cost: { input: 1, output: 0 } })).toBe(false)
    })
    test("returns false when output cost is positive", () => {
      expect(isFreeModel({ cost: { input: 0, output: 1 } })).toBe(false)
    })
    test("returns false when both costs are positive", () => {
      expect(isFreeModel({ cost: { input: 5, output: 10 } })).toBe(false)
    })
  })

  describe("maxReasoningProviderOptions", () => {
    test("returns empty for models without reasoning capability", () => {
      expect(maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/anthropic", false))).toEqual({})
    })
    test("returns Anthropic thinking config for @ai-sdk/anthropic", () => {
      const opts = maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/anthropic", true))
      expect(opts.anthropic?.thinking?.type).toBe("enabled")
      expect(opts.anthropic?.thinking?.budget_tokens).toBeGreaterThan(0)
    })
    test("returns OpenAI reasoningEffort for @ai-sdk/openai", () => {
      const opts = maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/openai", true))
      expect(opts.openai?.reasoningEffort).toBe("high")
    })
    test("returns Google thinkingConfig for @ai-sdk/google", () => {
      const opts = maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/google", true))
      expect(opts.google?.thinkingConfig?.thinkingBudget).toBeGreaterThan(0)
    })
    test("returns xai reasoning for @ai-sdk/xai", () => {
      const opts = maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/xai", true))
      expect(opts.xai?.reasoning?.effort).toBe("high")
    })
    test("returns groq reasoning_format for @ai-sdk/groq", () => {
      const opts = maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/groq", true))
      expect(opts.groq?.reasoning_format).toBe("parsed")
    })
    test("returns openrouter reasoning for @openrouter/ai-sdk-provider", () => {
      const opts = maxReasoningProviderOptions(fakeModelMeta("@openrouter/ai-sdk-provider", true))
      expect(opts.openrouter?.reasoning?.enabled).toBe(true)
      expect(opts.openrouter?.reasoning?.effort).toBe("high")
    })
    test("returns generic reasoning for @ai-sdk/openai-compatible", () => {
      const opts = maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/openai-compatible", true))
      expect(opts.reasoning?.effort).toBe("high")
    })
    test("returns empty for unknown npm packages", () => {
      expect(maxReasoningProviderOptions(fakeModelMeta("@ai-sdk/unknown", true))).toEqual({})
    })
  })

  describe("createConjuntoLanguageModel — static metadata", () => {
    test("exposes the correct static metadata", async () => {
      const model = createConjuntoLanguageModel({ members: async () => [] })
      expect(model.specificationVersion).toBe("v3")
      expect(model.provider).toBe("conjunto")
      expect(model.modelId).toBe("ensemble")
      expect(model.supportedUrls).toEqual({})
      expect(typeof model.doGenerate).toBe("function")
      expect(typeof model.doStream).toBe("function")
    })
  })

  describe("createConjuntoLanguageModel — ensemble mode (default)", () => {
    test("doStream returns an error event when no members are connected", async () => {
      const model = createConjuntoLanguageModel({ members: async () => [] })
      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events } = await readStream(result.stream)
      expect(events[0]).toEqual({ type: "stream-start", warnings: [] })
      expect(events[1]?.type).toBe("error")
    })

    test("doStream runs all members in parallel, each labeled with its model name", async () => {
      const memberA = makeMember(
        "alpha/quick",
        fakeLanguageModel(
          [
            { type: "text-start", id: "a1" },
            { type: "text-delta", id: "a1", delta: "Answer from A" },
            { type: "text-end", id: "a1" },
            FINISH_EVENT,
          ],
          { id: "a" },
        ),
      )
      const memberB = makeMember(
        "beta/slow",
        fakeLanguageModel(
          [
            { type: "text-start", id: "b1" },
            { type: "text-delta", id: "b1", delta: "Answer from B" },
            { type: "text-end", id: "b1" },
            FINISH_EVENT,
          ],
          { id: "b", delay: 5 },
        ),
      )
      const model = createConjuntoLanguageModel({ members: async () => [memberA, memberB] })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events } = await readStream(result.stream)

      // Expect: stream-start, then 2 member text blocks (each with header),
      // then 1 synthesis text block, then finish.
      const textStarts = events.filter((e) => e.type === "text-start")
      expect(textStarts.length).toBe(3) // 2 members + 1 synthesis

      // The first text-delta of each member block should be the header.
      const textDeltas = events
        .filter((e) => e.type === "text-delta")
        .map((e) => (e as { delta: string }).delta)
        .join("")
      expect(textDeltas).toContain("### alpha/quick")
      expect(textDeltas).toContain("### beta/slow")
      expect(textDeltas).toContain("Answer from A")
      expect(textDeltas).toContain("Answer from B")
      expect(textDeltas).toContain("### Conjunto Synthesis")

      // The stream should end with a finish event with aggregated usage.
      const finish = events.find((e) => e.type === "finish") as
        | { usage: { inputTokens: { total: number }; outputTokens: { total: number } } }
        | undefined
      expect(finish).toBeDefined()
      // 2 members + 1 synthesizer = 3 × (10 input + 20 output) = 30 input + 60 output
      expect(finish!.usage.outputTokens.total).toBeGreaterThanOrEqual(40)
    })

    test("doStream skips synthesis when a member emits a tool call", async () => {
      const memberA = makeMember(
        "alpha/quick",
        fakeLanguageModel(
          [
            { type: "text-start", id: "a1" },
            { type: "text-delta", id: "a1", delta: "Let me check" },
            { type: "text-end", id: "a1" },
            {
              type: "tool-call",
              toolCallId: "tc1",
              toolName: "search",
              input: { query: "test" },
            } as unknown as LanguageModelV3StreamPart,
            FINISH_EVENT,
          ],
          { id: "a" },
        ),
      )
      const memberB = makeMember(
        "beta/slow",
        fakeLanguageModel(
          [
            { type: "text-start", id: "b1" },
            { type: "text-delta", id: "b1", delta: "No tool needed" },
            { type: "text-end", id: "b1" },
            FINISH_EVENT,
          ],
          { id: "b", delay: 5 },
        ),
      )
      const model = createConjuntoLanguageModel({ members: async () => [memberA, memberB] })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events } = await readStream(result.stream)

      // Tool calls should be forwarded
      const toolCalls = events.filter((e) => e.type === "tool-call")
      expect(toolCalls.length).toBe(1)

      // No synthesis block should appear (since one member emitted a tool call)
      const textDeltas = events
        .filter((e) => e.type === "text-delta")
        .map((e) => (e as { delta: string }).delta)
        .join("")
      expect(textDeltas).not.toContain("### Conjunto Synthesis")
    })

    test("doStream continues if one member fails, still synthesizes the rest", async () => {
      const memberA = makeMember(
        "alpha/quick",
        fakeLanguageModel(
          [
            { type: "text-start", id: "a1" },
            { type: "text-delta", id: "a1", delta: "A's answer" },
            { type: "text-end", id: "a1" },
            FINISH_EVENT,
          ],
          { id: "a" },
        ),
      )
      const memberB = makeMember(
        "beta/bad",
        fakeLanguageModel([], { id: "b", errorOnStream: new Error("B is down") }),
      )
      const model = createConjuntoLanguageModel({ members: async () => [memberA, memberB] })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events } = await readStream(result.stream)

      // Only 1 successful contribution → no synthesis (need ≥ 2)
      const textDeltas = events
        .filter((e) => e.type === "text-delta")
        .map((e) => (e as { delta: string }).delta)
        .join("")
      expect(textDeltas).toContain("A's answer")
      expect(textDeltas).not.toContain("### Conjunto Synthesis")

      // Should still finish cleanly
      const finish = events.find((e) => e.type === "finish")
      expect(finish).toBeDefined()
    })

    test("doStream emits lifecycle events for visibility", async () => {
      const events: ConjuntoEvent[] = []
      const member = makeMember(
        "alpha/quick",
        fakeLanguageModel(
          [
            { type: "text-start", id: "a1" },
            { type: "text-delta", id: "a1", delta: "hi" },
            { type: "text-end", id: "a1" },
            FINISH_EVENT,
          ],
          { id: "a" },
        ),
      )
      const member2 = makeMember(
        "beta/slow",
        fakeLanguageModel(
          [
            { type: "text-start", id: "b1" },
            { type: "text-delta", id: "b1", delta: "hi" },
            { type: "text-end", id: "b1" },
            FINISH_EVENT,
          ],
          { id: "b" },
        ),
      )
      const model = createConjuntoLanguageModel({
        members: async () => [member, member2],
        onEvent: (e) => events.push(e),
      })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      await readStream(result.stream)

      const kinds = events.map((e) => e.kind)
      expect(kinds).toContain("members")
      expect(kinds.filter((k) => k === "start").length).toBe(2)
      expect(kinds.filter((k) => k === "contribution").length).toBe(2)
      expect(kinds).toContain("synthesis-start")
      expect(kinds).toContain("synthesis-end")
      // Phase events carry the phase name in `e.phase`, not in `e.kind`.
      const phases = events.filter((e) => e.kind === "phase").map((e) => (e as { phase: string }).phase)
      expect(phases).toContain("fanout")
      expect(phases).toContain("synthesis")
      expect(phases).toContain("done")
    })

    test("doStream injects max reasoning options when forceMaxReasoning is true", async () => {
      let capturedOpts: LanguageModelV3CallOptions | null = null
      const member = makeMember(
        "alpha/anthropic",
        {
          specificationVersion: "v3",
          provider: "alpha",
          modelId: "anthropic",
          supportedUrls: {},
          async doGenerate() {
            return {} as any
          },
          async doStream(opts) {
            capturedOpts = opts
            return {
              stream: new ReadableStream({
                start(c) {
                  c.enqueue({ type: "text-start", id: "x" })
                  c.enqueue({ type: "text-delta", id: "x", delta: "ok" })
                  c.enqueue({ type: "text-end", id: "x" })
                  c.enqueue(FINISH_EVENT)
                  c.close()
                },
              }),
            }
          },
        },
        fakeModelMeta("@ai-sdk/anthropic", true),
      )
      const member2 = makeMember(
        "beta/other",
        fakeLanguageModel(
          [
            { type: "text-start", id: "b1" },
            { type: "text-delta", id: "b1", delta: "hi" },
            { type: "text-end", id: "b1" },
            FINISH_EVENT,
          ],
          { id: "b" },
        ),
        fakeModelMeta("@ai-sdk/anthropic", true),
      )
      const model = createConjuntoLanguageModel({
        members: async () => [member, member2],
        forceMaxReasoning: true,
      })

      const result = await model.doStream({
        prompt: [],
        providerOptions: { existing: { foo: "bar" } },
      } as LanguageModelV3CallOptions)
      await readStream(result.stream)

      expect(capturedOpts).not.toBeNull()
      expect((capturedOpts!.providerOptions as any).anthropic?.thinking?.type).toBe("enabled")
      // Existing providerOptions should be preserved
      expect((capturedOpts!.providerOptions as any).existing?.foo).toBe("bar")
    })

    test("doStream respects maxConcurrency by slicing the member list", async () => {
      const members: ConjuntoMember[] = Array.from({ length: 10 }, (_, i) =>
        makeMember(
          `p${i}/m${i}`,
          fakeLanguageModel(
            [
              { type: "text-start", id: `t${i}` },
              { type: "text-delta", id: `t${i}`, delta: `m${i}` },
              { type: "text-end", id: `t${i}` },
              FINISH_EVENT,
            ],
            { id: `m${i}`, delay: i },
          ),
        ),
      )

      const startedEvents: string[] = []
      const model = createConjuntoLanguageModel({
        members: async () => members,
        maxConcurrency: 3,
        onEvent: (e) => {
          if (e.kind === "start") startedEvents.push(`${e.providerID}/${e.modelID}`)
        },
      })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      await readStream(result.stream)

      // Only the first 3 members should have started.
      expect(startedEvents.length).toBeLessThanOrEqual(3)
      expect(startedEvents).toEqual(expect.arrayContaining(["p0/m0", "p1/m1", "p2/m2"]))
    })

    test("doStream surfaces an error when every member fails", async () => {
      const failing = makeMember(
        "bad/bad",
        fakeLanguageModel([], { id: "bad", errorOnStream: new Error("stream init failed") }),
      )
      const model = createConjuntoLanguageModel({ members: async () => [failing] })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events } = await readStream(result.stream)
      const errorEvent = events.find((e) => e.type === "error")
      expect(errorEvent).toBeDefined()
    })
  })

  describe("createConjuntoLanguageModel — race mode", () => {
    test("doStream forwards the winning member's stream verbatim", async () => {
      const winner = makeMember(
        "fast/quick",
        fakeLanguageModel(
          [
            { type: "text-start", id: "t1" },
            { type: "text-delta", id: "t1", delta: "Hello" },
            { type: "text-delta", id: "t1", delta: " world" },
            { type: "text-end", id: "t1" },
            FINISH_EVENT,
          ],
          { id: "quick", delay: 1 },
        ),
      )
      const loser = makeMember(
        "slow/slow",
        fakeLanguageModel(
          [
            { type: "text-start", id: "t2" },
            { type: "text-delta", id: "t2", delta: "too late" },
            { type: "text-end", id: "t2" },
            FINISH_EVENT,
          ],
          { id: "slow", delay: 100 },
        ),
      )
      const model = createConjuntoLanguageModel({
        members: async () => [winner, loser],
        mode: "race",
      })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events } = await readStream(result.stream)

      expect(events[0]).toEqual({ type: "stream-start", warnings: [] })

      // Race mode does NOT label contributions (only one wins). The winner's
      // text deltas are forwarded as-is.
      const textDeltas = events
        .filter((e) => e.type === "text-delta")
        .map((e) => (e as { delta: string }).delta)
        .join("")
      expect(textDeltas).toContain("Hello")
      expect(textDeltas).toContain(" world")
      expect(textDeltas).not.toContain("### Conjunto Synthesis")

      const finish = events.find((e) => e.type === "finish")
      expect(finish).toBeDefined()
    })

    test("doStream surfaces an error when every member errors before producing output", async () => {
      const failing = makeMember(
        "bad/bad",
        fakeLanguageModel([], { id: "bad", errorOnStream: new Error("stream init failed") }),
      )
      const model = createConjuntoLanguageModel({
        members: async () => [failing],
        mode: "race",
      })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events } = await readStream(result.stream)
      const errorEvent = events.find((e) => e.type === "error")
      expect(errorEvent).toBeDefined()
    })

    test("doStream surfaces an error when the winning model errors mid-stream (race: no fallback)", async () => {
      const flaky = makeMember(
        "flaky/bad",
        fakeLanguageModel(
          [
            { type: "text-start", id: "t1" },
            { type: "text-delta", id: "t1", delta: "first token" },
            { type: "error", error: new Error("flaky model crashed") },
          ],
          { id: "flaky" },
        ),
      )
      const model = createConjuntoLanguageModel({
        members: async () => [flaky],
        mode: "race",
      })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const { events, error } = await readStream(result.stream)

      const textDeltas = events
        .filter((e) => e.type === "text-delta")
        .map((e) => (e as { delta: string }).delta)
      expect(textDeltas).toContain("first token")
      expect(error).toBeInstanceOf(Error)
      expect((error as Error).message).toBe("flaky model crashed")
    })
  })

  describe("createConjuntoLanguageModel — doGenerate (always races)", () => {
    test("races members and returns the first successful result", async () => {
      const fast = makeMember("fast/quick", fakeLanguageModel([], { id: "quick", delay: 1 }))
      const slow = makeMember("slow/slow", fakeLanguageModel([], { id: "slow", delay: 100 }))
      const model = createConjuntoLanguageModel({
        members: async () => [fast, slow],
        mode: "ensemble",
      })

      const result = await model.doGenerate({ prompt: [] } as LanguageModelV3CallOptions)
      expect(result.content).toEqual([{ type: "text", text: "result from quick" }])
    })

    test("throws when every member fails", async () => {
      const failing = makeMember(
        "bad/bad",
        fakeLanguageModel([], { id: "bad", errorOnGenerate: new Error("nope") }),
      )
      const model = createConjuntoLanguageModel({ members: async () => [failing] })

      await expect(model.doGenerate({ prompt: [] } as LanguageModelV3CallOptions)).rejects.toThrow(
        /Conjunto: every free model failed/,
      )
    })

    test("injects max reasoning options into doGenerate callOptions", async () => {
      let capturedOpts: LanguageModelV3CallOptions | null = null
      const member = makeMember(
        "alpha/openai",
        {
          specificationVersion: "v3",
          provider: "alpha",
          modelId: "openai",
          supportedUrls: {},
          async doGenerate(opts) {
            capturedOpts = opts
            return {
              content: [{ type: "text", text: "ok" }],
              usage: {
                inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
                outputTokens: { total: 1, text: 1, reasoning: 0 },
              } as any,
              finishReason: { unified: "stop", raw: undefined } as any,
              warnings: [],
            } as any
          },
          async doStream() {
            return { stream: new ReadableStream() }
          },
        },
        fakeModelMeta("@ai-sdk/openai", true),
      )
      const model = createConjuntoLanguageModel({
        members: async () => [member],
        forceMaxReasoning: true,
      })

      await model.doGenerate({ prompt: [] } as LanguageModelV3CallOptions)
      expect(capturedOpts).not.toBeNull()
      expect((capturedOpts!.providerOptions as any).openai?.reasoningEffort).toBe("high")
    })
  })
})
