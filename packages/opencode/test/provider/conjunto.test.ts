import { describe, test, expect } from "bun:test"
import {
  createConjuntoLanguageModel,
  isFreeModel,
  type ConjuntoMember,
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
  events: LanguageModelV3StreamPart[] | ((opts: LanguageModelV3CallOptions) => LanguageModelV3StreamPart[]),
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
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, reasoningTokens: 0 },
        finishReason: { type: "stop" },
        warnings: [],
        request: { body: {} },
        response: { headers: {}, id: opts.id, modelId: opts.id, timestamp: new Date() },
      }
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

function makeMember(id: string, language: LanguageModelV3): ConjuntoMember {
  const [providerID, modelID] = id.split("/")
  return { providerID, modelID, displayName: id, language }
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

  describe("createConjuntoLanguageModel", () => {
    test("exposes the correct static metadata", async () => {
      const model = createConjuntoLanguageModel({ members: async () => [] })
      expect(model.specificationVersion).toBe("v3")
      expect(model.provider).toBe("conjunto")
      expect(model.modelId).toBe("ensemble")
      expect(model.supportedUrls).toEqual({})
      expect(typeof model.doGenerate).toBe("function")
      expect(typeof model.doStream).toBe("function")
    })

    test("doStream returns an error event when no members are connected", async () => {
      const model = createConjuntoLanguageModel({ members: async () => [] })
      const result = await model.doStream({
        prompt: [],
      } as LanguageModelV3CallOptions)
      const reader = result.stream.getReader()
      const events: LanguageModelV3StreamPart[] = []
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        if (value) events.push(value)
      }
      expect(events[0]).toEqual({ type: "stream-start", warnings: [] })
      expect(events[1]?.type).toBe("error")
    })

    test("doStream forwards the winning member's stream verbatim", async () => {
      const winner = makeMember(
        "fast/quick",
        fakeLanguageModel(
          [
            { type: "text-start", id: "t1" },
            { type: "text-delta", id: "t1", delta: "Hello" },
            { type: "text-delta", id: "t1", delta: " world" },
            { type: "text-end", id: "t1" },
            { type: "finish", finishReason: { type: "stop" }, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, reasoningTokens: 0 } },
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
            { type: "finish", finishReason: { type: "stop" }, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, reasoningTokens: 0 } },
          ],
          { id: "slow", delay: 100 },
        ),
      )
      const model = createConjuntoLanguageModel({ members: async () => [winner, loser] })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const reader = result.stream.getReader()
      const events: LanguageModelV3StreamPart[] = []
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        if (value) events.push(value)
      }

      // First event is stream-start
      expect(events[0]).toEqual({ type: "stream-start", warnings: [] })

      // Winner's text deltas are forwarded
      const textDeltas = events.filter((e) => e.type === "text-delta")
      expect(textDeltas.length).toBe(2)
      expect((textDeltas[0] as { delta: string }).delta).toBe("Hello")
      expect((textDeltas[1] as { delta: string }).delta).toBe(" world")

      // Should end with a finish event
      const finish = events.find((e) => e.type === "finish")
      expect(finish).toBeDefined()
    })

    test("doStream surfaces an error when the winning model errors mid-stream (v1: no mid-stream fallback)", async () => {
      // Member that wins the race (emits a text-delta) but then errors
      // mid-stream. v1 behavior: once a winner emits content to upstream,
      // errors are surfaced directly (no mid-stream fallback). This is
      // documented in conjunto.ts and matches user expectations: if they
      // see partial output, they should see the error too.
      const flaky = makeMember(
        "flaky/bad",
        fakeLanguageModel(
          [
            { type: "text-start", id: "t1" },
            { type: "text-delta", id: "t1", delta: "first token" }, // makes it win
            { type: "error", error: new Error("flaky model crashed") },
          ],
          { id: "flaky" },
        ),
      )
      const model = createConjuntoLanguageModel({ members: async () => [flaky] })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const reader = result.stream.getReader()
      const events: LanguageModelV3StreamPart[] = []
      // When the upstream controller gets `error()`, the next read() rejects.
      // Capture that rejection as a sentinel and stop reading.
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

      // The flaky winner's first text-delta should be replayed before the error
      const textDeltas = events.filter((e) => e.type === "text-delta").map((e) => (e as { delta: string }).delta)
      expect(textDeltas).toContain("first token")

      // The stream should have errored out
      expect(streamErrored).toBeInstanceOf(Error)
      expect((streamErrored as Error).message).toBe("flaky model crashed")
    })

    test("doGenerate races members and returns the first successful result", async () => {
      const fast = makeMember(
        "fast/quick",
        fakeLanguageModel([], { id: "quick", delay: 1 }),
      )
      const slow = makeMember(
        "slow/slow",
        fakeLanguageModel([], { id: "slow", delay: 100 }),
      )
      const model = createConjuntoLanguageModel({ members: async () => [fast, slow] })

      const result = await model.doGenerate({ prompt: [] } as LanguageModelV3CallOptions)
      expect(result.content).toEqual([{ type: "text", text: "result from quick" }])
    })

    test("doGenerate throws when every member fails", async () => {
      const failing = makeMember(
        "bad/bad",
        fakeLanguageModel([], { id: "bad", errorOnGenerate: new Error("nope") }),
      )
      const model = createConjuntoLanguageModel({ members: async () => [failing] })

      await expect(model.doGenerate({ prompt: [] } as LanguageModelV3CallOptions)).rejects.toThrow(
        /Conjunto: every free model failed/,
      )
    })

    test("doStream surfaces an error when every member errors before producing output", async () => {
      const failing = makeMember(
        "bad/bad",
        fakeLanguageModel([], { id: "bad", errorOnStream: new Error("stream init failed") }),
      )
      const model = createConjuntoLanguageModel({ members: async () => [failing] })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const reader = result.stream.getReader()
      const events: LanguageModelV3StreamPart[] = []
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        if (value) events.push(value)
      }
      const errorEvent = events.find((e) => e.type === "error")
      expect(errorEvent).toBeDefined()
    })

    test("invokes onEvent lifecycle hooks for visibility", async () => {
      const events: Array<{ kind: string }> = []
      const member = makeMember(
        "fast/quick",
        fakeLanguageModel(
          [
            { type: "text-start", id: "t1" },
            { type: "text-delta", id: "t1", delta: "hi" },
            { type: "text-end", id: "t1" },
            { type: "finish", finishReason: { type: "stop" }, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, reasoningTokens: 0 } },
          ],
          { id: "quick" },
        ),
      )
      const model = createConjuntoLanguageModel({
        members: async () => [member],
        onEvent: (e) => events.push(e as { kind: string }),
      })

      const result = await model.doStream({ prompt: [] } as LanguageModelV3CallOptions)
      const reader = result.stream.getReader()
      while (true) {
        const { done } = await reader.read()
        if (done) break
      }

      const kinds = events.map((e) => e.kind)
      expect(kinds).toContain("members")
      expect(kinds).toContain("start")
      expect(kinds).toContain("won")
    })

    test("respects maxConcurrency by slicing the member list", async () => {
      const members: ConjuntoMember[] = Array.from({ length: 10 }, (_, i) =>
        makeMember(
          `p${i}/m${i}`,
          fakeLanguageModel(
            [
              { type: "text-start", id: `t${i}` },
              { type: "text-delta", id: `t${i}`, delta: `m${i}` },
              { type: "text-end", id: `t${i}` },
              { type: "finish", finishReason: { type: "stop" }, usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, reasoningTokens: 0 } },
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
      const reader = result.stream.getReader()
      while (true) {
        const { done } = await reader.read()
        if (done) break
      }

      // Only the first 3 members (p0, p1, p2) should have started.
      expect(startedEvents.length).toBeLessThanOrEqual(3)
      expect(startedEvents).toEqual(expect.arrayContaining(["p0/m0", "p1/m1", "p2/m2"]))
    })
  })
})
