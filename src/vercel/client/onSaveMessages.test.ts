import { expect, test, vi } from "vitest";
import { defineTestApp } from "convex-test";
import {
  createFunctionHandle,
  defineSchema,
  defineTable,
  makeFunctionReference,
  type FunctionReference,
  type GenericActionCtx,
  type GenericDataModel,
} from "convex/server";
import { v } from "convex/values";
import { isStepCount, type GenerateObjectResult } from "ai";
import { MockEmbeddingModelV4 } from "ai/test";
import { z } from "zod/v4";
import componentTest from "../../test.js";
import {
  Agent,
  createTool,
  saveMessage,
  vOnSaveMessagesArgs,
  type OnSaveMessagesArgs,
} from "../index.js";
import { vMessageStatus } from "../../validators.js";
import { mockModel } from "./mockModel.js";

// https://github.com/convex/agent/issues/147

// convex-test mints a handle for any reference, but a deployment rejects one
// whose function is gone, as during a deploy that removed the hook.
vi.mock("convex/server", async (importOriginal) => {
  const server = await importOriginal<typeof import("convex/server")>();
  return {
    ...server,
    createFunctionHandle: async (
      ref: Parameters<typeof server.createFunctionHandle>[0],
    ) => {
      if (server.getFunctionName(ref) === "hook:removed") {
        throw new Error("Couldn't resolve api.hook.removed");
      }
      return await server.createFunctionHandle(ref);
    },
  };
});

const vReported = v.object({
  kind: v.union(v.literal("insert"), v.literal("update"), v.literal("delete")),
  messageId: v.string(),
  order: v.number(),
  stepOrder: v.number(),
  status: vMessageStatus,
});
const app = defineTestApp({
  schema: defineSchema({
    reports: defineTable({
      threadId: v.string(),
      messages: v.array(vReported),
    }).index("threadId", ["threadId"]),
    failing: defineTable({ threadId: v.string() }).index("threadId", [
      "threadId",
    ]),
  }),
  components: { agent: componentTest },
});
const component = app.components.agent;

const record = app.internalMutation({
  args: vOnSaveMessagesArgs,
  handler: async (ctx, { threadId, messages }) => {
    await ctx.db.insert("reports", { threadId, messages });
    return null;
  },
});

const fail = app.internalMutation({
  args: vOnSaveMessagesArgs,
  handler: async (ctx, { threadId, messages }) => {
    await ctx.db.insert("reports", { threadId, messages });
    throw new Error("hook failed");
  },
});

// Throws for threads marked failing, after recording.
const picky = app.internalMutation({
  args: vOnSaveMessagesArgs,
  handler: async (ctx, { threadId, messages }) => {
    await ctx.db.insert("reports", { threadId, messages });
    const failing = await ctx.db
      .query("failing")
      .withIndex("threadId", (q) => q.eq("threadId", threadId))
      .first();
    if (failing) throw new Error(`hook failed for ${threadId}`);
    return null;
  },
});

// Saves a note to the thread it was told about, with the standalone
// saveMessage, which no hook hears about.
const reply = app.internalMutation({
  args: vOnSaveMessagesArgs,
  handler: async (ctx, { threadId, messages }) => {
    await ctx.db.insert("reports", { threadId, messages });
    await saveMessage(ctx, component, { threadId, prompt: "noted" });
    return null;
  },
});

const lookup = createTool({
  description: "Looks something up",
  inputSchema: z.object({ q: z.string() }),
  execute: async (_ctx, input) => `found ${input.q}`,
});

const embeddingModel = new MockEmbeddingModelV4({
  modelId: "mock-embedding",
  doEmbed: async ({ values }) => ({
    embeddings: values.map(() => Array.from({ length: 128 }, () => 0.5)),
    warnings: [],
  }),
});

type Hook = "record" | "fail" | "reply" | "picky" | "removed" | "none";

const agentWith = ({
  hook = "record",
  fails = false,
  embeds = false,
}: { hook?: Hook; fails?: boolean; embeds?: boolean } = {}): Agent =>
  new Agent(component, {
    name: "hooked",
    tools: { lookup },
    stopWhen: isStepCount(3),
    ...(embeds ? { embeddingModel } : {}),
    onSaveMessages:
      hook === "none"
        ? undefined
        : {
            record: internal.hook.record,
            fail: internal.hook.fail,
            reply: internal.hook.reply,
            picky: internal.hook.picky,
            removed: makeFunctionReference<"mutation", OnSaveMessagesArgs>(
              "hook:removed",
            ) as unknown as typeof internal.hook.record,
          }[hook],
    languageModel: mockModel(
      fails
        ? { fail: { error: "provider down" } }
        : {
            contentSteps: [
              [
                {
                  type: "tool-call",
                  toolCallId: "call-1",
                  toolName: "lookup",
                  input: JSON.stringify({ q: "x" }),
                },
              ],
              [{ type: "text", text: "reply" }],
            ],
          },
    ),
  });

const vHook = v.optional(
  v.union(
    v.literal("record"),
    v.literal("fail"),
    v.literal("reply"),
    v.literal("picky"),
    v.literal("removed"),
    v.literal("none"),
  ),
);

const createThread = app.mutation({
  args: {},
  handler: async (ctx): Promise<string> =>
    (await ctx.runMutation(component.threads.createThread, {}))._id,
});

const generate = app.action({
  args: {
    threadId: v.string(),
    stream: v.optional(v.boolean()),
    fails: v.optional(v.boolean()),
    hook: vHook,
    promptMessageId: v.optional(v.string()),
    embeds: v.optional(v.boolean()),
  },
  handler: async (ctx, { threadId, stream, promptMessageId, ...opts }) => {
    const agent = agentWith(opts);
    const prompt = promptMessageId ? { promptMessageId } : { prompt: "hello" };
    if (stream) {
      const result = await agent.streamText(ctx, { threadId }, prompt, {
        saveStreamDeltas: { chunking: "word", throttleMs: 0 },
      });
      await result.consumeStream();
    } else {
      await agent.generateText(ctx, { threadId }, prompt).catch((e) => {
        if (!opts.fails) throw e;
      });
    }
    return null;
  },
});

const save = app.mutation({
  args: {
    threadId: v.string(),
    text: v.optional(v.string()),
    pending: v.optional(v.boolean()),
    restart: v.optional(v.boolean()),
    hook: vHook,
    rollBack: v.optional(v.boolean()),
  },
  handler: async (ctx, args): Promise<string> => {
    const agent = agentWith({ hook: args.hook });
    const { messages } = await agent.saveMessages(ctx, {
      threadId: args.threadId,
      messages: [
        args.pending
          ? { role: "assistant", content: [] }
          : { role: "user", content: args.text ?? "hi" },
      ],
      metadata: args.pending ? [{ status: "pending" }] : undefined,
      failPendingSteps: args.restart,
      skipEmbeddings: true,
    });
    if (args.rollBack) throw new Error("rolled back");
    return messages[0]._id;
  },
});

const finalize = app.mutation({
  args: {
    messageId: v.string(),
    failed: v.boolean(),
    // Scheduled-argument bytes to leave before finalizing.
    scheduleRoom: v.optional(v.number()),
  },
  handler: async (ctx, { messageId, failed, scheduleRoom }) => {
    if (scheduleRoom !== undefined) {
      const remaining = async () =>
        (await ctx.meta.getTransactionMetrics()).scheduledFunctionArgsBytes
          .remaining;
      const before = await remaining();
      await ctx.scheduler.runAfter(0, internal.hook.noop, { filler: "" });
      const overhead = before - (await remaining());
      for (let left = (await remaining()) - scheduleRoom; left > 0; ) {
        const size = Math.min(left, 1 << 20);
        await ctx.scheduler.runAfter(0, internal.hook.noop, {
          filler: "x".repeat(size - overhead),
        });
        left -= size;
      }
      expect(await remaining()).toBe(scheduleRoom);
    }
    await agentWith().finalizeMessage(ctx, {
      messageId,
      result: failed
        ? { status: "failed", error: "stopped" }
        : { status: "success" },
    });
    return null;
  },
});

const update = app.mutation({
  args: { messageId: v.string() },
  handler: async (ctx, { messageId }) => {
    await agentWith().updateMessage(ctx, {
      messageId,
      patch: {
        message: { role: "user", content: "edited" },
        status: "success",
      },
    });
    return null;
  },
});

const embed = app.action({
  args: { messageId: v.string() },
  handler: async (ctx, { messageId }) => {
    await agentWith({ embeds: true }).generateAndSaveEmbeddings(ctx, {
      messageIds: [messageId],
    });
    return null;
  },
});

const requestAndApprove = app.mutation({
  args: { threadId: v.string(), deny: v.optional(v.boolean()) },
  handler: async (ctx, { threadId, deny }): Promise<string[]> => {
    const agent = agentWith();
    const { messageId: request } = await agent.saveMessage(ctx, {
      threadId,
      skipEmbeddings: true,
      message: {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call-a",
            toolName: "lookup",
            input: { q: "x" },
          },
          {
            type: "tool-approval-request",
            toolCallId: "call-a",
            approvalId: "a",
          },
        ],
      },
    });
    const { messageId: response } = await agent[
      deny ? "denyToolCall" : "approveToolCall"
    ](ctx, { threadId, approvalId: "a" });
    return [request, response];
  },
});

const remove = app.mutation({
  args: {
    threadId: v.string(),
    messageIds: v.optional(v.array(v.string())),
    range: v.optional(v.boolean()),
    thread: v.optional(v.boolean()),
    pageSize: v.optional(v.number()),
    hook: vHook,
  },
  // How many functions the delete scheduled.
  handler: async (ctx, args): Promise<number> => {
    const before = await ctx.meta.getTransactionMetrics();
    const agent = agentWith({ hook: args.hook });
    if (args.messageIds?.length === 1) {
      await agent.deleteMessage(ctx, { messageId: args.messageIds[0] });
    } else if (args.messageIds) {
      await agent.deleteMessages(ctx, { messageIds: args.messageIds });
    } else if (args.range) {
      await agent.deleteMessageRange(ctx, {
        threadId: args.threadId,
        startOrder: 0,
        endOrder: 100,
      });
    } else if (args.thread) {
      await agent.deleteThreadAsync(ctx, {
        threadId: args.threadId,
        pageSize: args.pageSize,
      });
    }
    const after = await ctx.meta.getTransactionMetrics();
    return after.functionsScheduled.used - before.functionsScheduled.used;
  },
});

// Saves a generated step and an object result straight to a new thread.
const saveStepAndObject = app.action({
  args: { threadId: v.string(), source: v.string() },
  handler: async (ctx, { threadId, source }) => {
    const { steps } = await agentWith({ hook: "none" }).generateText(
      ctx,
      { threadId: source },
      { prompt: "hello" },
    );
    const agent = agentWith();
    const { messageId: promptMessageId } = await agentWith({
      hook: "none",
    }).saveMessage(ctx, { threadId, prompt: "hello", skipEmbeddings: true });
    await agent.saveStep(ctx, { threadId, promptMessageId, step: steps[0] });
    await agent.saveObject(ctx, {
      userId: undefined,
      threadId,
      promptMessageId,
      model: undefined,
      provider: undefined,
      result: {
        object: { answer: 42 },
        finishReason: "stop",
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        warnings: [],
      } as unknown as GenerateObjectResult<unknown>,
    });
    return null;
  },
});

// Threads with one message each, saved without an Agent.
const seedThreads = app.mutation({
  args: { count: v.number() },
  handler: async (ctx, { count }): Promise<string[]> => {
    const messageIds: string[] = [];
    for (let i = 0; i < count; i++) {
      const { _id: threadId } = await ctx.runMutation(
        component.threads.createThread,
        {},
      );
      const { messages } = await ctx.runMutation(
        component.messages.addMessages,
        { threadId, messages: [{ message: { role: "user", content: "hi" } }] },
      );
      messageIds.push(messages[0]._id);
    }
    return messageIds;
  },
});

const markFailing = app.mutation({
  args: { threadId: v.string() },
  handler: async (ctx, { threadId }) => {
    await ctx.db.insert("failing", { threadId });
    return null;
  },
});

const noop = app.internalMutation({
  args: { filler: v.optional(v.string()) },
  handler: async () => null,
});

// Uses up the scheduler's per-transaction limit, then saves.
const saveAtSchedulerLimit = app.mutation({
  args: { threadId: v.string() },
  handler: async (ctx, { threadId }) => {
    for (let i = 0; i < 1000; i++) {
      await ctx.scheduler.runAfter(0, internal.hook.noop, {});
    }
    await agentWith().saveMessage(ctx, {
      threadId,
      prompt: "hi",
      skipEmbeddings: true,
    });
    return null;
  },
});

const objectAgent = (text: string) =>
  new Agent(component, {
    name: "objects",
    onSaveMessages: internal.hook.record,
    languageModel: mockModel({ content: [{ type: "text", text }] }),
  });
const answer = JSON.stringify({ hello: "x" });
const vSchemaFit = v.union(v.literal("fits"), v.literal("misfits"));
const schema = z.object({ hello: z.string() });

const generateOrStreamObject = app.action({
  args: { threadId: v.string(), stream: v.boolean(), fit: vSchemaFit },
  handler: async (ctx, { threadId, stream, fit }) => {
    const agent = objectAgent(
      fit === "fits" ? answer : JSON.stringify({ hello: 1 }),
    );
    if (!stream) {
      await agent
        .generateObject(ctx, { threadId }, { prompt: "Go.", schema })
        .catch(() => {});
      return null;
    }
    const result = await agent.streamObject(
      ctx,
      { threadId },
      { prompt: "Go.", schema },
    );
    for await (const _ of result.partialObjectStream) {
      // Drain.
    }
    await result.object.catch(() => {});
    await settled(ctx, threadId);
    return null;
  },
});

// Waits out saves a generation leaves running after it returns.
async function settled(
  ctx: Pick<GenericActionCtx<GenericDataModel>, "runQuery">,
  threadId: string,
) {
  await vi.waitFor(async () => {
    const { page } = await ctx.runQuery(
      component.messages.listMessagesByThreadId,
      {
        threadId,
        order: "asc",
        paginationOpts: { cursor: null, numItems: 100 },
      },
    );
    expect(page.filter((m) => m.status === "pending")).toEqual([]);
  });
}

const vDeltas = v.union(
  v.literal("off"),
  v.literal("awaited"),
  v.literal("immediate"),
);
const vEnding = v.union(
  v.literal("success"),
  v.literal("providerError"),
  v.literal("callerAbort"),
  v.literal("outOfBandAbort"),
  v.literal("callbackThrows"),
);

// A streamed reply, ending as told.
const streamEnding = app.action({
  args: { threadId: v.string(), deltas: vDeltas, ending: vEnding },
  handler: async (ctx, { threadId, deltas, ending }) => {
    const agent = new Agent(component, {
      name: "streamed",
      onSaveMessages: internal.hook.record,
      languageModel: mockModel({
        chunkDelayInMs: 5,
        ...(ending === "providerError"
          ? { fail: { error: "provider down" } }
          : {}),
        content: [{ type: "text", text: Array(40).fill("word").join(" ") }],
      }),
    });
    const controller = new AbortController();
    const generation = agent
      .streamText(
        ctx,
        { threadId },
        {
          prompt: "hello",
          abortSignal: controller.signal,
          onChunk: ({ chunk }) => {
            if (ending === "callerAbort" && chunk.type === "text-delta") {
              controller.abort(new Error("caller aborted"));
            }
          },
          onStepFinish: ({ text }) => {
            if (ending === "callbackThrows" && text) {
              throw new Error("callback failed");
            }
          },
        },
        {
          saveStreamDeltas:
            deltas === "off"
              ? false
              : {
                  chunking: "word",
                  throttleMs: 0,
                  returnImmediately: deltas === "immediate",
                },
        },
      )
      .then((result) => result.consumeStream());
    if (ending === "outOfBandAbort") {
      await vi.waitFor(async () => {
        const [streaming] = await ctx.runQuery(component.streams.list, {
          threadId,
          statuses: ["streaming"],
        });
        expect(streaming).toBeDefined();
        await ctx.runMutation(component.streams.abort, {
          streamId: streaming.streamId,
          reason: "external abort",
        });
      });
    }
    await generation.catch(() => {});
    await settled(ctx, threadId);
    return null;
  },
});

const vThreadMethod = v.union(
  v.literal("generateText"),
  v.literal("streamText"),
  v.literal("generateObject"),
  v.literal("streamObject"),
);

// Generates through a thread object instead of the Agent's methods.
const throughThread = app.action({
  args: {
    method: vThreadMethod,
    continued: v.boolean(),
  },
  handler: async (ctx, { method, continued }): Promise<string> => {
    const agent = objectAgent(answer);
    let threadId, thread;
    if (continued) {
      ({ _id: threadId } = await ctx.runMutation(
        component.threads.createThread,
        {},
      ));
      ({ thread } = await agent.continueThread(ctx, { threadId }));
    } else {
      ({ threadId, thread } = await agent.createThread(ctx, {}));
    }
    if (method === "generateText") {
      await thread!.generateText({ prompt: "Go." });
    } else if (method === "streamText") {
      await (await thread!.streamText({ prompt: "Go." })).consumeStream();
    } else if (method === "generateObject") {
      await thread!.generateObject({ prompt: "Go.", schema });
    } else {
      const result = await thread!.streamObject({ prompt: "Go.", schema });
      for await (const _ of result.partialObjectStream) {
        // Drain.
      }
      await result.object;
      await settled(ctx, threadId);
    }
    return threadId;
  },
});

// Refers to the hook by name: these are made before `internal` exists.
const wrapped = new Agent(component, {
  name: "wrapped",
  onSaveMessages: makeFunctionReference<"mutation", OnSaveMessagesArgs>(
    "hook:record",
  ) as unknown as FunctionReference<"mutation", "internal", OnSaveMessagesArgs>,
  languageModel: mockModel({ content: [{ type: "text", text: answer }] }),
});
const textAction = wrapped.asTextAction({});
const streamTextAction = wrapped.asTextAction({ stream: true });
const objectAction = wrapped.asObjectAction({ schema });
const saveMessagesMutation = wrapped.asSaveMessagesMutation();

// Saves at a deleted prompt, failing the thread's pending messages first.
const saveAtDeletedPrompt = app.mutation({
  args: { threadId: v.string() },
  // How many messages the save returned.
  handler: async (ctx, { threadId }): Promise<number> => {
    const { messageId: promptMessageId } = await saveMessage(ctx, component, {
      threadId,
      prompt: "gone",
    });
    await ctx.runMutation(component.messages.deleteByIds, {
      messageIds: [promptMessageId],
    });
    const { messages } = await ctx.runMutation(component.messages.addMessages, {
      threadId,
      promptMessageId,
      abandonIfPromptMissing: true,
      failPendingSteps: true,
      messages: [{ message: { role: "assistant", content: "late" } }],
      onSaveMessages: await createFunctionHandle(internal.hook.record),
    });
    return messages.length;
  },
});

const allReports = app.query({
  args: {},
  handler: async (ctx) =>
    (await ctx.db.query("reports").collect()).map(({ threadId, messages }) => ({
      threadId,
      messages,
    })),
});

const deleteThreadSync = app.action({
  args: { threadId: v.string(), pageSize: v.optional(v.number()) },
  handler: async (ctx, { threadId, pageSize }) => {
    await agentWith().deleteThreadSync(ctx, { threadId, pageSize });
    return null;
  },
});

// Every way an app writes messages without an Agent.
const writeWithoutAgent = app.action({
  args: { threadId: v.string() },
  handler: async (ctx, { threadId }) => {
    const { messageId } = await saveMessage(ctx, component, {
      threadId,
      prompt: "standalone",
    });
    await ctx.runMutation(component.messages.addMessages, {
      threadId,
      messages: [{ message: { role: "user", content: "direct" } }],
    });
    await ctx.runMutation(component.messages.updateMessage, {
      messageId,
      patch: { status: "failed" },
    });
    await ctx.runMutation(component.vector.index.insertBatch, {
      vectorDimension: 128,
      vectors: [
        {
          messageId,
          model: "mock",
          table: "messages",
          vector: Array.from({ length: 128 }, () => 0.5),
        },
      ],
    });
    await ctx.runMutation(component.messages.deleteByIds, {
      messageIds: [messageId],
    });
    return null;
  },
});

const saveWithHandle = app.mutation({
  args: { threadId: v.string(), handle: v.string() },
  handler: async (ctx, { threadId, handle }): Promise<string> =>
    (
      await ctx.runMutation(component.messages.addMessages, {
        threadId,
        messages: [{ message: { role: "user", content: "hi" } }],
        onSaveMessages: handle,
      })
    ).messages[0]._id,
});

const saveReasoning = app.mutation({
  args: { threadId: v.string() },
  handler: async (ctx, { threadId }) => {
    await ctx.runMutation(component.messages.addMessages, {
      threadId,
      messages: [
        {
          message: {
            role: "assistant",
            content: [
              { type: "reasoning", text: "r".repeat(200 * 1024) },
              { type: "text", text: "done" },
            ],
          },
        },
      ],
    });
    return null;
  },
});

const read = app.query({
  args: { threadId: v.string() },
  handler: async (ctx, { threadId }) => {
    const { page } = await ctx.runQuery(
      component.messages.listMessagesByThreadId,
      {
        threadId,
        order: "asc",
        paginationOpts: { cursor: null, numItems: 100 },
      },
    );
    const reports = await ctx.db
      .query("reports")
      .withIndex("threadId", (q) => q.eq("threadId", threadId))
      .collect();
    return {
      messages: page.map((m) => ({
        messageId: m._id,
        order: m.order,
        stepOrder: m.stepOrder,
        status: m.status,
      })),
      reports: reports.map((r) => r.messages),
    };
  },
});

const { api, internal, createTest } = app.defineModules({
  hook: {
    record,
    fail,
    reply,
    createThread,
    generate,
    save,
    finalize,
    update,
    embed,
    requestAndApprove,
    remove,
    deleteThreadSync,
    writeWithoutAgent,
    saveWithHandle,
    saveReasoning,
    read,
    picky,
    saveStepAndObject,
    seedThreads,
    markFailing,
    noop,
    saveAtSchedulerLimit,
    allReports,
    generateOrStreamObject,
    streamEnding,
    throughThread,
    textAction,
    streamTextAction,
    objectAction,
    saveMessagesMutation,
    saveAtDeletedPrompt,
  },
});

type Reported = OnSaveMessagesArgs["messages"][number];
type Message = Omit<Reported, "kind">;

async function setup(opts: Parameters<typeof createTest>[0] = {}) {
  const t = createTest(opts);
  const threadId = await t.mutation(api.hook.createThread, {});
  const state = async (id = threadId) => {
    await t.finishAllScheduledFunctions(() => {});
    return t.query(api.hook.read, { threadId: id });
  };
  return { t, threadId, state };
}

const as = (kind: Reported["kind"], m: Message): Reported => ({ kind, ...m });

// What replaying the reports in order says the thread now holds.
function replay(reports: Reported[][]) {
  const byId = new Map<string, Message>();
  for (const { kind, ...m } of reports.flat()) {
    if (kind === "delete") byId.delete(m.messageId);
    else byId.set(m.messageId, m);
  }
  return [...byId.values()].sort(
    (a, b) => a.order - b.order || a.stepOrder - b.stepOrder,
  );
}

test("a generation reports its prompt and pending reply, then each step", async () => {
  const { t, threadId, state } = await setup();
  await t.action(api.hook.generate, { threadId });
  const { messages, reports } = await state();
  const [prompt, call, result, final] = messages;
  expect(messages.map((m) => m.status)).toEqual([
    "success",
    "success",
    "success",
    "success",
  ]);
  expect(reports).toEqual([
    [as("insert", prompt), as("insert", { ...call, status: "pending" })],
    [as("update", call), as("insert", result)],
    [as("insert", final)],
  ]);
});

test("a streamed generation reports what it saves", async () => {
  const { t, threadId, state } = await setup();
  await t.action(api.hook.generate, { threadId, stream: true });
  const { messages, reports } = await state();
  expect(messages.length).toBeGreaterThan(1);
  expect(reports[0]).toEqual([
    as("insert", messages[0]),
    as("insert", { ...messages[1], status: "pending" }),
  ]);
  expect(replay(reports)).toEqual(messages);
});

test("a generation whose provider fails reports its reply failing", async () => {
  const { t, threadId, state } = await setup();
  await t.action(api.hook.generate, { threadId, fails: true });
  const { messages, reports } = await state();
  const [prompt, reply] = messages;
  expect(reply.status).toBe("failed");
  expect(reports).toEqual([
    [as("insert", prompt), as("insert", { ...reply, status: "pending" })],
    [as("update", reply)],
  ]);
});

test("a generation from a saved prompt reports the prompt's lazy embedding", async () => {
  const { t, threadId, state } = await setup();
  const promptMessageId = await t.mutation(api.hook.save, { threadId });
  await t.action(api.hook.generate, {
    threadId,
    promptMessageId,
    embeds: true,
  });
  const { messages, reports } = await state();
  expect(reports.slice(0, 2)).toEqual([
    [as("insert", messages[0])],
    [as("update", messages[0])],
  ]);
  expect(replay(reports)).toEqual(messages);
});

test("a restart reports the pending message it fails, then the new prompt", async () => {
  const { t, threadId, state } = await setup();
  const pending = await t.mutation(api.hook.save, { threadId, pending: true });
  const next = await t.mutation(api.hook.save, { threadId, restart: true });
  const { messages, reports } = await state();
  const byId = (id: string) => messages.find((m) => m.messageId === id)!;
  expect(byId(pending).status).toBe("failed");
  expect(reports).toEqual([
    [as("insert", { ...byId(pending), status: "pending" })],
    [as("update", byId(pending)), as("insert", byId(next))],
  ]);
});

test.each([false, true])(
  "finalizing reports the pending message's end once (failed: %s)",
  async (failed) => {
    const { t, threadId, state } = await setup();
    const pending = await t.mutation(api.hook.save, {
      threadId,
      pending: true,
    });
    await t.mutation(api.hook.finalize, { messageId: pending, failed });
    await t.mutation(api.hook.finalize, { messageId: pending, failed });
    const { messages, reports } = await state();
    expect(reports).toEqual([
      [as("insert", { ...messages[0], status: "pending" })],
      [as("update", messages[0])],
    ]);
    expect(messages[0].status).toBe(failed ? "failed" : "success");
  },
);

// A pending reply with a tool step and partial text left in its stream.
async function pendingWithStream(
  t: Awaited<ReturnType<typeof setup>>["t"],
  threadId: string,
) {
  const pending = await t.mutation(api.hook.save, { threadId, pending: true });
  const streamId = await t.mutation(component.streams.create, {
    threadId,
    order: 0,
    stepOrder: 0,
    format: "UIMessageChunk",
  });
  const parts = [
    { type: "start" },
    { type: "start-step" },
    {
      type: "tool-input-available",
      toolCallId: "call-1",
      toolName: "lookup",
      input: { q: "x" },
    },
    { type: "tool-output-available", toolCallId: "call-1", output: "found" },
    { type: "finish-step" },
    { type: "start-step" },
    { type: "text-start", id: "t" },
    { type: "text-delta", id: "t", delta: "partial" },
  ];
  await t.mutation(component.streams.addDelta, {
    streamId,
    start: 0,
    end: parts.length,
    parts,
  });
  return pending;
}

test("finalizing reports every message recovered from the stream once", async () => {
  const { t, threadId, state } = await setup();
  const pending = await pendingWithStream(t, threadId);
  await t.mutation(api.hook.finalize, { messageId: pending, failed: true });
  await t.mutation(api.hook.finalize, { messageId: pending, failed: true });
  const { messages, reports } = await state();
  expect(messages.length).toBeGreaterThan(1);
  expect(messages[0].messageId).toBe(pending);
  expect(reports).toEqual([
    [as("insert", { ...messages[0], status: "pending" })],
    messages.map((m, i) => as(i === 0 ? "update" : "insert", m)),
  ]);
});

// Room for a one-message report, not for the recovered batch's.
test("a finalize that recovers stream output fails whole when its report can't be scheduled", async () => {
  const { t, threadId, state } = await setup({ transactionLimits: true });
  const pending = await pendingWithStream(t, threadId);
  const { messages: before } = await state();
  await expect(
    t.mutation(api.hook.finalize, {
      messageId: pending,
      failed: true,
      scheduleRoom: 400,
    }),
  ).rejects.toThrow("Scheduled function arguments too large");
  expect(await state()).toEqual({
    messages: before,
    reports: [[as("insert", { ...before[0], status: "pending" })]],
  });
});

test("saving a step or an object result directly reports what each saved", async () => {
  const { t, threadId, state } = await setup();
  const source = await t.mutation(api.hook.createThread, {});
  await t.action(api.hook.saveStepAndObject, { threadId, source });
  const { messages, reports } = await state();
  const [, ...saved] = messages;
  expect(reports.length).toBe(2);
  expect(reports.flat()).toEqual(saved.map((m) => as("insert", m)));
});

test("an approval's response is reported", async () => {
  const { t, threadId, state } = await setup();
  await t.mutation(api.hook.requestAndApprove, { threadId });
  const { messages, reports } = await state();
  const [request, response] = messages;
  expect(reports).toEqual([[as("insert", request)], [as("insert", response)]]);
});

test("an update and an embedding are reported", async () => {
  const { t, threadId, state } = await setup();
  const messageId = await t.mutation(api.hook.save, { threadId });
  await t.mutation(api.hook.update, { messageId });
  await t.action(api.hook.embed, { messageId });
  const { messages, reports } = await state();
  expect(reports).toEqual([
    [as("insert", messages[0])],
    [as("update", messages[0])],
    [as("update", messages[0])],
  ]);
});

test("deleting a message, messages, or a range is reported", async () => {
  const { t, threadId, state } = await setup();
  const ids = [];
  for (const text of ["a", "b", "c", "d", "e"]) {
    ids.push(await t.mutation(api.hook.save, { threadId, text }));
  }
  const { messages: saved } = await state();
  await t.mutation(api.hook.remove, { threadId, messageIds: [ids[0]] });
  await t.mutation(api.hook.remove, { threadId, messageIds: ids.slice(1, 3) });
  await t.mutation(api.hook.remove, { threadId, range: true });
  const { messages, reports } = await state();
  expect(messages).toEqual([]);
  expect(reports.slice(5)).toEqual([
    [as("delete", saved[0])],
    [as("delete", saved[1]), as("delete", saved[2])],
    [as("delete", saved[3]), as("delete", saved[4])],
  ]);
});

test("deleting messages from two threads reports each thread's deletes to it", async () => {
  const { t, threadId, state } = await setup();
  const other = await t.mutation(api.hook.createThread, {});
  const a = await t.mutation(api.hook.save, { threadId });
  const b = await t.mutation(api.hook.save, { threadId: other });
  const { messages: before } = await state();
  const { messages: otherBefore } = await state(other);
  expect(
    await t.mutation(api.hook.remove, { threadId, messageIds: [a, b] }),
  ).toBe(1);
  expect((await state()).reports.at(-1)).toEqual([as("delete", before[0])]);
  expect((await state(other)).reports.at(-1)).toEqual([
    as("delete", otherBefore[0]),
  ]);
});

test("a delete spanning more than 1,000 threads schedules one function, and every thread hears", async () => {
  const { t } = await setup({ transactionLimits: true });
  const messageIds: string[] = [];
  for (let i = 0; i < 5; i++) {
    messageIds.push(
      ...(await t.mutation(api.hook.seedThreads, { count: i < 4 ? 250 : 1 })),
    );
  }
  expect(await t.mutation(api.hook.remove, { threadId: "", messageIds })).toBe(
    1,
  );
  await t.finishAllScheduledFunctions(() => {});
  const reports = await t.query(api.hook.allReports, {});
  expect(reports.length).toBe(1001);
  expect(new Set(reports.map((r) => r.threadId)).size).toBe(1001);
  expect(
    reports.flatMap((r) => r.messages.map((m) => m.messageId)).sort(),
  ).toEqual([...messageIds].sort());
  expect(reports.every((r) => r.messages[0].kind === "delete")).toBe(true);
}, 30_000);

test("a save whose report can't be scheduled fails whole, like any runAfter in it", async () => {
  const { t, threadId, state } = await setup({ transactionLimits: true });
  await expect(
    t.mutation(api.hook.saveAtSchedulerLimit, { threadId }),
  ).rejects.toThrow("Scheduled too many functions");
  expect(await state()).toEqual({ messages: [], reports: [] });
});

test("a thread whose hook throws doesn't drop another thread's report", async () => {
  const error = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    const { t, threadId, state } = await setup();
    const other = await t.mutation(api.hook.createThread, {});
    const a = await t.mutation(api.hook.save, { threadId, hook: "none" });
    const b = await t.mutation(api.hook.save, {
      threadId: other,
      hook: "none",
    });
    await t.mutation(api.hook.markFailing, { threadId: other });
    const { messages } = await state();
    await t.mutation(api.hook.remove, {
      threadId,
      messageIds: [a, b],
      hook: "picky",
    });
    expect((await state()).reports).toEqual([[as("delete", messages[0])]]);
    expect((await state(other)).reports).toEqual([]);
    expect(error).toHaveBeenCalledWith(
      expect.stringContaining(`onSaveMessages failed for thread ${other}`),
      expect.anything(),
    );
  } finally {
    error.mockRestore();
  }
});

test.each(["async", "sync"] as const)(
  "deleting a thread reports every message, page by page (%s)",
  async (mode) => {
    const { t, threadId, state } = await setup();
    for (const text of ["a", "b", "c"]) {
      await t.mutation(api.hook.save, { threadId, text });
    }
    const { messages: saved } = await state();
    if (mode === "async") {
      await t.mutation(api.hook.remove, {
        threadId,
        thread: true,
        pageSize: 2,
      });
    } else {
      await t.action(api.hook.deleteThreadSync, { threadId, pageSize: 2 });
    }
    const { messages, reports } = await state();
    expect(messages).toEqual([]);
    expect(reports.slice(3)).toEqual([
      saved.slice(0, 2).map((m) => as("delete", m)),
      [as("delete", saved[2])],
    ]);
  },
);

test("deleting a thread of large messages fits in the scheduler's limits", async () => {
  const { t, threadId, state } = await setup({ transactionLimits: true });
  for (let i = 0; i < 30; i++) {
    await t.mutation(api.hook.saveReasoning, { threadId });
  }
  await t.mutation(api.hook.remove, { threadId, thread: true, pageSize: 30 });
  const { messages } = await state();
  expect(messages).toEqual([]);
  const reports = await t.run((ctx) =>
    ctx.db
      .query("reports")
      .withIndex("threadId", (q) => q.eq("threadId", threadId))
      .collect(),
  );
  expect(reports.flatMap((r) => r.messages).length).toBe(30);
});

test("the hook gets ids, positions, statuses and kinds, not message docs", () => {
  expect(Object.keys(vOnSaveMessagesArgs.fields).sort()).toEqual([
    "messages",
    "threadId",
  ]);
  expect(
    Object.keys(vOnSaveMessagesArgs.fields.messages.element.fields).sort(),
  ).toEqual(["kind", "messageId", "order", "status", "stepOrder"]);
});

test("writes made without an Agent are not reported", async () => {
  const { t, threadId, state } = await setup();
  await t.action(api.hook.writeWithoutAgent, { threadId });
  await t.mutation(api.hook.save, { threadId, hook: "none" });
  const { messages, reports } = await state();
  expect(messages.length).toBe(2);
  expect(reports).toEqual([]);
});

test("a save its mutation rolls back is never reported", async () => {
  const { t, threadId, state } = await setup();
  await expect(
    t.mutation(api.hook.save, { threadId, rollBack: true }),
  ).rejects.toThrow("rolled back");
  expect(await state()).toEqual({ messages: [], reports: [] });
});

test("a hook that throws never fails the save or the generation", async () => {
  const error = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    const { t, threadId, state } = await setup();
    await t.action(api.hook.generate, { threadId, hook: "fail" });
    await t.mutation(api.hook.save, { threadId, hook: "fail" });
    const { messages, reports } = await state();
    expect(messages.map((m) => m.status)).toEqual([
      "success",
      "success",
      "success",
      "success",
      "success",
    ]);
    expect(reports).toEqual([]);
    expect(error).toHaveBeenCalledWith(
      expect.stringContaining(`onSaveMessages failed for thread ${threadId}`),
      expect.objectContaining({
        message: expect.stringContaining("hook failed"),
      }),
    );
  } finally {
    error.mockRestore();
  }
});

test("a handle that no longer resolves never fails the save", async () => {
  const error = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    const { t, threadId, state } = await setup();
    const messageId = await t.mutation(api.hook.saveWithHandle, {
      threadId,
      handle: "function://;hook:removed",
    });
    const { messages, reports } = await state();
    expect(messages.map((m) => m.messageId)).toEqual([messageId]);
    expect(reports).toEqual([]);
    expect(error).toHaveBeenCalledWith(
      expect.stringContaining(`onSaveMessages failed for thread ${threadId}`),
      expect.anything(),
    );
  } finally {
    error.mockRestore();
  }
});

test("a hook removed from the deployment never fails the save or the generation", async () => {
  const error = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    const { t, threadId, state } = await setup();
    await t.action(api.hook.generate, { threadId, hook: "removed" });
    await t.mutation(api.hook.save, { threadId, hook: "removed" });
    const { messages, reports } = await state();
    expect(messages.map((m) => m.status)).toEqual([
      "success",
      "success",
      "success",
      "success",
      "success",
    ]);
    expect(reports).toEqual([]);
    expect(error).toHaveBeenCalledWith(
      expect.stringContaining("onSaveMessages"),
      expect.objectContaining({
        message: expect.stringContaining("Couldn't resolve api.hook.removed"),
      }),
    );
  } finally {
    error.mockRestore();
  }
});

test("a hook that saves to its own thread without the Agent doesn't hear about it", async () => {
  const { t, threadId, state } = await setup();
  await t.mutation(api.hook.save, { threadId, hook: "reply" });
  const { messages, reports } = await state();
  expect(messages.length).toBe(2);
  expect(reports).toEqual([[as("insert", messages[0])]]);
});

test.each([
  { stream: false, fit: "fits" },
  { stream: false, fit: "misfits" },
  { stream: true, fit: "fits" },
  { stream: true, fit: "misfits" },
] as const)(
  "an object generation reports each save it commits (stream: $stream, schema $fit)",
  async ({ stream, fit }) => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const { t, threadId, state } = await setup();
      await t.action(api.hook.generateOrStreamObject, {
        threadId,
        stream,
        fit,
      });
      const { messages, reports } = await state();
      expect(messages.map((m) => m.status)).toEqual([
        "success",
        fit === "fits" ? "success" : "failed",
      ]);
      expect(reports[0]).toEqual([
        as("insert", messages[0]),
        as("insert", { ...messages[1], status: "pending" }),
      ]);
      expect(replay(reports)).toEqual(messages);
    } finally {
      error.mockRestore();
    }
  },
);

test.each(
  (["off", "awaited", "immediate"] as const).flatMap((deltas) =>
    (
      [
        "success",
        "providerError",
        "callerAbort",
        "outOfBandAbort",
        "callbackThrows",
      ] as const
    )
      .filter((ending) => deltas !== "off" || ending !== "outOfBandAbort")
      .map((ending) => ({ deltas, ending })),
  ),
)(
  "a streamed generation reports each save it commits (deltas $deltas, $ending)",
  async ({ deltas, ending }) => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const { t, threadId, state } = await setup();
      await t.action(api.hook.streamEnding, { threadId, deltas, ending });
      const { messages, reports } = await state();
      expect(messages.map((m) => m.status)).toEqual([
        "success",
        ending === "success" || ending === "callbackThrows"
          ? "success"
          : "failed",
      ]);
      expect(replay(reports)).toEqual(messages);
    } finally {
      error.mockRestore();
      warn.mockRestore();
    }
  },
);

test("a denied tool call's response is reported", async () => {
  const { t, threadId, state } = await setup();
  await t.mutation(api.hook.requestAndApprove, { threadId, deny: true });
  const { messages, reports } = await state();
  const [request, response] = messages;
  expect(reports).toEqual([[as("insert", request)], [as("insert", response)]]);
});

test("a save abandoned at a deleted prompt reports the pending messages it failed", async () => {
  const { t, threadId, state } = await setup();
  await t.mutation(api.hook.save, { threadId, pending: true, hook: "none" });
  const { messages: before } = await state();
  expect(await t.mutation(api.hook.saveAtDeletedPrompt, { threadId })).toBe(0);
  const { messages, reports } = await state();
  expect(messages).toEqual([{ ...before[0], status: "failed" }]);
  expect(reports).toEqual([[as("update", messages[0])]]);
});

// A pending reply whose stream holds a delta recovery can't materialize.
async function pendingWithMalformedStream(
  t: Awaited<ReturnType<typeof setup>>["t"],
  threadId: string,
) {
  const pending = await t.mutation(api.hook.save, { threadId, pending: true });
  const streamId = await t.mutation(component.streams.create, {
    threadId,
    order: 0,
    stepOrder: 0,
    format: "UIMessageChunk",
  });
  await t.mutation(component.streams.addDelta, {
    streamId,
    start: 0,
    end: 1,
    parts: [{ type: "text-delta", id: "missing", delta: "partial" }],
  });
  return pending;
}

// A pending reply whose recovered text is larger than the save may write.
async function pendingWithOversizedStream(
  t: Awaited<ReturnType<typeof setup>>["t"],
  threadId: string,
) {
  const pending = await t.mutation(api.hook.save, { threadId, pending: true });
  const streamId = await t.mutation(component.streams.create, {
    threadId,
    order: 0,
    stepOrder: 0,
    format: "UIMessageChunk",
  });
  const parts = [
    { type: "start" },
    { type: "start-step" },
    { type: "text-start", id: "t" },
  ];
  await t.mutation(component.streams.addDelta, {
    streamId,
    start: 0,
    end: parts.length,
    parts,
  });
  for (let i = 0; i < 3; i++) {
    await t.mutation(component.streams.addDelta, {
      streamId,
      start: parts.length + i,
      end: parts.length + i + 1,
      parts: [{ type: "text-delta", id: "t", delta: "x".repeat(300_000) }],
    });
  }
  return pending;
}

test.each([
  { name: "can't be materialized", make: pendingWithMalformedStream },
  { name: "can't be written", make: pendingWithOversizedStream },
])(
  "a finalize whose stream recovery $name reports its failure, or fails whole",
  async ({ make }) => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const { t, threadId, state } = await setup({
        transactionLimits: { bytesWritten: 1024 * 1024 },
      });
      const pending = await make(t, threadId);
      const before = await state();
      await expect(
        t.mutation(api.hook.finalize, {
          messageId: pending,
          failed: false,
          scheduleRoom: 0,
        }),
      ).rejects.toThrow("Scheduled function arguments too large");
      expect(await state()).toEqual(before);
      await t.mutation(api.hook.finalize, {
        messageId: pending,
        failed: false,
      });
      const { messages, reports } = await state();
      expect(messages).toEqual([{ ...before.messages[0], status: "failed" }]);
      expect(reports).toEqual([
        [as("insert", { ...messages[0], status: "pending" })],
        [as("update", messages[0])],
      ]);
      expect(error).toHaveBeenCalledWith(
        expect.stringMatching(/^Failed to (materialize|persist)/),
        expect.anything(),
      );
    } finally {
      error.mockRestore();
    }
  },
);

test.each(
  (
    ["generateText", "streamText", "generateObject", "streamObject"] as const
  ).flatMap((method) => [
    { method, continued: false },
    { method, continued: true },
  ]),
)(
  "a thread's $method reports each save it commits (continued: $continued)",
  async ({ method, continued }) => {
    const { t, state } = await setup();
    const threadId = await t.action(api.hook.throughThread, {
      method,
      continued,
    });
    const { messages, reports } = await state(threadId);
    expect(messages.map((m) => m.status)).toEqual(["success", "success"]);
    expect(reports[0]).toEqual([
      as("insert", messages[0]),
      as("insert", { ...messages[1], status: "pending" }),
    ]);
    expect(replay(reports)).toEqual(messages);
  },
);

test.each(["textAction", "streamTextAction", "objectAction"] as const)(
  "%s reports each save it commits",
  async (action) => {
    const { t, threadId, state } = await setup();
    await t.action(internal.hook[action], { threadId, prompt: "Go." });
    const { messages, reports } = await state();
    expect(messages.map((m) => m.status)).toEqual(["success", "success"]);
    expect(replay(reports)).toEqual(messages);
  },
);

test("asSaveMessagesMutation reports what it saves", async () => {
  const { t, threadId, state } = await setup();
  await t.mutation(internal.hook.saveMessagesMutation, {
    threadId,
    messages: [{ message: { role: "user", content: "hi" } }],
  });
  const { messages, reports } = await state();
  expect(reports).toEqual([[as("insert", messages[0])]]);
});
