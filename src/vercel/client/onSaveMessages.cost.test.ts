import { expect, test } from "vitest";
import { defineTestApp } from "convex-test";
import {
  defineSchema,
  getFunctionAddress,
  type FunctionReference,
  type QueryMeta,
} from "convex/server";
import { v } from "convex/values";
import { isStepCount } from "ai";
import { MockEmbeddingModelV4 } from "ai/test";
import { z } from "zod/v4";
import componentTest from "../../test.js";
import { Agent, createTool, vOnSaveMessagesArgs } from "../index.js";
import { mockModel } from "./mockModel.js";
import type { ActionCtx, MutationCtx } from "./types.js";

// What each Agent path asks of the component with no onSaveMessages: the
// calls in order, their argument keys and sizes, and what each call reads,
// writes and schedules. BASE was measured on main before the hook existed.
const app = defineTestApp({
  schema: defineSchema({}),
  components: { agent: componentTest },
});
const component = app.components.agent;

type Metrics = Record<(typeof metricKeys)[number], number>;
type Call = { fn: string; args: string[]; argBytes: number } & Partial<Metrics>;

const metricKeys = [
  "bytesRead",
  "documentsRead",
  "databaseQueries",
  "bytesWritten",
  "documentsWritten",
  "functionsScheduled",
  "scheduledFunctionArgsBytes",
] as const;

const prefix = "_reference/childComponent/agent/";
const componentPath = (ref: unknown) => {
  const { reference } = getFunctionAddress(
    ref as FunctionReference<"query" | "mutation" | "action">,
  );
  return reference?.startsWith(prefix)
    ? reference.slice(prefix.length)
    : undefined;
};
const resolve = (path: string) =>
  path
    .split("/")
    .reduce(
      (ref: Record<string, unknown>, key) =>
        ref[key] as Record<string, unknown>,
      component as unknown as Record<string, unknown>,
    ) as unknown as FunctionReference<"query" | "mutation">;

async function measured<T>(
  ctx: { meta: QueryMeta },
  run: () => Promise<T>,
): Promise<{ result: T; metrics: Metrics }> {
  const before = await ctx.meta.getTransactionMetrics();
  const result = await run();
  const after = await ctx.meta.getTransactionMetrics();
  const metrics = Object.fromEntries(
    metricKeys.map((k) => [k, after[k].used - before[k].used]),
  ) as Metrics;
  return { result, metrics };
}

const measureMutation = app.internalMutation({
  args: { path: v.string(), args: v.any() },
  handler: (ctx, { path, args }) =>
    measured(ctx, () =>
      ctx.runMutation(resolve(path) as FunctionReference<"mutation">, args),
    ),
});

const measureQuery = app.internalQuery({
  args: { path: v.string(), args: v.any() },
  handler: (ctx, { path, args }) =>
    measured(ctx, () =>
      ctx.runQuery(resolve(path) as FunctionReference<"query">, args),
    ),
});

function recording(ctx: ActionCtx, calls: Call[]): ActionCtx {
  const entry = (ref: unknown, args: Record<string, unknown> = {}): Call => ({
    fn: componentPath(ref) ?? "app function",
    args: Object.keys(args)
      .filter((k) => args[k] !== undefined)
      .sort(),
    argBytes: JSON.stringify(args).length,
  });
  const through =
    (kind: "query" | "mutation") =>
    async (ref: unknown, args?: Record<string, unknown>) => {
      const call = entry(ref, args);
      calls.push(call);
      if (componentPath(ref)) {
        const measure = { path: call.fn, args: args ?? {} };
        const { result, metrics } =
          kind === "query"
            ? await ctx.runQuery(internal.cost.measureQuery, measure)
            : await ctx.runMutation(internal.cost.measureMutation, measure);
        Object.assign(call, metrics);
        return result;
      }
      return kind === "query"
        ? ctx.runQuery(ref as FunctionReference<"query">, args)
        : ctx.runMutation(ref as FunctionReference<"mutation">, args);
    };
  return {
    ...ctx,
    runQuery: through("query"),
    runMutation: through("mutation"),
    runAction: async (ref: unknown, args?: Record<string, unknown>) => {
      calls.push(entry(ref, args));
      return ctx.runAction(ref as FunctionReference<"action">, args);
    },
  } as ActionCtx;
}

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

const hook = app.internalMutation({
  args: vOnSaveMessagesArgs,
  handler: async () => null,
});

const agentFor = (opts: {
  hooked: boolean;
  fails?: boolean;
  embeds?: boolean;
}): Agent =>
  new Agent(component, {
    name: "cost",
    tools: { lookup },
    stopWhen: isStepCount(3),
    ...(opts.embeds ? { embeddingModel } : {}),
    ...(opts.hooked ? { onSaveMessages: internal.cost.hook } : {}),
    languageModel: mockModel(
      opts.fails
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

const paths = [
  "generateText",
  "streamText",
  "generateText, provider fails",
  "generateText, lazy prompt embedding",
  "saveMessage",
  "saveMessages, restart",
  "finalizeMessage",
  "updateMessage",
  "approveToolCall",
  "deleteMessage",
  "deleteMessages",
  "deleteMessages, two threads",
  "deleteMessageRange",
  "deleteThreadAsync",
  "deleteThreadSync",
  "generateAndSaveEmbeddings",
] as const;

const run = app.action({
  args: { path: v.string(), hooked: v.boolean() },
  handler: async (ctx, { path, hooked }): Promise<Call[]> => {
    const agent = agentFor({ hooked });
    const plain = agentFor({ hooked: false });
    const { threadId } = await plain.createThread(ctx, {});
    const save = (text: string, extra = {}) =>
      plain.saveMessage(ctx, {
        threadId,
        prompt: text,
        skipEmbeddings: true,
        ...extra,
      });
    const savePending = async () =>
      (
        await plain.saveMessage(ctx, {
          threadId,
          message: { role: "assistant", content: [] },
          metadata: { status: "pending" },
          skipEmbeddings: true,
        })
      ).messageId;
    const calls: Call[] = [];
    const rec = recording(ctx, calls);
    const asMutation = rec as unknown as MutationCtx;
    switch (path as (typeof paths)[number]) {
      case "generateText":
        await agent.generateText(rec, { threadId }, { prompt: "hello" });
        break;
      case "streamText": {
        const result = await agent.streamText(
          rec,
          { threadId },
          { prompt: "hello" },
          { saveStreamDeltas: { chunking: "word", throttleMs: 0 } },
        );
        await result.consumeStream();
        break;
      }
      case "generateText, provider fails":
        await agentFor({ hooked, fails: true })
          .generateText(rec, { threadId }, { prompt: "hello" })
          .catch(() => {});
        break;
      case "generateText, lazy prompt embedding": {
        const { messageId } = await save("hello");
        await agentFor({ hooked, embeds: true }).generateText(
          rec,
          { threadId },
          { promptMessageId: messageId },
        );
        break;
      }
      case "saveMessage":
        await agent.saveMessage(asMutation, {
          threadId,
          prompt: "hi",
          skipEmbeddings: true,
        });
        break;
      case "saveMessages, restart":
        await savePending();
        await agent.saveMessages(asMutation, {
          threadId,
          messages: [{ role: "user", content: "again" }],
          failPendingSteps: true,
          skipEmbeddings: true,
        });
        break;
      case "finalizeMessage":
        await agent.finalizeMessage(asMutation, {
          messageId: await savePending(),
          result: { status: "success" },
        });
        break;
      case "updateMessage":
        await agent.updateMessage(asMutation, {
          messageId: (await save("hi")).messageId,
          patch: {
            message: { role: "user", content: "edited" },
            status: "success",
          },
        });
        break;
      case "approveToolCall":
        await plain.saveMessage(ctx, {
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
        await agent.approveToolCall(asMutation, { threadId, approvalId: "a" });
        break;
      case "deleteMessage":
        await agent.deleteMessage(asMutation, {
          messageId: (await save("hi")).messageId,
        });
        break;
      case "deleteMessages": {
        const a = (await save("a")).messageId;
        const b = (await save("b")).messageId;
        await agent.deleteMessages(asMutation, { messageIds: [a, b] });
        break;
      }
      case "deleteMessages, two threads": {
        const { threadId: other } = await plain.createThread(ctx, {});
        const a = (await save("a")).messageId;
        const { messageId: b } = await plain.saveMessage(ctx, {
          threadId: other,
          prompt: "b",
          skipEmbeddings: true,
        });
        await agent.deleteMessages(asMutation, { messageIds: [a, b] });
        break;
      }
      case "deleteMessageRange":
        await save("a");
        await save("b");
        await agent.deleteMessageRange(asMutation, {
          threadId,
          startOrder: 0,
          endOrder: 10,
        });
        break;
      case "deleteThreadAsync":
        await save("a");
        await agent.deleteThreadAsync(asMutation, { threadId });
        break;
      case "deleteThreadSync":
        await save("a");
        await agent.deleteThreadSync(rec, { threadId });
        break;
      case "generateAndSaveEmbeddings":
        await agentFor({ hooked, embeds: true }).generateAndSaveEmbeddings(
          rec,
          { messageIds: [(await save("hi")).messageId] },
        );
        break;
      default:
        throw new Error(`unknown path ${path}`);
    }
    return calls;
  },
});

const { internal, api, createTest } = app.defineModules({
  cost: { measureMutation, measureQuery, hook, run },
});

function describe(c: Call) {
  const head = `${c.fn}(${c.args.join(",")}) ${c.argBytes}B`;
  if (c.bytesRead === undefined) return head;
  return (
    head +
    ` read ${c.bytesRead}B/${c.documentsRead} docs/${c.databaseQueries} queries,` +
    ` wrote ${c.bytesWritten}B/${c.documentsWritten} docs,` +
    ` scheduled ${c.functionsScheduled}/${c.scheduledFunctionArgsBytes}B`
  );
}

async function costs(hooked: boolean) {
  const out: Record<string, Call[]> = {};
  for (const path of paths) {
    const t = createTest({ transactionLimits: true });
    out[path] = await t.action(api.cost.run, { path, hooked });
  }
  return out;
}

const mapCalls = (costs: Record<string, Call[]>, f: (call: Call) => unknown) =>
  Object.fromEntries(Object.entries(costs).map(([k, c]) => [k, c.map(f)]));

test("an Agent without onSaveMessages asks of the component exactly what it did before the hook", async () => {
  expect(mapCalls(await costs(false), describe)).toEqual(BASE);
});

const writes = [
  "messages/addMessages",
  "messages/finalizeMessage",
  "messages/updateMessage",
  "messages/respondToToolCallApprovals",
  "messages/deleteByIds",
  "messages/deleteByOrder",
  "threads/deleteAllForThreadIdAsync",
  "threads/deleteAllForThreadIdSync",
  "vector/index/insertBatch",
];

// Reporting an embedding reads the message back for its position.
test("with onSaveMessages, each write also passes its handle and schedules one report", async () => {
  const [plain, hooked] = [await costs(false), await costs(true)];
  const more = (n: number | undefined, by: number) =>
    n === undefined ? undefined : n + by;
  const expected = mapCalls(plain, (c) =>
    writes.includes(c.fn)
      ? {
          ...c,
          args: [...c.args, "onSaveMessages"].sort(),
          argBytes: expect.any(Number),
          functionsScheduled: more(c.functionsScheduled, 1),
          scheduledFunctionArgsBytes:
            c.scheduledFunctionArgsBytes === undefined
              ? undefined
              : expect.any(Number),
          ...(c.fn === "vector/index/insertBatch"
            ? {
                bytesRead: expect.any(Number),
                documentsRead: more(c.documentsRead, 1),
                databaseQueries: more(c.databaseQueries, 1),
              }
            : {}),
        }
      : c,
  );
  expect(hooked).toEqual(expected);
});

// Measured on 853c2860, before onSaveMessages.
const BASE: Record<(typeof paths)[number], string[]> = {
  generateText: [
    "messages/listMessagesByThreadId(order,paginationOpts,statuses,threadId) 133B read 0B/0 docs/2 queries, wrote 0B/0 docs, scheduled 0/0B",
    "messages/addMessages(failPendingSteps,messages,threadId) 196B read 486B/3 docs/9 queries, wrote 408B/2 docs, scheduled 0/0B",
    "messages/addMessages(abandonIfPromptMissing,agentName,failPendingSteps,messages,pendingMessageId,promptMessageId,threadId) 833B read 2086B/8 docs/11 queries, wrote 994B/2 docs, scheduled 0/0B",
    "messages/addMessages(abandonIfPromptMissing,agentName,failPendingSteps,messages,promptMessageId,threadId) 559B read 1460B/5 docs/9 queries, wrote 602B/1 docs, scheduled 0/0B",
  ],
  streamText: [
    "messages/listMessagesByThreadId(order,paginationOpts,statuses,threadId) 133B read 0B/0 docs/2 queries, wrote 0B/0 docs, scheduled 0/0B",
    "messages/addMessages(failPendingSteps,messages,threadId) 196B read 486B/3 docs/9 queries, wrote 408B/2 docs, scheduled 0/0B",
    "streams/create(agentName,format,model,order,provider,stepOrder,threadId) 167B read 271B/1 docs/0 queries, wrote 588B/2 docs, scheduled 1/45B",
    "streams/addDelta(end,parts,start,streamId) 92B read 317B/1 docs/1 queries, wrote 156B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 97B read 317B/1 docs/1 queries, wrote 161B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 167B read 317B/1 docs/1 queries, wrote 220B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 149B read 317B/1 docs/1 queries, wrote 207B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 98B read 317B/1 docs/1 queries, wrote 162B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 97B read 317B/1 docs/1 queries, wrote 161B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 110B read 317B/1 docs/1 queries, wrote 171B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 126B read 317B/1 docs/1 queries, wrote 184B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 108B read 317B/1 docs/1 queries, wrote 169B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 99B read 317B/1 docs/1 queries, wrote 162B/1 docs, scheduled 0/0B",
    "streams/addDelta(end,parts,start,streamId) 117B read 317B/1 docs/1 queries, wrote 176B/1 docs, scheduled 0/0B",
    "messages/addMessages(abandonIfPromptMissing,agentName,failPendingSteps,finishStreamId,messages,pendingMessageId,promptMessageId,threadId) 665B read 2826B/11 docs/13 queries, wrote 912B/2 docs, scheduled 1/45B",
  ],
  "generateText, provider fails": [
    "messages/listMessagesByThreadId(order,paginationOpts,statuses,threadId) 133B read 0B/0 docs/2 queries, wrote 0B/0 docs, scheduled 0/0B",
    "messages/addMessages(failPendingSteps,messages,threadId) 196B read 486B/3 docs/9 queries, wrote 408B/2 docs, scheduled 0/0B",
    "messages/finalizeMessage(messageId,result) 101B read 396B/2 docs/4 queries, wrote 218B/1 docs, scheduled 0/0B",
  ],
  "generateText, lazy prompt embedding": [
    "messages/listMessagesByThreadId(order,paginationOpts,statuses,threadId,upToAndIncludingMessageId) 196B read 452B/2 docs/3 queries, wrote 0B/0 docs, scheduled 0/0B",
    "vector/index/insertBatch(vectorDimension,vectors) 697B read 226B/1 docs/0 queries, wrote 1660B/2 docs, scheduled 0/0B",
    "messages/addMessages(failPendingSteps,messages,promptMessageId,threadId) 202B read 870B/4 docs/10 queries, wrote 248B/1 docs, scheduled 0/0B",
    "messages/addMessages(abandonIfPromptMissing,agentName,failPendingSteps,messages,pendingMessageId,promptMessageId,threadId) 833B read 2360B/8 docs/11 queries, wrote 994B/2 docs, scheduled 0/0B",
    "messages/addMessages(abandonIfPromptMissing,agentName,embeddings,failPendingSteps,messages,promptMessageId,threadId) 1141B read 1630B/5 docs/9 queries, wrote 2036B/2 docs, scheduled 0/0B",
  ],
  saveMessage: [
    "messages/addMessages(agentName,failPendingSteps,messages,threadId) 147B read 298B/2 docs/8 queries, wrote 220B/1 docs, scheduled 0/0B",
  ],
  "saveMessages, restart": [
    "messages/addMessages(agentName,failPendingSteps,messages,threadId) 149B read 963B/5 docs/9 queries, wrote 457B/2 docs, scheduled 0/0B",
  ],
  finalizeMessage: [
    "messages/finalizeMessage(messageId,result) 78B read 428B/2 docs/4 queries, wrote 214B/1 docs, scheduled 0/0B",
  ],
  updateMessage: [
    "messages/updateMessage(messageId,patch) 122B read 668B/3 docs/2 queries, wrote 228B/1 docs, scheduled 0/0B",
  ],
  approveToolCall: [
    "messages/respondToToolCallApprovals(agentName,decisions,threadId) 115B read 1460B/5 docs/15 queries, wrote 314B/1 docs, scheduled 0/0B",
  ],
  deleteMessage: [
    "messages/deleteByIds(messageIds) 51B read 440B/2 docs/2 queries, wrote 0B/1 docs, scheduled 0/0B",
  ],
  deleteMessages: [
    "messages/deleteByIds(messageIds) 86B read 872B/4 docs/4 queries, wrote 0B/2 docs, scheduled 0/0B",
  ],
  "deleteMessages, two threads": [
    "messages/deleteByIds(messageIds) 86B read 872B/4 docs/4 queries, wrote 0B/2 docs, scheduled 0/0B",
  ],
  deleteMessageRange: [
    "messages/deleteByOrder(endOrder,startOrder,threadId) 76B read 872B/4 docs/8 queries, wrote 0B/2 docs, scheduled 0/0B",
  ],
  deleteThreadAsync: [
    "threads/deleteAllForThreadIdAsync(threadId) 47B read 592B/4 docs/5 queries, wrote 0B/2 docs, scheduled 0/0B",
  ],
  deleteThreadSync: ["threads/deleteAllForThreadIdSync(threadId) 47B"],
  generateAndSaveEmbeddings: [
    "messages/getMessagesByIds(messageIds) 51B read 220B/1 docs/1 queries, wrote 0B/0 docs, scheduled 0/0B",
    "vector/index/insertBatch(vectorDimension,vectors) 697B read 220B/1 docs/0 queries, wrote 1654B/2 docs, scheduled 0/0B",
  ],
};
