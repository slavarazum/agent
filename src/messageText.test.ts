/// <reference types="vite/client" />

import { omit, pick } from "convex-helpers";
import type { FunctionArgs } from "convex/server";
import { convexTest } from "convex-test";
import { convexToJson, getDocumentSize, type Value } from "convex/values";
import { describe, expect, test } from "vitest";
import { listMessages } from "./client/messages.js";
import { extractText, isTool, withDerivedReasoning } from "./shared.js";
import type { Message, MessageDoc } from "./validators.js";
import { fetchContextMessages } from "./vercel/client/search.js";
import { api } from "./component/_generated/api.js";
import type { ComponentApi } from "./component/_generated/component.js";
import type { Doc, Id } from "./component/_generated/dataModel.js";
import schema from "./component/schema.js";
import { initConvexTest, modules } from "./component/setup.test.js";

const component = api as unknown as ComponentApi;
// Enforces Convex's transaction limits, like 4,096 index ranges read.
const limited = () => convexTest({ schema, modules, transactionLimits: true });

type T = ReturnType<typeof initConvexTest>;
type Ctx = Parameters<Parameters<T["run"]>[0]>[0];
type Sent = FunctionArgs<typeof api.messages.addMessages>["messages"][number];
type Options = Record<string, Record<string, unknown>>;
type PaginationResult = Awaited<ReturnType<typeof listMessages>>;

const METADATA = [
  "providerOptions",
  "usage",
  "providerMetadata",
  "sources",
  "warnings",
  "reasoning",
  "reasoningDetails",
] as const;

let seed = 7;
function words(chars: number) {
  const vocabulary = "the agent thread message context query index tool result";
  let out = "";
  while (out.length < chars) {
    seed = (seed * 1_103_515_245 + 12_345) % 2 ** 31;
    const list = vocabulary.split(" ");
    out += list[seed % list.length] + " ";
  }
  return out.slice(0, chars);
}
const blob = (chars: number) =>
  "Zm9v".repeat(Math.ceil(chars / 4)).slice(0, chars);

const openaiUsage = {
  promptTokens: 2412,
  completionTokens: 812,
  totalTokens: 3224,
  reasoningTokens: 0,
  cachedInputTokens: 1920,
  nonCachedInputTokens: 492,
  textOutputTokens: 812,
  raw: {
    input_tokens: 2412,
    input_tokens_details: { cached_tokens: 1920 },
    output_tokens: 812,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 3224,
  },
};
const anthropicUsage = {
  cache_creation: {
    ephemeral_1h_input_tokens: 0,
    ephemeral_5m_input_tokens: 4,
  },
  cache_creation_input_tokens: 412,
  cache_read_input_tokens: 5120,
  input_tokens: 18,
  output_tokens: 734,
  server_tool_use: { web_fetch_requests: 0, web_search_requests: 0 },
  service_tier: "standard",
};
const openai = {
  model: "gpt-4.1",
  provider: "openai.responses",
  providerMetadata: {
    openai: { responseId: `resp_${blob(48)}`, serviceTier: "default" },
  },
  usage: openaiUsage,
  finishReason: "stop" as const,
};
const anthropic = (finishReason: "stop" | "tool-calls") => ({
  model: "claude-sonnet-4-5",
  provider: "anthropic.messages",
  providerMetadata: {
    anthropic: { container: null, stopSequence: null, usage: anthropicUsage },
  },
  usage: {
    cacheWriteInputTokens: 412,
    cachedInputTokens: 5120,
    completionTokens: 734,
    nonCachedInputTokens: 18,
    promptTokens: 5550,
    totalTokens: 6284,
    raw: anthropicUsage,
  },
  finishReason,
});

const prompt = (): Sent => ({
  message: { role: "user", content: `needle ${words(160)}` },
});
const text = (chars: number, providerOptions?: Options) => ({
  type: "text" as const,
  text: `needle ${words(chars)}`,
  ...(providerOptions && { providerOptions }),
});
const reasoning = (chars: number, providerOptions: Options) => ({
  type: "reasoning" as const,
  text: words(chars),
  providerOptions,
});
const assistant = (content: Message["content"], fields: object): Sent =>
  ({ message: { role: "assistant", content }, ...fields }) as Sent;
let call = 0;
function toolStep(output: unknown): Sent[] {
  const toolCallId = `toolu_${call++}`;
  return [
    assistant(
      [
        reasoning(800, { anthropic: { signature: blob(1200) } }),
        {
          type: "tool-call",
          toolCallId,
          toolName: "search",
          input: { q: "q" },
        },
      ],
      anthropic("tool-calls"),
    ),
    {
      message: {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId,
            toolName: "search",
            output: { type: "json", value: output as never },
          },
        ],
      },
    },
  ];
}

/** One turn of each kind of thread, sized like what providers return. */
const turns: Record<string, () => Sent[]> = {
  "plain chat": () => [
    prompt(),
    assistant([text(1200, { openai: { itemId: `msg_${blob(48)}` } })], openai),
  ],
  "encrypted reasoning": () => [
    prompt(),
    assistant(
      [
        reasoning(900, {
          openai: {
            itemId: `rs_${blob(48)}`,
            reasoningEncryptedContent: blob(6000),
          },
        }),
        text(1200, { openai: { itemId: `msg_${blob(48)}` } }),
      ],
      { ...openai, model: "gpt-5" },
    ),
  ],
  "tools with large results": () => [
    prompt(),
    ...toolStep({ results: words(18_000) }),
    ...toolStep({ text: words(50_000) }),
    assistant(
      [reasoning(400, { anthropic: { signature: blob(1200) } }), text(1500)],
      anthropic("stop"),
    ),
  ],
  "tool calls and short answers": () => {
    const toolCallId = `call_${call++}`;
    return [
      prompt(),
      assistant(
        [
          {
            type: "tool-call",
            toolCallId,
            toolName: "search",
            input: { q: "q" },
          },
        ],
        { ...openai, finishReason: "tool-calls" },
      ),
      {
        message: {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId,
              toolName: "search",
              output: { type: "text", value: words(200) },
            },
          ],
        },
      },
      assistant([text(20)], openai),
    ];
  },
  "grounded answers": () => [
    prompt(),
    assistant(
      [
        reasoning(1200, { google: { thoughtSignature: blob(1800) } }),
        text(1400, { google: { thoughtSignature: blob(600) } }),
      ],
      {
        model: "gemini-2.5-flash",
        provider: "google.generative-ai",
        finishReason: "stop",
        usage: {
          promptTokens: 1540,
          completionTokens: 1580,
          totalTokens: 3120,
        },
        providerMetadata: {
          google: {
            groundingMetadata: {
              groundingChunks: Array.from({ length: 8 }, (_, i) => ({
                web: { title: `site${i}.example.com`, uri: blob(283) },
              })),
              searchEntryPoint: { renderedContent: words(6000) },
              webSearchQueries: ["convex agent cost"],
            },
          },
        },
        sources: Array.from({ length: 8 }, (_, i) => ({
          type: "source",
          sourceType: "url",
          id: `s${i}`,
          title: `site${i}.example.com`,
          url: blob(283),
        })),
      },
    ),
  ],
  "images and files": () => [
    {
      message: {
        role: "user",
        content: [
          { type: "text", text: "needle what is in these?" },
          {
            type: "image",
            image: new ArrayBuffer(40_000),
            mediaType: "image/png",
          },
          { type: "file", data: blob(20_000), mediaType: "application/pdf" },
        ],
      },
    },
    assistant([text(800)], openai),
  ],
};

async function createThread(t: T, userId?: string) {
  const thread = await t.mutation(api.threads.createThread, { userId });
  return thread._id as Id<"threads">;
}

async function save(t: T, threadId: Id<"threads">, turns: Sent[][]) {
  const saved: MessageDoc[] = [];
  for (const messages of turns) {
    const result = await t.mutation(api.messages.addMessages, {
      threadId,
      messages,
    });
    saved.push(...result.messages);
  }
  return saved;
}

/** Inserts the turns as rows the way releases before text was stored once saved them. */
async function saveWhole(t: T, threadId: Id<"threads">, turns: Sent[][]) {
  await t.run(async (ctx) => {
    let order = -1;
    let stepOrder = 0;
    for (const message of turns.flat()) {
      if (message.message.role === "user") {
        order++;
        stepOrder = 0;
      } else {
        stepOrder++;
      }
      await ctx.db.insert("messages", {
        ...message,
        threadId,
        order,
        stepOrder,
        status: message.status ?? "success",
        tool: isTool(message.message),
        text: extractText(message.message),
      } as Omit<Doc<"messages">, "_id" | "_creationTime">);
    }
  });
}

async function cost<R>(t: T, fn: (ctx: Ctx) => Promise<R>) {
  return t.run(async (ctx) => {
    const before = await ctx.meta.getTransactionMetrics();
    const result = await fn(ctx);
    const after = await ctx.meta.getTransactionMetrics();
    return {
      result,
      bytesRead: after.bytesRead.used - before.bytesRead.used,
      documentsRead: after.documentsRead.used - before.documentsRead.used,
      bytesWritten: after.bytesWritten.used - before.bytesWritten.used,
    };
  });
}

const page = (numItems: number) => ({ cursor: null, numItems });

/** Each read and save path an Agent app runs on every turn. */
const paths = {
  context: (threadId: Id<"threads">) => (ctx: Ctx) =>
    fetchContextMessages(ctx, component, {
      threadId,
      userId: undefined,
      contextOptions: { recentMessages: 100 },
    }),
  page: (threadId: Id<"threads">) => (ctx: Ctx) =>
    listMessages(ctx, component, { threadId, paginationOpts: page(20) }),
  textPage: (threadId: Id<"threads">, once: boolean) => (ctx: Ctx) =>
    listMessages(ctx, component, {
      threadId,
      paginationOpts: page(20),
      ...(once ? { textOnly: true } : { excludeToolMessages: true }),
    }),
  pageWithoutTools: (threadId: Id<"threads">) => (ctx: Ctx) =>
    listMessages(ctx, component, {
      threadId,
      paginationOpts: page(20),
      excludeToolMessages: true,
    }),
  search: (threadId: Id<"threads">) => (ctx: Ctx) =>
    ctx.runQuery(api.messages.textSearch, {
      threadId,
      text: "needle",
      limit: 10,
    }),
  save: (threadId: Id<"threads">) => (ctx: Ctx) =>
    ctx.runMutation(api.messages.addMessages, {
      threadId,
      messages: [{ message: { role: "assistant", content: "next step" } }],
    }),
  stepSave: (threadId: Id<"threads">) => async (ctx: Ctx) => {
    const [promptMessage, pending] = (
      await ctx.runMutation(api.messages.addMessages, {
        threadId,
        messages: [
          prompt(),
          { message: { role: "assistant", content: [] }, status: "pending" },
        ],
      })
    ).messages;
    return ctx.runMutation(api.messages.addMessages, {
      threadId,
      promptMessageId: promptMessage._id as Id<"messages">,
      pendingMessageId: pending._id as Id<"messages">,
      messages: toolStep({ results: words(2_000) }),
    });
  },
};

const exact = (doc: object) =>
  convexToJson(omit(doc as MessageDoc, ["_id", "_creationTime"]) as Value);

describe("each hot path reads no more than rows saved whole", () => {
  for (const [name, turn] of Object.entries(turns)) {
    test(name, async () => {
      const t = initConvexTest();
      const sent = Array.from({ length: 6 }, turn);
      const once = await createThread(t);
      const whole = await createThread(t);
      await save(t, once, sent);
      await saveWhole(t, whole, sent);
      for (const [path, fn] of Object.entries(paths)) {
        const run = fn as (
          threadId: Id<"threads">,
          once: boolean,
        ) => (ctx: Ctx) => Promise<unknown>;
        const [a, b] = [
          await cost(t, run(once, true)),
          await cost(t, run(whole, false)),
        ];
        const where = `${name}: ${path}`;
        expect(a.bytesRead, where).toBeLessThanOrEqual(b.bytesRead);
        expect(a.documentsRead, where).toBeLessThanOrEqual(b.documentsRead);
      }
    });
  }
});

test("each message's text is stored once", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  await save(t, threadId, [turns["plain chat"]()]);
  const rows = await t.run((ctx) => ctx.db.query("messages").collect());
  for (const row of rows) {
    expect(row.text?.length).toBeGreaterThan(100);
    expect(JSON.stringify(row.message)).not.toContain(row.text!.slice(0, 40));
  }
});

test("context and search return each message as saved", async () => {
  const t = initConvexTest();
  const threadId = await createThread(t);
  const sent = Object.values(turns).map((turn) => turn());
  const saved = await save(t, threadId, sent);
  const context = await t.run(paths.context(threadId));
  expect(context.map(exact)).toEqual(saved.map(exact));
  const found = await t.run(paths.search(threadId));
  expect(found.length).toBeGreaterThan(0);
  for (const doc of found) {
    expect(exact(doc)).toEqual(exact(saved.find((m) => m._id === doc._id)!));
  }
});

describe("text-only lists", () => {
  test("return each message's role and text, and no tool steps or messages without text", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const toolCallId = "call_text";
    const saved = await save(t, threadId, [
      [
        prompt(),
        {
          message: {
            role: "user",
            content: [
              {
                type: "image",
                image: new ArrayBuffer(3),
                mediaType: "image/png",
              },
            ],
          },
        },
        assistant(
          [
            text(40),
            { type: "tool-call", toolCallId, toolName: "search", input: {} },
          ],
          { ...openai, finishReason: "tool-calls" },
        ),
        {
          message: {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId,
                toolName: "search",
                output: { type: "text", value: "v" },
              },
            ],
          },
        },
        assistant(
          [reasoning(100, { anthropic: { signature: blob(100) } })],
          anthropic("stop"),
        ),
        assistant(
          [reasoning(100, { anthropic: { signature: blob(100) } }), text(60)],
          anthropic("stop"),
        ),
        {
          message: { role: "assistant", content: [] },
          status: "pending",
        },
      ],
    ]);
    const [user, , , , , answer, pending] = saved;
    const { page: listed } = await t.run(paths.textPage(threadId, true));
    expect(listed.map(exact)).toEqual(
      [user, answer, pending].reverse().map((m) =>
        exact({
          ...omit(m, [...METADATA, "message"]),
          message: { role: m.message!.role, content: m.text ?? "" },
        }),
      ),
    );
  });

  test("follow their cursors to every message with text", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const sent = Object.values(turns).flatMap((turn) => turn());
    for (let i = 0; i < 4; i++) {
      await saveWhole(t, threadId, [sent]);
      await save(t, threadId, [sent]);
    }
    const all = await readAll(t, threadId);
    for (const textOnly of [false, true]) {
      for (const maximumBytesRead of [undefined, 30_000, 200_000]) {
        const seen: string[] = [];
        let cursor: string | null = null;
        let reads = 0;
        do {
          const result: PaginationResult = await t.query(
            api.messages.listMessagesByThreadId,
            {
              threadId,
              order: "asc",
              textOnly,
              paginationOpts: { cursor, numItems: 7, maximumBytesRead },
            },
          );
          if (result.splitCursor && result.pageStatus === "SplitRequired") {
            const half: PaginationResult = await t.query(
              api.messages.listMessagesByThreadId,
              {
                threadId,
                order: "asc",
                textOnly,
                paginationOpts: {
                  cursor,
                  numItems: 7,
                  endCursor: result.splitCursor,
                },
              },
            );
            seen.push(...half.page.map((m) => m._id));
            cursor = result.splitCursor;
          } else {
            seen.push(...result.page.map((m) => m._id));
            cursor = result.isDone ? null : result.continueCursor;
          }
          expect(++reads).toBeLessThan(1_000);
        } while (cursor);
        const expected = all
          .filter((m) => !textOnly || (!m.tool && m.text))
          .map((m) => m._id);
        expect(seen, `${textOnly} ${maximumBytesRead}`).toEqual(expected);
      }
    }
  }, 120_000);
});

/**
 * Messages whose text is easy to split wrongly: several text parts, empty
 * ones, spaces at either end or doubled, text between other parts and parts
 * with provider options. Some carry result metadata.
 */
function tricky(count: number): Sent[] {
  let state = count;
  const next = () => {
    state = (state * 1_103_515_245 + 12_345) % 2 ** 31;
    return state / 2 ** 31;
  };
  const pick = <V>(values: readonly V[]) =>
    values[Math.floor(next() * values.length)];
  const pieces = ["", " ", "  ", "a", "one two", " lead", "trail ", "a  b"];
  const someText = () =>
    Array.from({ length: Math.floor(next() * 4) }, () =>
      pick([...pieces, "line\nbreak", "é ü 🧪", "x".repeat(300)]),
    ).join(pick(["", " "]));
  const options = () =>
    next() < 0.4 ? { providerOptions: { p: { id: someText() } } } : {};
  const part = {
    text: () => ({ type: "text", text: someText(), ...options() }),
    reasoning: () => ({ type: "reasoning", text: someText(), ...options() }),
    image: () => ({ type: "image", image: new ArrayBuffer(3) }),
    file: () => ({ type: "file", data: someText(), mediaType: "text/plain" }),
    call: () => ({
      type: "tool-call",
      toolCallId: "c",
      toolName: "t",
      input: { q: someText() },
    }),
  };
  const parts = (...kinds: (keyof typeof part)[]) =>
    Array.from({ length: Math.floor(next() * 6) }, () => part[pick(kinds)]());
  const messages = [
    () => ({ role: "user", content: someText() }),
    () => ({ role: "user", content: parts("text", "text", "image", "file") }),
    () => ({ role: "assistant", content: someText() }),
    () => ({
      role: "assistant",
      content: parts("text", "text", "reasoning", "call", "file"),
    }),
    () => ({ role: "system", content: someText() }),
  ];
  return Array.from({ length: count }, () => ({
    message: {
      ...pick(messages)(),
      ...(next() < 0.2 && { providerOptions: { p: { m: 1 } } }),
    } as Message,
    ...(next() < 0.4 && omit(openai, ["finishReason"])),
  }));
}

async function readAll(t: T, threadId: Id<"threads">) {
  const all = await t.query(api.messages.listMessagesByThreadId, {
    threadId,
    order: "asc",
    paginationOpts: page(10_000),
  });
  const byId = await t.query(api.messages.getMessagesByIds, {
    messageIds: all.page.map((m) => m._id as Id<"messages">),
  });
  expect(byId).toEqual(all.page);
  return all.page;
}

describe("full reads return each message as saved", () => {
  const sent = [
    ...tricky(80),
    ...Object.values(turns).flatMap((turn) => turn()),
  ];
  for (const hideFromUserIdSearch of [false, true]) {
    test(
      hideFromUserIdSearch ? "hidden from search" : "searchable",
      async () => {
        const t = initConvexTest();
        const threadId = await createThread(t, "u");
        const { messages: saved } = await t.mutation(api.messages.addMessages, {
          threadId,
          messages: sent,
          hideFromUserIdSearch,
        });
        saved.forEach((m, i) => {
          expect(exact(m.message!)).toEqual(exact(sent[i].message));
          expect(m.text).toBe(
            hideFromUserIdSearch ? undefined : extractText(sent[i].message),
          );
          expect(m).toMatchObject(omit(sent[i], ["message"]));
        });
        expect((await readAll(t, threadId)).map(exact)).toEqual(
          saved.map(exact),
        );
      },
    );
  }

  test("rows saved whole, before text was stored once", async () => {
    const t = initConvexTest();
    const whole = await createThread(t);
    const split = await createThread(t);
    await saveWhole(t, whole, [sent]);
    await save(t, split, [sent]);
    const strip = (docs: MessageDoc[]) =>
      docs.map((m) => exact(omit(m, ["threadId", "order", "stepOrder"])));
    expect(strip(await readAll(t, whole))).toEqual(
      strip(await readAll(t, split)),
    );
  });
});

describe("writes keep each message whole", () => {
  async function setup() {
    const t = initConvexTest();
    const threadId = await createThread(t, "u");
    const sent = [...tricky(30), ...turns["tools with large results"]()];
    const saved = await save(t, threadId, [sent]);
    return { t, threadId, sent, saved };
  }

  test("updates", async () => {
    const { t, threadId, sent, saved } = await setup();
    const expected = [];
    for (const [i, doc] of saved.entries()) {
      const { message } = sent[(i + 7) % sent.length];
      const patch = [
        { status: "failed" as const, error: "nope" },
        { providerOptions: { p: { updated: true } } },
        { message },
        { message, model: "m2", finishReason: "length" as const },
      ][i % 4];
      const updated = await t.mutation(api.messages.updateMessage, {
        messageId: doc._id as Id<"messages">,
        patch,
      });
      const next = withDerivedReasoning({
        ...omit(doc, ["reasoning", "reasoningDetails"]),
        ...patch,
        ...(patch.message && {
          tool: isTool(patch.message),
          text: extractText(patch.message),
        }),
      });
      expect(exact(updated)).toEqual(exact(next));
      expected.push(next);
    }
    expect((await readAll(t, threadId)).map(exact)).toEqual(
      expected.map(exact),
    );
  });

  test("clones, and deletes of the source", async () => {
    const { t, threadId, saved } = await setup();
    const targetThreadId = await createThread(t);
    await t.action(api.messages.cloneThread, {
      sourceThreadId: threadId,
      targetThreadId,
      batchSize: 7,
    });
    await t.action(api.threads.deleteAllForThreadIdSync, { threadId });
    const strip = (docs: MessageDoc[]) =>
      docs.map((m) => exact(omit(m, ["threadId"])));
    expect(strip(await readAll(t, targetThreadId))).toEqual(strip(saved));
  });

  test("deletes in the middle of a paginated read", async () => {
    const { t, threadId, saved } = await setup();
    const first = await t.query(api.messages.listMessagesByThreadId, {
      threadId,
      order: "asc",
      paginationOpts: page(10),
    });
    await t.mutation(api.messages.deleteByIds, {
      messageIds: saved.slice(5, 20).map((m) => m._id as Id<"messages">),
    });
    const rest = await t.query(api.messages.listMessagesByThreadId, {
      threadId,
      order: "asc",
      paginationOpts: { cursor: first.continueCursor, numItems: 1_000 },
    });
    expect([...first.page, ...rest.page].map(exact)).toEqual(
      [...saved.slice(0, 10), ...saved.slice(20)].map(exact),
    );
  });

  test("a pending message replaced by its step, and concurrent saves and updates", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const [promptMessage, pending] = await save(t, threadId, [
      [
        prompt(),
        { message: { role: "assistant", content: [] }, status: "pending" },
      ],
    ]);
    const step = toolStep({ results: words(5_000) });
    const [, saved] = await Promise.all([
      t.mutation(api.messages.updateMessage, {
        messageId: promptMessage._id as Id<"messages">,
        patch: { message: { role: "user", content: "needle edited" } },
      }),
      t.mutation(api.messages.addMessages, {
        threadId,
        promptMessageId: promptMessage._id as Id<"messages">,
        pendingMessageId: pending._id as Id<"messages">,
        messages: step,
      }),
      t.mutation(api.messages.addMessages, {
        threadId,
        messages: turns["grounded answers"](),
      }),
    ]);
    expect(saved.messages[0]._id).toBe(pending._id);
    expect(saved.messages.map((m) => exact(m.message!))).toEqual(
      step.map((m) => exact(m.message)),
    );
    const all = await readAll(t, threadId);
    expect(all[0].message).toEqual({ role: "user", content: "needle edited" });
    expect(all).toHaveLength(5);
  });

  test("a step replacing a pending message with metadata, and concurrent steps at one order", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const [promptMessage, pending] = await save(t, threadId, [
      [
        prompt(),
        {
          message: { role: "assistant", content: [] },
          status: "pending",
          ...openai,
        },
      ],
    ]);
    const promptMessageId = promptMessage._id as Id<"messages">;
    await Promise.all([
      t.mutation(api.messages.addMessages, {
        threadId,
        promptMessageId,
        pendingMessageId: pending._id as Id<"messages">,
        messages: [{ message: { role: "assistant", content: "plain" } }],
      }),
      ...Array.from({ length: 4 }, () =>
        t.mutation(api.messages.addMessages, {
          threadId,
          promptMessageId,
          messages: toolStep({ results: words(3_000) }),
        }),
      ),
    ]);
    const all = await readAll(t, threadId);
    expect(all.map((m) => m.stepOrder)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(all[1]).not.toHaveProperty("usage");
    expect(all.filter((m) => m.usage)).toHaveLength(4);
  });

  test("an update or clone of a row saved whole is read as before", async () => {
    const t = initConvexTest();
    const whole = await createThread(t);
    const sent = [...tricky(20), ...turns["grounded answers"]()];
    await saveWhole(t, whole, [sent]);
    const before = await readAll(t, whole);
    const updated = await Promise.all(
      before.map((doc) =>
        t.mutation(api.messages.updateMessage, {
          messageId: doc._id as Id<"messages">,
          patch: { model: "m2" },
        }),
      ),
    );
    expect(updated.map(exact)).toEqual(
      before.map((m) => exact({ ...m, model: "m2" })),
    );
    const targetThreadId = await createThread(t);
    await t.action(api.messages.cloneThread, {
      sourceThreadId: whole,
      targetThreadId,
    });
    const strip = (docs: MessageDoc[]) =>
      docs.map((m) => exact(omit(m, ["threadId"])));
    expect(strip(await readAll(t, targetThreadId))).toEqual(strip(updated));
    expect(strip(await readAll(t, whole))).toEqual(strip(updated));
  });

  test("finalizing a pending message with text keeps its text", async () => {
    const t = initConvexTest();
    const partial = words(200);
    const threadId = await createThread(t);
    const [, pending] = await save(t, threadId, [
      [
        prompt(),
        {
          message: { role: "assistant", content: partial },
          status: "pending",
        },
      ],
    ]);
    await t.run(async (ctx) => {
      const streamId = await ctx.db.insert("streamingMessages", {
        threadId,
        order: pending.order,
        stepOrder: pending.stepOrder,
        format: "UIMessageChunk",
        state: { kind: "aborted", reason: "interrupted" },
      });
      await ctx.db.insert("streamDeltas", {
        streamId,
        start: 0,
        end: 2,
        parts: [
          { type: "text-start", id: "t" },
          { type: "text-delta", id: "t", delta: "streamed" },
        ],
      });
    });
    await t.mutation(api.messages.finalizeMessage, {
      messageId: pending._id as Id<"messages">,
      result: { status: "success" },
    });
    const [final] = await t.query(api.messages.getMessagesByIds, {
      messageIds: [pending._id as Id<"messages">],
    });
    expect(final).toMatchObject({
      status: "success",
      message: { role: "assistant", content: partial },
    });
  });

  test("a step keeps the prompt it answers", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const [promptMessage, pending] = await save(t, threadId, [
      [
        prompt(),
        { message: { role: "assistant", content: [] }, status: "pending" },
      ],
    ]);
    await t.mutation(api.messages.addMessages, {
      threadId,
      promptMessageId: promptMessage._id as Id<"messages">,
      pendingMessageId: pending._id as Id<"messages">,
      messages: toolStep({ results: words(500) }),
    });
    const rows = await t.run((ctx) => ctx.db.query("messages").collect());
    expect(rows.slice(1).map((m) => m.parentMessageId)).toEqual([
      promptMessage._id,
      promptMessage._id,
    ]);
  });
});

describe("near the limits", () => {
  test("each row is no larger than the message saved whole, and a 600 KB answer fits", async () => {
    const t = initConvexTest();
    const threadId = await createThread(t);
    const sent: Sent[] = [
      prompt(),
      assistant([text(600_000)], openai),
      ...toolStep({ text: words(900_000) }),
      assistant(
        [reasoning(700_000, { anthropic: { signature: blob(100) } }), text(10)],
        anthropic("stop"),
      ),
      assistant(
        Array.from({ length: 8_000 }, (_, i) => ({
          type: "text" as const,
          text: i ? "" : "a",
        })),
        openai,
      ),
    ];
    await save(t, threadId, [sent]);
    const rows = await t.run((ctx) => ctx.db.query("messages").collect());
    for (const [i, row] of rows.entries()) {
      const whole = getDocumentSize({
        ...sent[i],
        ...pick(row, [
          "_id",
          "_creationTime",
          "threadId",
          "order",
          "stepOrder",
          "status",
          "tool",
          "parentMessageId",
        ]),
        text: extractText(sent[i].message),
      } as Record<string, Value>);
      expect(getDocumentSize(row)).toBeLessThanOrEqual(whole);
    }
    // A 600 KB answer saved whole holds its text twice, past 1 MiB.
    expect(getDocumentSize(rows[1])).toBeLessThan(700_000);
    expect((await readAll(t, threadId)).map((m) => exact(m.message!))).toEqual(
      sent.map((m) => exact(m.message)),
    );
  });

  test("getMessagesByIds and a page read 4,000 messages, as rows saved whole do", async () => {
    const t = limited();
    const threadId = await createThread(t);
    const answer = assistant([text(400)], openai);
    for (let i = 0; i < 4_000; i += 500) {
      await t.mutation(api.messages.addMessages, {
        threadId,
        messages: Array.from({ length: 500 }, () => answer),
      });
    }
    const { page: listed } = await t.query(
      api.messages.listMessagesByThreadId,
      { threadId, order: "asc", paginationOpts: page(4_000) },
    );
    expect(listed).toHaveLength(4_000);
    const read = await t.query(api.messages.getMessagesByIds, {
      messageIds: listed.map((m) => m._id as Id<"messages">),
    });
    expect(read.map((m) => m?._id)).toEqual(listed.map((m) => m._id));
    expect(read.every((m) => m?.usage?.totalTokens === 3224)).toBe(true);
    expect(read.every((m) => m?.message?.content[0])).toBe(true);
  }, 120_000);
});
