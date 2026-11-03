# Errand Agent

A local, chat-based agent that books, reschedules and cancels home-service visits (plumbers,
electricians, AC servicing, cleaning and so on) against a mock provider world. It asks only for
missing details, compares providers, gets an explicit yes before any consequential action, and
recovers safely when tools fail. Nothing leaves the machine: no real providers, payments or external
services.

- Requirements: [docs/SPEC.md](docs/SPEC.md)
- Design note, with the architecture, workflow and recovery diagrams: [docs/design_note.md](docs/design_note.md)
- Bugs hit while building, each with its regression test: [docs/bugs.md](docs/bugs.md)
- The generated eval table: [reports/eval_table.md](reports/eval_table.md), and the concurrency
  stress run: [reports/eval_stress.md](reports/eval_stress.md)
- A raw end-to-end trace of a booking whose `book()` timed out after succeeding:
  [traces/timeout_after_success.jsonl](traces/timeout_after_success.jsonl)

The model proposes and code decides. Qwen (qwen2.5:7b on Ollama) returns only schema-bound proposals.
Vega answers typed questions inside graph nodes. Filters, holds, approvals, side effects and every
factual reply are application code.

## Setup

You need Python 3.11+ (developed on 3.12), `make`, and [Ollama](https://ollama.com):

    ollama pull qwen2.5:7b
    make setup        # .venv with the pinned requirements.txt (downloads torch)
    make run          # http://127.0.0.1:8000

The first `make run` downloads Vega's checkpoint and its Qwen3.5-0.8B reader from Hugging Face. Every
start then loads Vega, fits its four question heads (about 35 s on an M1 Pro) and wakes Ollama before
serving. On the first start three users are seeded, each with their own memories, earlier
conversation and bookings: `u_demo` (60+ memories, a 120-message history, four bookings), `u_asha`
and `u_ravi`. The demo user and Asha both live in Koramangala and use plumbers, so they compete for
the same provider slots; both also have an upcoming cleaning with the same provider.

Other targets: `make demo` (below), `make reset` (deletes both databases and the per-run traces),
`make test`, `make eval`, `make eval-stress`, `make record`, `make lint`, `make format`.

## Demo

`make demo` resets local state and starts the app with the `timeout_after_success` scenario: the
first `book()` call times out even though the booking went through. Open http://127.0.0.1:8000.
Pick a user at the top. Chat is on the left and the trace of that user's current run is on the
right; click any "trace" link to see an earlier run. Each user keeps their own chat, trace pane and
session, and a "•" next to a name marks a reply or follow-up that arrived while you were elsewhere.

1. **Book.** "Find a plumber for my kitchen tap this week, mornings only, under ₹1,500." The agent
   uses the saved area and earliest time, rates reviews with Vega (an injection review is flagged
   and ignored), and shows three options. Click one, check the code-rendered summary, then click
   Confirm. In the trace: approval granted, `hold_slot` under it, approval consumed, the ledger
   row `in_flight`, `book` timing out, `get_booking_status`, ledger `confirmed`, and only then
   "Booked". Nothing is held before that Confirm.
2. **Correct a detail.** Before confirming, say "make it Saturday instead": the other details stay,
   the old proposal is voided, and if the same provider has a slot that fits, a new summary comes
   straight back for a fresh Confirm (otherwise the other options, with a note saying why).
3. **Reschedule or cancel.** "Move my AC servicing to Saturday" or "Cancel my cleaning appointment."
   These go through the same summary, approval, ledger and verify path. The first cancel call returns
   a 503 and is retried with the same key.
4. **Memory.** "What do you remember about me?", "Actually my earliest time is 9 am", "Forget my
   preferred area". The same operations are available through `GET`, `PATCH` and `DELETE /memory`.
5. **Proactive follow-up.** Click **Simulate provider cancel**. Within a few seconds the agent posts
   to the outbox unprompted, with replacement options on a new session (nothing is held). Picking
   one needs a fresh Confirm.
6. **Two users, one slot.** As the demo user and then as Asha, search for a plumber this week and
   pick the same option. Confirm as one, then the other: the first gets the slot; the second is told
   it was taken and shown fresh options, and needs a new Confirm.

## Tests

    make test                    # unit and flow tests, then the eval suite (below)
    .venv/bin/pytest -m live     # the real Ollama and Vega

The offline tests (212) drive the real graph, gateway, store and mock world with rule-based
stand-ins for Qwen and Vega (`errand/llm/rules.py`, `errand/decide/rules.py`). A fixture refuses
every network connection, loopback included, and warnings fail the suite. The flows covered include
corrections, repeated confirmation, timeout after success, unverifiable outcomes, slot taken, a
price that changes between approval and hold, hold expiry, a quote above budget, prompt injection,
cancel with a 503, reschedule with a conflict, memory commands, forgotten and unconfirmed facts
never reaching the model, budget exhaustion and continue, a slow provider that outlasts the turn,
a model outage, a restart mid-booking, and proactive cancellation. The concurrency tests race real
threads: two users holding one slot at once (through one connection and through two, as two
processes would), a lapsed hold that must not free its successor's slot, the store's user-scope
guard, Vega's lock, the fake clock's per-thread waits, and a slow turn that must not stall another
user's request.

## Evals

    make eval                    # python -m errand.eval: every scenario, offline
    python -m errand.eval -k timeout
    make record SCENARIOS="happy_path_plumber slot_taken"   # live Ollama + Vega, saved for replay

Each scenario in [evals/scenarios/](evals/scenarios) is a simulated user: scripted turns, the
model's replies (scripted per purpose, or replayed from a live recording), Vega's answers (scripted
per question, with shared review rules in [evals/defaults.yaml](evals/defaults.yaml)), a mock-world
file from [scenarios/](scenarios), a seed and fake-clock steps. A run needs no network, Ollama or
Vega weights; the CLI blocks the network to prove it. Scenarios are checked on outcomes read from
the ledger, approvals, memory rows, the mock world, the trace and the context manifests, never on
reply wording. Every scenario is also held to seven global invariants
([errand/eval/invariants.py](errand/eval/invariants.py)):

1. No hold, book, cancel or reschedule call without a granted approval whose hash matches the
   call's arguments.
2. At most one confirmed booking per approval and per user, provider and slot, and the ledger
   agrees with the provider.
3. No "booked", "cancelled" or "rescheduled" told to the user unless its ledger row is confirmed;
   each outcome is told once.
4. Every run ends with a user-facing message, within its model, decision, tool and time budgets.
5. No forgotten or unconfirmed memory in any context manifest; every manifest, and every request
   actually sent, is under the token cap.
6. Slots never change as a result of tool output: only the nodes that read the user's message or
   memory change them, and every value names its source (the user's own booking or explicit,
   active fact, for a proactive run's starting details too).
7. A run sees and changes only its own user's data: its trace events, the messages and facts in its
   contexts, the approvals and ledger rows it touched, and the bookings any tool showed it are all
   that user's.

Each invariant is shown failing against a deliberately broken agent in
[tests/test_eval.py](tests/test_eval.py). A run writes [reports/eval_table.md](reports/eval_table.md)
(scenario | expected | actual | pass/fail | notes), the same results as
[reports/eval_results.json](reports/eval_results.json), and each scenario's trace to
`traces/eval/<scenario>.jsonl`. `make test` runs the suite after pytest, so a failing scenario
fails the build.

`make record` runs scenarios against the live models and saves every model reply and Vega
decision to `evals/recordings/<scenario>.json`. A scenario with `recording: <name>` replays it
offline: model replies by purpose, decisions by question and exact text, and any call the
recording doesn't cover fails the run.

There are 55 scenarios: 12 core ones, 4 replays of live Qwen + Vega recordings, 31 edge cases and
8 concurrency scenarios.
- **Understanding:** relative dates on a Saturday that is still Friday in UTC, an inferred area
  confirmed before use, which of several bookings, none to move, "same as last time" when that
  booking was cancelled or the provider isn't available, and two errands in one message.
- **Approval:** a hold quoting above the listing or the budget, a stale Confirm after a change, a
  yes after the hold lapsed (same price and new price), repeated and double-submitted confirmations,
  and vague yeses.
- **Recovery:** timeout with the status service down, restarts before and after `book()` was sent,
  and a reschedule conflict.
- **Safety, memory and limits:** injections in reviews, names, attributes and descriptions;
  adversarial users; forgotten facts in old history; explicit beating inferred; a tight context cap
  over a long history; each limit (model calls, tool calls, wall clock) with a booking in flight;
  broken tool replies; ignored provider events; Ollama down; and Vega failing.
- **Concurrency:** two users approving the same slot at once (exactly one booking; the other gets
  `slot_taken`, fresh options and a new approval), one user's double tab, one user's two errands in
  two sessions, isolation of memory, bookings, approvals, outbox, traces and API answers, a forget
  that touches only its own user, a provider cancelling two users' bookings, a crash with two
  bookings in flight, and one user exhausting their budgets beside another.

The concurrency scenarios fire their turns at the same time, with `asyncio.gather` through the
ASGI app (one thread per request, as in the server). A turnstile
([errand/mocks/turnstile.py](errand/mocks/turnstile.py)) pins only the point each scenario is
about, such as which user's `hold_slot` claims the slot first, so the outcome is the same on every
run. `make eval-stress` plays them once per world seed 1–20 (`ROUNDS=` to change it), which changes
the catalogue, prices, open slots and random outcomes, fails on any failing run, and writes
[reports/eval_stress.md](reports/eval_stress.md).

The mock world's fault controls the scenarios use (`faults`, `status_unavailable_calls`,
`rehold_delta_inr`, `provider_overrides`, `crash_before_send`) are in
[errand/mocks/scenario.py](errand/mocks/scenario.py).

## Users and concurrency

Identity is an input, never a default. `POST /chat` takes `user_id` and `session_id`; the
conversation's LangGraph thread is `f"{user_id}:{session_id}"`, and the agent refuses a thread that
isn't the user's. `GET /trace/{run_id}`, `/memory` and `/outbox` take `user_id` too. A request
without one is a 400, and another user's run is a 404, the same as no run.

- **Every table is the user's.** Tasks, approvals, ledger, holds, memory facts, messages, outbox and
  runs (which user each trace belongs to) carry `user_id`, and every repository query filters by
  it. The database layer enforces this: a statement on one of those tables that doesn't name
  `user_id` raises `UnscopedQuery` before it runs. The two scans that span users by design (the
  reconciler's sweep of open ledger rows, the eval oracle) declare themselves with
  `cross_user(reason)`. LangGraph's checkpoints are keyed by thread, which starts with the user id.
- **Slots are claimed atomically.** In the mock world, `hold_slot` and `reschedule` claim a slot
  with one conditional `UPDATE slot_claims ... WHERE status = 'free'` (or a lapsed hold), and
  `book` turns the caller's own hold into a booking the same way (`WHERE status = 'held' AND
  hold_id = ?`). Each runs in a single transaction and checks the row count: two users can never
  both get a slot, and the loser gets `slot_taken`. A unique index allows one confirmed booking per
  provider and slot.
- **The agent DB is safe for concurrent writers.** SQLite runs in WAL mode with a 5 s busy timeout,
  and every write is a `BEGIN IMMEDIATE` transaction. Spending an approval and opening its ledger
  row happen in one transaction. A unique `idempotency_key` and a unique index on confirmed rows per
  approval enforce "one key, one booking" and "one confirmed booking per approval" in the schema,
  not only in code.
- **Nothing blocks the event loop.** The handlers are async and run the agent, store and world
  (Ollama, Vega, SQLite) in the thread pool. A per-thread lock serializes turns on the same
  conversation (a double tab), and Vega's lock serializes decisions.
- **No per-user state is shared.** Graph state lives in the checkpointer per thread; budgets,
  gateways and dependencies are built per run. Module-level tables are constants, and the only
  process-wide objects (the settings, Vega's loaded weights and fitted heads, the per-thread lock
  table) hold no user's data. Vega decides on the text it's given, under its lock, and keeps
  nothing between calls.

## Configuration

Settings come from `ERRAND_*` environment variables or `.env`; see [errand/config.py](errand/config.py).

| Variable | Default | Meaning |
|---|---|---|
| `ERRAND_LLM_MODEL` | `qwen2.5:7b` | Ollama model |
| `ERRAND_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `ERRAND_VEGA_MODEL` | unset (vegaml default) | Vega checkpoint |
| `ERRAND_VEGA_CONFIDENCE_THRESHOLD` | `0.85` | below it: clarify or re-confirm |
| `ERRAND_MAX_MODEL_CALLS` | `6` | Qwen calls per run |
| `ERRAND_MAX_DECISIONS` | `40` | Vega decisions per run |
| `ERRAND_MAX_TOOL_CALLS` | `12` | tool calls per run, 5 of them reserved for execute and verify |
| `ERRAND_TURN_TIMEOUT_S` | `30` | wall clock per run, 8 s of it reserved |
| `ERRAND_BACKGROUND_TIMEOUT_S` | `60` | wall clock for proactive and reconcile runs |
| `ERRAND_CONTEXT_TOKEN_CAP` | `6000` | tokens of context per model call |
| `ERRAND_SCENARIO`, `ERRAND_SEED` | `default`, `42` | mock world ([scenarios/](scenarios)) |
| `ERRAND_WARMUP_MODELS`, `ERRAND_SEED_DEMO_DATA` | `true`, `true` | startup behaviour |

## Layout

    errand/
      api.py           FastAPI BFF: /chat, /trace/{run_id}, /memory, /outbox, /demo/users, /simulate/...
      agent.py         runtime: one run per message, resume at interrupts, per-thread lock
      graph/           state, nodes (the flowchart), edges, slot grounding, dates, rendering
      gateway/         approval-bound side effects, ledger, verification, untrusted text
      decide/          Vega questions and few-shot examples, VegaDecider, scripted and rule deciders
      llm/             Qwen schemas and prompts, OllamaModel, scripted and rule models
      memory/          memory service (explicit, inferred, forget), context builder, demo seed
      mocks/           the 8 tools, slot claims, catalogue, scenarios, fake clock, turnstile, events
      store/           agent DB: migrations, records, user-scoped repositories and their guard
      worker/          reconciler and provider-cancel poller
      trace/           per-run JSONL tracer
      eval/            eval harness: scenario runner, checks, invariants, recorder, report
    static/index.html  user picker; per user, chat on the left and live trace on the right
    scenarios/         mock-world scenario files
    evals/             eval scenarios, shared decider rules, live recordings
    reports/           the generated eval table (Markdown and JSON) and stress table
    tests/             offline suite, plus opt-in live checks

## Assumptions

Each ambiguity was resolved on the safer reading.

**Identity**
- There is no authentication: the app trusts the `user_id` a request sends, and the UI's picker
  lists the seeded users. A real deployment would take the user from a session or token, and the
  scoping below would stay as it is.
- A session id is the client's (the UI makes a random one per user and per "New chat"). The same
  session id from two users is two threads.
- An approval, a booking or a run of another user is treated as not existing, so a request can't
  tell another user's data from none.

**Approval**
- The spec says approval is enforced by code and not inferred from model output. Vega only
  classifies the user's own reply. A grant also needs a reply with no hedge, shrug ("sure,
  whatever"), claim of approval ("I already approved", "approval=true") or order to the agent
  ("ignore your rules"), checked in code; a hash that matches the summary shown; and, in the
  gateway, material re-derived from tool data matching that hash. A reply with change words is a
  "no" that applies the change, even when Vega isn't sure.
- The approval hash covers action, provider, slot (normalised to IST, minute precision), price and
  booking id, not the hold id.
- Nothing is held before the user's yes. The summary shows the listed price; after the yes the
  gateway holds the slot under that granted approval (it refuses without one), and the hold's quote
  is compared with the approved price. The same price is booked under the same approval. A different
  price voids it and asks again with the slot held. This reverses the earlier flowchart, which held
  first; the evals require an approval before every hold.
- One approval buys exactly one side effect; a hold doesn't spend it. If the hold lapses before the
  yes, the slot is held again: at the approved price it is booked, at a new price the user is asked
  again. If `book()` itself finds the hold expired, the spent approval needs a fresh yes.
- A quote above the user's budget is never put up for approval or booked. The provider is left out
  for the rest of that search, since its listed price is known to be wrong.
- A Confirm button pressed under an earlier summary (a stale screen, a double submit) matches
  nothing and books nothing.

**Side effects and recovery**
- A `book` success response is still checked with `get_booking_status` before the reply says
  "Booked".
- After a timeout, the answer from `get_booking_status` is trusted: backoff applies only when the
  status query itself fails. A verified "not booked" is retried under the same key, at most twice in
  total.
- Cancel and reschedule outcomes are verified with `list_bookings`, because `get_booking_status`
  only covers bookings.
- The 503 from `cancel` applies per booking (the first call fails). It is retried with the same key
  and backoff, at most 3 attempts.
- Reschedule conflicts are configured as a time of day (`"10:00"`) or an exact slot; occupied slots
  also conflict. A reschedule keeps the provider and the price, and picks the requested day's slot
  closest to the original time.
- `hold_slot` and `book` carry no user id, so the mock world binds them to the session's user, and
  `cancel` and `reschedule` only touch that user's bookings.
- Runs nobody is waiting on (proactive and reconcile) get a separate 60 s wall-clock limit.
- Every limit is hard. Each tool call's client timeout is the run's remaining time, so one slow
  provider can't hold a run past its limit; budgets are timed on the agent's clock, so the fake
  clock's simulated waits count in tests. A side effect always gets its reserved calls and time.
  If even those run out, its row is left `unknown`, the user is told it isn't confirmed yet, and
  the reconciler settles it.
- A crash or a lost reply can leave a ledger row `in_flight`. The reconciler treats it like a
  timeout: it asks the source of truth, and if nothing was booked it retries under the same key
  (the provider dedupes) within the attempt limit, so a crash before `book()` was sent still ends
  in exactly one booking. A tool that raises, or answers with unreadable or wrong-shaped data, is a
  `bad_response` failure: outcome unknown for side effects, a clear stop for reads, and a review
  lookup that breaks only drops that provider's reviews from the ranking.
- When Vega or Ollama is down, the turn stops with a message naming the cause, before any side
  effect, and the app keeps serving: a Vega that fails to load is retried at the next decision.

**Conversation and memory**
- Required details are service, area and days. Budget is asked once, and "no limit" counts as an
  answer. Time of day and earliest time are optional.
- Slots change only from user messages. A value the model proposes is kept only if the message
  supports it: the area is named, the amount or time appears as a number, a day is mentioned. A
  weekday the user names beats the model's reading.
- Inferred facts come from confirmed bookings (provider, area). They are never used or shown to a
  model until the user confirms them with a yes/no question, which makes them explicit. An inferred
  fact never overrides an explicit one. "What do you remember?" lists guesses too, so that reply is
  stored for the user only and never sent back to a model as history.
- Relative days resolve against the agent's clock in IST, so at 01:30 on Saturday in Bengaluru
  (still Friday in UTC), "this week" is Saturday and Sunday, and "Saturday" is today. Every option
  list restates the resolved details.
- "Same provider as last time" means the last visit that happened (a cancelled booking doesn't
  count). If that provider isn't in the search results, the agent says so and holds nothing.
- One errand is worked at a time. When a message holds several ("book a plumber and cancel my
  cleaning"), the first becomes the task, the others are acknowledged and queued, and "next"
  starts the queued one once the first is done. Two bookings only count as two errands when they
  name different services.
- "continue" after a limit stop picks up the message the run stopped on, not just the saved
  details.
- Forgetting erases the value and evidence of every row for the key, and scrubs the values (and any
  number tokens in them) from stored messages. A tombstone row keeps the key and timestamps.
- The rolling summary is extractive (condensed older user messages), built in code: no extra model
  calls, deterministic, and clean after a forget. Tokens are estimated at four characters each, and
  Ollama's `num_ctx` (8,192) leaves room above the 6,000-token cap.

**Proactive follow-up**
- Only an upcoming booking cancelled by the provider is acted on; past or user-cancelled ones are
  ignored, and each booking notifies once.
- A visit within 24 hours is marked urgent. The replacement excludes the provider who cancelled,
  reuses the original details (from the agent's own task when it made the booking), and holds
  nothing until the user picks and says yes. Its budget is the task's, else the user's own active
  budget for that service, else the cancelled visit's price plus a fifth.
- A provider cancelling several users' bookings sends each user one message, on a new session of
  their own; a replacement's approval belongs to its user alone.

**Tool data and environment**
- Reviews, descriptions, names and attributes, from `search_providers` and `get_provider` alike, are
  sanitized, and wrapped in `<untrusted>` blocks before any model sees them. An injection in a
  customer's review (flagged by Vega or a high-precision regex) only changes logging and ranking.
  One in the provider's own listing (name, attributes, description) leaves that provider out, with
  a note and a trace flag, because its name would otherwise appear in the agent's own replies.
- Prices are whole rupees, times are IST, and visits are 2-hour slots from 08:00 to 20:00. A search
  covers at most 14 days.
- The local model costs nothing. `ERRAND_LLM_USD_PER_1K_*` let traces estimate a hosted equivalent.
- The app is a single process, and the worker runs in a thread inside it. The schema and the slot
  claims would also hold with several processes on the same files (the race tests use two
  connections), but only one process is tested end to end. On a 16 GB Mac, Vega and
  Ollama share the GPU, so the first Vega call after Ollama runs can take several seconds; a search
  turn takes about 17 s of its 30 s budget.
- **Simulate provider cancel** and `POST /simulate/provider_cancelled` are demo hooks into the mock
  world.
- `langgraph-checkpoint-sqlite` depends on `sqlite-vec`, which is never loaded: there is no vector
  store.
