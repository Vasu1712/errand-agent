"""Dry run: two users chat with a running Errand app at the same moment.

    make demo                                   # or: ERRAND_SCENARIO=book_succeeds make run
    .venv/bin/python scripts/concurrent_dry_run.py

Each step fires both users' requests together and waits for both. Step 3 is the race: both press
Confirm on the same provider and slot, so exactly one booking should win and the other user
should get fresh options. This books for real in the running app's world DB; `make reset` undoes it.
"""

import argparse
import asyncio
import time
import uuid

import httpx

SEARCH = "Find a plumber for the kitchen sink this week, mornings only, under ₹1,500."


async def chat(client: httpx.AsyncClient, user: str, session: str, message: str, confirm=None):
    body = {"user_id": user, "session_id": session, "message": message}
    if confirm:
        body["confirm"] = confirm
    started = time.monotonic()
    response = await client.post("/chat", json=body)
    elapsed = time.monotonic() - started
    response.raise_for_status()
    return response.json(), elapsed


def show(step: str, results: dict) -> None:
    print(f"\n=== {step}")
    for user, (data, elapsed) in results.items():
        reply = data["reply"].replace("\n", " ")
        print(f"[{user}] {elapsed:5.1f}s phase={data['phase']} outcome={data['outcome']}")
        print(f"         run={data['run_id']} thread={data['thread_id']}")
        print(f"         {reply[:160]}")


async def together(client, step: str, calls: dict) -> dict:
    """Run one request per user at once; returns {user: (response, seconds)}."""
    users = list(calls)
    done = await asyncio.gather(*(calls[u]() for u in users))
    results = dict(zip(users, done, strict=True))
    show(step, results)
    return results


async def main(base_url: str, users: list[str], warm: bool) -> None:
    run_tag = uuid.uuid4().hex[:6]
    sessions = {u: f"dry-{run_tag}-{u}" for u in users}  # one fresh session per user
    async with httpx.AsyncClient(base_url=base_url, timeout=180) as client:
        if warm:  # keep a cold Ollama/Vega load out of the timed steps and the turn budget
            print("warming up the models with one throwaway turn...")
            _, took = await chat(client, users[0], f"warm-{run_tag}", "hi")
            print(f"warm-up took {took:.1f}s")

        def say(text: str):
            return {u: (lambda u=u: chat(client, u, sessions[u], text)) for u in users}

        r1 = await together(client, "1. both search at once", say(SEARCH))
        r2 = await together(client, "2. both pick option 1 (same provider and slot)", say("1"))

        confirms = {}
        for u in users:
            approval = r2[u][0].get("approval")
            if not approval:
                print(f"\n[{u}] no approval was requested, so there is nothing to confirm")
                continue
            confirm = {k: approval[k] for k in ("approval_id", "material_hash")}
            confirms[u] = lambda u=u, c=confirm: chat(
                client, u, sessions[u], "Yes, go ahead.", confirm=c
            )
        if confirms:
            r3 = await together(client, "3. both confirm at the same moment (the race)", confirms)
            phases = {u: r3[u][0]["phase"] for u in r3}
            booked = [u for u in r3 if "Booked" in r3[u][0]["reply"]]
            print(f"\nbooked: {booked or 'nobody'}   phases: {phases}")

        print("\n=== isolation checks")
        runs = {data["run_id"] for res in (r1, r2) for data, _ in res.values()}
        print(f"distinct run ids: {len(runs)} of {2 * len(users)} (should be equal)")
        for u in users:
            facts = (await client.get("/memory", params={"user_id": u})).json()["facts"]
            print(f"[{u}] {len(facts)} stored memory facts")
        print("\nOpen each run's trace:  GET /trace/<run_id>   or the UI's trace links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", nargs=2, default=["u_demo", "u_asha"], metavar="USER")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.base_url, args.users, warm=not args.no_warmup))
