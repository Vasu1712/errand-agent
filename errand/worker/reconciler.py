"""The reconciler: settles side effects whose outcome isn't known yet.

Runs once at startup (rows left open by a crash) and then on every worker tick (rows a run left
``unknown``). Each row is settled from the source of truth through the gateway, under the same
idempotency key, and the user is told the result through the outbox. Rows whose thread has a run
in progress are left for the next tick, so the reconciler never races a live execute.
"""

from errand.gateway import Execution
from errand.graph import Phase, render
from errand.ids import session_of
from errand.store import LedgerStatus
from errand.trace import EventKind, new_run_id


def reconcile(agent) -> list[Execution]:
    settled = []
    # The one sweep across users; everything after it is done as the row's own user.
    for entry in agent.store.ledger.unresolved_everywhere():
        user_id = entry.user_id
        task = agent.store.tasks.get(entry.task_id, user_id=user_id) if entry.task_id else None
        if task is None:
            continue
        lock = agent.thread_lock(task.thread_id)
        if not lock.acquire(blocking=False):
            continue  # a run on this thread is busy with it right now
        try:
            settled.append(_settle(agent, entry, task))
        finally:
            lock.release()
    return settled


def _settle(agent, entry, task) -> Execution:
    run_id = new_run_id()
    user_id = task.user_id
    with agent.tracer(
        run_id, user_id=user_id, thread_id=task.thread_id, kind="reconcile"
    ) as tracer:
        tracer.emit(
            EventKind.RUN_START,
            kind="reconcile",
            user_id=user_id,
            thread_id=task.thread_id,
            idempotency_key=entry.idempotency_key,
            status=entry.status,
        )
        budget = agent.budget(timeout_s=agent.settings.background_timeout_s)
        result = agent.gateway(tracer, user_id, budget).verify(entry.idempotency_key)
        if result.status in (LedgerStatus.CONFIRMED, LedgerStatus.NOT_BOOKED):
            confirmed = result.status is LedgerStatus.CONFIRMED
            phase = Phase.DONE if confirmed else Phase.FAILED
            if confirmed:
                body = "Update: " + render.done(entry.op.value, result.booking)
                tracer.emit(
                    EventKind.CLAIM,
                    op=entry.op,
                    idempotency_key=entry.idempotency_key,
                    booking_id=result.booking.booking_id,
                    channel="outbox",
                )
                agent.store.tasks.update(
                    task.task_id,
                    user_id=user_id,
                    phase=phase,
                    booking_id=result.booking.booking_id,
                )
            else:
                body = "Update: your earlier request didn't go through after all, so nothing was changed. Want me to try again?"
                agent.store.tasks.update(task.task_id, user_id=user_id, phase=phase)
            agent.store.messages.add(user_id, task.thread_id, "assistant", body, run_id=run_id)
            agent.store.outbox.enqueue(
                user_id,
                "reconciled",
                body,
                task_id=task.task_id,
                payload={
                    "thread_id": task.thread_id,
                    "session_id": session_of(task.thread_id),
                    "run_id": run_id,
                    "status": result.status,
                },
                dedupe_key=f"reconciled:{entry.idempotency_key}",
            )
            # The thread's next run must not report the outcome a second time.
            config = {"configurable": {"thread_id": task.thread_id}}
            if agent.graph.get_state(config).values:
                agent.graph.update_state(
                    config,
                    {
                        "phase": phase,
                        "execution": result.model_dump(mode="json"),
                        "approval": None,
                        "hold": None,
                    },
                )
        tracer.emit(
            EventKind.RUN_END,
            outcome=result.status,
            idempotency_key=entry.idempotency_key,
            budget=budget.snapshot(),
        )
    return result
