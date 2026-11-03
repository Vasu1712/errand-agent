"""``python -m errand.eval``: run the offline eval suite and write reports/eval_table.md.

    python -m errand.eval                    every scenario, offline; exit 1 if any fails
    python -m errand.eval -k timeout         only scenarios whose name contains "timeout"
    python -m errand.eval --list             list the scenarios
    python -m errand.eval --stress 20        every concurrency scenario once per world seed 1..20;
                                             writes reports/eval_stress.md, exit 1 on any failure
    python -m errand.eval --record NAME...   run each NAME against live Ollama and Vega and save
                                             evals/recordings/NAME.json for offline replay

Offline runs refuse every network connection, so a scenario can't reach Ollama by accident.
"""

import argparse
import os
import socket
import sys

from errand.config import Settings
from errand.eval.report import write_reports, write_stress_report
from errand.eval.spec import EvalScenario, load_scenarios
from errand.eval.suite import ScenarioResult, record_scenarios, run_scenario


def forbid_network() -> None:
    """Make any IPv4 or IPv6 connection fail loudly, loopback included."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def refuse(sock: socket.socket, address: object) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.close()
            raise RuntimeError(f"network access attempted in an offline eval: {address!r}")

    def connect(self: socket.socket, address: object) -> None:
        refuse(self, address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        refuse(self, address)
        return real_connect_ex(self, address)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m errand.eval", description=__doc__.split("\n")[0]
    )
    parser.add_argument("-k", dest="match", help="only scenarios whose name contains this")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument(
        "--record", metavar="NAME", nargs="+", help="record scenarios against live Ollama and Vega"
    )
    parser.add_argument(
        "--stress",
        metavar="N",
        type=int,
        help="play the concurrency scenarios N times, with world seeds 1..N; fail on any failure",
    )
    args = parser.parse_args(argv)

    # Hermetic: a developer's ERRAND_* variables or LangSmith settings must not change a run.
    for name in [n for n in os.environ if n.startswith("ERRAND_")]:
        del os.environ[name]
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"] = "false"
    settings = Settings(_env_file=None)
    scenarios = load_scenarios(settings.evals_dir)

    if args.list:
        for s in scenarios:
            print(f"{s.name:40} {s.title}")
        return 0
    if args.record:
        by_name = {s.name: s for s in scenarios}
        if unknown := [name for name in args.record if name not in by_name]:
            parser.error(f"no scenario named {', '.join(unknown)}")
        passed = True
        for result, path in record_scenarios([by_name[n] for n in args.record], settings):
            _print(result)
            print(f"      saved the live model replies and decisions to {path}")
            passed = passed and result.passed
        return 0 if passed else 1

    forbid_network()
    if args.stress:
        return stress(scenarios, settings, rounds=args.stress, match=args.match)
    chosen = [s for s in scenarios if not args.match or args.match in s.name]
    results = []
    for scenario in chosen:
        result = run_scenario(scenario, settings)
        _print(result)
        results.append(result)
    if args.match is None:
        md, js = write_reports(results, settings.reports_dir)
        print(f"wrote {md.relative_to(settings.reports_dir.parent)} and {js.name}")
    failed = [r.name for r in results if not r.passed]
    print(f"{len(results) - len(failed)} of {len(results)} scenarios pass")
    return 1 if failed else 0


def stress(
    scenarios: list[EvalScenario], settings: Settings, *, rounds: int, match: str | None
) -> int:
    """Each concurrency scenario once per world seed 1..rounds. A seed changes the catalogue,
    prices, open slots and random provider outcomes; every run fires its turns on real threads.
    One failing run fails the loop: under concurrency, a failure that shows up once is a race."""
    chosen = [s for s in scenarios if "concurrency" in s.covers and (not match or match in s.name)]
    seeds = list(range(1, rounds + 1))
    results: dict[str, list[ScenarioResult]] = {s.name: [] for s in chosen}
    for seed in seeds:
        failed = []
        for scenario in chosen:
            result = run_scenario(scenario, settings, seed=seed)
            results[scenario.name].append(result)
            if not result.passed:
                _print(result)
                failed.append(scenario.name)
        note = f"; FAILED {', '.join(failed)}" if failed else ""
        print(f"seed {seed:>3}: {len(chosen) - len(failed)} of {len(chosen)} pass{note}")
    path = write_stress_report(results, seeds, settings.reports_dir)
    runs = sum(len(r) for r in results.values())
    passed = sum(r.passed for rs in results.values() for r in rs)
    print(f"wrote {path.relative_to(settings.reports_dir.parent)}")
    print(f"stress: {passed} of {runs} runs pass ({len(chosen)} scenarios x {rounds} seeds)")
    return 0 if passed == runs else 1


def _print(result) -> None:
    status = "pass" if result.passed else "FAIL"
    held = sum(i.passed for i in result.invariants)
    print(
        f"{status}  {result.name:40} invariants {held}/{len(result.invariants)}  {result.duration_s}s"
    )
    for check in result.checks:
        if not check.passed:
            print(f"      {check.check}: expected {check.expected}; got {check.actual}")
    for invariant in result.invariants:
        for violation in invariant.violations:
            print(f"      invariant {invariant.id}: {violation}")


if __name__ == "__main__":
    sys.exit(main())
