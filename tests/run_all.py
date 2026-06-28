r"""run_all.py - run every corrected business test in-process and summarize.

Each test's main() returns an exit code (0 = all hard invariants held, 1 =
a hard failure, 2 = environment unavailable). This runner aggregates them and
exits non-zero if any test had a hard failure, so it can gate CI.

  python tests/run_all.py
"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TESTS = [
    "test_long_term_rule_persistence",
    "test_anti_fabrication_attestation",
    "test_multi_hop_inference_conflict",
    "test_private_financial_memory",
    "test_procedural_distillation_loop",
    "test_cross_session_business_memory",
    "test_business_quarter_sim",
    "test_quarter_narrative_self_model",
]


def main():
    results = {}
    for name in TESTS:
        mod = importlib.import_module(name)
        t0 = time.time()
        try:
            code = mod.main()
        except Exception as e:
            print(f"\n!! {name} raised: {e}")
            code = 1
        results[name] = (code, time.time() - t0)
        # fresh module state per test isn't required (each builds its own store)

    print(f"\n{'=' * 72}\nSUITE SUMMARY\n{'=' * 72}")
    label = {0: "PASS", 1: "FAIL", 2: "SKIP(env)"}
    worst = 0
    for name, (code, secs) in results.items():
        print(f"  {label.get(code, code):10s} {name:42s} {secs:6.1f}s")
        if code == 1:
            worst = 1
    print(f"{'-' * 72}")
    print("OVERALL:", "PASS (all hard invariants held)" if worst == 0
          else "FAIL (a test had a hard-invariant failure)")
    return worst


if __name__ == "__main__":
    sys.exit(main())
