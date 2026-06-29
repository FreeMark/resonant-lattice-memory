# Resonant Lattice Monitor (`rl_monitor`)

A live, **read-only**, `nvtop`-style terminal dashboard for the Resonant Lattice
memory store. Point it at a running agent's memory DB and watch the lattice do its
thing: facts get added and reinforced, tiers promote, dream cycles decay and prune,
conflicts surface and resolve.

It attaches over a **separate read-only WAL connection**, so it never locks or
blocks the agent. The only thing it can write is an explicit pin/unpin toggle you
trigger by hand (and even that can be disabled with `--read-only`).

## Install

```bash
pip install textual          # or: pip install -r tools/requirements.txt
```

## Run

```bash
# watch a live agent's memory
python tools/rl_monitor.py --db ~/.hermes/resonant_lattice_memory.db

# no agent handy? drive synthetic activity into a throwaway DB and watch that
python tools/rl_monitor.py --demo

# print one static snapshot and exit (great over SSH / for a screenshot)
python tools/rl_monitor.py --demo --once
```

### Flags

| flag | meaning |
|---|---|
| `--db PATH` | store DB to watch (default `~/.hermes/resonant_lattice_memory.db`) |
| `--demo` | generate synthetic activity into a throwaway DB and watch it |
| `--refresh interval\|cycle` | repaint every heartbeat (`interval`, default) **or** only when the memory clock advances (`cycle`) |
| `--interval SECONDS` | heartbeat seconds (default `1.0`) |
| `--read-only` | disable the pin/unpin write action (pure observation) |
| `--once` | print one static snapshot and exit (no live TUI) |
| `--demo-interval SECONDS` | seconds per synthetic cycle in `--demo` (default `1.2`) |

### Keys

- **↑ / ↓** — move the selection in the Top-resonance table
- **p** — pin / unpin the selected fact
- **q** — quit

## What you see

- **Header** — memory/dream cycle counters, fact/pinned/conflict/entity/episode totals, DB size.
- **Tiers** — short / mid / long gauges (the memory-pressure view).
- **Top resonance** — the loudest-ringing facts (selectable; `★` = pinned).
- **Activity (cycles)** — the cycle-structured feed: per-cycle `add` / `reinforce` / `promote` / `conflict` / `supersede`, with a `DREAM` banner summarizing each dream cycle's bulk decay/prune/promote.
- **Conflicts** — active conflict groups and their contradicting snippets.
- **Health** — category mix, superseded history, near-cap saturation, max resonance, orphan entities.

```text
┌───────────── Resonant Lattice Monitor ─────────────┐
│ LIVE · WAL read-only · 12.4 MB                      │
│ memory cycle 1,284   dream cycle 42                 │
│ facts 3,610  pinned 7  conflicts 2  entities 511    │
└────────────────────────────────────────────────────┘
┌──── Tiers ─────┐ ┌──────── Top resonance ──────────┐
│ short ██████ 2841│ 48.2 ██████████ ★long never auto…│
│ mid   ███    449 │ 31.0 ███████··· long  Acme Boston│
│ long  ██     320 │ 12.5 █████····· mid   Net-45 …   │
└────────────────┘ └─────────────────────────────────┘
┌──── Activity (cycles) ────────┐ ┌──── Conflicts ────┐
│ c1283 + add  Globex renewed   │ │ ⚔ cg-amt (2)      │
│ c1283 ↑ reinforce Acme  3→7   │ │   dollars ⟂ cents │
│ c1284 ⤴ promote short→mid     │ │                   │
│ c1284 🧠 DREAM decayed 31 …   │ │                   │
└───────────────────────────────┘ └───────────────────┘
```

## Why it's safe

- **Monitoring is strictly read-only.** The dashboard opens the DB with
  `mode=ro` (a URI read-only connection). WAL mode lets it read a consistent
  snapshot concurrently with the agent's writes; it never takes a write lock.
- **The one write is explicit.** `p` flips a fact's `pinned` flag (exactly what
  the store's pin action is) via a brief, separate read-write connection with a
  short busy timeout. The read connection is never a writer. `--read-only`
  turns even that off.
- **Cycles, not seconds.** With `--refresh cycle`, the dashboard advances only
  when the memory's own logical clock (`memory_cycle` / `dream_cycle`) ticks —
  matching the substrate. The heartbeat then merely *detects* a cycle; it isn't
  the cadence.

## Notes

- Use a modern terminal (Windows Terminal, iTerm2, most Linux terminals) so the
  block-glyphs and the `🧠`/`⚔` markers render.
- Tested on `textual` 8.2.7, Python 3.10+.
