#!/usr/bin/env python
"""rl_monitor.py - a live, nvtop-style TUI for the Resonant Lattice memory store.

Read-only by design: it attaches to the store's SQLite DB over a SEPARATE
read-only connection (WAL allows concurrent readers), so it never locks or blocks
the running agent. It only ever issues SELECTs; it never opens the store, runs
migrations, or writes.

Refresh model (the seam is the same either way; pick with --refresh):
  * interval - repaint every heartbeat (simple; good while developing).
  * cycle    - repaint ONLY when the memory's own logical clock advances
               (the memory_cycle / dream_cycle counters in the `meta` table) -
               "cycles, not seconds." The heartbeat then merely DETECTS that a
               cycle happened; it is not the cadence. SQLite can't push to an
               external process, so a tiny counter read is the cheapest detector.

Run against a live agent:
    python tools/rl_monitor.py --db ~/.hermes/resonant_lattice_memory.db

Watch it move with no agent (synthetic activity into a throwaway DB):
    python tools/rl_monitor.py --demo
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import sqlite3
import tempfile
import threading
from typing import Optional

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog, Static


# ───────────────────────── read-only store view ─────────────────────────
class MemoryReader:
    """Read-only window onto a Lattice store DB. Never writes; reconnects on demand."""

    def __init__(self, db_path: str):
        self.db_path = str(pathlib.Path(db_path).expanduser())
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # mode=ro keeps it read-only AND prevents SQLite from creating the
            # file if it doesn't exist yet. Use a proper file URI so Windows
            # drive-letter paths (file:///C:/...) parse correctly.
            p = pathlib.Path(self.db_path).expanduser()
            try:
                uri = p.resolve().as_uri() + "?mode=ro"
            except ValueError:
                uri = "file:" + p.as_posix() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=1.0)
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def available(self) -> bool:
        try:
            self._connect().execute("SELECT 1")
            return True
        except sqlite3.Error:
            self.close()
            return False

    def _scalar(self, conn, sql, params=()):
        try:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row and row[0] is not None else 0
        except sqlite3.Error:
            return 0

    def data_version(self) -> Optional[int]:
        """PRAGMA data_version changes whenever ANY other connection commits -
        the cheapest 'did something happen?' probe for an external reader."""
        try:
            return self._connect().execute("PRAGMA data_version").fetchone()[0]
        except sqlite3.Error:
            self.close()
            return None

    def cycle_counts(self) -> tuple[int, int]:
        """(memory_cycle, dream_cycle) from the meta table - the logical clock."""
        out = {"memory_cycle": 0, "dream_cycle": 0}
        try:
            for r in self._connect().execute(
                "SELECT key, value FROM meta WHERE key IN ('memory_cycle','dream_cycle')"
            ):
                try:
                    out[r["key"]] = int(r["value"])
                except (TypeError, ValueError):
                    pass
        except sqlite3.Error:
            pass
        return out["memory_cycle"], out["dream_cycle"]

    def snapshot(self) -> dict:
        conn = self._connect()
        by_tier: dict[str, int] = {}
        try:
            for r in conn.execute(
                "SELECT tier, COUNT(*) AS c FROM semantic_facts GROUP BY tier"
            ):
                by_tier[r["tier"]] = r["c"]
        except sqlite3.Error:
            pass
        return {
            "total": sum(by_tier.values()),
            "by_tier": by_tier,
            "pinned": self._scalar(conn, "SELECT COUNT(*) FROM semantic_facts WHERE pinned=1"),
            "entities": self._scalar(conn, "SELECT COUNT(*) FROM entities"),
            "episodes": self._scalar(conn, "SELECT COUNT(*) FROM episodes"),
            "conflicts": self._scalar(
                conn,
                "SELECT COUNT(DISTINCT conflict_group_id) FROM semantic_facts "
                "WHERE conflict_group_id IS NOT NULL",
            ),
        }

    def db_size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += os.path.getsize(self.db_path + suffix)
            except OSError:
                pass
        return total

    # order of the per-fact state tuple returned by fact_states()
    _STATE_COLS = ("tier", "resonance_count", "conflict_group_id", "superseded_by", "pinned")

    def fact_states(self) -> dict:
        """Lightweight per-fact state for diffing the activity feed: no content
        (fetched lazily for only the few facts that produced events), so the cost
        stays low even on a large live store. Adapts to whatever columns the
        store actually has, so it never silently returns an empty board on an
        older or partial schema."""
        out: dict[int, tuple] = {}
        try:
            conn = self._connect()
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(semantic_facts)")}
            if "id" not in cols:
                return out
            sel = ["id"] + [c for c in self._STATE_COLS if c in cols]
            for r in conn.execute(f"SELECT {', '.join(sel)} FROM semantic_facts"):
                keys = set(r.keys())
                out[r["id"]] = tuple(r[c] if c in keys else None for c in self._STATE_COLS)
        except sqlite3.Error:
            pass
        return out

    def contents(self, ids) -> dict:
        ids = list(ids)
        if not ids:
            return {}
        out: dict[int, str] = {}
        qs = ",".join("?" * len(ids))
        try:
            for r in self._connect().execute(
                f"SELECT id, content FROM semantic_facts WHERE id IN ({qs})", ids
            ):
                out[r["id"]] = r["content"]
        except sqlite3.Error:
            pass
        return out


# ───────────────────────── synthetic demo source ─────────────────────────
_ENTITIES = ["Acme Corp", "Globex", "Initech", "Stark Industries", "Vantyx", "Tanager", "Borealis"]
_ATTRS = [
    "spend approved {n} cents", "located in {city}", "payment terms Net-{n}",
    "contract status {state}", "renewed the {plan} plan", "invoice INV-{n} issued",
]
_CITIES = ["Boston", "Boise", "Lisbon", "Denver", "Austin"]
_STATES = ["active", "trial", "expired", "pending"]
_PLANS = ["Growth", "Scale", "Starter"]


def _rand_fact() -> tuple[str, str]:
    ent = random.choice(_ENTITIES)
    attr = random.choice(_ATTRS).format(
        n=random.randint(10, 9999), city=random.choice(_CITIES),
        state=random.choice(_STATES), plan=random.choice(_PLANS),
    )
    return f"{ent} {attr}", ent.lower()


class DemoSimulator:
    """Writes synthetic, cycle-paced activity into a throwaway DB so the monitor
    has something to show without a live agent. It mimics the store's OBSERVABLE
    surface (tiers, resonance, cycle counters) - it is NOT the real engine."""

    PROMOTE_MID = 8.0    # resonance to graduate short -> mid
    PROMOTE_LONG = 20.0  # mid -> long
    DREAM_EVERY = 4      # a dream cycle every N memory cycles
    MAX_SHORT = 60       # keep the live set bounded

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.mc = 0
        self.dc = 0
        self._init()

    def _init(self) -> None:
        c = self._conn
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
                content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS semantic_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT UNIQUE,
                category TEXT DEFAULT 'general',
                tier TEXT DEFAULT 'short',
                resonance_count REAL DEFAULT 3.0,
                cycles_in_tier INTEGER DEFAULT 0,
                pinned INTEGER NOT NULL DEFAULT 0,
                conflict_group_id TEXT,
                superseded_by INTEGER,
                max_resonance_seen REAL,
                learned_at_cycle INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        for content in (
            "POLICY: never auto-approve any spend; require human approval.",
            "POLICY: amounts are always recorded in cents, never dollars.",
        ):
            c.execute(
                "INSERT OR IGNORE INTO semantic_facts"
                "(content, category, tier, resonance_count, pinned, max_resonance_seen, learned_at_cycle) "
                "VALUES (?, 'policy', 'long', 3.0, 1, 3.0, 0)",
                (content,),
            )
        self._set_cycles()
        c.commit()

    def _set_cycles(self) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('memory_cycle', ?)", (str(self.mc),))
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('dream_cycle', ?)", (str(self.dc),))

    def _add_fact(self) -> None:
        content, ent = _rand_fact()
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO semantic_facts"
                "(content, tier, resonance_count, max_resonance_seen, learned_at_cycle) "
                "VALUES (?, 'short', 3.0, 3.0, ?)",
                (content, self.mc),
            )
            self._conn.execute("INSERT OR IGNORE INTO entities(name) VALUES (?)", (ent,))
        except sqlite3.Error:
            pass

    def step(self) -> bool:
        """Advance one memory cycle of synthetic activity (a dream cycle every N).
        Returns True if this was a dream cycle."""
        c = self._conn
        self.mc += 1
        for _ in range(random.randint(2, 5)):
            self._add_fact()
        ids = [r["id"] for r in c.execute(
            "SELECT id FROM semantic_facts WHERE pinned=0 AND tier!='superseded' "
            "ORDER BY RANDOM() LIMIT 6")]
        for fid in ids:
            bump = random.uniform(1.0, 4.0)
            c.execute(
                "UPDATE semantic_facts SET resonance_count = resonance_count + ?, "
                "max_resonance_seen = MAX(COALESCE(max_resonance_seen, 0), resonance_count + ?), "
                "cycles_in_tier = cycles_in_tier + 1 WHERE id = ?",
                (bump, bump, fid))
        is_dream = (self.mc % self.DREAM_EVERY == 0)
        if is_dream:
            self.dc += 1
            c.execute("UPDATE semantic_facts SET resonance_count = resonance_count - 1.5, "
                      "cycles_in_tier = cycles_in_tier + 1 WHERE pinned = 0")
            c.execute("UPDATE semantic_facts SET tier='mid', cycles_in_tier=0 "
                      "WHERE tier='short' AND resonance_count >= ?", (self.PROMOTE_MID,))
            c.execute("UPDATE semantic_facts SET tier='long', cycles_in_tier=0 "
                      "WHERE tier='mid' AND resonance_count >= ?", (self.PROMOTE_LONG,))
            c.execute("DELETE FROM semantic_facts WHERE pinned=0 AND resonance_count <= 0")
            if random.random() < 0.35:
                pair = [r["id"] for r in c.execute(
                    "SELECT id FROM semantic_facts WHERE conflict_group_id IS NULL AND pinned=0 "
                    "AND tier!='superseded' ORDER BY RANDOM() LIMIT 2")]
                if len(pair) == 2:
                    grp = f"cg-{self.mc}"
                    c.executemany(
                        "UPDATE semantic_facts SET conflict_group_id=? WHERE id=?",
                        [(grp, pair[0]), (grp, pair[1])])
            # occasionally RESOLVE an existing conflict by superseding the weaker side
            if random.random() < 0.5:
                grp = c.execute(
                    "SELECT conflict_group_id AS g FROM semantic_facts "
                    "WHERE conflict_group_id IS NOT NULL ORDER BY RANDOM() LIMIT 1").fetchone()
                if grp:
                    members = [r["id"] for r in c.execute(
                        "SELECT id FROM semantic_facts WHERE conflict_group_id=? "
                        "ORDER BY resonance_count DESC", (grp["g"],))]
                    if len(members) >= 2:
                        winner, loser = members[0], members[-1]
                        c.execute(
                            "UPDATE semantic_facts SET conflict_group_id=NULL WHERE id=?",
                            (winner,))
                        c.execute(
                            "UPDATE semantic_facts SET superseded_by=?, conflict_group_id=NULL, "
                            "tier='superseded' WHERE id=?", (winner, loser))
            short_n = c.execute(
                "SELECT COUNT(*) FROM semantic_facts WHERE tier='short'").fetchone()[0]
            if short_n > self.MAX_SHORT:
                c.execute(
                    "DELETE FROM semantic_facts WHERE id IN ("
                    "  SELECT id FROM semantic_facts WHERE tier='short' AND pinned=0 "
                    "  ORDER BY resonance_count ASC LIMIT ?)",
                    (short_n - self.MAX_SHORT,))
        self._set_cycles()
        c.commit()
        return is_dream

    def run_forever(self, interval: float, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.step()
            except sqlite3.Error:
                pass
            stop.wait(interval)


# ───────────────────────── snapshot diff (the feed engine) ─────────────────────────
_TIER_RANK = {"short": 0, "mid": 1, "long": 2}


def diff_states(prev: dict, cur: dict) -> dict:
    """Diff two per-fact state maps into discrete activity events.

    Bulk effects (decay/prune) are returned as lists too, but the UI rolls them
    into the per-dream-cycle summary rather than printing one line each."""
    added, pruned = [], []
    reinforced, decayed, promoted, demoted = [], [], [], []
    conflicts, superseded = [], []
    new_conflict_groups = set()

    prev_ids, cur_ids = set(prev), set(cur)
    added = sorted(cur_ids - prev_ids)
    pruned = sorted(prev_ids - cur_ids)
    for fid in cur_ids & prev_ids:
        pt, pr, pcg, psup, _ = prev[fid]
        ct, cr, ccg, csup, _ = cur[fid]
        if _TIER_RANK.get(ct, 0) > _TIER_RANK.get(pt, 0):
            promoted.append((fid, pt, ct))
        elif _TIER_RANK.get(ct, 0) < _TIER_RANK.get(pt, 0):
            demoted.append((fid, pt, ct))
        if pr is not None and cr is not None:
            if cr > pr + 1e-9:
                reinforced.append((fid, pr, cr))
            elif cr < pr - 1e-9:
                decayed.append((fid, pr, cr))
        if ccg and not pcg:
            conflicts.append(fid)
            new_conflict_groups.add(ccg)
        if csup and not psup:
            superseded.append((fid, csup))
    return {
        "added": added, "pruned": pruned, "reinforced": reinforced, "decayed": decayed,
        "promoted": promoted, "demoted": demoted, "conflicts": conflicts,
        "conflict_groups": new_conflict_groups, "superseded": superseded,
    }


# ───────────────────────── the dashboard ─────────────────────────
_TIERS = [("short", "cyan"), ("mid", "yellow"), ("long", "green")]


class MonitorApp(App):
    CSS = """
    Screen { background: $surface; }
    #header { border: round $accent; height: auto; padding: 0 1; margin: 0 0 1 0; }
    #tiers  { border: round $accent; height: auto; padding: 1 1; }
    #feed   { border: round $accent; height: 1fr; padding: 0 1; margin: 1 0 0 0; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, reader: MemoryReader, *, interval: float = 1.0,
                 refresh_mode: str = "interval", demo: bool = False):
        super().__init__()
        self.reader = reader
        self.interval = interval
        self.refresh_mode = refresh_mode
        self.demo = demo
        self._last_dv: Optional[int] = None
        self._last_cycle: tuple[int, int] = (-1, -1)
        self._last_dream: int = -1
        self._prev_states: Optional[dict] = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="header")
            yield Static(id="tiers")
            yield RichLog(id="feed", max_lines=1000, wrap=False,
                          highlight=False, markup=False)

    def on_mount(self) -> None:
        self.query_one("#header", Static).border_title = "Resonant Lattice Monitor"
        self.query_one("#tiers", Static).border_title = "Tiers"
        self.query_one("#feed", RichLog).border_title = "Activity (cycles)"
        self._refresh()
        self.set_interval(self.interval, self._tick)

    def _tick(self) -> None:
        """Heartbeat. In 'cycle' mode it only repaints when the memory clock
        advances; in 'interval' mode it repaints whenever anything changed."""
        if self.refresh_mode == "cycle":
            if self.reader.cycle_counts() == self._last_cycle:
                return
        else:
            dv = self.reader.data_version()
            if dv is not None and dv == self._last_dv:
                return
        self._refresh()

    def _refresh(self) -> None:
        self._last_dv = self.reader.data_version()
        header = self.query_one("#header", Static)
        tiers = self.query_one("#tiers", Static)
        if not self.reader.available():
            header.update(Text(f"waiting for memory DB…\n{self.reader.db_path}", style="yellow"))
            tiers.update(Text(""))
            return
        mc, dc = self.reader.cycle_counts()
        snap = self.reader.snapshot()
        states = self.reader.fact_states()
        if self._prev_states is None:
            self._feed_line(mc, "•", "monitor", f"{len(states)} facts", "dim")
        else:
            self._emit_events(mc, dc, diff_states(self._prev_states, states))
        self._prev_states = states
        self._last_cycle = (mc, dc)
        self._last_dream = dc
        header.update(self._render_header(mc, dc, snap))
        tiers.update(self._render_tiers(snap))

    def _emit_events(self, mc: int, dc: int, d: dict) -> None:
        dream = dc > self._last_dream
        label_ids = set(d["added"])
        label_ids |= {f for f, _, _ in d["reinforced"]}
        label_ids |= {f for f, _, _ in d["promoted"]}
        label_ids |= set(d["conflicts"])
        label_ids |= {f for f, _ in d["superseded"]}
        labels = self.reader.contents(label_ids)

        def snip(fid: int, n: int = 44) -> str:
            c = labels.get(fid) or f"#{fid}"
            return (c[: n - 1] + "…") if len(c) > n else c

        cap = 3
        for fid in d["added"][:cap]:
            self._feed_line(mc, "+", "add", snip(fid), "green")
        if len(d["added"]) > cap:
            self._feed_line(mc, "+", "add", f"…+{len(d['added']) - cap} more", "green dim")
        for fid, pr, cr in d["reinforced"][:cap]:
            self._feed_line(mc, "↑", "reinforce", f"{snip(fid)}  r {pr:.1f}→{cr:.1f}", "cyan")
        for fid, pt, ct in d["promoted"][:cap]:
            self._feed_line(mc, "⤴", "promote", f"{pt}→{ct}  {snip(fid)}", "bold yellow")
        for fid in d["conflicts"][:cap]:
            self._feed_line(mc, "⚔", "conflict", snip(fid), "bold red")
        for fid, sup in d["superseded"][:cap]:
            self._feed_line(mc, "⤳", "supersede", f"{snip(fid)} → #{sup}", "magenta")
        if dream:
            parts = (f"promoted {len(d['promoted'])} · decayed {len(d['decayed'])} "
                     f"· pruned {len(d['pruned'])}")
            if d["conflict_groups"]:
                parts += f" · +conflict {len(d['conflict_groups'])}"
            self._feed_line(dc, "🧠", "DREAM", parts, "bold blue", prefix="d")

    def _feed_line(self, n: int, glyph: str, kind: str, detail: str,
                   style: str, prefix: str = "c") -> None:
        line = Text()
        line.append(f"{prefix}{n:<5}", style="grey50")
        line.append(f" {glyph} ", style=style)
        line.append(f"{kind:<10}", style=style)
        line.append(detail, style=style)
        self.query_one("#feed", RichLog).write(line)

    def _render_header(self, mc: int, dc: int, snap: dict) -> Text:
        bt = snap["by_tier"]
        size_mb = self.reader.db_size_bytes() / (1024 * 1024)
        t = Text()
        t.append("DEMO" if self.demo else "LIVE",
                 style="bold magenta" if self.demo else "bold green")
        t.append("  ·  WAL read-only  ·  ", style="dim")
        t.append(f"{size_mb:,.1f} MB\n", style="cyan")
        t.append("memory cycle ", style="dim"); t.append(f"{mc:,}", style="bold white")
        t.append("    dream cycle ", style="dim"); t.append(f"{dc:,}\n", style="bold white")
        t.append("facts ", style="dim"); t.append(f"{snap['total']:,}", style="bold")
        t.append("   pinned ", style="dim"); t.append(f"{snap['pinned']:,}", style="bold blue")
        t.append("   conflicts ", style="dim")
        t.append(f"{snap['conflicts']:,}", style="bold red" if snap["conflicts"] else "dim")
        t.append("   entities ", style="dim"); t.append(f"{snap['entities']:,}", style="white")
        t.append("   episodes ", style="dim"); t.append(f"{snap['episodes']:,}", style="white")
        return t

    def _render_tiers(self, snap: dict) -> Table:
        bt = snap["by_tier"]
        maxv = max([bt.get(name, 0) for name, _ in _TIERS] + [1])
        width = 38
        tbl = Table.grid(padding=(0, 1))
        tbl.add_column(justify="right")
        tbl.add_column()
        tbl.add_column(justify="right")
        for name, color in _TIERS:
            v = bt.get(name, 0)
            filled = int(round(width * v / maxv)) if maxv else 0
            bar = Text("█" * filled + "·" * (width - filled), style=color)
            tbl.add_row(Text(name, style=f"bold {color}"), bar, Text(f"{v:,}", style=color))
        return tbl


# ───────────────────────── entry point ─────────────────────────
def _default_db_path() -> str:
    return os.path.join(os.path.expanduser("~/.hermes"), "resonant_lattice_memory.db")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Live, read-only TUI for the Resonant Lattice memory store.")
    ap.add_argument("--db", help=f"path to the store DB (default: {_default_db_path()})")
    ap.add_argument("--demo", action="store_true",
                    help="generate synthetic activity into a throwaway DB and watch it")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="heartbeat seconds (default 1.0)")
    ap.add_argument("--refresh", choices=["interval", "cycle"], default="interval",
                    help="repaint every heartbeat (interval) or only when the "
                         "memory clock advances (cycle)")
    ap.add_argument("--demo-interval", type=float, default=1.2,
                    help="seconds per synthetic cycle in --demo (default 1.2)")
    args = ap.parse_args(argv)

    stop = threading.Event()
    if args.demo:
        tmpdir = tempfile.mkdtemp(prefix="rl_monitor_demo_")
        db_path = os.path.join(tmpdir, "demo_memory.db")
        sim = DemoSimulator(db_path)
        sim.step()  # one cycle so the board isn't empty at launch
        threading.Thread(
            target=sim.run_forever, args=(args.demo_interval, stop), daemon=True).start()
    else:
        db_path = args.db or _default_db_path()

    reader = MemoryReader(db_path)
    app = MonitorApp(reader, interval=args.interval, refresh_mode=args.refresh, demo=args.demo)
    try:
        app.run()
    finally:
        stop.set()
        reader.close()


if __name__ == "__main__":
    main()
