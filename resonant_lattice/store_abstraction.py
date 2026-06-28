"""store_abstraction.py — AbstractionMixin: LLM-driven generalization and
procedural distillation, provenance/explainability, shared JSON cleanup.

Mixed into LatticeStore; calls sibling-mixin methods via self."""

import logging
import json
import re
import urllib.request
from typing import List, Dict, Optional, Tuple

from store_common import serialize_vector, hrr, _HRR_AVAILABLE, sqlite3

logger = logging.getLogger(__name__)


class AbstractionMixin:

    def perform_abstraction_pass(
        self,
        reason_model: str,
        ollama_endpoint: str,
        prompt: str = None,
        max_facts: int = 180,
        max_clusters: int = 6,
        min_cluster_size: int = 3,
        max_cluster_size: int = 8,
        cluster_hrr_similarity: float = 0.68,
        cluster_entity_overlap: float = 0.55,
        dedup_threshold: float = 0.82,
    ) -> None:
        """Memory Abstraction / Generalization Layer.
 
        Clusters long-term facts using HRR similarity + entity overlap,
        then asks the reasoning model to synthesize higher-level abstractions.
 
        All numeric thresholds are configurable from config.yaml under
        plugins.resonant_lattice, passed through the LatticeMemoryProvider
        constructor.
        """
        if not _HRR_AVAILABLE:
            return
 
        # ── Phase 1: Data gathering ───────────
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, content, hrr_vector
                FROM semantic_facts
                WHERE tier = 'long' AND hrr_vector IS NOT NULL
                ORDER BY resonance_count DESC, updated_at DESC
                LIMIT ?
                """,
                (max_facts,),
            ).fetchall()
 
            if len(rows) < 5:
                logger.debug("Abstraction pass: not enough long-term facts (%d)", len(rows))
                return
 
            entity_rows = self._conn.execute(
                """
                SELECT fe.fact_id, e.name
                FROM fact_entities fe
                JOIN entities e ON e.entity_id = fe.entity_id
                WHERE fe.fact_id IN (
                    SELECT id FROM semantic_facts WHERE tier = 'long'
                )
                """
            ).fetchall()
 
        entity_map: Dict[int, set] = {}
        for r in entity_rows:
            entity_map.setdefault(r["fact_id"], set()).add(r["name"])
 
        rows = list(rows)
 
        # ── Phase 2: Clustering ──────
        clusters = []
        used: set = set()
 
        for i in range(len(rows)):
            if i in used:
                continue
            f1 = rows[i]
            v1 = self._phases_from_blob(f1["hrr_vector"])
            if v1 is None:
                continue
            cluster = [f1]
            used.add(i)
            ents1 = entity_map.get(f1["id"], set())
 
            for j in range(i + 1, len(rows)):
                if j in used:
                    continue
                f2 = rows[j]
                v2 = self._phases_from_blob(f2["hrr_vector"])
                if v2 is None:
                    continue
                sim = hrr.similarity(v1, v2)
                ents2 = entity_map.get(f2["id"], set())
                overlap = len(ents1 & ents2) / max(len(ents1 | ents2), 1)

                # Cluster on a strong signal on EITHER axis, but guard
                # entity-only matches with a semantic floor: now that entity
                # extraction is high-precision, overlap is trustworthy, yet a
                # single shared entity should not merge semantically unrelated
                # facts. (Was a bare OR, which over-clustered on noisy entities.)
                strong_semantic = sim >= cluster_hrr_similarity
                strong_entity = overlap >= cluster_entity_overlap
                if strong_semantic or (strong_entity and sim >= 0.30):
                    cluster.append(f2)
                    used.add(j)
 
                if len(cluster) >= max_cluster_size:
                    break
 
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
 
        if not clusters:
            logger.debug("Abstraction pass: no meaningful clusters found")
            return
 
        logger.info("🧠 Abstraction pass: %d clusters from %d long-term facts", len(clusters), len(rows))
 
        # ── Phase 3: LLM calls + embedding fetches ─
        pending: List[Tuple[str, Optional[List[float]], list]] = []

        for cluster in clusters[:max_clusters]:
            fact_texts = [f"• {r['content']}" for r in cluster]
            cluster_text = "\n".join(fact_texts)

            # Use the config prompt if provided, otherwise fallback to default
            base_prompt = prompt if prompt else "You are an expert memory abstraction engine for an AI agent.\n\nGiven the following group of related long-term facts, synthesize 1-2 higher-level abstractions that capture the general pattern WITHOUT erasing what makes each fact true in its OWN situation.\n\nRules:\n- Abstraction is CONTEXTUALIZATION, not erasure. When the facts give different answers in different conditions, state the DEFAULT and PRESERVE the exceptions as scoped conditions (e.g. 'X by default; Y when <condition>'). The conditions that make each fact correct are the POINT — keep them.\n- Do NOT collapse facts that hold in distinct contexts into one detail-free claim, and NEVER invent a generalization the facts do not support.\n- Concise but meaningful; one principle per object\n- Output ONLY a valid JSON array (no extra text)\n- Each object must have keys: \"content\" and \"category\" (use \"abstract\")"
            
            # Append the cluster text
            final_prompt = f"{base_prompt}\n\nFACTS:\n{cluster_text}\n\nJSON OUTPUT:"

            try:
                payload = {"model": reason_model, "prompt": final_prompt,
                           "stream": False, "options": {"temperature": 0.2}}
                req = urllib.request.Request(
                    f"{ollama_endpoint}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=300.0) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    response_text = result.get("response", "[]").strip()

                response_text = self._clean_llm_json(response_text)

                abstractions = []
                try:
                    start = response_text.find('[')
                    end = response_text.rfind(']')
                    if start != -1 and end != -1:
                        abstractions = json.loads(response_text[start:end + 1])
                    else:
                        abstractions = json.loads(response_text)
                except json.JSONDecodeError:
                    content_pat = r'"content"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
                    for m in re.finditer(content_pat, response_text):
                        abstractions.append({"content": m.group(1), "category": "abstract"})

                if isinstance(abstractions, dict):
                    abstractions = (
                        abstractions["facts"] if isinstance(abstractions.get("facts"), list)
                        else [abstractions] if "content" in abstractions
                        else []
                    )
                if not isinstance(abstractions, list):
                    abstractions = []
                for abstract in abstractions:
                    if not isinstance(abstract, dict):
                        continue
                    content = abstract.get("content", "").strip()
                    if not content or len(content) < 15:
                        continue
                    emb = self._get_embedding_for_abstraction(content, ollama_endpoint)
                    pending.append((content, emb, cluster))   # capture cluster for provenance

            except Exception as e:
                logger.warning("Abstraction cluster LLM call failed: %s", e)

        if not pending:
            logger.debug("Abstraction pass: LLM produced no candidates")
            return

        # ── Phase 4: Dedup + INSERT + Entity Extraction ──────────
        abstractions_created = 0
        with self._lock:
            for content, emb, cluster in pending:
                if emb:
                    # Pass dedup_threshold as the real cutoff so the lookup and the
                    # gate agree. _find_semantic_match returns None unless the top
                    # hit already clears dedup_threshold, so the row is a true dup.
                    similar = self._find_semantic_match(emb, threshold=dedup_threshold)
                    if similar:
                        continue

                # 1. Extract entities BEFORE inserting
                entities = self._extract_entities(content)
                
                # 2. Use encode_fact so the abstract vector math matches normal facts
                hrr_vec = None
                if _HRR_AVAILABLE:
                    try:
                        hrr_vec = hrr.phases_to_bytes(hrr.encode_fact(content, entities, dim=self.hrr_dim))
                    except Exception as e:
                        logger.debug(f"HRR encoding failed for abstract fact: {e}")

                try:
                    cur = self._conn.execute(
                        """
                        INSERT INTO semantic_facts
                            (content, category, tier, resonance_count, source_session, hrr_vector)
                        VALUES (?, 'abstract', 'long', 5, 'abstraction', ?)
                        """,
                        (content, hrr_vec)
                    )
                    fact_id = cur.lastrowid
                    if emb and not self.degraded:
                        self._conn.execute(
                            "INSERT INTO semantic_vec (id, embedding) VALUES (?, ?)",
                            (fact_id, serialize_vector(emb))
                        )
                    
                    if entities:
                        self._link_entities(fact_id, entities)

                    # Record provenance using the *captured* cluster for this abstract
                    for src in cluster:
                        self._conn.execute(
                            """
                            INSERT OR IGNORE INTO abstraction_sources 
                            (abstract_id, source_id, cluster_size_at_creation)
                            VALUES (?, ?, ?)
                            """,
                            (fact_id, src["id"], len(cluster))
                        )

                    self._conn.commit()
                    abstractions_created += 1
                    logger.debug("Created abstraction: %s…", content[:60])

                except sqlite3.IntegrityError:
                    # Duplicate content (UNIQUE collision) — abstraction already
                    # exists. Roll back the aborted statement so the shared
                    # connection is left clean for the next iteration.
                    self._conn.rollback()
                except Exception as e:
                    # Any other DB error (vec dim, provenance, etc.) must NOT
                    # leave a half-applied transaction open on the shared
                    # connection for the next locked op to inherit.
                    self._conn.rollback()
                    logger.debug("Abstraction insert failed, rolled back: %s", e)
 
        if abstractions_created > 0:
            logger.info("✅ Abstraction pass complete — %d new abstract facts", abstractions_created)
        else:
            logger.debug("Abstraction pass completed with no new abstractions")


    def _cluster_by_hrr_entity(self, rows, entity_map, cluster_hrr_similarity,
                               cluster_entity_overlap, min_cluster_size,
                               max_cluster_size):
        """Greedy single-pass clustering by HRR content similarity OR entity overlap.

        Shared, LLM-free, deterministic grouping used by Phase-4 pre-prune gisting.
        Mirrors the inline clustering in perform_abstraction_pass (kept inline there
        to avoid disturbing the proven path; this is the extracted, unit-tested
        version — a future refactor can point the abstraction pass here too). An
        entity-only match still needs a low semantic floor (sim >= 0.30) so a single
        shared entity doesn't merge unrelated facts. `rows` are mapping-like with
        id/hrr_vector; returns a list of clusters (each a list of the input rows).
        """
        clusters = []
        used = set()
        for i in range(len(rows)):
            if i in used:
                continue
            f1 = rows[i]
            v1 = self._phases_from_blob(f1["hrr_vector"])
            if v1 is None:
                continue
            cluster = [f1]
            used.add(i)
            ents1 = entity_map.get(f1["id"], set())
            for j in range(i + 1, len(rows)):
                if j in used:
                    continue
                f2 = rows[j]
                v2 = self._phases_from_blob(f2["hrr_vector"])
                if v2 is None:
                    continue
                sim = hrr.similarity(v1, v2)
                ents2 = entity_map.get(f2["id"], set())
                overlap = len(ents1 & ents2) / max(len(ents1 | ents2), 1)
                if sim >= cluster_hrr_similarity or (
                        overlap >= cluster_entity_overlap and sim >= 0.30):
                    cluster.append(f2)
                    used.add(j)
                if len(cluster) >= max_cluster_size:
                    break
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        return clusters


    def _select_gist_candidates(self, gist_floor, min_peak_resonance, limit):
        """Phase 4: dying facts that EARNED their place and aren't already preserved.

        A candidate is fading (resonance_count <= gist_floor) AND was important once
        (tier in mid/long, OR max_resonance_seen >= min_peak_resonance), AND is an
        ordinary fact (not an abstraction/gist/procedural synthesis, not superseded
        history), AND has an HRR vector to cluster on, AND is NOT already represented
        in an abstraction/gist (so we never gist a fact whose meaning is already
        preserved, and never re-gist one we already gisted). Short-tier noise that
        was never important is deliberately excluded. Read-only; caller holds the
        lock. Returns the candidate rows.
        """
        return self._conn.execute(
            """
            SELECT id, content, hrr_vector, tier, resonance_count, max_resonance_seen
            FROM semantic_facts
            WHERE resonance_count <= ?
              AND tier != 'superseded'
              AND category NOT IN ('abstract', 'gist', 'procedural')
              AND hrr_vector IS NOT NULL
              AND (tier IN ('mid', 'long') OR COALESCE(max_resonance_seen, 0) >= ?)
              AND id NOT IN (
                  SELECT source_id FROM abstraction_sources WHERE source_id IS NOT NULL
              )
            ORDER BY COALESCE(max_resonance_seen, 0) DESC, id ASC
            LIMIT ?
            """,
            (gist_floor, min_peak_resonance, limit),
        ).fetchall()


    def consolidate_before_prune(
        self,
        reason_model: str,
        ollama_endpoint: str,
        prompt: str = None,
        gist_floor: float = 0.0,
        min_peak_resonance: float = 4.0,
        cluster_hrr_similarity: float = 0.68,
        cluster_entity_overlap: float = 0.55,
        min_cluster_size: int = 2,
        max_cluster_size: int = 8,
        max_clusters: int = 3,
        dedup_threshold: float = 0.82,
    ) -> int:
        """Phase 4 — gist-preserving forgetting (hippocampal→neocortical analogue).

        Run in the dream cycle BEFORE prune_weak_facts: cluster the dying-but-
        once-important facts and LLM-summarise each cluster into ONE category='gist'
        fact (clearly framed as a remembered summary, NOT verbatim), recording
        provenance in abstraction_sources. The originals are then pruned normally —
        their detail is gone, but their MEANING survives in the gist. Because each
        gisted source becomes an abstraction_sources row, it is excluded from re-
        gisting on the next cycle even if it lingers above the prune threshold.

        LLM-bound and conservative: returns early (no LLM call) when there is no
        qualifying cluster. Returns the number of gists created. Mirrors the
        perform_abstraction_pass structure: gather (locked) → cluster + LLM + embed
        (unlocked) → insert + provenance (locked)."""
        if not _HRR_AVAILABLE:
            return 0

        # ── Phase 1: gather candidates + entity map (locked) ──────
        with self._lock:
            rows = self._select_gist_candidates(
                gist_floor, min_peak_resonance, max_clusters * max_cluster_size * 2
            )
            if len(rows) < min_cluster_size:
                return 0
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            ent_rows = self._conn.execute(
                f"""
                SELECT fe.fact_id, e.name
                FROM fact_entities fe
                JOIN entities e ON e.entity_id = fe.entity_id
                WHERE fe.fact_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
        entity_map: Dict[int, set] = {}
        for r in ent_rows:
            entity_map.setdefault(r["fact_id"], set()).add(r["name"])
        rows = [dict(r) for r in rows]

        # ── Phase 2: cluster (no lock) + LLM summarise + embed ──────
        clusters = self._cluster_by_hrr_entity(
            rows, entity_map, cluster_hrr_similarity, cluster_entity_overlap,
            min_cluster_size, max_cluster_size,
        )
        if not clusters:
            return 0

        base_prompt = prompt or (
            "You are a memory consolidation engine. The facts below are fading from "
            "memory but mattered once. Write ONE concise gist that preserves their "
            "shared meaning while letting specific details go. Frame it as a summary, "
            "never invent specifics. Output ONLY a JSON array with a single object "
            "with keys \"content\" and \"category\" (use \"gist\"), or []."
        )

        pending: List[Tuple[str, Optional[List[float]], list]] = []
        for cluster in clusters[:max_clusters]:
            cluster_text = "\n".join(f"• {r['content']}" for r in cluster)
            final_prompt = f"{base_prompt}\n\nFADING FACTS:\n{cluster_text}\n\nJSON OUTPUT:"
            try:
                payload = {"model": reason_model, "prompt": final_prompt,
                           "stream": False, "options": {"temperature": 0.2}}
                req = urllib.request.Request(
                    f"{ollama_endpoint}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                # Generous timeout: this runs off the hot path (dream cycle) and a
                # cold-start model load can exceed 60s. A failed gist here lets the
                # originals prune unpreserved, so it's worth waiting for the model.
                with urllib.request.urlopen(req, timeout=300.0) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    response_text = result.get("response", "[]").strip()

                response_text = self._clean_llm_json(response_text)
                gists = []
                try:
                    start, end = response_text.find('['), response_text.rfind(']')
                    if start != -1 and end != -1:
                        gists = json.loads(response_text[start:end + 1])
                    else:
                        gists = json.loads(response_text)
                except json.JSONDecodeError:
                    gists = [{"content": m.group(1), "category": "gist"}
                             for m in re.finditer(
                                 r'"content"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', response_text)]
                if isinstance(gists, dict):
                    gists = (gists["facts"] if isinstance(gists.get("facts"), list)
                             else [gists] if "content" in gists else [])
                if not isinstance(gists, list):
                    gists = []

                # One gist per cluster — take the first valid object.
                for g in gists:
                    if not isinstance(g, dict):
                        continue
                    content = (g.get("content") or "").strip()
                    if not content or len(content) < 15:
                        continue
                    emb = self._get_embedding_for_abstraction(content, ollama_endpoint)
                    pending.append((content, emb, cluster))
                    break
            except Exception as e:
                logger.warning("Gist consolidation LLM call failed: %s", e)

        if not pending:
            return 0

        # ── Phase 3: insert gists + provenance (locked) ──────
        created = 0
        with self._lock:
            for content, emb, cluster in pending:
                # Dedup against EXISTING facts, but NEVER against this gist's own
                # still-present source facts — a gist is by construction similar to
                # the dying facts it summarises (they prune moments later). Only a
                # NON-source near-duplicate should block it. The abstraction_sources
                # re-gist guard + UNIQUE content already prevent duplicate gists.
                if emb:
                    dup = self._find_semantic_match(emb, threshold=dedup_threshold)
                    if dup and dup["id"] not in {src["id"] for src in cluster}:
                        continue
                entities = self._extract_entities(content)
                hrr_vec = None
                if _HRR_AVAILABLE:
                    try:
                        hrr_vec = hrr.phases_to_bytes(
                            hrr.encode_fact(content, entities, dim=self.hrr_dim))
                    except Exception as e:
                        logger.debug("HRR encoding failed for gist: %s", e)
                try:
                    # Gists enter the long tier at the promotion bar: durable (long
                    # is decay-exempt) but not artificially maximal — the preserved
                    # meaning should survive, then earn/lose its keep like any fact.
                    cur = self._conn.execute(
                        """
                        INSERT INTO semantic_facts
                            (content, category, tier, resonance_count, source_session,
                             hrr_vector, max_resonance_seen)
                        VALUES (?, 'gist', 'long', ?, 'gist_consolidation', ?, ?)
                        """,
                        (content, float(self.promotion_threshold), hrr_vec,
                         float(self.promotion_threshold)),
                    )
                    gist_id = cur.lastrowid
                    if emb and not self.degraded:
                        self._conn.execute(
                            "INSERT INTO semantic_vec (id, embedding) VALUES (?, ?)",
                            (gist_id, serialize_vector(emb)),
                        )
                    if entities:
                        self._link_entities(gist_id, entities)
                    for src in cluster:
                        self._conn.execute(
                            """
                            INSERT OR IGNORE INTO abstraction_sources
                                (abstract_id, source_id, cluster_size_at_creation)
                            VALUES (?, ?, ?)
                            """,
                            (gist_id, src["id"], len(cluster)),
                        )
                    self._conn.commit()
                    created += 1
                    logger.debug("Created gist from %d fading facts: %s…",
                                 len(cluster), content[:60])
                except sqlite3.IntegrityError:
                    self._conn.rollback()  # duplicate content (UNIQUE) — already gisted
                except Exception as e:
                    self._conn.rollback()
                    logger.debug("Gist insert failed, rolled back: %s", e)

        if created:
            logger.info("🌫️  Gist consolidation — preserved %d fading theme(s) as gists", created)
        return created


    def _get_embedding_for_abstraction(self, text: str, ollama_endpoint: str) -> Optional[List[float]]:
        """Internal helper — uses self.embed_model (was hardcoded)."""
        try:
            payload = {"model": self.embed_model, "prompt": text}   # ← was "nomic-embed-text"
            req = urllib.request.Request(
                f"{ollama_endpoint}/api/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("embedding")
        except Exception:
            return None


    def get_abstraction_sources(self, abstract_id: int) -> List[Dict]:
        """Return the original source facts that contributed to this abstraction."""
        with self._lock:
            rows = self._conn.execute("""
                SELECT s.source_id as id, f.content, f.tier, f.resonance_count, 
                       f.category, s.cluster_size_at_creation, s.created_at
                FROM abstraction_sources s
                JOIN semantic_facts f ON f.id = s.source_id
                WHERE s.abstract_id = ?
                ORDER BY f.resonance_count DESC
            """, (abstract_id,)).fetchall()
            return [dict(r) for r in rows]


    def get_abstraction_explanation(self, abstract_id: int) -> Dict:
        """Return a rich, explainable view of an abstraction.

        Includes:
        - The abstraction content + metadata
        - All original source facts (with current resonance/tier)
        - Entities linked to the abstraction
        - Staleness signal: how much of the original evidence still exists and is strong
        """
        with self._lock:
            # Get the abstraction itself
            abstract_row = self._conn.execute(
                "SELECT id, content, category, tier, resonance_count, created_at, updated_at "
                "FROM semantic_facts WHERE id = ?",
                (abstract_id,)
            ).fetchone()

            if not abstract_row:
                return {"error": "Abstraction not found", "abstract_id": abstract_id}

            abstract = dict(abstract_row)

            # Get source facts + their current state
            sources = self.get_abstraction_sources(abstract_id)

            # Get entities for the abstraction
            entities = self.get_entities_for_fact(abstract_id)

            # === Staleness calculation ===
            original_count = len(sources)
            if original_count == 0:
                staleness = 1.0  # No provenance recorded (legacy abstraction)
            else:
                active_sources = [s for s in sources if s.get("resonance_count", 0) > 0]
                active_count = len(active_sources)

                if active_count == 0:
                    staleness = 1.0  # All evidence has been pruned or zeroed
                else:
                    avg_resonance = sum(s["resonance_count"] for s in active_sources) / active_count
                    coverage = active_count / original_count
                    # Reference "strong" resonance = the promotion bar (a source at
                    # promotion strength counts as fully-supporting). Tracks config
                    # instead of a magic 5.0 that breaks once caps/deltas change.
                    strong_ref = float(max(self.promotion_threshold, 1))
                    staleness = round(1.0 - (coverage * min(avg_resonance / strong_ref, 1.0)), 3)

            return {
                "abstract": abstract,
                "source_facts": sources,
                "entities": entities,
                "provenance": {
                    "original_source_count": original_count,
                    "current_active_sources": len([s for s in sources if s.get("resonance_count", 0) > 0]),
                    "staleness": staleness,           # 0.0 = fresh, 1.0 = fully stale
                    "staleness_label": (
                        "fresh" if staleness < 0.2 else
                        "aging" if staleness < 0.5 else
                        "stale" if staleness < 0.8 else
                        "orphaned"
                    )
                }
            }

            

    def distill_procedural_facts(
        self,
        reason_model: str,
        ollama_endpoint: str,
        prompt: str = None,
        min_episodes: int = 4,
        max_tools: int = 8,
        sample_size: int = 12,
    ) -> int:
        """Generalize raw tool episodes into reusable 'procedural' semantic facts.

        Returns the number of procedural facts created. Episodes that are fed to
        the LLM are marked distilled=1 so they are processed exactly once.
        """
        base_prompt = prompt or (
            "You are a procedural memory engine for an AI agent. Below are recent "
            "records of the agent calling one tool, each tagged SUCCESS or FAILURE.\n"
            "Synthesize concise, REUSABLE rules to use this tool better next time: "
            "argument/context patterns that SUCCEED, patterns that FAIL (especially "
            "valuable), and gotchas. Each rule must GENERALIZE across the records, "
            "never restate one call. If any FAILUREs occurred, include at least one "
            "failure-avoidance rule. Output ONLY a JSON array; each object has keys "
            "\"content\" and \"category\" (use \"procedural\"). 1-4 rules, or []."
        )

        # ── Phase 1: gather qualifying tools + sample episodes (locked) ──────
        with self._lock:
            tool_rows = self._conn.execute(
                """
                SELECT tool_name, COUNT(*) AS n, SUM(success) AS succ
                FROM tool_episodes
                WHERE distilled = 0
                GROUP BY tool_name
                HAVING COUNT(*) >= ?
                ORDER BY n DESC
                LIMIT ?
                """,
                (min_episodes, max_tools),
            ).fetchall()
            if not tool_rows:
                return 0
            batches = []  # (tool_name, n, succ, [sample dicts], [ids])
            for tr in tool_rows:
                samples = self._conn.execute(
                    """
                    SELECT id, arguments, result, success
                    FROM tool_episodes
                    WHERE distilled = 0 AND tool_name = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (tr["tool_name"], sample_size),
                ).fetchall()
                ids = [s["id"] for s in samples]
                batches.append((tr["tool_name"], tr["n"], tr["succ"] or 0,
                                [dict(s) for s in samples], ids))

        # ── Phase 2: LLM distillation + embeddings (NO lock — network I/O) ───
        pending: List[Tuple[str, Optional[List[float]], str]] = []
        consumed_ids: List[int] = []
        for tool_name, n, succ, samples, ids in batches:
            fails = n - succ
            lines = []
            for s in samples:
                tag = "SUCCESS" if s["success"] else "FAILURE"
                res = (s["result"] or "").replace("\n", " ")[:200]
                lines.append(f"{tag} | args={s['arguments']} | outcome: {res}")
            final_prompt = (
                f"{base_prompt}\n\nTOOL: {tool_name}\n"
                f"Total: {n} (successes: {succ}, failures: {fails})\n"
                f"RECORDS:\n" + "\n".join(lines) + "\n\nJSON OUTPUT:"
            )
            try:
                payload = {"model": reason_model, "prompt": final_prompt,
                           "stream": False, "options": {"temperature": 0.2}}
                req = urllib.request.Request(
                    f"{ollama_endpoint}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=300.0) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    response_text = result.get("response", "[]").strip()
            except Exception as e:
                logger.warning("Procedural distillation LLM call failed for %s: %s", tool_name, e)
                continue  # don't consume ids — retry this batch next cycle

            # Attempted this batch — consume it regardless of yield so we don't reprocess.
            consumed_ids.extend(ids)

            response_text = self._clean_llm_json(response_text)
            try:
                start, end = response_text.find('['), response_text.rfind(']')
                if start != -1 and end != -1:
                    rules = json.loads(response_text[start:end + 1])
                else:
                    rules = json.loads(response_text)
            except json.JSONDecodeError:
                rules = [{"content": m.group(1), "category": "procedural"}
                         for m in re.finditer(
                             r'"content"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', response_text)]
            if isinstance(rules, dict):
                rules = (rules["facts"] if isinstance(rules.get("facts"), list)
                         else [rules] if "content" in rules else [])
            if not isinstance(rules, list):
                rules = []

            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                content = (rule.get("content") or "").strip()
                if not content or len(content) < 15:
                    continue
                if not content.lower().startswith(tool_name.lower()):
                    content = f"[{tool_name}] {content}"
                emb = self._get_embedding_for_abstraction(content, ollama_endpoint)
                pending.append((content, emb, tool_name))

        # ── Phase 3: insert procedural facts + mark episodes distilled (locked) ─
        created = 0
        with self._lock:
            for content, emb, tname in pending:
                if not emb:
                    continue
                entities = self._extract_entities(content)
                tl = tname.lower()
                if tl not in entities:
                    entities.append(tl)  # always link the tool entity
                hrr_vec = None
                if _HRR_AVAILABLE:
                    try:
                        hrr_vec = hrr.encode_fact(content, entities, dim=self.hrr_dim)
                    except Exception:
                        hrr_vec = None
                try:
                    _action, fid = self.add_or_reinforce_fact(
                        content, emb, category="procedural",
                        source_session="tool_distillation",
                        hrr_vector=hrr_vec, entities=entities,
                    )
                    if fid > 0:
                        created += 1
                except Exception as e:
                    logger.debug("Procedural fact insert failed: %s", e)

            if consumed_ids:
                ph = ",".join("?" * len(consumed_ids))
                self._conn.execute(
                    f"UPDATE tool_episodes SET distilled = 1 WHERE id IN ({ph})",
                    consumed_ids,
                )
            self._conn.commit()

        if created:
            logger.info("✅ Procedural distillation — %d procedural facts from tool use", created)
        return created

            
    @staticmethod
    def _clean_llm_json(text: str) -> str:
        """Shared robust JSON cleaning for both consolidation and abstraction LLM responses."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE).strip()
        return text
