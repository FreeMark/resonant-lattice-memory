"""tool_handler.py — ToolHandlerMixin + the lattice_store tool schema.

Named tool_handler (NOT tools) on purpose: the plugin dir sits on
sys.path[0], so a module named tools.py would shadow Hermes' tools
package and break `from tools.registry import tool_error`.

Mixed into LatticeMemoryProvider; uses self._store/_retriever/config attrs
and sibling methods (self._run_consolidation_epoch, etc.) via the composite."""

import json
import logging
import threading
from typing import Any, Dict, List

from tools.registry import tool_error
from store_common import hrr, _HRR_AVAILABLE

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tool Schema (expanded with new actions)
# ----------------------------------------------------------------------
LATTICE_STORE_SCHEMA = {
    "name": "lattice_store",
    "description": (
        "Resonant Lattice Memory tool — full neuroplastic control.\n"
        "Actions: add, search, get_fact, fact_history, feedback, pin, unpin, "
        "request_abstraction, force_consolidation, force_dream_cycle, stats, "
        "memory_audit (includes feature_status), pending_conflicts, resolve_conflict, facts_about_entity, "
        "entities_for_fact, related_entities, explain_abstraction, tool_history, "
        "relational, infer, get_self_model, set_self_model, narrative, "
        "set_canonical, get_canonical.\n"
        "Canonical state: set_canonical(key, value[, category]) records the CURRENT "
        "value of a key as a single authoritative field (updating it preserves "
        "history); get_canonical(key) reads it (no key = list all / by category). "
        "Use it for facts that have one current truth (vendor terms, an address, a "
        "config value) instead of inferring 'current' from recall ranking.\n"
        "You INFLUENCE memory, you never destroy it. There is deliberately NO delete: "
        "to retire a fact you believe is wrong or stale, call feedback with "
        "feedback='unhelpful' — that lowers its resonance so it FADES toward dormancy "
        "(recoverable, still pluckable by a strong cue), rather than erasing it. "
        "feedback='helpful' raises resonance. The system alone owns decay/forget.\n"
        "Use pin to mark a fact identity-level / never-forget (e.g. the user's name, a "
        "standing preference, a safety rule): a pinned fact is exempt from ALL forgetting "
        "— decay, dormancy-prune, and long-tier eviction — until you unpin it. Pin only "
        "PROTECTS; it does not inflate importance. Reserve it for the few facts that must "
        "never fade; unpin to release one back to normal Hebbian dynamics.\n"
        "Use request_abstraction to ask the memory to run a generalization pass NOW — "
        "cluster related long-term facts and synthesize higher-level abstractions "
        "(contextualized: a default plus its scoped exceptions), instead of waiting for "
        "the periodic dream-cycle abstraction. Distinct from force_dream_cycle (which only "
        "sometimes abstracts).\n"
        "Use narrative to read the durable cross-session history — one-paragraph "
        "summaries of past sessions (a remembered gist, not verbatim) for continuity "
        "on what you and the user have been doing together.\n"
        "Use get_self_model / set_self_model to read or deliberately curate the "
        "agent's own identity (a separate, authoritative self-model that is NEVER "
        "auto-ingested): set_self_model(key, value) records who you are, your "
        "capabilities, or your standing relationship with the user. This is the one "
        "place self-knowledge is written on purpose; ordinary consolidation can read "
        "it but never writes it.\n"
        "Use relational to answer who/where/what questions over the extracted "
        "(subject, relation, object) graph: pass a natural-language 'query', or "
        "structured 'subject'/'relation'/'object' (any can be omitted as the "
        "unknown). Results are exact graph matches first, then graceful HRR "
        "fuzzy matches; each carries its source fact.\n"
        "Use infer for bounded multi-hop reasoning from a 'subject' (optionally to "
        "an 'object'): it chains stored triples (<= hops) into DERIVED connections. "
        "These are INFERENCES, not facts — each is flagged inferred=true, carries "
        "the supporting path + a decayed confidence, and is NEVER stored. Treat "
        "them as leads to confirm, not as attested memory.\n"
        "Use pending_conflicts to see unresolved contradictory memories (the system "
        "duels them automatically, but you can disambiguate): each group lists the "
        "competing facts; call resolve_conflict with fact_id = the correct one to "
        "boost it and retire the rest as superseded history.\n"
        "Use get_fact (exact ID lookup) — NOT search — when you need to confirm "
        "the verbatim content of a specific stored fact; search returns similar "
        "neighbours, not the exact row. get_fact also returns quote_status: only "
        "'attested' means the fact's source_quote was verified verbatim against "
        "its source; treat 'soft'/'unattested'/'unverified'/null as unconfirmed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "get_fact", "fact_history", "feedback", "pin", "unpin", "request_abstraction", "force_consolidation", "force_dream_cycle", "stats", "memory_audit", "pending_conflicts", "resolve_conflict", "facts_about_entity", "entities_for_fact", "related_entities", "explain_abstraction", "tool_history", "relational", "infer", "get_self_model", "set_self_model", "narrative", "set_canonical", "get_canonical", "list_batches", "rollback_batch"],
            },
            "content": {"type": "string", "description": "Fact content (for add/feedback)"},
            "query": {"type": "string", "description": "Search query, entity/tool name for entity-graph actions, or a natural-language relational question (relational)"},
            "subject": {"type": "string", "description": "relational/infer: subject slot — the start node for infer (omit = unknown for relational)"},
            "relation": {"type": "string", "description": "relational: relation slot, e.g. works_at/lives_in (omit = unknown)"},
            "object": {"type": "string", "description": "relational/infer: object slot — for infer, only chains ending here are returned (optional)"},
            "hops": {"type": "integer", "description": "infer: max chain length (defaults to max_inference_hops)"},
            "key": {"type": "string", "description": "set_self_model/get_self_model: identity key, e.g. name/role/relationship_with_user (get with no key returns the whole self-model)"},
            "value": {"type": "string", "description": "set_self_model: the identity value to record for key"},
            "fact_id": {"type": "integer", "description": "Fact ID (get_fact/fact_history/feedback/pin/unpin/entities_for_fact/explain_abstraction; for resolve_conflict it is the WINNING fact's id)"},
            "min_age": {"type": "integer", "description": "pending_conflicts: only list conflict groups at least this many cycles old (0 = all)"},
            "feedback": {"type": "string", "enum": ["helpful", "unhelpful"], "description": "User feedback for Hebbian adjustment"},
            "category": {"type": "string", "description": "Category for new facts or filtering (e.g., procedural)"},
            "entity": {"type": "string", "description": "Entity name (facts_about_entity/related_entities)"},
            "tool_name": {"type": "string", "description": "Tool name to query history for (tool_history)"},
            "abstract_id": {"type": "integer", "description": "Abstraction fact id (explain_abstraction)"},
            "batch_id": {"type": "integer", "description": "Write-batch id (list_batches with batch_id = its facts; rollback_batch)"},
            "tier": {"type": "string", "enum": ["short", "mid", "long"], "description": "Optional tier filter"},
            "limit": {"type": "integer", "description": "Max results to return"},
            "min_shared": {"type": "integer", "description": "Min shared facts for related_entities co-occurrence"},
        },
        "required": ["action"],
    },
}


class ToolHandlerMixin:

    # ====================== TOOL HANDLING ======================
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [LATTICE_STORE_SCHEMA]


    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Full tool handler for lattice_store with all actions implemented."""
        if tool_name != "lattice_store" or not self._store or not self._retriever:
            return tool_error(
                "Resonant Lattice Memory is not available in this session "
                "(store or retriever failed to initialize)."
            )

        action = args.get("action")
        if action in ("add", "feedback", "pin", "unpin", "request_abstraction", "remove", "resolve_conflict", "force_consolidation", "force_dream_cycle", "set_self_model", "set_canonical", "rollback_batch") \
                and not self._write_enabled:
            return tool_error("Memory is read-only in this agent context (non-primary).")
        try:
            if action == "add":
                content = args.get("content", "")
                if not content:
                    return tool_error("Missing required parameter: content")
                
                emb = self._retriever._get_embedding(content)
                if not emb:
                    return tool_error("Failed to generate embedding for fact.")

                entities = self._store._extract_entities(content)
                hrr_vec = hrr.encode_fact(content, entities, dim=self._hrr_dim) if _HRR_AVAILABLE else None

                action_taken, fid = self._store.add_or_reinforce_fact(
                    content, emb, args.get("category", "general"), self._session_id,
                    hrr_vector=hrr_vec, entities=entities
                )
                return json.dumps({"status": action_taken, "fact_id": fid})

            elif action == "search":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                results = self._retriever.search(query, limit=10)
                # Conflict CONTAINMENT (the explicit search surface is contained too,
                # not just autonomous prefetch): withhold high-stakes unresolved-conflict
                # unpinned values and return only the conflict metadata, so an agent that
                # searches when prefetch is insufficient still cannot read a disputed
                # high-stakes value before resolution. No-op when quarantine is off.
                results, withheld = self._quarantine_conflicts(results)
                self._apply_recall_reinforcement(results)
                resp = {"results": results, "count": len(results)}
                if withheld:
                    resp["withheld_conflicts"] = [
                        {"conflict_group_id": g, "withheld": n}
                        for g, n in sorted(withheld.items())]
                    resp["notice"] = ("High-stakes facts in unresolved conflict(s) were "
                                      "WITHHELD; call pending_conflicts / resolve_conflict "
                                      "before acting on the disputed value.")
                return json.dumps(resp)

            elif action == "get_fact":
                # Exact-ID lookup. The return shape is DELIBERATELY distinct from
                # search (no "results"/"count" keys, no neighbours) so the model
                # cannot confuse a direct hit with a similarity match. A miss
                # returns found:false and NEVER neighbour rows — preventing the
                # phantom-fact confabulation that arises from narrating around
                # search neighbours when asked for a specific row.
                fact_id = args.get("fact_id")
                if fact_id is None:
                    return tool_error("Missing required parameter: fact_id")
                fact = self._store.get_fact(fact_id)
                if fact is None:
                    return json.dumps({"found": False, "fact_id": fact_id})
                return json.dumps({"found": True, "fact": fact})

            elif action == "fact_history":
                # Read-only supersedion lineage (Phase 1b): the forward chain of
                # facts that replaced this one + the facts it superseded. Not
                # write-gated (like get_fact/stats). A miss returns found:false.
                fact_id = args.get("fact_id")
                if fact_id is None:
                    return tool_error("Missing required parameter: fact_id")
                history = self._store.get_fact_history(fact_id)
                if history is None:
                    return json.dumps({"found": False, "fact_id": fact_id})
                return json.dumps({"found": True, "history": history})

            elif action == "remove":
                # A21 no-delete: the AGENT influences memory, it never destroys it. The
                # destructive remove path is admin-only (config agent_can_delete, default
                # False). When disabled, point the model at the non-destructive route —
                # unhelpful feedback fades the fact to dormancy (recoverable), and pin
                # protects a vital one. (remove_fact stays for audited human/admin cleanup.)
                if not getattr(self, "_agent_can_delete", False):
                    return tool_error(
                        "Memory has no agent delete (by design). To retire a wrong/stale "
                        "fact use feedback with feedback='unhelpful' (it fades to dormancy, "
                        "recoverable); to protect a vital one use pin. Hard delete is "
                        "admin-only (config agent_can_delete)."
                    )
                fact_id = args.get("fact_id")
                if fact_id is None:
                    return tool_error("Missing required parameter: fact_id")
                removed = self._store.remove_fact(fact_id)   # ← uses lock internally
                if removed:
                    return json.dumps({"status": "removed", "fact_id": fact_id})
                else:
                    return tool_error(f"Fact {fact_id} not found.")

            elif action in ("pin", "unpin"):
                # A5 identity-level durability: pin marks a fact never-forget (exempt from
                # decay/prune/eviction); unpin releases it to normal Hebbian dynamics.
                # Protective only — never changes resonance, so it can't be abused to make a
                # fact runaway-immortal; the system still owns forgetting for everything else.
                fact_id = args.get("fact_id")
                if fact_id is None:
                    return tool_error("Missing required parameter: fact_id")
                pinned = (action == "pin")
                ok = self._store.set_pinned(fact_id, pinned)
                if ok:
                    return json.dumps({"status": "pinned" if pinned else "unpinned",
                                       "fact_id": fact_id, "pinned": pinned})
                return tool_error(f"Fact {fact_id} not found.")

            elif action == "request_abstraction":
                # On-demand generalization (A8 contextualization): run the abstraction pass
                # NOW instead of waiting for the periodic dream-cycle one. Non-daemon +
                # tracked as the dream thread so shutdown() drains it before close() (it
                # writes synthesized facts to the DB, same handle-safety as a dream cycle).
                _t = threading.Thread(target=self._perform_abstraction_pass, daemon=False)
                self._last_dream_thread = _t
                _t.start()
                return json.dumps({"status": "Abstraction pass triggered"})

            elif action == "feedback":
                fact_id = args.get("fact_id")
                feedback = args.get("feedback")
                if fact_id is None or feedback not in ["helpful", "unhelpful"]:
                    return tool_error("feedback requires fact_id and feedback=helpful|unhelpful")
                delta = 2 if feedback == "helpful" else -3
                success = self._store.adjust_resonance(fact_id, delta)
                if success:
                    return json.dumps({"status": "feedback recorded", "delta": delta})
                else:
                    return tool_error("Fact not found")

            elif action == "force_consolidation":
                # Non-daemon: this is an intentional agent action that must complete
                # even if the session ends immediately after the tool call.
                threading.Thread(
                    target=self._run_consolidation_epoch,
                    args=(self._session_id,),
                    daemon=False    # was daemon=True — would be killed on process exit
                ).start()
                return json.dumps({"status": "Waking cycle (consolidation) triggered"})
 
            elif action == "force_dream_cycle":
                # Non-daemon: same reasoning as force_consolidation above. Tracked
                # so shutdown() drains it before closing the DB handle.
                _t = threading.Thread(
                    target=self._run_dream_cycle,
                    daemon=False    # was daemon=True — would be killed on process exit
                )
                self._last_dream_thread = _t
                _t.start()
                return json.dumps({"status": "Dream Cycle triggered"})

            elif action == "stats":
                stats = self._store.get_stats()              # ← uses lock internally
                stats["memory_cycles"] = self._memory_cycle
                stats["dream_cycles"] = self._dream_cycle_count
                stats["feature_status"] = self.get_feature_status()
                stats["status"] = "healthy"
                return json.dumps(stats)

            elif action == "memory_audit":
                # Read-only health snapshot (same data the cycle-based audit logs).
                # No side effects — not write-gated, like stats.
                health = self._store.get_memory_health(near_cap=self._health_near_cap)
                health["memory_cycle"] = self._memory_cycle
                health["dream_cycle"] = self._dream_cycle_count
                health["feature_status"] = self.get_feature_status()
                # Phase 7: expose blind tier status if active (more diagnostics for encrypted setups)
                if getattr(self, '_blind_tier', None):
                    bt = self._blind_tier
                    health["blind_active"] = True
                    health["blind_reconcile_batch"] = getattr(bt, 'reconcile_batch', 0)
                    health["blind_has_recall"] = bt.recall is not None if bt else False
                    health["blind_has_hrr"] = bt.hrr is not None if bt else False
                else:
                    health["blind_active"] = False
                return json.dumps(health)

            elif action == "pending_conflicts":
                # Read-only (not write-gated): list unresolved conflict groups so the
                # agent can disambiguate. min_age defaults to 0 here (an explicit query
                # shows everything; the passive recall nudge is what age-gates).
                min_age = int(args.get("min_age", 0))
                limit = int(args.get("limit", 20))
                conflicts = self._store.get_pending_conflicts(min_age_cycles=min_age, limit=limit)
                return json.dumps({"conflicts": conflicts, "count": len(conflicts)})

            elif action == "resolve_conflict":
                winner_id = args.get("fact_id", args.get("winner_id"))
                if winner_id is None:
                    return tool_error("resolve_conflict requires fact_id (the winning fact's id)")
                result = self._store.resolve_conflict(winner_id, current_cycle=self._memory_cycle)
                if result is None:
                    return tool_error(f"Fact {winner_id} is not in an active conflict group.")
                return json.dumps({"status": "conflict resolved", **result})

            elif action == "facts_about_entity":
                entity = args.get("entity") or args.get("query")
                if not entity:
                    return tool_error("Missing required parameter: entity or query")
                limit = int(args.get("limit", 15))
                tier = args.get("tier")
                category = args.get("category")
                results = self._store.get_facts_for_entity(entity, limit=limit, tier=tier, category=category)
                return json.dumps({"entity": entity, "results": results, "count": len(results)})

            elif action == "entities_for_fact":
                fact_id = args.get("fact_id")
                if fact_id is None:
                    return tool_error("Missing required parameter: fact_id")
                entities = self._store.get_entities_for_fact(fact_id)
                return json.dumps({"fact_id": fact_id, "entities": entities})

            elif action == "related_entities":
                entity = args.get("entity") or args.get("query")
                if not entity:
                    return tool_error("Missing required parameter: entity or query")
                min_shared = int(args.get("min_shared", 2))
                limit = int(args.get("limit", 20))
                results = self._store.get_related_entities(entity, min_shared=min_shared, limit=limit)
                return json.dumps({"entity": entity, "related": results})
                
            elif action == "explain_abstraction":
                abstract_id = args.get("abstract_id") or args.get("fact_id")
                if abstract_id is None:
                    return tool_error("Missing required parameter: abstract_id or fact_id")
                try:
                    explanation = self._store.get_abstraction_explanation(abstract_id)
                    return json.dumps(explanation, default=str)
                except Exception as e:
                    return tool_error(f"Failed to explain abstraction: {e}")
                    
            elif action == "tool_history":
                tool_name = args.get("tool_name") or args.get("entity") or args.get("query")
                if not tool_name:
                    return tool_error("Missing required parameter: tool_name (or entity/query)")
                limit = int(args.get("limit", 10))
                tier = args.get("tier")
                results = self._store.get_tool_history(tool_name, limit=limit, tier=tier)
                return json.dumps({"tool_name": tool_name, "results": results, "count": len(results)})

            elif action == "relational":
                # Phase 5b: read-only relational recall over the (s,r,o) graph.
                # Accepts a free-text query or structured subject/relation/object.
                query = args.get("query")
                subject = args.get("subject")
                relation = args.get("relation")
                obj = args.get("object")
                if not (query or subject or relation or obj):
                    return tool_error("relational requires a query or at least one of subject/relation/object")
                limit = int(args.get("limit", 10))
                results = self._store.relational_recall(
                    subject=subject, relation=relation, object=obj, query=query,
                    max_results=limit, hrr_floor=self._relation_recall_hrr_floor,
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "infer":
                # Phase 5c: bounded transitive inference. Read-only and never
                # persisted — results are DERIVED, flagged inferred=true with their
                # supporting path. subject (start node) is required.
                subject = args.get("subject") or args.get("query")
                if not subject:
                    return tool_error("infer requires a subject (the start node)")
                obj = args.get("object")
                hops = int(args.get("hops", self._max_inference_hops))
                limit = int(args.get("limit", 10))
                inferences = self._store.infer_relations(
                    subject=subject, object=obj, max_hops=hops, max_results=limit,
                )
                return json.dumps({"inferences": inferences, "count": len(inferences),
                                   "note": "inferred (derived) — not stored facts"})

            elif action == "get_self_model":
                # Phase 7: read the deliberate self-model. Read-only (not write-
                # gated). Whole model when no key is given.
                if not self._enable_self_model:
                    return tool_error("self-model is disabled (set enable_self_model=true to use it).")
                result = self._store.get_self_model(args.get("key"))
                return json.dumps({"self_model": result})

            elif action == "set_self_model":
                # Phase 7: deliberately curate the self-model. Write-gated to the
                # primary context above — this is the ONLY write path to the
                # agent_identity store (autonomous ingest can never reach it).
                if not self._enable_self_model:
                    return tool_error("self-model is disabled (set enable_self_model=true to use it).")
                key = args.get("key")
                value = args.get("value")
                if not key or value is None:
                    return tool_error("set_self_model requires key and value")
                row = self._store.set_self_model(key, value, current_cycle=self._memory_cycle)
                if row is None:
                    return tool_error("Invalid key/value for set_self_model")
                return json.dumps({"status": "self-model updated", **row})

            elif action == "narrative":
                # Phase 8: read-only durable cross-session narrative history.
                if not self._enable_narrative:
                    return tool_error("narrative is disabled (set enable_narrative=true to use it).")
                limit = int(args.get("limit", self._narrative_keep))
                results = self._store.get_recent_narrative(limit=limit)
                return json.dumps({"narrative": results, "count": len(results)})

            elif action == "set_canonical":
                # Canonical-state projection: the explicit CURRENT-value layer over
                # the lattice. Write-gated (primary context only). Updating a key
                # closes the prior value (history preserved) and makes this current.
                key = args.get("key")
                value = args.get("value")
                if not key or value is None:
                    return tool_error("set_canonical requires key and value")
                cid = self._store.set_canonical(
                    key, value, category=args.get("category", "general"),
                    source_fact_id=args.get("fact_id"), cycle=self._memory_cycle)
                return json.dumps({"status": "canonical set", "canonical_id": cid,
                                   **(self._store.get_canonical(key) or {})})

            elif action == "get_canonical":
                # Read the current canonical value for a key; with no key, list all
                # current canonical records (optionally filtered by category). Read-only.
                key = args.get("key")
                if key:
                    rec = self._store.get_canonical(key)
                    return json.dumps({"canonical": rec, "found": rec is not None})
                recs = self._store.list_canonical(category=args.get("category"))
                return json.dumps({"canonical": recs, "count": len(recs)})

            elif action == "list_batches":
                # Semantic-rollback provenance: list recent consolidation/dream write
                # batches, or (with batch_id) the facts a batch wrote (the diff surface).
                bid = args.get("batch_id")
                if bid is not None:
                    return json.dumps({"batch_id": int(bid),
                                       "facts": self._store.get_batch_facts(int(bid))})
                return json.dumps({"batches": self._store.list_write_batches(
                    limit=int(args.get("limit", 50)))})

            elif action == "rollback_batch":
                # Undo a bad consolidation/dream batch's NEW facts (pinned kept). Write-gated.
                bid = args.get("batch_id")
                if bid is None:
                    return tool_error("rollback_batch requires batch_id")
                return json.dumps(self._store.rollback_write_batch(int(bid)))

            else:
                return tool_error(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"handle_tool_call failed for action {action}: {e}", exc_info=True)
            return tool_error(f"Tool call failed: {e}")
