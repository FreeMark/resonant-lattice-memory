"""blind_tier.py — BlindTier: the Tier-1 blind store as ONE cohesive collaborator.

The SEAM between the COGNITION layer (tiers / decay / conflict / abstraction / dream cycles — all of
which run on the plaintext store regardless of encryption mode) and the blind-store implementation.
Before this, the provider carried six ``self._blind*`` fields, two resolver methods, a retriever
swap, and the reconcile pass spread across ``__init__`` + ``consolidation``. BlindTier owns all of
it, so the provider holds ONE optional ``self._blind_tier`` and the entire blind-vs-plaintext
divergence lives behind a single boundary — and the future §5 change ("blind ciphertext becomes the
source of truth") is localized here rather than scattered through the provider.

Today the blind tier is a MIRROR alongside the plaintext store (embedding → ``semantic_he``,
HRR lift → ``semantic_he_hrr``, entity-name set → ``semantic_he_entities``); it is not yet a
replacement backend. Any SUBSET may be present: the HE recall/HRR/maint contexts need openfhe; the
AEAD entity store is independent (argon2 + cryptography only) and comes up even without openfhe, so
an openfhe-less host still gets entity-at-rest protection.

crypto_keys / he_crypto are imported LAZILY (openfhe only on the blind path). Module-level imports
are the dependency-free ``store_common`` (holographic + _HRR_AVAILABLE) only. Helper layer — like
retrieval / crypto_keys / he_crypto, this file is UNTRACKED by the acceptance harness."""

import logging
import os
from typing import Optional

from store_common import hrr as _hg, _HRR_AVAILABLE

logger = logging.getLogger(__name__)


class BlindTier:
    """Owns the Tier-1 blind components + the reconcile pass. Construct via ``resolve`` (returns
    None when the blind tier can't be brought up at all). The provider holds the result as one
    optional field and routes the retriever swap + reconcile through it."""

    def __init__(self, store, *, recall=None, hrr_client=None, maint=None,
                 writer=None, hrr_writer=None, entities=None, reconcile_batch=200):
        self.store = store
        self.recall = recall            # BlindRecallPRE @ embed-dim (cosine + PRE) | None
        self.hrr = hrr_client           # BlindRecallPRE @ 2*hrr_dim (HRR lift)     | None
        self.maint = maint              # BlindMaintenance (held for E5 2b-ii)       | None
        self.writer = writer            # BlindWriter -> semantic_he                 | None
        self.hrr_writer = hrr_writer    # BlindWriter -> semantic_he_hrr             | None
        self.entities = entities        # BlindEntityStore -> semantic_he_entities   | None
        self.reconcile_batch = int(reconcile_batch)

    # ── construction ──────────────────────────────────────────────────────────────
    @classmethod
    def resolve(cls, store, *, db_path, keystore_path, he_keystore_path, hrr_dim,
                reconcile_batch=200) -> "Optional[BlindTier]":
        """Bring up the blind tier for ``encryption_mode=blind``. Returns a BlindTier holding
        whatever subset came up, or None if NOTHING did (neither the HE contexts nor the entity
        store). Every failure is logged + non-fatal — the caller falls back to plaintext recall and
        a no-op reconcile rather than disabling memory. ``db_path`` is accepted for symmetry/logging;
        ``keystore_path`` / ``he_keystore_path`` are the already-resolved sidecar paths."""
        contexts = cls._resolve_contexts(store, keystore_path, he_keystore_path, hrr_dim)
        recall = hrr_client = maint = writer = hrr_writer = None
        if contexts is not None:
            recall = contexts.get("recall")
            hrr_client = contexts.get("hrr")
            maint = contexts.get("maint")     # held for E5 dream-cycle wiring (2b-ii)
            from retrieval import BlindWriter
            writer = BlindWriter(store, recall)
            # HRR write mirror: encrypted (cos,sin) lift -> semantic_he_hrr (E4 4b), sized 2*hrr_dim.
            if hrr_client is not None:
                hrr_writer = BlindWriter(store, hrr_client, table="semantic_he_hrr")
            logger.info("\U0001f512 Blind tier ACTIVE — homomorphic recall over semantic_he "
                        "(+ HRR lift mirror to semantic_he_hrr).")
        else:
            logger.warning("encryption_mode=blind but the blind tier is unavailable (see prior errors); "
                           "falling back to plaintext recall and skipping blind writes this session. "
                           "This is expected if openfhe/argon2 not installed or passphrase missing.")
        # Entity sets are AEAD (no openfhe) → set up INDEPENDENTLY of the HE contexts, so
        # entity-at-rest protection holds even on a host without openfhe.
        entities = cls._resolve_entities(store, keystore_path)
        if entities is not None:
            logger.info("\U0001f512 Blind entity sets ACTIVE — AEAD-encrypted names in "
                        "semantic_he_entities (client-side overlap, store learns nothing).")
        if contexts is None and entities is None:
            return None
        return cls(store, recall=recall, hrr_client=hrr_client, maint=maint, writer=writer,
                   hrr_writer=hrr_writer, entities=entities, reconcile_batch=reconcile_batch)

    @staticmethod
    def _resolve_contexts(store, keystore_path, he_keystore_path, hrr_dim):
        """Set up or load ALL HE contexts from the MULTI-keyset keystore — ``recall``
        (BlindRecallPRE @ embed-dim), ``hrr`` (BlindRecallPRE @ 2*hrr_dim), ``maint`` (light
        decay-only BlindMaintenance). Returns the ``{recall, hrr, maint}`` client dict, or None.
        Needs argon2 + openfhe + a passphrase; creates the at-rest keystore (salt/master) + the
        multi HE keystore (``<db>.he``) on first run. The keysets coexist in one process (OpenFHE
        keys the global eval-key store by context tag; node-proven, Option A 2a)."""
        import crypto_keys
        if not crypto_keys.kdf_available():
            logger.error("encryption_mode=blind requires argon2-cffi; blind tier disabled.")
            return None
        try:
            import he_crypto
        except Exception as e:
            logger.error("encryption_mode=blind requires openfhe (%s); blind tier disabled.", e)
            return None
        if not he_crypto.he_available():
            logger.error("encryption_mode=blind: openfhe/numpy unavailable; blind tier disabled.")
            return None
        passphrase = crypto_keys.get_passphrase(prompt=False)
        if not passphrase:
            logger.error("encryption_mode=blind but no passphrase available. Set %s in the "
                         "environment. Blind tier disabled.", crypto_keys.ENV_PASSPHRASE)
            return None
        try:
            if not os.path.exists(keystore_path):
                keystore = crypto_keys.create_keystore(passphrase)
                crypto_keys.save_keystore(keystore_path, keystore)
                logger.warning("Blind-tier keystore CREATED at %s. The passphrase is the ONLY "
                               "way to recover this memory — there is NO recovery.", keystore_path)
            else:
                keystore = crypto_keys.load_keystore(keystore_path)
            clients, _he_ks, created = crypto_keys.setup_or_load_blind_contexts(
                passphrase, keystore, he_keystore_path,
                embed_dim=int(store.vector_dim), hrr_dim=int(hrr_dim), role="user")
            logger.info("Blind tier MULTI HE keystore %s at %s (recall/hrr/maint).",
                        "INITIALIZED" if created else "loaded", he_keystore_path)
            return clients
        except crypto_keys.WrongPassphraseError:
            logger.error("Blind tier: passphrase does not match the keystore at %s. Disabled.",
                         keystore_path)
            return None
        except Exception as e:
            logger.error("Blind tier (multi keystore) setup failed (%s); blind tier disabled.",
                         e, exc_info=True)
            return None
        finally:
            if isinstance(passphrase, bytearray):
                crypto_keys.secure_zero(passphrase)

    @staticmethod
    def _resolve_entities(store, keystore_path):
        """Build a BlindEntityStore that AEAD-encrypts each fact's entity-NAME set at rest in
        ``semantic_he_entities`` (client-side, no-leak; roadmap §7.4). Returns it or None. Needs
        argon2 + cryptography + a passphrase + the keystore; **no openfhe** (pure AEAD), so entity-
        at-rest protection is independent of the HE recall contexts and comes up even when openfhe
        is absent. The derived entity key is held for the session inside the encrypt/decrypt
        closures — the trusted-client RAM key, like the HE secret."""
        import crypto_keys
        if not crypto_keys.kdf_available():
            logger.error("blind entity store requires argon2-cffi; entity encryption disabled.")
            return None
        if not crypto_keys.aead_available():
            logger.error("blind entity store requires cryptography; entity encryption disabled.")
            return None
        passphrase = crypto_keys.get_passphrase(prompt=False)
        if not passphrase:
            logger.error("encryption_mode=blind but no passphrase available for entity "
                         "encryption. Set %s in the environment. Entity encryption disabled.",
                         crypto_keys.ENV_PASSPHRASE)
            return None
        try:
            if not os.path.exists(keystore_path):
                keystore = crypto_keys.create_keystore(passphrase)
                crypto_keys.save_keystore(keystore_path, keystore)
                logger.warning("Blind-tier keystore CREATED at %s. The passphrase is the ONLY "
                               "way to recover this memory — there is NO recovery.", keystore_path)
            else:
                keystore = crypto_keys.load_keystore(keystore_path)
            ent_key = crypto_keys.derive_entity_key(passphrase, keystore)  # verifies key-check
            # Bind the session entity key into closures so BlindEntityStore carries no crypto import
            # (duck-typed encrypt_fn/decrypt_fn over crypto_keys.encrypt/decrypt_entities).
            from retrieval import BlindEntityStore
            enc = lambda ents: crypto_keys.encrypt_entities(ents, ent_key)
            dec = lambda blob: crypto_keys.decrypt_entities(blob, ent_key)
            return BlindEntityStore(store, enc, dec)
        except crypto_keys.WrongPassphraseError:
            logger.error("Blind entity store: passphrase does not match the keystore at %s. "
                         "Entity encryption disabled.", keystore_path)
            return None
        except Exception as e:
            logger.error("Blind entity store setup failed (%s); entity encryption disabled.",
                         e, exc_info=True)
            return None
        finally:
            if isinstance(passphrase, bytearray):
                crypto_keys.secure_zero(passphrase)

    # ── recall + write ────────────────────────────────────────────────────────────
    def decorate_retriever(self, plaintext_retriever, ollama_endpoint, embed_model, min_similarity):
        """Return a BlindRetriever (homomorphic vector + HRR recall over the encrypted tables) when
        the recall context is up, else the plaintext retriever unchanged (entity-only / openfhe-less
        sessions keep plaintext recall). HRR recall routes through the separate 2*hrr_dim context."""
        if self.recall is None:
            return plaintext_retriever
        from retrieval import BlindRetriever
        return BlindRetriever(self.store, ollama_endpoint, embed_model, blind=self.recall,
                              min_similarity=min_similarity, blind_hrr=self.hrr)

    def reconcile(self, store=None, limit: int = 0) -> int:
        """Write-path completeness (roadmap §14 6a): mirror every fact that has a plaintext row but
        NO blind ciphertext into the encrypted tables, reading its embedding/HRR/entities back from
        the plaintext store (semantic_vec / semantic_facts.hrr_vector / fact_entities — NO Ollama).

        The single mechanism for ALL write paths — catches facts created OUTSIDE the consolidation
        epoch (abstraction / gist / procedural distillation; the builtin-memory mirror) and BACKFILLS
        a store on first blind-enable. Idempotent (the worklists are LEFT JOINs, so a mirrored fact
        drops off; the entity set re-mirrors when ``entities_dirty`` flips) and non-fatal (each
        writer swallows + logs). ``limit`` (>0, else ``reconcile_batch``) bounds the per-pass work.
        Returns the count of embedding cts written."""
        st = store if store is not None else self.store
        if st is None:
            return 0
        eff_limit = limit if limit > 0 else self.reconcile_batch
        n = 0
        if self.writer is not None:
            for fid in st.facts_missing_blind("semantic_he", eff_limit):
                emb = st.get_fact_embedding(fid)
                if emb and self.writer.write_fact(fid, emb):
                    n += 1
        if self.hrr_writer is not None and _HRR_AVAILABLE and _hg is not None:
            for fid in st.facts_missing_blind("semantic_he_hrr", eff_limit):
                phases = st.get_fact_hrr_phases(fid)
                if phases is None:
                    continue
                try:
                    self.hrr_writer.write_fact(fid, _hg.hrr_lift(phases).tolist())
                except Exception as e:
                    logger.debug("Blind reconcile HRR mirror failed for %s (non-fatal): %s", fid, e)
        if self.entities is not None:
            for fid in st.facts_needing_entity_mirror(eff_limit):
                try:
                    if self.entities.set_entities(fid, st.get_entities_for_fact(fid)):
                        st.mark_entities_mirrored(fid)
                except Exception as e:
                    logger.debug("Blind reconcile entity mirror failed for %s (non-fatal): %s", fid, e)
        if n:
            logger.debug("Blind reconcile mirrored %d embedding ct(s).", n)
        return n
