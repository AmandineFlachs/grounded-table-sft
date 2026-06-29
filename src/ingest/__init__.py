"""Real-table ingestion adapters (Phase 2).

Each adapter parses a public dataset's table representation into our canonical
``Table`` schema (see ``src/schema.py``), so the same question/trace/validation
machinery built for synthetic tables runs unchanged on real ones.
"""
