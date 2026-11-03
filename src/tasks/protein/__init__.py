"""Protein task package; importing it must not import the agent stack.

Nothing here is eager, because ``tasks.protein`` is a parent package of
``tasks.protein.cm.poet2``, which has to import inside the separate PoET-2
interpreter -- a 3.10 venv holding PoET-2's pinned flash-attn build and little
else. An eager ``from . import primitives`` here would put ``recital`` (and so
tenacity, rapidfuzz, the LLM client) on that path.

The residue-calibrated primitives therefore register from the adapter that
offers them (``tasks/protein/cm/adapter.py``).

The DNA packages (``tasks/promoter/__init__.py`` and ``tasks/loopseq/__init__.py``)
register eagerly at package scope. That is safe there because neither is a
parent of a module that has to import under a foreign interpreter; Loop-seq's
DNAcycP2 scorer runs in a subprocess that is never re-imported as a
``tasks.loopseq`` submodule.
"""
