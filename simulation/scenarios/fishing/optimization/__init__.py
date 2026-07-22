"""Norm-optimization layer for the fishing commons.

Wraps the simulation as an evaluation function and searches over *norms* (the
NL contract/law text) for ones that produce better commons outcomes, using
prompt/context-optimization methods:

- BaselineOptimizer  -- holds a fixed norm (control: does learning beat nothing?)
- TextGradOptimizer  -- textual-gradient refinement of the norm (TextGrad)
- ACEOptimizer       -- evolving playbook via Generator/Reflector/Curator (ACE)

The research question is which norms score best and why; see reward.py for the
scalar objective that operationalizes "better".
"""
