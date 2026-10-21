"""Diameter-constrained clustering for the replay scripts — re-exported from
production, no longer implemented here.

`lib.recurrence.complete_linkage` is the shipped grouping rule (canonical
complete linkage: merge the closest admissible pair each round, stop when the
smallest exceeds the threshold; every cluster has diameter <= th). The replays
measured with this module's own pure-Python copy while the pipeline still ran
single linkage; the day the swap shipped, the copy became a drift risk and was
replaced by the import. The recorded findings stand because the two were
verified to produce identical clusters, tie-breaks included, before the copy
was deleted.

One finding worth keeping at the definition site even though the code moved:
CANONICAL is not a detail. A greedy first-admissible-pair version is
order-dependent, and on the corpus sweep it reported NO threshold at which all
six verified true cards survive, where the canonical form has a clean window
at 0.065-0.070. Split sizes differ on 2 of 19 published docu-batch groups.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # repo root

from lib.recurrence import complete_linkage as _complete_linkage  # noqa: E402


def complete_linkage(D, th):
    """Production clustering, in the replays' historical order (largest
    cluster first — some replays report "the biggest subcluster")."""
    return sorted(_complete_linkage(D, th), key=len, reverse=True)


def diameter(D):
    """Largest pairwise distance in a full distance matrix."""
    return max((D[i][j] for i in range(len(D)) for j in range(i + 1, len(D))),
               default=0.0)
