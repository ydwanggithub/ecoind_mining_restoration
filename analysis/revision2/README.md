# Second revision analyses

These scripts produced the result tables in `data/results/revision2/` (the
README there maps each folder to its scripts). They extend the analytical
sequence in `analysis/` with the checks added in the second revision: false
discovery rate control of the trend tests, recovery status under 12
combinations of endpoint, threshold and trend criterion and the agreement of
each cell's label across them, repeated spatial block assignments, buffered
block validation and residual dependence, refits without rapidly urbanizing
cells, distance from the sites, cumulative neighborhoods, the footprint check
in the core mining area, and the climate context of annual ECI.

The scripts read internal tables that carry cell identifiers, coordinates or
registered site locations. These are not released, so the scripts document the
computations rather than run on the released files alone. `_paths.py` reads
the internal folders from the environment variables `REGAIN_ROUND1_ANALYSIS`,
`REGAIN_ROUND2_ANALYSIS`, `REGAIN_MODEL_INPUTS`, `REGAIN_VECTORS` and
`REGAIN_BOUNDARY`, and `revision_common.py` joins the released deidentified
model matrix with the internal cell table.
