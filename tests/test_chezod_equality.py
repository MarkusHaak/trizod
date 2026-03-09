"""
TODO: Port legacy CheZOD equality test.

Requires:
- test/data/CheZOD_results_all/ — legacy CheZOD reference output files
- Update unpacking of get_offset_corrected_wSCS() (now returns 9 values, old test expected 5)

The test validates that TriZOD Z-scores match the original CheZOD implementation
by comparing against pre-computed CheZOD reference data for known BMRB entries.
"""
