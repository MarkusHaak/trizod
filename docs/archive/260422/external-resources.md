# External NMR Validation & Re-Referencing Resources

## Re-Referencing Tools

### LACS (Linear Analysis of Chemical Shifts)
- **What:** Detects systematic referencing errors using reference-independent CA-CB correlation
- **Paper:** Wang & Wishart 2005/2009, doi:10.1007/s10858-005-2515-0
- **Source code:** https://github.com/bmrb-io/LACS (MATLAB)
- **Our reimplementation:** `trizod/lacs/` (Python, 640 lines)
- **BMRB pre-computed:** https://bmrb.io/ftp/pub/bmrb/validation_reports/LACS/ (6,774 entries, frozen July 2020)
- **Local data:** `data/bmrb_lacs/` (downloaded NMR-STAR files)
- **Atom types:** CA, CB, C', HA, H, N

### PANAV (Probabilistic Approach for NMR Assignment Validation)
- **What:** Detects referencing errors AND individual mis-assignments using BMRB population statistics with secondary structure prediction
- **Paper:** Wang, Wang & Wishart 2010, doi:10.1007/s10858-010-9407-y
- **Source code:** Closed source (Java JAR only)
- **JAR file:** https://github.com/bmrb-io/BMRB-API/blob/master/server/wsgi/bmrbapi/submodules/panav/panav.jar
- **BMRB API:** https://api.bmrb.io/v2/entry/{ID}/validate (returns PANAV + AVS, ~1.8s/entry)
- **Local JAR:** `tools/panav.jar` (103KB, run with `java -cp panav.jar CLI -f star -i <file> -j`)
- **Local data:** `data/panav_offsets.json` (computed locally, 17,388 entries)
- **Atom types for offsets:** CO, CA, CB, N (no H, HA)

### CheZOD / TriZOD Offset Correction
- **What:** AIC-based offset detection using POTENCI random coil predictions as reference
- **Paper:** Nielsen & Mulder 2016, doi:10.3389/fmolb.2016.00004
- **Source code:** https://github.com/protein-nmr/CheZOD
- **Our implementation:** `trizod/scoring/scoring.py` (`compute_offsets`, `compute_running_offsets`)
- **Atom types:** All 7 backbone

## Shift Prediction Tools

### POTENCI
- **What:** Predicts random coil chemical shifts accounting for sequence, temperature, pH, ionic strength
- **Paper:** Nielsen & Mulder 2018, doi:10.1007/s10858-018-0175-y
- **Source code:** https://github.com/protein-nmr/POTENCI (Python)
- **Our copy:** `trizod/potenci/` (modernized, CSV data tables)

### SPARTA+
- **What:** Predicts chemical shifts from 3D protein structure (backbone torsion angles, ring currents, H-bonds)
- **Paper:** Shen & Bax 2010, doi:10.1007/s10858-010-9433-9
- **Source:** https://spin.niddk.nih.gov/bax/software/SPARTA+/
- **BMRB pre-computed:** https://bmrb.io/ftp/pub/bmrb/validation_reports/SPARTA/
- **Requires:** PDB 3D structure (not applicable to TriZOD)

### ncIDP
- **What:** Neighbor-corrected IDP random coil chemical shift library (predecessor to POTENCI)
- **Paper:** Tamiola et al. 2010, doi:10.1021/ja105656t
- **Note:** Superseded by POTENCI. The original CheZOD sigma values were derived using ncIDP.

## Validation Tools

### AVS (Assignment Validation Suite)
- **What:** Checks if individual shifts are plausible for their amino acid type (flags consistent/suspicious/outlier)
- **BMRB API:** returned alongside PANAV from https://api.bmrb.io/v2/entry/{ID}/validate
- **BMRB pre-computed:** https://bmrb.io/ftp/pub/bmrb/validation_reports/AVS/
- **Note:** Detects individual mis-assignments, not systematic referencing errors

### CheSPI (Chemical Shift Secondary structure Population Inference)
- **What:** Secondary structure and disorder prediction from chemical shifts (includes CheZOD scoring)
- **Paper:** Nielsen & Mulder 2021, doi:10.1007/s10858-021-00374-w
- **Source code:** https://github.com/protein-nmr/CheSPI
- **Note:** Uses same POTENCI+AIC offset approach as our scoring.py

## BMRB Resources

- **BMRB main:** https://bmrb.io/
- **Validation portal:** https://bmrb.io/validate/
- **FTP validation reports:** https://bmrb.io/ftp/pub/bmrb/validation_reports/
- **API docs:** https://api.bmrb.io/v2/
- **API source:** https://github.com/bmrb-io/BMRB-API
- **Chemical shift statistics:** https://bmrb.io/ref_info/csstats.php
