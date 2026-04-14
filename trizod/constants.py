BACKBONE_ATOMS = ["C", "CA", "CB", "HA", "H", "N", "HB"]
# Expected RMSD of secondary chemical shifts (observed - POTENCI predicted) for
# intrinsically disordered residues, in ppm. These are the POTENCI-updated
# equivalents of the sigma values in Eq. 2 of Nielsen & Mulder 2016
# (doi:10.3389/fmolb.2016.00004), computed on a 117-entry IDP reference set
# (13,069 residues) using POTENCI instead of ncIDP as the random coil predictor.
# Values taken from the original CheZOD source code:
# https://github.com/protein-nmr/CheZOD/blob/master/chezod1_1.py
REFINED_WEIGHTS = {
    "C": 0.1846,
    "CA": 0.1982,
    "CB": 0.1544,
    "HA": 0.02631,
    "H": 0.06708,
    "N": 0.4722,
    "HB": 0.02154,
}
AA3TO1 = {
    "CYS": "C",
    "GLN": "Q",
    "ILE": "I",
    "SER": "S",
    "VAL": "V",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "LYS": "K",
    "THR": "T",
    "PHE": "F",
    "ALA": "A",
    "HIS": "H",
    "GLY": "G",
    "ASP": "D",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "GLU": "E",
    "TYR": "Y",
}
AA1TO3 = {v: k for k, v in AA3TO1.items()}
CANONICAL_AA_MASK = str.maketrans("ARNDCQEGHILKMFPSTYWVX", "#####################")
