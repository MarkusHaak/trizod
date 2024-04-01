BBATNS = ['C','CA','CB','HA','H','N','HB']
REFINED_WEIGHTS = {'C':0.1846, 'CA':0.1982, 'CB':0.1544, 'HA':0.02631, 'H':0.06708, 'N':0.4722, 'HB':0.02154}
AA3TO1 = {'CYS': 'C', 'GLN': 'Q', 'ILE': 'I', 'SER': 'S', 'VAL': 'V', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'LYS': 'K', 'THR': 'T', 'PHE': 'F', 'ALA': 'A', 'HIS': 'H', 'GLY': 'G', 'ASP': 'D', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'GLU': 'E', 'TYR': 'Y'}
AA1TO3 = {v:k for k,v in AA3TO1.items()}
CAN_TRANS = str.maketrans("ARNDCQEGHILKMFPSTYWVX", "#####################")