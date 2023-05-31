import pytest
import os,sys
import logging
import numpy as np

import pprint
pprint.pprint(sys.path)
sys.path.append("..")

import TriZOD.potenci as potenci
import TriZOD.trizod as trizod
import TriZOD.bmrb as bmrb

ids_to_test = [int(fn[7:-4]) for fn in os.listdir(os.path.join('data', 'CheZOD_results')) if fn.startswith('zscores')]
bmrb_dir = os.path.join('data', 'bmrb_entries')

def parse_zscore_file(fn, seq=None, fill=999.):
    """
    Parses a zscore file and returns a (gap-filled if fill != False, fill value = fill) list of zscores
    and the number of gaps compared to the sequence seq.
    """
    vals = []
    gaps = 0
    seq_ = []
    with open(fn, 'r') as f:
        i = 1
        for line in f.readlines():
            if line:
                aa,pos,val = line.strip().split()
                pos, val = int(pos), float(val)
                if seq:
                    if seq[pos-1].upper() != aa.upper():
                        #print(f"Error: AA mismatch at position {i} of {id_}: ", seq[pos-1].upper(), "!=", aa.upper())
                        raise ValueError
                if fill is not False:
                    while i < pos:
                        vals.append(fill)
                        i += 1
                        gaps += 1
                        seq_.append(" ")
                vals.append(val)
                seq_.append(aa)
                i += 1
    if seq and fill is not False:
        while len(vals) < len(seq):
            vals.append(fill)
            gaps += 1
            seq_.append(" ")
    return vals, gaps, "".join(seq_).upper()

#@pytest.mark.parametrize("id_", ids_to_test)
def test_equaltity_to_chezod(id_):
    '''Checks if any protein-shift combination in the corresponding BMRB entry produces
    equivalent Z-scores as those output by the original CheZOD script'''
    zscores_fn = os.path.join('data', 'CheZOD_results', f'zscores{id_}.txt')
    entry = bmrb.BmrbEntry(id_, bmrb_dir)
    peptide_shifts = entry.get_peptide_shifts()
    for (stID, condID, assemID, assem_entityID, entityID), shifts in peptide_shifts.items():
        # get polymer sequence and chemical backbone shifts
        seq = entry.entities[entityID].seq
        if seq is None:
            logging.warning(f"skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)}, no sequence information")
            continue
        elif len(seq) < 5:
            logging.warning(f"skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)}, sequence shorter than 5 residues")
            continue
        # try to parse CheZOD zscores with this seq - fails in case of amino acid mismatches!
        try:
            chezod_zscores,_,_ = parse_zscore_file(zscores_fn, seq=seq, fill=np.nan)
            chezod_zscores = np.array(chezod_zscores)
        except ValueError:
            logging.warning(f"skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)}, sequence mismatch")
            continue
        # use POTENCI to predict shifts
        ion = entry.conditions[condID].get_ionic_strength()
        pH = entry.conditions[condID].get_pH()
        temperature = entry.conditions[condID].get_temperature()
        # reject if out of certain ranges
        if ion > 3. or pH > 13. or temperature < 273.15 or temperature > 373.15:
            logging.error(f"skipping {(stID, condID, assemID, assem_entityID, entityID)} due to extreme experiment conditions")
            continue
        # predict random coil chemical shifts using POTENCI
        usephcor = pH != 7.0
        try:
            predshiftdct = potenci.getpredshifts(seq,temperature,pH,ion,usephcor,pkacsvfile=None)
        except:
            logging.error(f"POTENCI failed for {(stID, condID, assemID, assem_entityID, entityID)} due to the following error:", exc_info=True)
            continue
        ret = trizod.get_offset_corrected_wSCS(seq, shifts, predshiftdct)
        if ret is None:
            logging.warning(f'skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)} due to a previous error')
            continue
        shw, ashwi, cmp_mask, olf, offf = ret
        ashwi3, k3 = trizod.convert_to_triplet_data(ashwi, cmp_mask)
        zscores = trizod.compute_zscores(ashwi3, k3, cmp_mask)
        # compare zscores
        try:
            np.testing.assert_allclose(chezod_zscores, zscores, rtol=0.1, equal_nan=True)
            break
        except:
            pass
    else:
        breakpoint()
        #raise AssertionError(f'No matching zscores could be produced for BMRB entry {id_}')


for id_ in ids_to_test:
    #try:
    test_equaltity_to_chezod(id_)
    #except:
    #    logging.error(f'failed: {id_}')