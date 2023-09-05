import pytest
import os,sys
import logging
import numpy as np
import re

import pprint
sys.path.append("..")

import TriZOD.potenci as potenci
import TriZOD.trizod as trizod
import TriZOD.bmrb as bmrb

class BiostarV2FileMissing(Exception):
    pass
class PotenciError(Exception):
    pass
class TriZODError(Exception):
    pass
class ChiZODError(Exception):
    pass
class BMRBEntryError(Exception):
    pass
class ZscoreDiversion(Exception):
    pass

bmrb_dir = os.path.join(os.path.dirname(__file__), 'data', 'bmrb_entries')
ids_to_test = [int(dn[3:]) for dn in os.listdir(os.path.join(os.path.dirname(__file__), 'data', 'CheZOD_results_all')) if dn.startswith('bmr')]
#ids_to_test = [id_ for id_ in ids_to_test if os.path.exists(os.path.join(bmrb_dir, f"bmr{id_}", f"bmr{id_}_3.str"))]
ids_value_error = [
'6753',
'27272',
'26904',
'26953',
]

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

def parse_shift_file(fn, seq=None):
    '''
    Parses a shift data file ceated by CheZOD and returns a set of all parsed shifts
    '''
    shift_data = {}
    offsets = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            if line.startswith('off'):
                _,at,val = line.strip().split()
                val = float(val)
                offsets[at] = val
                continue
            pos,aa,at,val,_,_ = line.strip().split() #po is 1-based
            pos,val = int(pos), float(val)
            if seq:
                if seq[pos-1].upper() != aa.upper():
                    logging.warning(f'amino acid mismatch at position {pos-1}')
                    raise ValueError(f'amino acid mismatch at position {pos-1}')
            if (pos-1, at) in shift_data:
                logging.warning(f'duplicate shift data entries')
                raise ValueError(f'duplicate shift data entries')
            shift_data[(pos-1, at)] = val
    return shift_data, offsets
    
#@pytest.mark.parametrize("id_", ids_to_test)
def test_equaltity_to_chezod(id_):
    '''Checks if any protein-shift combination in the corresponding BMRB entry produces
    equivalent Z-scores as those output by the original CheZOD script'''
    str_fn = os.path.join(os.path.dirname(__file__), 'data', 'CheZOD_results_all', f'bmr{id_}', f'bmr{id_}.str')
    if not os.path.exists(str_fn):
        logging.warning(f"skipping entry {id_}, Biostar V2 file not found")
        if __name__ == '__main__':
            raise BiostarV2FileMissing
        return
    zscores_fn = os.path.join(os.path.dirname(__file__), 'data', 'CheZOD_results_all', f'bmr{id_}', f'zscores{id_}.txt')
    if not os.path.exists(zscores_fn):
        logging.warning(f"skipping entry {id_}, CheZOD zscore file not found")
        if __name__ == '__main__':
            raise ChiZODError
        return 
    shifts_fn = os.path.join(os.path.dirname(__file__), 'data', 'CheZOD_results_all', f'bmr{id_}', f'shifts{id_}.txt')
    potenci_fn = [fn for fn in os.listdir(os.path.join(os.path.dirname(__file__), 'data', 'CheZOD_results_all', f'bmr{id_}')) if fn.startswith(f'outPOTENCI')][0]
    m = re.search("_T([^_]*)_I([^_]*)_pH([^_]*)\.", potenci_fn)
    chezod_conditions = (
        float(m.groups()[0]),
        float(m.groups()[1]),
        float(m.groups()[2])
    )
    try:
        entry = bmrb.BmrbEntry(id_, bmrb_dir)
    except:
        logging.warning(f"skipping entry {id_}, BMRB entry could not be parsed")
        if __name__ == '__main__':
            raise BMRBEntryError
        return 
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
        try:
            chezod_shifts, chezod_offsets = parse_shift_file(shifts_fn, seq=seq)
        except ValueError:
            logging.warning(f"skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)}, duplicates or sequence mismatch in chezod shifts file")
            continue
        # check if shift data is identical
        bbshifts,_,_ = bmrb.get_valid_bbshifts(shifts, seq)
        trizod_shifts = {(pos,at) : bbshifts[pos][at][0] for pos in bbshifts for at in bbshifts[pos] if pos not in [0,len(seq)-1]}
        if set(chezod_shifts.keys()) != set(trizod_shifts.keys()):
            logging.warning(f"skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)}, backbone shift count not identical")
            #breakpoint()
            continue
        all_close = True
        for key in chezod_shifts.keys():
            if not np.isclose(chezod_shifts[key], trizod_shifts[key], atol=0.1):
                break
        else:
            all_close = True
        if not all_close:
            logging.warning(f"skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)}, backbone shifts not identical")
            continue
            #breakpoint()
        # use POTENCI to predict shifts
        ion = entry.conditions[condID].get_ionic_strength()
        pH = entry.conditions[condID].get_pH()
        temperature = entry.conditions[condID].get_temperature()
        if not np.allclose(np.array((temperature, ion, pH)), np.array(chezod_conditions), atol=0.01):
            logging.warning(f"skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)}, experiment conditions do not match")
            continue
        # reject if out of certain ranges
        #if ion > 3. or pH > 13. or temperature < 273.15 or temperature > 373.15:
        #    logging.error(f"skipping {(stID, condID, assemID, assem_entityID, entityID)} due to extreme experiment conditions")
        #    continue
        # predict random coil chemical shifts using POTENCI
        usephcor = pH != 7.0
        try:
            predshiftdct = potenci.getpredshifts(seq,temperature,pH,ion,usephcor,pkacsvfile=None)
        except:
            logging.error(f"POTENCI failed for {(stID, condID, assemID, assem_entityID, entityID)} due to the following error:", exc_info=True)
            if __name__ == '__main__':
                raise PotenciError
            continue
        ret = trizod.get_offset_corrected_wSCS(seq, shifts, predshiftdct)
        if ret is None:
            logging.warning(f'skipping shifts for {(stID, condID, assemID, assem_entityID, entityID)} due to a previous error')
            if __name__ == '__main__':
                raise TriZODError
            continue
        shw, ashwi, cmp_mask, olf, offf = ret
        ashwi3, k3 = trizod.convert_to_triplet_data(ashwi, cmp_mask)
        zscores = trizod.compute_zscores(ashwi3, k3, cmp_mask)
        # compare zscores
        try:
            np.testing.assert_allclose(chezod_zscores, zscores, atol=0.1, equal_nan=True)
            break
        except:
            logging.exception(f'{id_}, {(id_, stID, condID, assemID, assem_entityID, entityID)}: Z-scores diverge substantially between CheZOD and TriZOD')
            if __name__ == '__main__':
                raise ZscoreDiversion(f'Z-scores diverge substantially between CheZOD and TriZOD')
    else:
        raise AssertionError(f'No matching entities found for BMRB entry {id_}')
    return (id_, stID, condID, assemID, assem_entityID, entityID)

if __name__ == '__main__':
    logfile = os.path.join(os.path.dirname(__file__), "test_results.csv")
    with open(logfile, 'w') as lf:
        for id_ in ids_to_test:
        #for id_ in ids_value_error:
            try:
                entity_info = test_equaltity_to_chezod(id_)
                print("\t".join([str(id_), "pass", ",".join([str(i) for i in entity_info])]), file=lf)
            except Exception as ex:
                print("\t".join([str(id_),"fail",ex.__class__.__name__]), file=lf)