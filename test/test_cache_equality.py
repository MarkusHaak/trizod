import os,sys
import logging
import numpy as np
import pandas as pd
import argparse
from trizod.constants import BBATNS

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('cache_dir',
        help='Cache directory that shall be tested.')
    parser.add_argument('cache_comp',
        help='Cache directory to compare against.')

    args = parser.parse_args()
    return args

def main():
    cache_files = {f for f in os.listdir(os.path.join(args.cache_dir, 'wSCS')) if os.path.isfile(os.path.join(args.cache_dir, 'wSCS', f)) and f.endswith('.npz')}
    comp_files = {f for f in os.listdir(os.path.join(args.cache_comp, 'wSCS')) if os.path.isfile(os.path.join(args.cache_comp, 'wSCS', f)) and f.endswith('.npz')}
    common_files = list(cache_files & comp_files)
    common_files.sort()
    print(len(common_files))
    res = []
    for fn in common_files:
        with open(os.path.join(args.cache_dir, 'wSCS', fn), 'rb') as f1, open(os.path.join(args.cache_comp, 'wSCS', fn), 'rb') as f2:
            z = np.load(f1)
            shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = z['shw'], z['ashwi'], z['cmp_mask'], z['olf'], z['offf'], z['shw0'], z['ashwi0'], z['ol0'], z['off0']
            offf, off0 = {at:off for at,off in zip(BBATNS, offf)}, {at:off for at,off in zip(BBATNS, off0)}
            
            z =  np.load(f2)
            shw_, ashwi_, cmp_mask_, olf_, offf_, shw0_, ashwi0_, ol0_, off0_ = z['shw'], z['ashwi'], z['cmp_mask'], z['olf'], z['offf'], z['shw0'], z['ashwi0'], z['ol0'], z['off0']
            offf_, off0_ = {at:off for at,off in zip(BBATNS, offf_)}, {at:off for at,off in zip(BBATNS, off0_)}
        if ashwi.shape != ashwi_.shape:
            res.append("shape mismatch")
            continue
        if not np.allclose(ashwi, ashwi_):
            res.append('not close')
            continue
        res.append('close')
    df = pd.DataFrame(res, columns=['result'], index=common_files)
    breakpoint()

if __name__ == '__main__':
    level = logging.INFO
    logging.basicConfig(level=level, format=f'%(levelname)s : %(message)s')
    args = parse_args()
    main()