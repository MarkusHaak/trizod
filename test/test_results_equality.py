import os,sys
import logging
import numpy as np
import pandas as pd
import argparse
from trizod.constants import BBATNS

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('results_file',
        help='')
    parser.add_argument('results_comp',
        help='')

    args = parser.parse_args()
    return args

def main():
    df1 = pd.read_csv(args.results_file, index_col=[0,1,2,3,4,5])
    df2 = pd.read_csv(args.results_comp, index_col=[0,1,2,3,4,5])
    diff1 = df1.index.difference(df2.index)
    logging.info(f"results file contains {diff1.size} entries that are not in the comparison file")
    diff2 = df2.index.difference(df1.index)
    logging.info(f"comparison file contains {diff2.size} entries that are not in the results file")
    idx = df1.index.intersection(df2.index)
    logging.info(f"{idx.size} entries are comparible")
    close = np.isclose(df1.loc[idx].scores, df2.loc[idx].scores, equal_nan=True)
    breakpoint()

if __name__ == '__main__':
    level = logging.INFO
    logging.basicConfig(level=level, format=f'%(levelname)s : %(message)s')
    args = parse_args()
    main()