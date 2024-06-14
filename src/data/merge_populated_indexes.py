"""
Merge two populated indexes into one.
Example usage:
    python merge_populated_indexes.py --populated-index-first ../../data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated_0_to_499.index \ 
        --populated-index-second ../../data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated_500_to_999.index \
        --output-index ../../data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index
"""

import faiss
import json
import numpy as np
import time
from pathlib import Path
from typing import List

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--populated-index-first", required=True, type=Path)
parser.add_argument("--populated-index-second", required=True, type=Path)
parser.add_argument("--output-index", required=True, type=Path)
args = parser.parse_args()

def merge_invlists(il_src, il_dest):
    """ 
    merge inverted lists from two ArrayInvertedLists 
    may be added to main Faiss at some point
    """
    assert il_src.nlist == il_dest.nlist
    assert il_src.code_size == il_dest.code_size
    
    start = time.time()
    for list_no in range(il_src.nlist): 
        # print("Merging list {}".format(list_no))
        il_dest.add_entries(
            list_no,
            il_src.list_size(list_no), 
            il_src.get_ids(list_no), 
            il_src.get_codes(list_no)
        )   
    print("Merging took {} seconds".format(time.time() - start))

# Load empty (but trained) index
start = time.time()
print("Loading first index...")
first_index = faiss.read_index(str(args.populated_index_first))
print("Loading second index...")
second_index = faiss.read_index(str(args.populated_index_second))
print("Loading indexes took {} seconds".format(time.time() - start))

ntotal_first = first_index.ntotal
ntotal_second = second_index.ntotal
print("First index has {} vectors".format(ntotal_first))
print("Second index has {} vectors".format(ntotal_second))

assert first_index.nlist == second_index.nlist, "Indexes must have same number of clusters"

# https://gist.github.com/mdouze/7331e6fc1da2334f30706b9b9962068b
print("Merging indexes...")
merge_invlists(
    faiss.extract_index_ivf(second_index).invlists,
    faiss.extract_index_ivf(first_index).invlists 
)
print(first_index.ntotal)
first_index.ntotal = faiss.extract_index_ivf(first_index).ntotal =  ntotal_first + ntotal_second 

print("Saving index...")
faiss.write_index(first_index, str(args.output_index))
end = time.time()
print("Total time elapsed: {} seconds".format(end - start))