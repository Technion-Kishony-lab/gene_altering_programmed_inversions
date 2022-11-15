import os
import os.path
import pickle

import numpy as np
import pandas as pd

from generic import generic_utils
from searching_for_pis import py_repeats_finder

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

def get_pairs_fully_contained_in_other_ir_pairs(pairs_df):
    pairs_df = pairs_df.copy()
    column_name_to_other_column_name = {x: f'other_{x}' for x in list(pairs_df)}
    other_pair_column_names = list(column_name_to_other_column_name.values())

    pairs_df['dummy_for_cross_product_merge'] = 0
    merged_df = pairs_df.merge(pairs_df.rename(columns=column_name_to_other_column_name)).drop('dummy_for_cross_product_merge', axis=1)
    merged_df = merged_df[
        (merged_df['left1'] != merged_df['other_left1']) |
        (merged_df['right1'] != merged_df['other_right1']) |
        (merged_df['left2'] != merged_df['other_left2']) |
        (merged_df['right2'] != merged_df['other_right2'])
    ]
    pairs_fully_contained_in_other_pairs_df = merged_df[
        (merged_df['other_left1'] <= merged_df['left1']) &
        (merged_df['other_left2'] <= merged_df['left2']) &
        (merged_df['right1'] <= merged_df['other_right1']) &
        (merged_df['right2'] <= merged_df['other_right2'])
    ].drop(other_pair_column_names, axis=1)

    return pairs_fully_contained_in_other_pairs_df

def discard_ir_pairs_fully_contained_in_other_ir_pairs(pairs_df):
    orig_num_of_pairs = len(pairs_df)
    CHUNK_SIZE = 1000
    num_of_chunks = np.ceil(orig_num_of_pairs / CHUNK_SIZE)
    if num_of_chunks <= 1:
        pairs_fully_contained_in_other_pairs_df = get_pairs_fully_contained_in_other_ir_pairs(pairs_df)
    else:
        # Why is all of this needed? because sometimes the number of IR pairs is big enough such that the cross product in get_pairs_fully_contained_in_other_ir_pairs
        # is really not a good idea (even with a reasonable amount of RAM).
        all_repeat1s_region = (pairs_df['left1'].min(), pairs_df['right1'].max())
        chunk_edge_indices = np.arange(0, orig_num_of_pairs, CHUNK_SIZE)
        chunk_edges = pairs_df['left1'].sort_values().iloc[chunk_edge_indices]
        chunk_edges = list(chunk_edges) + [all_repeat1s_region[1] + 1]
        assert chunk_edges[0] == all_repeat1s_region[0]
        generic_utils.print_and_write_to_log(f'(discard_ir_pairs_fully_contained_in_other_ir_pairs) splitting to chunks. num_of_chunks: {num_of_chunks}, '
                                             f'chunk_edges: {chunk_edges}')
        assert pd.Series(chunk_edges).is_monotonic_increasing
        chunk_pairs_fully_contained_in_other_pairs_dfs = []
        for region_that_curr_repeat1s_must_overlap_start, region_that_curr_repeat1s_must_overlap_end in zip(chunk_edges[:-1], chunk_edges[1:]):
            # if a pair is covered by another pair, then their repeat1s must overlap in at least one base, obviously, so both will belong to curr_pairs_df at least
            # for one region_that_curr_repeat1s_must_overlap. alright.
            curr_pairs_df = pairs_df[(pairs_df['right1'] >= region_that_curr_repeat1s_must_overlap_start) &
                                     (pairs_df['left1'] < region_that_curr_repeat1s_must_overlap_end)]
            chunk_pairs_fully_contained_in_other_pairs_dfs.append(get_pairs_fully_contained_in_other_ir_pairs(curr_pairs_df))
        pairs_fully_contained_in_other_pairs_df = pd.concat(chunk_pairs_fully_contained_in_other_pairs_dfs, ignore_index=True).drop_duplicates()

    filtered_pairs_df = pairs_df.merge(pairs_fully_contained_in_other_pairs_df, how='left', indicator=True)
    filtered_pairs_df = filtered_pairs_df[filtered_pairs_df['_merge'] == 'left_only'].drop('_merge', axis=1)
    return filtered_pairs_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_entry_ir_pairs(
        input_file_path_fasta,
        seed_len,
        min_repeat_len,
        max_spacer_len,
        min_spacer_len,
        max_evalue,
        inverted_or_direct_or_both,
        output_file_path_ir_pairs_csv,
        output_file_path_ir_pairs_minimal_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with generic_utils.timing_context_manager(f'cached_write_nuccore_entry_ir_pairs'):
        orig_ir_pairs_df_csv_file_path = f'{output_file_path_ir_pairs_csv}.orig.csv'
        py_repeats_finder.find_imperfect_repeats_pairs(
            input_file_path_fasta=input_file_path_fasta,
            seed_len=seed_len,
            min_repeat_len=min_repeat_len,
            max_spacer_len=max_spacer_len,
            min_spacer_len=min_spacer_len,
            max_evalue=max_evalue,
            inverted_or_direct_or_both=inverted_or_direct_or_both,
            output_file_path_imperfect_repeats_pairs_csv=orig_ir_pairs_df_csv_file_path,
        )
        ir_pairs_df = py_repeats_finder.read_repeats_pairs_df_from_csv_in_my_blast_like_format(orig_ir_pairs_df_csv_file_path)[[
            'left1', 'right1', 'left2', 'right2', 'mismatch']].drop_duplicates()
        filtered_ir_pairs_df = discard_ir_pairs_fully_contained_in_other_ir_pairs(ir_pairs_df)
        filtered_ir_pairs_df.to_csv(output_file_path_ir_pairs_csv, sep='\t', index=False)

        num_of_filtered_ir_pairs = len(filtered_ir_pairs_df)

        ir_pairs_minimal_info = {
            'orig_ir_pairs_csv_file_path': orig_ir_pairs_df_csv_file_path,
            'filtered_ir_pairs_csv_file_path': output_file_path_ir_pairs_csv,
            'num_of_filtered_ir_pairs': num_of_filtered_ir_pairs,
            'num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs': (len(ir_pairs_df) - num_of_filtered_ir_pairs),
        }
        with open(output_file_path_ir_pairs_minimal_info_pickle, 'wb') as f:
            pickle.dump(ir_pairs_minimal_info, f, protocol=4)

def write_nuccore_entry_ir_pairs(
        input_file_path_fasta,
        seed_len,
        min_repeat_len,
        max_spacer_len,
        min_spacer_len,
        max_evalue,
        inverted_or_direct_or_both,
        output_file_path_ir_pairs_csv,
        output_file_path_ir_pairs_minimal_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_entry_ir_pairs(
        input_file_path_fasta=input_file_path_fasta,
        seed_len=seed_len,
        min_repeat_len=min_repeat_len,
        max_spacer_len=max_spacer_len,
        min_spacer_len=min_spacer_len,
        max_evalue=max_evalue,
        inverted_or_direct_or_both=inverted_or_direct_or_both,
        output_file_path_ir_pairs_csv=output_file_path_ir_pairs_csv,
        output_file_path_ir_pairs_minimal_info_pickle=output_file_path_ir_pairs_minimal_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=7,
    )

def get_nuccore_entry_ir_pairs_info(
        nuccore_entry_info,
        nuccore_entry_ir_pairs_root_dir_path,
        search_for_pis_args,
        nuccore_uid,
):
    chrom_len = nuccore_entry_info['chrom_len']
    fasta_file_path = nuccore_entry_info['fasta_file_path']
    ir_pairs_csv_file_path = os.path.join(nuccore_entry_ir_pairs_root_dir_path, f'ir_pairs.csv')
    ir_pairs_minimal_info_pickle_file_path = os.path.join(nuccore_entry_ir_pairs_root_dir_path, f'ir_pairs_minimal_info.pickle')

    write_nuccore_entry_ir_pairs(
        input_file_path_fasta=fasta_file_path,
        seed_len=search_for_pis_args['stage2']['repeat_pairs']['seed_len'],
        min_repeat_len=search_for_pis_args['stage2']['repeat_pairs']['min_repeat_len'],
        max_spacer_len=search_for_pis_args['stage2']['repeat_pairs']['max_spacer_len'],
        min_spacer_len=search_for_pis_args['stage2']['repeat_pairs']['min_spacer_len'],
        max_evalue=search_for_pis_args['stage2']['repeat_pairs']['max_evalue'],
        inverted_or_direct_or_both='inverted',
        output_file_path_ir_pairs_csv=ir_pairs_csv_file_path,
        output_file_path_ir_pairs_minimal_info_pickle=ir_pairs_minimal_info_pickle_file_path,
    )

    with open(ir_pairs_minimal_info_pickle_file_path, 'rb') as f:
        ir_pairs_minimal_info = pickle.load(f)

    ir_pairs_info = {
        'ir_pairs_minimal_info': ir_pairs_minimal_info,
        'nuccore_total_num_of_base_pairs_that_were_searched': chrom_len,
    }

    return ir_pairs_info
