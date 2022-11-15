import logging
import os
import os.path
import pathlib
import pickle

import numpy as np
import pandas as pd

from generic import generic_utils
from searching_for_pis import find_all_ir_pairs
from searching_for_pis import massive_screening_stage_1
from searching_for_pis import massive_screening_configuration

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

DO_STAGE2 = True
# DO_STAGE2 = False

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_all_ir_pairs_df_csv(
        input_file_path_filtered_nuccore_accession_to_ir_pairs_info_pickle,
        output_file_path_all_ir_pairs_df_csv,
):
    with open(input_file_path_filtered_nuccore_accession_to_ir_pairs_info_pickle, 'rb') as f:
        filtered_nuccore_accession_to_ir_pairs_info = pickle.load(f)

    num_of_nuccore_entries = len(filtered_nuccore_accession_to_ir_pairs_info)

    ir_pairs_dfs = []
    for i, (nuccore_accession, ir_pairs_info) in enumerate(filtered_nuccore_accession_to_ir_pairs_info.items()):
        generic_utils.print_and_write_to_log(f'starting work on nuccore {i + 1}/{num_of_nuccore_entries}: {nuccore_accession}')
        ir_pairs_minimal_info = ir_pairs_info['ir_pairs_minimal_info']
        ir_pairs_csv_file_path = ir_pairs_minimal_info['filtered_ir_pairs_csv_file_path']
        curr_ir_pairs_df = pd.read_csv(ir_pairs_csv_file_path, sep='\t', low_memory=False)
        curr_ir_pairs_df = curr_ir_pairs_df[['left1', 'right1', 'left2', 'right2', 'mismatch']].reset_index().rename(
            columns={'index': 'index_in_nuccore_ir_pairs_df_csv_file',
                     'mismatch': 'num_of_mismatches'})
        curr_ir_pairs_df['nuccore_accession'] = nuccore_accession
        ir_pairs_dfs.append(curr_ir_pairs_df)

    final_column_names = ['left1', 'right1', 'left2', 'right2', 'index_in_nuccore_ir_pairs_df_csv_file', 'num_of_mismatches', 'nuccore_accession']
    if ir_pairs_dfs:
        all_ir_pairs_df_csv = pd.concat(ir_pairs_dfs)
        assert set(all_ir_pairs_df_csv) == set(final_column_names)
    else:
        all_ir_pairs_df_csv = pd.DataFrame([], columns=final_column_names)


    all_ir_pairs_df_csv.to_csv(output_file_path_all_ir_pairs_df_csv, sep='\t', index=False)

def write_all_ir_pairs_df_csv(
        input_file_path_filtered_nuccore_accession_to_ir_pairs_info_pickle,
        output_file_path_all_ir_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_all_ir_pairs_df_csv(
        input_file_path_filtered_nuccore_accession_to_ir_pairs_info_pickle=input_file_path_filtered_nuccore_accession_to_ir_pairs_info_pickle,
        output_file_path_all_ir_pairs_df_csv=output_file_path_all_ir_pairs_df_csv,
    )

def do_massive_screening_stage2(
        search_for_pis_args,
):
    massive_screening_stage2_out_dir_path = search_for_pis_args['stage2']['output_dir_path']

    debug___num_of_taxa_to_go_over = search_for_pis_args['debug___num_of_taxa_to_go_over']
    debug___num_of_nuccore_entries_to_go_over = search_for_pis_args['debug___num_of_nuccore_entries_to_go_over']
    debug___taxon_uids = search_for_pis_args['debug___taxon_uids']

    stage_out_file_name_suffix = massive_screening_stage_1.get_stage_out_file_name_suffix(
        debug___num_of_taxa_to_go_over=debug___num_of_taxa_to_go_over,
        debug___taxon_uids=debug___taxon_uids,
    )

    pathlib.Path(massive_screening_stage2_out_dir_path).mkdir(parents=True, exist_ok=True)
    massive_screening_log_file_path = os.path.join(massive_screening_stage2_out_dir_path, 'massive_screening_stage2_log.txt')
    logging.basicConfig(filename=massive_screening_log_file_path, level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug(f'---------------starting do_massive_screening_stage2({massive_screening_stage2_out_dir_path})---------------')

    # stage1_results_info = massive_screening_stages_1_and_2.do_massive_screening_stage1(
    #     search_for_pis_args,
    #     debug___num_of_taxa_to_go_over=stage1_debug___num_of_taxa_to_go_over,
    # )
    stage1_results_info_pickle_file_path = os.path.join(search_for_pis_args['stage1']['output_dir_path'], search_for_pis_args['stage1']['results_pickle_file_name'])
    stage1_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage1_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage1_results_info_pickle_file_path, 'rb') as f:
        stage1_results_info = pickle.load(f)

    all_ir_pairs_extended_info = find_all_ir_pairs.find_all_ir_pairs(
        stage1_results_info=stage1_results_info,
        output_dir_path=massive_screening_stage2_out_dir_path,
        search_for_pis_args=search_for_pis_args,
        debug___num_of_nuccore_entries_to_go_over=debug___num_of_nuccore_entries_to_go_over,
    )

    all_ir_pairs_info = all_ir_pairs_extended_info['all_ir_pairs_info']
    nuccore_accession_to_ir_pairs_info = all_ir_pairs_extended_info['nuccore_accession_to_ir_pairs_info']
    filtered_nuccore_accession_to_ir_pairs_info = all_ir_pairs_extended_info['filtered_nuccore_accession_to_ir_pairs_info']


    all_ir_pairs_info_pickle_file_path = os.path.join(massive_screening_stage2_out_dir_path, 'all_ir_pairs_info.pickle')
    all_ir_pairs_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        all_ir_pairs_info_pickle_file_path, stage_out_file_name_suffix)
    with open(all_ir_pairs_info_pickle_file_path, 'wb') as f:
        pickle.dump(all_ir_pairs_info, f, protocol=4)

    nuccore_accession_to_ir_pairs_info_pickle_file_path = os.path.join(massive_screening_stage2_out_dir_path, 'nuccore_accession_to_ir_pairs_info.pickle')
    nuccore_accession_to_ir_pairs_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        nuccore_accession_to_ir_pairs_info_pickle_file_path, stage_out_file_name_suffix)
    with open(nuccore_accession_to_ir_pairs_info_pickle_file_path, 'wb') as f:
        pickle.dump(nuccore_accession_to_ir_pairs_info, f, protocol=4)

    filtered_nuccore_accession_to_ir_pairs_info_pickle_file_path = os.path.join(massive_screening_stage2_out_dir_path, 'filtered_nuccore_accession_to_ir_pairs_info.pickle')
    filtered_nuccore_accession_to_ir_pairs_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        filtered_nuccore_accession_to_ir_pairs_info_pickle_file_path, stage_out_file_name_suffix)
    with open(filtered_nuccore_accession_to_ir_pairs_info_pickle_file_path, 'wb') as f:
        pickle.dump(filtered_nuccore_accession_to_ir_pairs_info, f, protocol=4)

    all_ir_pairs_df_csv_file_path = os.path.join(massive_screening_stage2_out_dir_path, 'all_ir_pairs_df.csv')
    all_ir_pairs_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        all_ir_pairs_df_csv_file_path, stage_out_file_name_suffix)

    write_all_ir_pairs_df_csv(
        input_file_path_filtered_nuccore_accession_to_ir_pairs_info_pickle=filtered_nuccore_accession_to_ir_pairs_info_pickle_file_path,
        output_file_path_all_ir_pairs_df_csv=all_ir_pairs_df_csv_file_path,
    )
    all_ir_pairs_df = pd.read_csv(all_ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
    num_of_all_ir_pairs = len(all_ir_pairs_df)
    print(f'num_of_all_ir_pairs: {num_of_all_ir_pairs}')


    if not all_ir_pairs_df.empty:
        # a sanity check to make sure the BLAST max evalue argument is high enough so that we don't miss short repeats for very long nuccores.
        nuccore_df_csv_file_path = stage1_results_info['nuccore_df_csv_file_path']
        nuccore_df = pd.read_csv(nuccore_df_csv_file_path, sep='\t', low_memory=False)
        relevant_nuccore_df = nuccore_df[nuccore_df['num_of_filtered_cds_features'] > 0]
        longest_relevant_nuccore_row = relevant_nuccore_df.sort_values('chrom_len', ascending=False).iloc[0]
        # print(f'longest_relevant_nuccore_row: {longest_relevant_nuccore_row}')
        longest_nuccore_len = longest_relevant_nuccore_row['chrom_len']
        if longest_nuccore_len > 10e6:
            longest_nuccore_ir_pairs_df = all_ir_pairs_df[all_ir_pairs_df['nuccore_accession'] == longest_relevant_nuccore_row['nuccore_accession']]
            assert (longest_nuccore_ir_pairs_df['right1'] - longest_nuccore_ir_pairs_df['left1'] + 1).min() == search_for_pis_args['stage2']['repeat_pairs']['min_repeat_len']
            # print("((longest_nuccore_ir_pairs_df['right1'] - longest_nuccore_ir_pairs_df['left1'] + 1) == search_for_pis_args['stage2']['repeat_pairs']['min_repeat_len']).sum()")
            # print(((longest_nuccore_ir_pairs_df['right1'] - longest_nuccore_ir_pairs_df['left1'] + 1) == search_for_pis_args['stage2']['repeat_pairs']['min_repeat_len']).sum())

        # a sanity check to verify that most species were indeed scan and ir pairs were found in them.
        num_of_species = nuccore_df['taxon_uid'].nunique()
        num_of_species_with_at_least_one_ir_pair = all_ir_pairs_df.merge(nuccore_df)['taxon_uid'].nunique()
        proportion_of_species_with_at_least_one_ir_pair = num_of_species_with_at_least_one_ir_pair / num_of_species
        print(f'num_of_species: {num_of_species}')
        print(f'num_of_species_with_at_least_one_ir_pair: {num_of_species_with_at_least_one_ir_pair}')
        print(f'proportion_of_species_with_at_least_one_ir_pair: {proportion_of_species_with_at_least_one_ir_pair}')
        if num_of_species > 200:
            assert proportion_of_species_with_at_least_one_ir_pair >= 0.95

    stage2_results_info_pickle_file_path = os.path.join(massive_screening_stage2_out_dir_path, search_for_pis_args['stage2']['results_pickle_file_name'])
    stage2_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage2_results_info_pickle_file_path, stage_out_file_name_suffix)

    stage2_results_info = {
        'all_ir_pairs_info_pickle_file_path': all_ir_pairs_info_pickle_file_path,
        'nuccore_accession_to_ir_pairs_info_pickle_file_path': nuccore_accession_to_ir_pairs_info_pickle_file_path,
        'filtered_nuccore_accession_to_ir_pairs_info_pickle_file_path': filtered_nuccore_accession_to_ir_pairs_info_pickle_file_path,
        'all_ir_pairs_df_csv_file_path': all_ir_pairs_df_csv_file_path,

        'num_of_all_ir_pairs': num_of_all_ir_pairs,
    }

    with open(stage2_results_info_pickle_file_path, 'wb') as f:
        pickle.dump(stage2_results_info, f, protocol=4)

    return stage2_results_info


def main():
    with generic_utils.timing_context_manager('massive_screening_stage_2.py'):
        if DO_STAGE2:
            do_massive_screening_stage2(
                search_for_pis_args=massive_screening_configuration.SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT,
            )

        print('\n')


if __name__ == '__main__':
    main()
