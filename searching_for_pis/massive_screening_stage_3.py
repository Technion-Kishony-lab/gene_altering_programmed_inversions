import logging
import os
import pathlib
import pickle

import numpy as np
import pandas as pd

from generic import generic_utils
from searching_for_pis import ir_pairs_linkage
from searching_for_pis import ir_pairs_linked_df_extension
from searching_for_pis import massive_screening_stage_1
from searching_for_pis import massive_screening_configuration

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

DO_STAGE3 = True
# DO_STAGE3 = False

@generic_utils.execute_if_output_doesnt_exist_already
def cached_discard_ir_pairs_with_short_repeat_len(
    input_file_path_ir_pairs_df_csv,
    min_repeat_len,
    output_file_path_filtered_ir_pairs_df_csv,
):
    ir_pairs_df = pd.read_csv(input_file_path_ir_pairs_df_csv, sep='\t', low_memory=False)
    ir_pairs_df['repeat_len'] = ir_pairs_df['right1'] - ir_pairs_df['left1'] + 1

    ir_pairs_df = ir_pairs_df[ir_pairs_df['repeat_len'] >= min_repeat_len]

    ir_pairs_df.to_csv(output_file_path_filtered_ir_pairs_df_csv, sep='\t', index=False)

def discard_ir_pairs_with_short_repeat_len(
    input_file_path_ir_pairs_df_csv,
    min_repeat_len,
    output_file_path_filtered_ir_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_discard_ir_pairs_with_short_repeat_len(
        input_file_path_ir_pairs_df_csv=input_file_path_ir_pairs_df_csv,
        min_repeat_len=min_repeat_len,
        output_file_path_filtered_ir_pairs_df_csv=output_file_path_filtered_ir_pairs_df_csv,
    )


def get_pairs_linked_df_with_taxon_uid(
        pairs_linked_df_csv_file_path,
        nuccore_df_csv_file_path,
):
    pairs_linked_df = pd.read_csv(pairs_linked_df_csv_file_path, sep='\t', low_memory=False)
    nuccore_df = pd.read_csv(nuccore_df_csv_file_path, sep='\t', low_memory=False)
    assert set(pairs_linked_df['nuccore_accession'].unique()) <= set(nuccore_df['nuccore_accession'].unique())
    return pairs_linked_df.merge(nuccore_df[['taxon_uid', 'nuccore_accession']])

def discard_ir_pairs_linked_to_bad_cds(pairs_df, bad_cds_df):
    if bad_cds_df.empty:
        return pairs_df.copy()

    filtered_pairs_df = pairs_df.copy()
    for repeat_num in (1, 2):
        filtered_pairs_df = filtered_pairs_df.merge(bad_cds_df.rename(
            columns={'index_in_nuccore_cds_features_gb_file': f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'}), how='left', indicator=True)
        filtered_pairs_df = filtered_pairs_df[filtered_pairs_df['_merge'] == 'left_only']
        filtered_pairs_df.drop('_merge', axis=1, inplace=True)

    return filtered_pairs_df


def discard_ir_pairs_linked_to_cds_containing_repeats_of_bad_ir_pairs(pairs_df, bad_pairs_df):
    if bad_pairs_df.empty:
        return pairs_df.copy()
    assert (
            (bad_pairs_df[['repeat1_strictly_contained_in_its_cds', 'repeat2_strictly_contained_in_its_cds']] == True) |
            (bad_pairs_df[['repeat1_strictly_contained_in_its_cds', 'repeat2_strictly_contained_in_its_cds']] == False)
    ).all(axis=None)
    cds_containing_repeats_of_bad_pairs_df = pd.concat(
        [
            bad_pairs_df[
                bad_pairs_df[f'repeat{repeat_num}_strictly_contained_in_its_cds']
            ][['nuccore_accession', f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file']].drop_duplicates().rename(columns={
                f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file': 'index_in_nuccore_cds_features_gb_file'})
            for repeat_num in (1, 2)
        ],
        ignore_index=True,
    ).drop_duplicates()

    return discard_ir_pairs_linked_to_bad_cds(
        pairs_df=pairs_df,
        bad_cds_df=cds_containing_repeats_of_bad_pairs_df,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num(
        input_file_path_pairs_df_csv,
        min_max_estimated_copy_num_to_classify_as_mobile_element,
        output_file_path_filtered_pairs_df_csv,
        output_file_path_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    pairs_df = pd.read_csv(input_file_path_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_pairs = len(pairs_df)

    bad_pairs_df = pairs_df[pairs_df['repeat_estimated_copy_num'] >= min_max_estimated_copy_num_to_classify_as_mobile_element]
    filtered_pairs_df = discard_ir_pairs_linked_to_cds_containing_repeats_of_bad_ir_pairs(pairs_df, bad_pairs_df)

    num_of_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num = orig_num_of_pairs - len(filtered_pairs_df)
    ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_proportion = (
            num_of_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num / orig_num_of_pairs) if orig_num_of_pairs else None

    filtered_pairs_df.to_csv(output_file_path_filtered_pairs_df_csv, sep='\t', index=False)
    discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info = {
        'num_of_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num': (
            num_of_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num),
        'ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_proportion': (
            ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_proportion),
    }
    with open(output_file_path_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle, 'wb') as f:
        pickle.dump(discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info, f, protocol=4)

def discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num(
        input_file_path_pairs_df_csv,
        min_max_estimated_copy_num_to_classify_as_mobile_element,
        output_file_path_filtered_pairs_df_csv,
        output_file_path_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num(
        input_file_path_pairs_df_csv=input_file_path_pairs_df_csv,
        min_max_estimated_copy_num_to_classify_as_mobile_element=min_max_estimated_copy_num_to_classify_as_mobile_element,
        output_file_path_filtered_pairs_df_csv=output_file_path_filtered_pairs_df_csv,
        output_file_path_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle=output_file_path_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )


def get_pairs_with_both_repeats_linked(pairs_df):
    return pairs_df[ir_pairs_linked_df_extension.get_both_repeats_linked_filter(pairs_df)]


def do_massive_screening_stage3(
        search_for_pis_args,
):
    massive_screening_stage3_out_dir_path = search_for_pis_args['stage3']['output_dir_path']

    debug___num_of_taxa_to_go_over = search_for_pis_args['debug___num_of_taxa_to_go_over']
    debug___num_of_nuccore_entries_to_go_over = search_for_pis_args['debug___num_of_nuccore_entries_to_go_over']
    debug___taxon_uids = search_for_pis_args['debug___taxon_uids']

    stage_out_file_name_suffix = massive_screening_stage_1.get_stage_out_file_name_suffix(
        debug___num_of_taxa_to_go_over=debug___num_of_taxa_to_go_over,
        debug___taxon_uids=debug___taxon_uids,
    )

    stage3_results_info_pickle_file_path = os.path.join(massive_screening_stage3_out_dir_path, search_for_pis_args['stage3']['results_pickle_file_name'])
    stage3_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage3_results_info_pickle_file_path, stage_out_file_name_suffix)

    stage3_nuccore_entries_out_dir_path = os.path.join(massive_screening_stage3_out_dir_path, 'nuccore_accessions')
    stage3_taxa_out_dir_path = os.path.join(massive_screening_stage3_out_dir_path, 'taxa')

    pathlib.Path(massive_screening_stage3_out_dir_path).mkdir(parents=True, exist_ok=True)
    massive_screening_log_file_path = os.path.join(massive_screening_stage3_out_dir_path, 'massive_screening_stage3_log.txt')
    logging.basicConfig(filename=massive_screening_log_file_path, level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug(f'---------------starting do_massive_screening_stage3({massive_screening_stage3_out_dir_path})---------------')

    massive_screening_stage1_out_dir_path = search_for_pis_args['stage1']['output_dir_path']
    stage1_results_info_pickle_file_path = os.path.join(massive_screening_stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])
    stage1_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage1_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage1_results_info_pickle_file_path, 'rb') as f:
        stage1_results_info = pickle.load(f)
    all_cds_df_csv_file_path = stage1_results_info['all_cds_df_csv_file_path']
    nuccore_df_csv_file_path = stage1_results_info['nuccore_df_csv_file_path']
    taxon_uid_to_taxon_info_pickle_file_path = stage1_results_info['taxon_uid_to_taxon_info_pickle_file_path']

    massive_screening_stage2_out_dir_path = search_for_pis_args['stage2']['output_dir_path']
    stage2_results_info_pickle_file_path = os.path.join(massive_screening_stage2_out_dir_path, search_for_pis_args['stage2']['results_pickle_file_name'])
    stage2_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage2_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage2_results_info_pickle_file_path, 'rb') as f:
        stage2_results_info = pickle.load(f)
    all_ir_pairs_df_csv_file_path = stage2_results_info['all_ir_pairs_df_csv_file_path']

    ir_pairs_filtered_by_repeat_len_df_csv_file_path = os.path.join(massive_screening_stage3_out_dir_path, 'ir_pairs_filtered_by_repeat_len_df.csv')
    ir_pairs_filtered_by_repeat_len_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        ir_pairs_filtered_by_repeat_len_df_csv_file_path, stage_out_file_name_suffix)
    discard_ir_pairs_with_short_repeat_len(
        input_file_path_ir_pairs_df_csv=all_ir_pairs_df_csv_file_path,
        min_repeat_len=search_for_pis_args['stage3']['min_repeat_len'],
        output_file_path_filtered_ir_pairs_df_csv=ir_pairs_filtered_by_repeat_len_df_csv_file_path,
    )

    ir_pairs_linked_df_csv_file_path = os.path.join(massive_screening_stage3_out_dir_path, 'ir_pairs_linked_df.csv')
    ir_pairs_linked_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        ir_pairs_linked_df_csv_file_path, stage_out_file_name_suffix)
    ir_pair_linkage_info_pickle_file_path = os.path.join(massive_screening_stage3_out_dir_path, 'ir_pair_linkage_info.pickle')
    ir_pair_linkage_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        ir_pair_linkage_info_pickle_file_path, stage_out_file_name_suffix)

    ir_pairs_linkage.write_ir_pairs_linked_df_csv(
        input_file_path_all_cds_df_csv=all_cds_df_csv_file_path,
        input_file_path_all_ir_pairs_df_csv=ir_pairs_filtered_by_repeat_len_df_csv_file_path,
        output_file_path_ir_pairs_linked_df_csv=ir_pairs_linked_df_csv_file_path,
        output_file_path_ir_pair_linkage_info_pickle=ir_pair_linkage_info_pickle_file_path,
        debug___num_of_nuccore_entries_to_go_over=debug___num_of_nuccore_entries_to_go_over,
    )
    with open(ir_pair_linkage_info_pickle_file_path, 'rb') as f:
        ir_pair_linkage_info = pickle.load(f)
    print(f'\nir_pair_linkage_info:\n{ir_pair_linkage_info}\n')

    pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df_csv_file_path = os.path.join(
        massive_screening_stage3_out_dir_path, 'pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df.csv')
    pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df_csv_file_path, stage_out_file_name_suffix)
    nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path = os.path.join(
        massive_screening_stage3_out_dir_path, 'nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path.pickle') # not a mistake!
    nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path, stage_out_file_name_suffix)
    ir_pairs_linked_df_extension.find_evidence_for_repeat_overlapping_mobile_element(
        input_file_path_ir_pairs_linked_df_csv=ir_pairs_linked_df_csv_file_path,
        input_file_path_taxon_uid_to_taxon_info_pickle=taxon_uid_to_taxon_info_pickle_file_path,
        input_file_path_nuccore_df_csv=nuccore_df_csv_file_path,
        min_dist_from_ir_pair_region_for_alignments=search_for_pis_args['stage3']['blast_repeat_to_its_taxon_genome_to_find_copy_num'][
            'min_dist_from_ir_pair_region_for_alignments'],
        seed_len=search_for_pis_args['stage3']['blast_repeat_to_its_taxon_genome_to_find_copy_num']['seed_len'],
        max_evalue=search_for_pis_args['stage3']['blast_repeat_to_its_taxon_genome_to_find_copy_num']['max_evalue'],
        output_file_path_ir_pairs_linked_with_evidence_df_csv=pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df_csv_file_path,
        output_file_path_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle=(
            nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path),
        nuccore_entries_out_dir_path=stage3_nuccore_entries_out_dir_path,
        taxa_out_dir_path=stage3_taxa_out_dir_path,
    )

    pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path = os.path.join(
        massive_screening_stage3_out_dir_path, 'pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df.csv')
    pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path, stage_out_file_name_suffix)
    discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle_file_path = os.path.join(
        massive_screening_stage3_out_dir_path, 'discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info.pickle')
    discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle_file_path, stage_out_file_name_suffix)
    discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num(
        input_file_path_pairs_df_csv=pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df_csv_file_path,
        min_max_estimated_copy_num_to_classify_as_mobile_element=search_for_pis_args['stage3']['min_max_estimated_copy_num_to_classify_as_mobile_element'],
        output_file_path_filtered_pairs_df_csv=pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path,
        output_file_path_discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle=discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle_file_path,
    )
    with open(discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info_pickle_file_path, 'rb') as f:
        discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info = pickle.load(f)
    print(f'\ndiscard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info:\n{discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info}\n')

    stage3_results_info = {
        'ir_pairs_linked_df_csv_file_path': ir_pairs_linked_df_csv_file_path,
        'pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df_csv_file_path': (
            pairs_linked_with_evidence_for_repeat_overlapping_mobile_element_df_csv_file_path),
        'pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path': (
            pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path),

        'nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path': (
            nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path),

        'ir_pair_linkage_info': ir_pair_linkage_info,
        'discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info': (
            discard_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_info),
    }
    print(f'\nstage3_results_info:\n{stage3_results_info}\n')
    with open(stage3_results_info_pickle_file_path, 'wb') as f:
        pickle.dump(stage3_results_info, f, protocol=4)

    return stage3_results_info

def main():
    with generic_utils.timing_context_manager('massive_screening_stage_3.py'):
        if DO_STAGE3:
            do_massive_screening_stage3(
                search_for_pis_args=massive_screening_configuration.SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT,
            )

        print('\n')

if __name__ == '__main__':
    main()
