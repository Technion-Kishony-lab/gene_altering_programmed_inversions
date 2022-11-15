import logging
import os
import pathlib
import pickle
import warnings
import scipy.interpolate

from generic import bio_utils
from generic import generic_utils
from searching_for_pis import cds_pair_clustering
from searching_for_pis import index_column_names
from searching_for_pis import ir_pairs_linkage
from searching_for_pis import ir_pairs_linked_df_extension
from searching_for_pis import massive_screening_stage_1
from searching_for_pis import massive_screening_stage_3
from searching_for_pis import massive_screening_stage_5
from searching_for_pis import massive_screening_configuration
from searching_for_pis import writing_repeat_cdss_to_fasta

import statsmodels.tools.sm_exceptions
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import statsmodels.api
import pandas as pd
import numpy as np


# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

pd.set_option('display.max_colwidth', None)

MIN_TOTAL_SAMPLE_SIZE_FOR_G_TEST = 1000 # according to https://en.wikipedia.org/wiki/G-test#Distribution_and_use and http://www.biostathandbook.com/small.html.
MIN_MIN_SAMPLE_SIZE_FOR_G_TEST = 20 # according to https://en.wikipedia.org/wiki/G-test#Distribution_and_use and http://www.biostathandbook.com/small.html.

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_any_high_confidence_ir_pair_linked_to_cds_pair_column(
        input_file_path_ir_pairs_df_csv,
        output_file_path_extended_ir_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    ir_pairs_df = pd.read_csv(input_file_path_ir_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_ir_pairs = len(ir_pairs_df)
    # orig_num_of_cds_pairs = len(ir_pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].drop_duplicates())

    pairs_with_high_confidence_df = ir_pairs_df[ir_pairs_df['high_confidence_bp_for_both_repeats']]
    
    cds_pairs_with_high_confidence_ir_pairs_df = pairs_with_high_confidence_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].drop_duplicates()

    assert len(cds_pairs_with_high_confidence_ir_pairs_df) == len(cds_pairs_with_high_confidence_ir_pairs_df.drop_duplicates())
    extended_ir_pairs_df = ir_pairs_df.merge(cds_pairs_with_high_confidence_ir_pairs_df, how='left', indicator=True)
    extended_ir_pairs_df.loc[extended_ir_pairs_df['_merge'] == 'both', 'any_high_confidence_ir_pair_linked_to_cds_pair'] = True
    extended_ir_pairs_df.loc[extended_ir_pairs_df['_merge'] == 'left_only', 'any_high_confidence_ir_pair_linked_to_cds_pair'] = False
    extended_ir_pairs_df.drop('_merge', axis=1, inplace=True)

    assert not extended_ir_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'].isna().any()
    extended_ir_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'] = extended_ir_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'].astype(bool)

    assert len(extended_ir_pairs_df) == orig_num_of_ir_pairs
    extended_ir_pairs_df.to_csv(output_file_path_extended_ir_pairs_df_csv, sep='\t', index=False)

def add_any_high_confidence_ir_pair_linked_to_cds_pair_column(
        input_file_path_ir_pairs_df_csv,
        output_file_path_extended_ir_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_any_high_confidence_ir_pair_linked_to_cds_pair_column(
        input_file_path_ir_pairs_df_csv=input_file_path_ir_pairs_df_csv,
        output_file_path_extended_ir_pairs_df_csv=output_file_path_extended_ir_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_cds_pairs_relevant_for_genomic_architecture_analysis_and_logistic_training(
        input_file_path_cds_pairs_df_csv,
        input_file_path_merged_cds_pair_region_df_csv,
        min_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds,
        output_file_path_filtered_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_cds_pairs = len(cds_pairs_df)
    print(f'orig_num_of_cds_pairs: {orig_num_of_cds_pairs}')

    minimal_merged_cds_pair_region_df = pd.read_csv(input_file_path_merged_cds_pair_region_df_csv, sep='\t', low_memory=False)[
        index_column_names.MERGED_CDS_PAIR_REGION_INDEX_COLUMN_NAMES +
        ['num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds']
    ]
    cds_pairs_df = cds_pairs_df.merge(minimal_merged_cds_pair_region_df)
    cds_pairs_df = cds_pairs_df[
        cds_pairs_df['num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds'] >=
        min_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds
    ]
    num_of_cds_pairs_with_regions_in_other_nuccores_satisfying_thresholds = len(cds_pairs_df)
    print(f'num_of_cds_pairs_with_regions_in_other_nuccores_satisfying_thresholds: {num_of_cds_pairs_with_regions_in_other_nuccores_satisfying_thresholds}')
    print(cds_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'].value_counts())

    if 'cds_pair_cluster_index' in cds_pairs_df:
        cds_pairs_df = cds_pairs_df.groupby('cds_pair_cluster_index').sample(n=1, random_state=0)
        num_of_cluster_representative_cds_pairs = len(cds_pairs_df)
        print(f'num_of_cluster_representative_cds_pairs: {num_of_cluster_representative_cds_pairs}')
        print(cds_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'].value_counts())
    else:
        print('no cds_pair_cluster_index column!')

    cds_pairs_df.to_csv(output_file_path_filtered_cds_pairs_df_csv, sep='\t', index=False)

def write_cds_pairs_relevant_for_genomic_architecture_analysis_and_logistic_training(
        input_file_path_cds_pairs_df_csv,
        input_file_path_merged_cds_pair_region_df_csv,
        min_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds,
        output_file_path_filtered_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_cds_pairs_relevant_for_genomic_architecture_analysis_and_logistic_training(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_merged_cds_pair_region_df_csv=input_file_path_merged_cds_pair_region_df_csv,
        min_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds=(
            min_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds),
        output_file_path_filtered_cds_pairs_df_csv=output_file_path_filtered_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_accession_and_taxon_scientific_names_df(
        input_file_path_nuccore_df_csv,
        input_file_path_taxa_df_csv,
        output_file_path_nuccore_accession_and_taxon_scientific_names_df_csv,
):
    nuccore_df = pd.read_csv(input_file_path_nuccore_df_csv, sep='\t', low_memory=False)
    orig_num_of_nuccores = len(nuccore_df)
    taxa_df = pd.read_csv(input_file_path_taxa_df_csv, sep='\t', low_memory=False)

    taxa_df_column_names = set(taxa_df)
    taxon_scientific_name_column_names = [f'taxon_{x}' for x in bio_utils.TAXON_RANKS]
    taxon_scientific_name_column_names = [x for x in taxon_scientific_name_column_names if x in taxa_df_column_names]
    nuccore_accession_and_taxon_scientific_names_df = nuccore_df[['nuccore_accession', 'taxon_uid']].merge(taxa_df[['taxon_uid'] + taxon_scientific_name_column_names])

    assert len(nuccore_accession_and_taxon_scientific_names_df) == orig_num_of_nuccores
    nuccore_accession_and_taxon_scientific_names_df.to_csv(output_file_path_nuccore_accession_and_taxon_scientific_names_df_csv, sep='\t', index=False)

def write_nuccore_accession_and_taxon_scientific_names_df(
        input_file_path_nuccore_df_csv,
        input_file_path_taxa_df_csv,
        output_file_path_nuccore_accession_and_taxon_scientific_names_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_accession_and_taxon_scientific_names_df(
        input_file_path_nuccore_df_csv=input_file_path_nuccore_df_csv,
        input_file_path_taxa_df_csv=input_file_path_taxa_df_csv,
        output_file_path_nuccore_accession_and_taxon_scientific_names_df_csv=output_file_path_nuccore_accession_and_taxon_scientific_names_df_csv,
    )




@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_relevant_nuccore_accession_to_nuccore_entry_info(
        input_file_path_ir_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_relevant_nuccore_accession_to_nuccore_entry_info_pickle,
):
    with open(input_file_path_nuccore_accession_to_nuccore_entry_info_pickle, 'rb') as f:
        nuccore_accession_to_nuccore_entry_info = pickle.load(f)
    pairs_df = pd.read_csv(input_file_path_ir_pairs_df_csv, sep='\t', low_memory=False)

    relevant_nuccore_accession_to_nuccore_entry_info = {
        x: nuccore_accession_to_nuccore_entry_info[x]
        for x in pairs_df['nuccore_accession'].unique()
    }

    with open(output_file_path_relevant_nuccore_accession_to_nuccore_entry_info_pickle, 'wb') as f:
        pickle.dump(relevant_nuccore_accession_to_nuccore_entry_info, f, protocol=4)

def write_relevant_nuccore_accession_to_nuccore_entry_info(
        input_file_path_ir_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_relevant_nuccore_accession_to_nuccore_entry_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_relevant_nuccore_accession_to_nuccore_entry_info(
        input_file_path_ir_pairs_df_csv=input_file_path_ir_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_relevant_nuccore_accession_to_nuccore_entry_info_pickle=output_file_path_relevant_nuccore_accession_to_nuccore_entry_info_pickle,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_operon_df(
        input_file_path_nuccore_cds_df_csv,
        output_file_path_operon_df_csv,
        max_dist_between_cds_in_operon,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    # max_dist_between_cds_in_operon == 0 means the cds must overlap (i.e., at least one bp is shared).

    nuccore_cds_df = pd.read_csv(input_file_path_nuccore_cds_df_csv, sep='\t', low_memory=False)

    curr_operon_cds_indices = [next(iter(nuccore_cds_df.iterrows()))[1]['index_in_nuccore_cds_features_gb_file']]
    assert curr_operon_cds_indices == [0]

    list_of_operon_cds_indices = []
    for (_, prev_cds_row), (_, curr_cds_row) in zip(list(nuccore_cds_df.iterrows())[:-1], list(nuccore_cds_df.iterrows())[1:]):
        cds_strand = curr_cds_row['strand']
        prev_cds_strand = prev_cds_row['strand']
        curr_cds_start = curr_cds_row['start_pos']
        # curr_cds_end = curr_cds_row['end_pos']
        cds_index_in_nuccore_cds_features_gb_file = curr_cds_row['index_in_nuccore_cds_features_gb_file']
        prev_cds_start = prev_cds_row['start_pos']
        prev_cds_end = prev_cds_row['end_pos']
        # it would be better to have > instead of >=, but this isn't always true, unfortunately. ugh.
        if not (curr_cds_start >= prev_cds_start):
            print(f'(curr_cds_start, prev_cds_start): {(curr_cds_start, prev_cds_start)}')
        assert curr_cds_start >= prev_cds_start

        if (prev_cds_strand == cds_strand) and (prev_cds_end >= curr_cds_start - max_dist_between_cds_in_operon):
            curr_operon_cds_indices.append(cds_index_in_nuccore_cds_features_gb_file)
        else:
            if len(curr_operon_cds_indices) > 1:
                list_of_operon_cds_indices.append(curr_operon_cds_indices)
            curr_operon_cds_indices = [cds_index_in_nuccore_cds_features_gb_file]

    if len(curr_operon_cds_indices) > 1:
        list_of_operon_cds_indices.append(curr_operon_cds_indices)

    operon_flat_dicts = []
    for operon_index, operon_cds_indices in enumerate(list_of_operon_cds_indices):
        operon_flat_dicts.extend([
            {
                'index_in_nuccore_cds_features_gb_file': index,
                'operon_index': operon_index,
            }
            for index in operon_cds_indices
        ])

    if operon_flat_dicts:
        assert nuccore_cds_df['nuccore_accession'].nunique() == 1
        nuccore_accession = nuccore_cds_df['nuccore_accession'].iloc[0]
        operon_df = pd.DataFrame(operon_flat_dicts)
        operon_df['nuccore_accession'] = nuccore_accession
        operon_df.to_csv(output_file_path_operon_df_csv, sep='\t', index=False)
    else:
        generic_utils.write_empty_file(output_file_path_operon_df_csv)

    # extended_cds_df = nuccore_cds_df.merge(operon_df, how='left')
    # extended_cds_df.to_csv(output_file_path_extended_cds_df_csv, sep='\t', index=False)

def write_nuccore_operon_df(
        input_file_path_nuccore_cds_df_csv,
        output_file_path_operon_df_csv,
        max_dist_between_cds_in_operon,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_operon_df(
        input_file_path_nuccore_cds_df_csv=input_file_path_nuccore_cds_df_csv,
        output_file_path_operon_df_csv=output_file_path_operon_df_csv,
        max_dist_between_cds_in_operon=max_dist_between_cds_in_operon,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_operon_df(
        input_file_path_all_cds_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_operon_df_csv,
        max_dist_between_cds_in_operon,
        nuccore_entries_output_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    all_cds_df = pd.read_csv(input_file_path_all_cds_df_csv, sep='\t', low_memory=False)
    with open(input_file_path_nuccore_accession_to_nuccore_entry_info_pickle, 'rb') as f:
        nuccore_accession_to_nuccore_entry_info = pickle.load(f)

    nuccore_accessions = set(all_cds_df['nuccore_accession']) & set(nuccore_accession_to_nuccore_entry_info)
    num_of_nuccore_entries = len(nuccore_accessions)
    nuccore_operon_df_csv_file_paths = []
    for i, nuccore_accession in enumerate(sorted(nuccore_accessions)):
        generic_utils.print_and_write_to_log(f'(cached_write_operon_df) starting work on nuccore {i + 1}/{num_of_nuccore_entries}: {nuccore_accession}')
        nuccore_entry_info = nuccore_accession_to_nuccore_entry_info[nuccore_accession]
        nuccore_cds_df_csv_file_path = nuccore_entry_info['cds_df_csv_file_path']
        nuccore_out_dir_path = os.path.join(nuccore_entries_output_dir_path, nuccore_accession)
        pathlib.Path(nuccore_out_dir_path).mkdir(parents=True, exist_ok=True)
        nuccore_operon_df_csv_file_path = os.path.join(nuccore_out_dir_path, 'nuccore_operon_df.csv')

        write_nuccore_operon_df(
            input_file_path_nuccore_cds_df_csv=nuccore_cds_df_csv_file_path,
            output_file_path_operon_df_csv=nuccore_operon_df_csv_file_path,
            max_dist_between_cds_in_operon=max_dist_between_cds_in_operon,
        )
        nuccore_operon_df_csv_file_paths.append(nuccore_operon_df_csv_file_path)

    operon_df = pd.concat((pd.read_csv(x, sep='\t') for x in nuccore_operon_df_csv_file_paths if not generic_utils.is_file_empty(x)), ignore_index=True)

    orig_num_of_operons = len(operon_df)

    operon_df_grouped_by_operon = operon_df.groupby(index_column_names.OPERON_INDEX_COLUMN_NAMES)
    operon_df = operon_df.merge(
        operon_df_grouped_by_operon['index_in_nuccore_cds_features_gb_file'].min().reset_index(name='operon_min_index_in_nuccore_cds_features_gb_file')
    ).merge(
        operon_df_grouped_by_operon['index_in_nuccore_cds_features_gb_file'].max().reset_index(name='operon_max_index_in_nuccore_cds_features_gb_file')
    ).merge(
        all_cds_df[index_column_names.CDS_INDEX_COLUMN_NAMES + ['start_pos']].rename(
            columns={'index_in_nuccore_cds_features_gb_file': 'operon_min_index_in_nuccore_cds_features_gb_file',
                     'start_pos': 'operon_start'})
    ).merge(
        all_cds_df[index_column_names.CDS_INDEX_COLUMN_NAMES + ['end_pos']].rename(
            columns={'index_in_nuccore_cds_features_gb_file': 'operon_max_index_in_nuccore_cds_features_gb_file',
                     'end_pos': 'operon_end'})
    )
    assert len(operon_df) == orig_num_of_operons

    operon_df.to_csv(output_file_path_operon_df_csv, sep='\t', index=False)

def write_operon_df(
        input_file_path_all_cds_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_operon_df_csv,
        max_dist_between_cds_in_operon,
        nuccore_entries_output_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_operon_df(
        input_file_path_all_cds_df_csv=input_file_path_all_cds_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_operon_df_csv=output_file_path_operon_df_csv,
        max_dist_between_cds_in_operon=max_dist_between_cds_in_operon,
        nuccore_entries_output_dir_path=nuccore_entries_output_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_ir_pairs_with_operons_df(
        input_file_path_pairs_df_csv,
        input_file_path_operon_df_csv,
        output_file_path_extended_pairs_df,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    pairs_df = pd.read_csv(input_file_path_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_pairs = len(pairs_df)

    operon_df = pd.read_csv(input_file_path_operon_df_csv, sep='\t', low_memory=False)
    operon_df_column_names = set(operon_df)
    assert operon_df_column_names == {
        'nuccore_accession', 'index_in_nuccore_cds_features_gb_file', 'operon_index',
        'operon_min_index_in_nuccore_cds_features_gb_file',
        'operon_max_index_in_nuccore_cds_features_gb_file',
        'operon_start',
        'operon_end',
    }
    for repeat_num in (1, 2):
        pairs_df = pairs_df.merge(operon_df.rename(columns={x: f'repeat{repeat_num}_cds_{x}'
                                                            for x in sorted(operon_df_column_names - {'nuccore_accession'})}), how='left')
        pairs_df[f'repeat{repeat_num}_cds_is_in_operon'] = ~pairs_df[f'repeat{repeat_num}_cds_operon_index'].isna()

        pairs_df.loc[~pairs_df[f'repeat{repeat_num}_cds_is_in_operon'], f'repeat{repeat_num}_cds_operon_start'] = pairs_df.loc[
            ~pairs_df[f'repeat{repeat_num}_cds_is_in_operon'], f'repeat{repeat_num}_cds_start_pos']
        pairs_df.loc[~pairs_df[f'repeat{repeat_num}_cds_is_in_operon'], f'repeat{repeat_num}_cds_operon_end'] = pairs_df.loc[
            ~pairs_df[f'repeat{repeat_num}_cds_is_in_operon'], f'repeat{repeat_num}_cds_end_pos']

    pairs_df['num_of_repeat_cds_in_operons'] = pairs_df[['repeat1_cds_is_in_operon', 'repeat2_cds_is_in_operon']].sum(axis=1)

    assert len(pairs_df) == orig_num_of_pairs
    pairs_df.to_csv(output_file_path_extended_pairs_df, sep='\t', index=False)

def write_ir_pairs_with_operons_df(
        input_file_path_pairs_df_csv,
        input_file_path_operon_df_csv,
        output_file_path_extended_pairs_df,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_ir_pairs_with_operons_df(
        input_file_path_pairs_df_csv=input_file_path_pairs_df_csv,
        input_file_path_operon_df_csv=input_file_path_operon_df_csv,
        output_file_path_extended_pairs_df=output_file_path_extended_pairs_df,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_cds_with_operons_df(
        input_file_path_cds_df_csv,
        input_file_path_operon_df_csv,
        output_file_path_extended_cds_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_df = pd.read_csv(input_file_path_cds_df_csv, sep='\t', low_memory=False)
    operon_df = pd.read_csv(input_file_path_operon_df_csv, sep='\t', low_memory=False)
    assert set(operon_df) == {'nuccore_accession', 'index_in_nuccore_cds_features_gb_file', 'operon_index'}
    extended_cds_df = cds_df.merge(operon_df, how='left')
    assert len(extended_cds_df) == len(cds_df)
    extended_cds_df.to_csv(output_file_path_extended_cds_df_csv, sep='\t', index=False)

def write_cds_with_operons_df(
        input_file_path_cds_df_csv,
        input_file_path_operon_df_csv,
        output_file_path_extended_cds_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_cds_with_operons_df(
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        input_file_path_operon_df_csv=input_file_path_operon_df_csv,
        output_file_path_extended_cds_df_csv=output_file_path_extended_cds_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_cds_pairs_with_operons_df(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        input_file_path_operon_df_csv,
        recombinase_products_to_use_in_logistic_regression,
        output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    cds_df = pd.read_csv(input_file_path_cds_df_csv, sep='\t', low_memory=False)
    operon_df = pd.read_csv(input_file_path_operon_df_csv, sep='\t', low_memory=False)
    orig_num_of_operons = len(operon_df)

    cds_with_operon_and_product_df = operon_df.merge(cds_df[index_column_names.CDS_INDEX_COLUMN_NAMES + ['product']])
    assert len(cds_with_operon_and_product_df) == orig_num_of_operons
    assert set(cds_with_operon_and_product_df) == {*index_column_names.CDS_INDEX_COLUMN_NAMES, 'operon_index', 'product'}
    cds_with_operon_and_product_df['is_recombinase'] = False
    for recombinase_product in recombinase_products_to_use_in_logistic_regression:
        cds_with_operon_and_product_df['is_recombinase'] |= cds_with_operon_and_product_df['product'] == recombinase_product

    operon_df = operon_df.merge(cds_with_operon_and_product_df.groupby(index_column_names.OPERON_INDEX_COLUMN_NAMES)['is_recombinase'].any().reset_index(
        name='any_recombinase_in_operon'))
    assert len(operon_df) == orig_num_of_operons

    extended_cds_pairs_df = cds_pairs_df
    for repeat_num in (1, 2):
        extended_cds_pairs_df = extended_cds_pairs_df.merge(operon_df.rename(columns={
            x: f'repeat{repeat_num}_cds_{x}'
            for x in (set(operon_df) - {'nuccore_accession'})
        }), how='left')
    assert len(extended_cds_pairs_df) == len(cds_pairs_df)

    extended_cds_pairs_df['any_cds_in_operon'] = (~(extended_cds_pairs_df[['repeat1_cds_operon_index', 'repeat2_cds_operon_index']].isna())).any(axis=1)
    extended_cds_pairs_df['any_cds_in_operon_with_recombinase'] = (extended_cds_pairs_df[[
        'repeat1_cds_any_recombinase_in_operon', 'repeat2_cds_any_recombinase_in_operon']] == True).any(axis=1)

    extended_cds_pairs_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)

def write_cds_pairs_with_operons_df(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        input_file_path_operon_df_csv,
        recombinase_products_to_use_in_logistic_regression,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_cds_pairs_with_operons_df(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        input_file_path_operon_df_csv=input_file_path_operon_df_csv,
        recombinase_products_to_use_in_logistic_regression=recombinase_products_to_use_in_logistic_regression,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

def get_product_path_suitable_name(product):
    product_with_special_chars_converted = ''.join(x if x.isalnum() else '_' for x in product)
    return product_with_special_chars_converted[:35] + '_' + generic_utils.get_str_sha256_in_hex(product)[:10]



def get_cds_pairs_df(ir_pairs_df):
    ir_pairs_df = ir_pairs_df.copy()
    assert len(ir_pairs_df) == len(massive_screening_stage_3.get_pairs_with_both_repeats_linked(ir_pairs_df))
    # pairs_df = massive_screening_stage_3.get_pairs_with_both_repeats_linked(pairs_df)

    assert (ir_pairs_df['repeat1_cds_index_in_nuccore_cds_features_gb_file'] != ir_pairs_df['repeat2_cds_index_in_nuccore_cds_features_gb_file']).all()
    assert (ir_pairs_df['repeat1_cds_strand'] != ir_pairs_df['repeat2_cds_strand']).all()

    ir_pairs_df = ir_pairs_linked_df_extension.add_spacer_len_and_repeat_len_and_mismatch_fraction_columns(ir_pairs_df)
    ir_pairs_df = ir_pairs_linked_df_extension.add_repeat_cds_len_columns(ir_pairs_df)
    # pairs_df = ir_pairs_linked_df_extension.add_repeat_position_columns(pairs_df)
    ir_pairs_df = ir_pairs_linked_df_extension.add_longer_repeat_cds_columns(ir_pairs_df)

    more_shared_column_names = [
        x for x in list(ir_pairs_df) if (
                x.startswith('repeat1_cds_') or
                x.startswith('repeat2_cds_') or
                x.startswith('repeat_cds_pair_') or
                x.startswith('longer_repeat_cds_')
        )
    ] + [
        'any_high_confidence_ir_pair_linked_to_cds_pair',
        'merged_cds_pair_region_start',
        'merged_cds_pair_region_end',
        'max_repeat_cds_len',
        'min_repeat_cds_len',
    ]
    column_names_to_keep = set(index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES + more_shared_column_names)

    num_of_cds_pairs = len(ir_pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].drop_duplicates())
    assert len(ir_pairs_df[list(column_names_to_keep & set(ir_pairs_df))].drop_duplicates()) == num_of_cds_pairs

    pairs_grouped_by_cds_pairs_df = ir_pairs_df.groupby(list(column_names_to_keep), dropna=False)
    cds_pairs_df = pairs_grouped_by_cds_pairs_df.size().reset_index(name='num_of_linked_ir_pairs').merge(
        pairs_grouped_by_cds_pairs_df['spacer_len'].min().reset_index(name='min_spacer_len')).merge(
        pairs_grouped_by_cds_pairs_df['left1'].min().reset_index(name='min_left1')).merge(
        pairs_grouped_by_cds_pairs_df['left1'].max().reset_index(name='max_left1')).merge(
        pairs_grouped_by_cds_pairs_df['right1'].min().reset_index(name='min_right1')).merge(
        pairs_grouped_by_cds_pairs_df['right1'].max().reset_index(name='max_right1')).merge(
        pairs_grouped_by_cds_pairs_df['left2'].min().reset_index(name='min_left2')).merge(
        pairs_grouped_by_cds_pairs_df['left2'].max().reset_index(name='max_left2')).merge(
        pairs_grouped_by_cds_pairs_df['right2'].min().reset_index(name='min_right2')).merge(
        pairs_grouped_by_cds_pairs_df['right2'].max().reset_index(name='max_right2')).merge(
        pairs_grouped_by_cds_pairs_df['repeat_len'].min().reset_index(name='min_repeat_len')).merge(
        pairs_grouped_by_cds_pairs_df['repeat_len'].max().reset_index(name='max_repeat_len')).merge(
        pairs_grouped_by_cds_pairs_df['repeat_len'].sum().reset_index(name='repeat_len_sum')).merge(
        pairs_grouped_by_cds_pairs_df['num_of_mismatches'].sum().reset_index(name='total_num_of_mismatches'))

    assert len(cds_pairs_df) == num_of_cds_pairs

    cds_pairs_df['repeat_cdss_are_head_to_head'] = cds_pairs_df['repeat1_cds_strand'] == 1
    assert (cds_pairs_df['repeat_cdss_are_head_to_head'] == (cds_pairs_df['repeat2_cds_strand'] == -1)).all()

    cds_pairs_df['cds_spacer_len'] = cds_pairs_df['repeat2_cds_start_pos'] - cds_pairs_df['repeat1_cds_end_pos'] - 1
    cds_pairs_df['operon_spacer_len'] = cds_pairs_df['repeat2_cds_operon_start'] - cds_pairs_df['repeat1_cds_operon_end'] - 1

    for repeat_num in (1, 2):
        for min_or_max in ('min', 'max'):
            cds_pairs_df[f'len_of_cds_region_left_to_{min_or_max}_left{repeat_num}'] = (cds_pairs_df[f'{min_or_max}_left{repeat_num}'] -
                                                                                        cds_pairs_df[f'repeat{repeat_num}_cds_start_pos'])
            cds_pairs_df.loc[cds_pairs_df[f'len_of_cds_region_left_to_{min_or_max}_left{repeat_num}'] < 0,
                             f'len_of_cds_region_left_to_{min_or_max}_left{repeat_num}'] = 0

            cds_pairs_df[f'len_of_cds_region_right_to_{min_or_max}_right{repeat_num}'] = (cds_pairs_df[f'repeat{repeat_num}_cds_end_pos'] -
                                                                                          cds_pairs_df[f'{min_or_max}_right{repeat_num}'])
            cds_pairs_df.loc[cds_pairs_df[f'len_of_cds_region_right_to_{min_or_max}_right{repeat_num}'] < 0,
                             f'len_of_cds_region_right_to_{min_or_max}_right{repeat_num}'] = 0


            cds_pairs_df[f'len_of_operon_region_left_to_{min_or_max}_left{repeat_num}'] = (cds_pairs_df[f'{min_or_max}_left{repeat_num}'] -
                                                                                           cds_pairs_df[f'repeat{repeat_num}_cds_operon_start'])
            cds_pairs_df.loc[cds_pairs_df[f'len_of_operon_region_left_to_{min_or_max}_left{repeat_num}'] < 0,
                             f'len_of_operon_region_left_to_{min_or_max}_left{repeat_num}'] = 0

            cds_pairs_df[f'len_of_operon_region_right_to_{min_or_max}_right{repeat_num}'] = (cds_pairs_df[f'repeat{repeat_num}_cds_operon_end'] -
                                                                                             cds_pairs_df[f'{min_or_max}_right{repeat_num}'])
            cds_pairs_df.loc[cds_pairs_df[f'len_of_operon_region_right_to_{min_or_max}_right{repeat_num}'] < 0,
                             f'len_of_operon_region_right_to_{min_or_max}_right{repeat_num}'] = 0




    for region_type in ('cds', 'operon'):
        cds_pairs_df[f'len_of_shorter_{region_type}_region_outside_outermost_repeats'] = cds_pairs_df[[f'len_of_{region_type}_region_left_to_min_left1',
                                                                                            f'len_of_{region_type}_region_right_to_max_right2']].min(axis=1)
        cds_pairs_df[f'len_of_longer_{region_type}_region_outside_outermost_repeats'] = cds_pairs_df[[f'len_of_{region_type}_region_left_to_min_left1',
                                                                                            f'len_of_{region_type}_region_right_to_max_right2']].max(axis=1)
        cds_pairs_df[f'len_of_shorter_{region_type}_region_between_innermost_repeats'] = cds_pairs_df[[f'len_of_{region_type}_region_left_to_min_left2',
                                                                                            f'len_of_{region_type}_region_right_to_max_right1']].min(axis=1)
        cds_pairs_df[f'len_of_longer_{region_type}_region_between_innermost_repeats'] = cds_pairs_df[[f'len_of_{region_type}_region_left_to_min_left2',
                                                                                            f'len_of_{region_type}_region_right_to_max_right1']].max(axis=1)

        cds_pairs_df.loc[cds_pairs_df['repeat_cdss_are_head_to_head'], f'{region_type}_asymmetry'] = 1 - (
                cds_pairs_df[f'len_of_shorter_{region_type}_region_outside_outermost_repeats'] /
                cds_pairs_df[f'len_of_longer_{region_type}_region_outside_outermost_repeats']
        )
        cds_pairs_df.loc[~cds_pairs_df['repeat_cdss_are_head_to_head'], f'{region_type}_asymmetry'] = 1 - (
                cds_pairs_df[f'len_of_shorter_{region_type}_region_between_innermost_repeats'] /
                cds_pairs_df[f'len_of_longer_{region_type}_region_between_innermost_repeats']
        )
        # 0 == 0, so asymmetry should be 0 rather than nan.
        cds_pairs_df.loc[cds_pairs_df['repeat_cdss_are_head_to_head'] & (cds_pairs_df[f'len_of_longer_{region_type}_region_outside_outermost_repeats'] == 0),
                         f'{region_type}_asymmetry'] = 0
        cds_pairs_df.loc[(~cds_pairs_df['repeat_cdss_are_head_to_head']) & (cds_pairs_df[f'len_of_longer_{region_type}_region_between_innermost_repeats'] == 0),
                         f'{region_type}_asymmetry'] = 0
        # print(cds_pairs_df[f'{region_typea_symmetry'].describe())
        assert ((cds_pairs_df[f'{region_type}_asymmetry'] >= 0) & (cds_pairs_df[f'{region_type}_asymmetry'] <= 1)).all()


        cds_pairs_df[f'total_len_of_{region_type}_regions_outside_outermost_repeats'] = (cds_pairs_df[f'len_of_{region_type}_region_left_to_min_left1'] +
                                                                              cds_pairs_df[f'len_of_{region_type}_region_right_to_max_right2'])
        cds_pairs_df[f'total_len_of_{region_type}_regions_between_innermost_repeats'] = (cds_pairs_df[f'len_of_{region_type}_region_right_to_max_right1'] +
                                                                              cds_pairs_df[f'len_of_{region_type}_region_left_to_min_left2'])
        # Roy noted that another option, if we want the asymmetry and this attribute to be independent, is to use max length instead of sum of lengths...
        # i.e., max(alpha1,alpha2)/(max(alpha1,alpha2)+max(gamma1,gamma2)) (see Fig 2).
        cds_pairs_df[f'{region_type}_closest_repeat_position_orientation_matching'] = (
                (cds_pairs_df[f'total_len_of_{region_type}_regions_outside_outermost_repeats'] + cds_pairs_df[f'{region_type}_spacer_len']) /
                (cds_pairs_df[f'total_len_of_{region_type}_regions_outside_outermost_repeats'] +
                 cds_pairs_df[f'total_len_of_{region_type}_regions_between_innermost_repeats'] + 2 * cds_pairs_df[f'{region_type}_spacer_len'])
        )
        cds_pairs_df.loc[cds_pairs_df[f'{region_type}_closest_repeat_position_orientation_matching'] < 0,
                             f'{region_type}_closest_repeat_position_orientation_matching'] = 0


        cds_pairs_df[f'total_len_of_{region_type}_regions_outside_innermost_repeats'] = (cds_pairs_df[f'len_of_{region_type}_region_left_to_max_left1'] +
                                                                              cds_pairs_df[f'len_of_{region_type}_region_right_to_min_right2'])
        cds_pairs_df[f'total_len_of_{region_type}_regions_between_outermost_repeats'] = (cds_pairs_df[f'len_of_{region_type}_region_right_to_min_right1'] +
                                                                              cds_pairs_df[f'len_of_{region_type}_region_left_to_max_left2'])
        cds_pairs_df[f'{region_type}_furthest_repeat_position_orientation_matching'] = (
                cds_pairs_df[f'total_len_of_{region_type}_regions_outside_innermost_repeats'] /
                (cds_pairs_df[f'total_len_of_{region_type}_regions_outside_innermost_repeats'] +
                 cds_pairs_df[f'total_len_of_{region_type}_regions_between_outermost_repeats'])
        )

    assert len(cds_pairs_df) == num_of_cds_pairs

    assert (cds_pairs_df['num_of_linked_ir_pairs'] >= 1).all()
    cds_pairs_df['only_a_single_ir_pair_linked'] = cds_pairs_df['num_of_linked_ir_pairs'] == 1
    cds_pairs_df['total_mismatch_fraction'] = cds_pairs_df['total_num_of_mismatches'] / cds_pairs_df['repeat_len_sum']

    assert len(cds_pairs_df) == len(cds_pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].drop_duplicates())
    assert len(cds_pairs_df) == num_of_cds_pairs
    return cds_pairs_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_cds_pairs_df(
        input_file_path_ir_pairs_df_csv,
        output_file_path_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    ir_pairs_df = pd.read_csv(input_file_path_ir_pairs_df_csv, sep='\t', low_memory=False)
    assert not ir_pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].isna().any().any()
    orig_num_of_cds_pairs = len(ir_pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].drop_duplicates())
    cds_pairs_df = get_cds_pairs_df(ir_pairs_df)
    cds_pairs_df.sort_values(index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES, inplace=True) # so that the order would always be the same.
    # print(f'orig_num_of_cds_pairs: {orig_num_of_cds_pairs}')
    assert len(cds_pairs_df) == orig_num_of_cds_pairs
    cds_pairs_df.to_csv(output_file_path_cds_pairs_df_csv, sep='\t', index=False)

    # print('aoeu')
    # print(cds_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'].value_counts(dropna=False))
    # for column_name in (
    #     # 'cds_asymmetry',
    #     # 'operon_asymmetry',
    #     'cds_strand_efficiency',
    #     'operon_strand_efficiency',
    # ):
    #     print(cds_pairs_df[cds_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair']][column_name].describe())
    #     print(cds_pairs_df[~(cds_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'])][column_name].describe())
    # exit()


def write_cds_pairs_df(
        input_file_path_ir_pairs_df_csv,
        output_file_path_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_cds_pairs_df(
        input_file_path_ir_pairs_df_csv=input_file_path_ir_pairs_df_csv,
        output_file_path_cds_pairs_df_csv=output_file_path_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=59,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_column_name_to_critical_val_and_passed_threshold_column_name_pickle(
        input_file_path_cds_pairs_df_csv,
        column_names,
        output_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    num_of_cds_pairs = len(cds_pairs_df)
    curr_df_grouped = cds_pairs_df.groupby('any_high_confidence_ir_pair_linked_to_cds_pair')

    column_name_to_critical_val_and_passed_threshold_column_name = {}
    for column_name in column_names:
        column = cds_pairs_df[column_name]
        assert not column.isna().any()
        bins = sorted(column.unique())
        high_confidence_to_cumsum_hist = {high_confidence: np.cumsum(np.histogram(group_df[column_name], bins=bins)[0])
                                for high_confidence, group_df in curr_df_grouped}
        assert sum(x[-1] for x in high_confidence_to_cumsum_hist.values()) == num_of_cds_pairs
        high_confidence_to_normalized_cumsum_hist = {k: (v / v[-1]) for k, v in high_confidence_to_cumsum_hist.items()}

        critical_val_index_in_bins = np.argmax(np.abs(high_confidence_to_normalized_cumsum_hist[True] - high_confidence_to_normalized_cumsum_hist[False]))
        critical_val = bins[critical_val_index_in_bins]

        is_upper_bound = (high_confidence_to_normalized_cumsum_hist[True][critical_val_index_in_bins] >
                          high_confidence_to_normalized_cumsum_hist[False][critical_val_index_in_bins])

        column_name_to_critical_val_and_passed_threshold_column_name[column_name] = (critical_val, f'low_{column_name}' if is_upper_bound else f'high_{column_name}')

    # print('column_name_to_critical_val_and_passed_threshold_column_name')
    # print(column_name_to_critical_val_and_passed_threshold_column_name)
    print(f'column_name_to_critical_val_and_passed_threshold_column_name: {column_name_to_critical_val_and_passed_threshold_column_name}')
    # exit()

    with open(output_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle, 'wb') as f:
        pickle.dump(column_name_to_critical_val_and_passed_threshold_column_name, f, protocol=4)

def write_column_name_to_critical_val_and_passed_threshold_column_name_pickle(
        input_file_path_cds_pairs_df_csv,
        column_names,
        output_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_column_name_to_critical_val_and_passed_threshold_column_name_pickle(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        column_names=column_names,
        output_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle=output_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_passed_threshold_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle,
        output_file_path_extended_cds_pairs_df_csv,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    with open(input_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle, 'rb') as f:
        column_name_to_critical_val_and_passed_threshold_column_name = pickle.load(f)

    for column_name, (critical_val, passed_threshold_column_name) in column_name_to_critical_val_and_passed_threshold_column_name.items():
        if passed_threshold_column_name.startswith('low_'):
            cds_pairs_df[passed_threshold_column_name] = cds_pairs_df[column_name] <= critical_val
        else:
            assert passed_threshold_column_name.startswith('high_')
            cds_pairs_df[passed_threshold_column_name] = cds_pairs_df[column_name] >= critical_val

    cds_pairs_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)

def add_passed_threshold_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_passed_threshold_columns(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle=input_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
    )



@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_cds_pairs_and_adjacent_products_df(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        output_file_path_cds_pairs_and_adjacent_products_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    cds_pairs_df = cds_pairs_df[['nuccore_accession',
                                 'repeat1_cds_index_in_nuccore_cds_features_gb_file',
                                 'repeat2_cds_index_in_nuccore_cds_features_gb_file']]
    cds_df = pd.read_csv(input_file_path_cds_df_csv, sep='\t', low_memory=False)[[
        'nuccore_accession', 'product',
        'index_in_nuccore_cds_features_gb_file',
    ]].merge(cds_pairs_df['nuccore_accession'].drop_duplicates())

    orig_num_of_cds = len(cds_df)
    orig_num_of_pairs = len(cds_pairs_df)

    chunks_df = ir_pairs_linkage.get_chunks_df(
        df1=cds_df,
        df2=cds_pairs_df,
        df1_num_column_name='num_of_cds',
        df2_num_column_name='num_of_cds_pairs',
    )
    cds_df = cds_df.merge(chunks_df)
    cds_pairs_df = cds_pairs_df.merge(chunks_df)

    assert len(cds_df) == orig_num_of_cds
    assert len(cds_pairs_df) == orig_num_of_pairs

    chunk_index_to_chunk_size = chunks_df['chunk_index'].value_counts().to_dict()
    num_of_chunks = chunks_df['chunk_index'].nunique()

    cds_pairs_df_grouped = cds_pairs_df.groupby('chunk_index', sort=False)
    cds_df_grouped = cds_df.groupby('chunk_index', sort=False)

    chunk_cds_pairs_and_adjacent_products_dfs = []
    num_of_nuccore_entries_we_went_over = 0
    for i, (chunk_index, curr_ir_pairs_df) in enumerate(cds_pairs_df_grouped):
        chunk_size = chunk_index_to_chunk_size[chunk_index]
        generic_utils.print_and_write_to_log(f'starting work on chunk {i + 1}/{num_of_chunks} (chunk_index: {chunk_index}, chunk_size: {chunk_size})')
        curr_chunk_pairs_and_products_df = curr_ir_pairs_df.drop('chunk_index', axis=1).merge(
            cds_df_grouped.get_group(chunk_index).drop('chunk_index', axis=1), on='nuccore_accession', how='inner')
        generic_utils.print_and_write_to_log(f'num of rows in the cross product df: {len(curr_chunk_pairs_and_products_df)}')

        curr_chunk_pairs_and_products_df = curr_chunk_pairs_and_products_df[
            (
                (curr_chunk_pairs_and_products_df['repeat1_cds_index_in_nuccore_cds_features_gb_file'] - 1 <=
                 curr_chunk_pairs_and_products_df['index_in_nuccore_cds_features_gb_file']) &
                (curr_chunk_pairs_and_products_df['index_in_nuccore_cds_features_gb_file'] <=
                 curr_chunk_pairs_and_products_df['repeat1_cds_index_in_nuccore_cds_features_gb_file'] + 1)
            )
            |
            (
                (curr_chunk_pairs_and_products_df['repeat2_cds_index_in_nuccore_cds_features_gb_file'] - 1 <=
                 curr_chunk_pairs_and_products_df['index_in_nuccore_cds_features_gb_file']) &
                (curr_chunk_pairs_and_products_df['index_in_nuccore_cds_features_gb_file'] <=
                 curr_chunk_pairs_and_products_df['repeat2_cds_index_in_nuccore_cds_features_gb_file'] + 1)
            )
        ]

        # drop_duplicates() here is required because we don't care whether a product appears more than once near an ir_pair
        # (we just care whether it appears at least once).
        chunk_cds_pairs_and_adjacent_products_dfs.append(curr_chunk_pairs_and_products_df.drop('index_in_nuccore_cds_features_gb_file', axis=1).drop_duplicates())

        num_of_nuccore_entries_we_went_over += chunk_size

    cds_pairs_and_adjacent_products_df = pd.concat(chunk_cds_pairs_and_adjacent_products_dfs, ignore_index=True)

    assert len(cds_pairs_and_adjacent_products_df[[
        'nuccore_accession', 'repeat1_cds_index_in_nuccore_cds_features_gb_file', 'repeat2_cds_index_in_nuccore_cds_features_gb_file'
    ]].drop_duplicates()) == orig_num_of_pairs

    with generic_utils.timing_context_manager('cds_pairs_and_adjacent_products_df.to_csv()'):
        cds_pairs_and_adjacent_products_df.to_csv(output_file_path_cds_pairs_and_adjacent_products_df_csv, sep='\t', index=False)

def write_cds_pairs_and_adjacent_products_df(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        output_file_path_cds_pairs_and_adjacent_products_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_cds_pairs_and_adjacent_products_df(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        output_file_path_cds_pairs_and_adjacent_products_df_csv=output_file_path_cds_pairs_and_adjacent_products_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_cluster_cds_pairs(
        min_pairwise_identity_with_cluster_centroid,
        input_file_path_cds_pairs_df_csv,
        input_file_path_repeat_cds_seqs_fasta,
        input_file_path_repeat_seq_name_df_csv,
        output_file_path_pair_clustering_extended_info_pickle,
        pairwise_identity_definition_type,
        clustering_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    pairs_with_cluster_indices_df_csv_file_path = os.path.join(clustering_out_dir_path, 'pairs_with_cluster_indices_df.csv')
    pair_clustering_info_pickle_file_path = os.path.join(clustering_out_dir_path, 'pair_clustering.pickle')
    with generic_utils.timing_context_manager('write_pairs_df_with_cluster_indices'):
        cds_pair_clustering.write_pairs_df_with_cluster_indices(
            input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
            input_file_path_repeat_cds_seqs_fasta=input_file_path_repeat_cds_seqs_fasta,
            input_file_path_repeat_seq_name_df_csv=input_file_path_repeat_seq_name_df_csv,
            min_pairwise_identity_with_cluster_centroid=min_pairwise_identity_with_cluster_centroid,
            pairwise_identity_definition_type=pairwise_identity_definition_type,
            output_file_path_pairs_with_cluster_indices_df_csv=pairs_with_cluster_indices_df_csv_file_path,
            output_file_path_pair_clustering_info_pickle=pair_clustering_info_pickle_file_path,
            clustering_out_dir_path=clustering_out_dir_path,
        )
    with open(pair_clustering_info_pickle_file_path, 'rb') as f:
        pair_clustering_info = pickle.load(f)

    pair_clustering_extended_info = {
        'pairs_with_cluster_indices_df_csv_file_path': pairs_with_cluster_indices_df_csv_file_path,
        'clustering_out_dir_path': clustering_out_dir_path,
        'pair_clustering_info': pair_clustering_info,
    }
    with open(output_file_path_pair_clustering_extended_info_pickle, 'wb') as f:
        pickle.dump(pair_clustering_extended_info, f, protocol=4)

def cluster_cds_pairs(
        min_pairwise_identity_with_cluster_centroid,
        input_file_path_cds_pairs_df_csv,
        input_file_path_repeat_cds_seqs_fasta,
        input_file_path_repeat_seq_name_df_csv,
        output_file_path_pair_clustering_extended_info_pickle,
        pairwise_identity_definition_type,
        clustering_out_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_cluster_cds_pairs(
        min_pairwise_identity_with_cluster_centroid=min_pairwise_identity_with_cluster_centroid,
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_repeat_cds_seqs_fasta=input_file_path_repeat_cds_seqs_fasta,
        input_file_path_repeat_seq_name_df_csv=input_file_path_repeat_seq_name_df_csv,
        output_file_path_pair_clustering_extended_info_pickle=output_file_path_pair_clustering_extended_info_pickle,
        pairwise_identity_definition_type=pairwise_identity_definition_type,
        clustering_out_dir_path=clustering_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=5,
    )




def get_pairs_with_products_linked_df(pairs_df, linked_products):
    pairs_with_products_linked_df = pd.concat(
        [
            pairs_df.merge(linked_products.rename('repeat1_cds_product')),
            pairs_df.merge(linked_products.rename('repeat2_cds_product')),
        ],
        ignore_index=True,
    ).drop_duplicates()

    # np.nan might appear in the left side of the assert, but this is ok.
    assert ((set(pairs_with_products_linked_df['repeat1_cds_product']) |
             set(pairs_with_products_linked_df['repeat2_cds_product'])) >=
            set(linked_products))

    return pairs_with_products_linked_df



def add_longer_repeat_cds_my_product_family_column(df, product_regex_to_my_product_family):
    df['longer_repeat_cds_my_product_family'] = np.nan
    for product_regex, my_product_family in product_regex_to_my_product_family.items():
        df.loc[df['longer_repeat_cds_my_product_family'].isna() & df['longer_repeat_cds_product'].str.contains(product_regex, regex=True),
                    'longer_repeat_cds_my_product_family'] = my_product_family

    wasnt_classified_to_family_filter = df['longer_repeat_cds_my_product_family'].isna()
    df.loc[wasnt_classified_to_family_filter, 'longer_repeat_cds_my_product_family'] = df.loc[wasnt_classified_to_family_filter, 'longer_repeat_cds_product']
    assert not df['longer_repeat_cds_my_product_family'].isna().any()
    return df


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_cds_pair_representatives_df_for_enrichment_analysis(
        input_file_path_cds_pairs_df_csv,
        output_file_path_filtered_cds_pairs_df_csv,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    generic_utils.print_and_write_to_log('2')
    if 'cds_pair_cluster_index' in cds_pairs_df:
        print('before choosing random cluster representatives:')
        print(cds_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'].value_counts())
        generic_utils.print_and_write_to_log('2.1')
        cds_pairs_df = cds_pairs_df.groupby('cds_pair_cluster_index').sample(n=1, random_state=0)
        generic_utils.print_and_write_to_log('2.2')
        print('after choosing random cluster representatives:')
        print(cds_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'].value_counts())
        generic_utils.print_and_write_to_log('2.3')
    else:
        print('no cds_pair_cluster_index column!')
    generic_utils.print_and_write_to_log('3')

    cds_pairs_df.to_csv(output_file_path_filtered_cds_pairs_df_csv, sep='\t', index=False)

def write_cds_pair_representatives_df_for_enrichment_analysis(
        input_file_path_cds_pairs_df_csv,
        output_file_path_filtered_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_cds_pair_representatives_df_for_enrichment_analysis(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        output_file_path_filtered_cds_pairs_df_csv=output_file_path_filtered_cds_pairs_df_csv,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_perform_fisher_or_g_for_each_product_comparing_cds_pairs(
        input_file_path_cds_pairs_df_csv,
        product_column_names,
        min_num_of_cds_pairs_with_product_for_enrichment_test,
        test_column_name,
        output_file_path_repeat_cds_product_fisher_or_g_result_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    generic_utils.print_and_write_to_log('1')
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)

    cds_pairs_df = cds_pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES + [test_column_name] + product_column_names]
    generic_utils.print_and_write_to_log('4')
    num_of_cds_pairs = len(cds_pairs_df)

    product_and_cds_pairs_df = pd.concat(
        [
            cds_pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES + [product_column_name]].rename(columns={product_column_name: 'product'})
            for product_column_name in product_column_names
        ],
        ignore_index=True,
    ).drop_duplicates()
    cds_pairs_df.drop(product_column_names, axis=1, inplace=True)
    generic_utils.print_and_write_to_log('5')

    product_cds_pair_count_df = product_and_cds_pairs_df['product'].value_counts().reset_index(name='num_of_cds_pairs').rename(columns={'index': 'product'})
    generic_utils.print_and_write_to_log('6')
    products_to_test = sorted(
        product_cds_pair_count_df[product_cds_pair_count_df['num_of_cds_pairs'] >= min_num_of_cds_pairs_with_product_for_enrichment_test]['product'])
    generic_utils.print_and_write_to_log('7')
    num_of_products_to_test = len(products_to_test)
    product_and_cds_pairs_df_grouped_by_product = product_and_cds_pairs_df.groupby('product')
    generic_utils.print_and_write_to_log('8')

    flat_dicts = []
    for i, product in enumerate(products_to_test):
        # if product != 'DUF4965 domain-containing protein':
        # if ('PsrA' not in product) and (product != 'DUF4965 domain-containing protein'):
        #     continue
        generic_utils.print_and_write_to_log(f'(cached_perform_fisher_or_g_for_each_product_comparing_cds_pairs) '
                                             f'starting work on product {i + 1}/{num_of_products_to_test}: {product}')
        curr_cds_pairs_df = cds_pairs_df.merge(
            product_and_cds_pairs_df_grouped_by_product.get_group(product)[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES], how='left', indicator=True)
        assert len(curr_cds_pairs_df) == len(cds_pairs_df)

        curr_cds_pairs_df.loc[curr_cds_pairs_df['_merge'] == 'both', 'product_appears'] = True
        curr_cds_pairs_df.loc[curr_cds_pairs_df['_merge'] == 'left_only', 'product_appears'] = False

        fisher_or_g_result = generic_utils.perform_g_test_or_fisher_exact_test(curr_cds_pairs_df, 'product_appears', test_column_name,
                                                                               alternative='two-sided', return_matrix_in_4_keys=True)
        assert fisher_or_g_result['matrix_total_count'] == num_of_cds_pairs
        flat_dicts.append({
            'product': product,
            **fisher_or_g_result,
        })
        # print(flat_dicts)
        # exit()

    repeat_cds_product_fisher_or_g_result_df = pd.DataFrame(flat_dicts)
    repeat_cds_product_fisher_or_g_result_df['corrected_pvalue'] = (
            repeat_cds_product_fisher_or_g_result_df['pvalue'] * num_of_products_to_test)
    repeat_cds_product_fisher_or_g_result_df.sort_values(by='pvalue', inplace=True)
    repeat_cds_product_fisher_or_g_result_df.to_csv(output_file_path_repeat_cds_product_fisher_or_g_result_df_csv,
                                                                           sep='\t', index=False)

def perform_fisher_or_g_for_each_product_comparing_cds_pairs(
        input_file_path_cds_pairs_df_csv,
        product_column_names,
        min_num_of_cds_pairs_with_product_for_enrichment_test,
        test_column_name,
        output_file_path_repeat_cds_product_fisher_or_g_result_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_perform_fisher_or_g_for_each_product_comparing_cds_pairs(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        product_column_names=product_column_names,
        min_num_of_cds_pairs_with_product_for_enrichment_test=min_num_of_cds_pairs_with_product_for_enrichment_test,
        test_column_name=test_column_name,
        output_file_path_repeat_cds_product_fisher_or_g_result_df_csv=(
            output_file_path_repeat_cds_product_fisher_or_g_result_df_csv),
        dummy_arg_to_make_caching_mechanism_not_skip_execution=14,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_significantly_enriched_recombinase_cds_df(
        input_file_path_cds_df_csv,
        significantly_enriched_recombinase_products,
        output_significantly_enriched_recombinase_cds_df_csv_file_path,
):
    cds_df = pd.read_csv(input_file_path_cds_df_csv, sep='\t', low_memory=False)

    if significantly_enriched_recombinase_products:
        significantly_enriched_recombinase_cds_df = pd.concat(
            [
                cds_df[cds_df['product'] == product]
                for product in significantly_enriched_recombinase_products
            ],
            ignore_index=True,
        ).drop_duplicates()
    else:
        significantly_enriched_recombinase_cds_df = cds_df.iloc[0:0].copy()

    significantly_enriched_recombinase_cds_df.to_csv(output_significantly_enriched_recombinase_cds_df_csv_file_path, sep='\t', index=False)

def write_significantly_enriched_recombinase_cds_df(
        input_file_path_cds_df_csv,
        significantly_enriched_recombinase_products,
        output_significantly_enriched_recombinase_cds_df_csv_file_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_significantly_enriched_recombinase_cds_df(
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        significantly_enriched_recombinase_products=significantly_enriched_recombinase_products,
        output_significantly_enriched_recombinase_cds_df_csv_file_path=output_significantly_enriched_recombinase_cds_df_csv_file_path,
    )

def get_logistic_regression_fit_result_info(X, y, discard_const_row=True):
    logit = statsmodels.api.Logit(y, statsmodels.api.add_constant(X))

    # https://stats.stackexchange.com/questions/463324/logistic-regression-failed-in-statsmodel-but-works-in-sklearn-breast-cancer-dat/463335#463335
    # I guess that if X can perfectly predict y, then 'numpy.linalg.LinAlgError: Singular matrix' is raised??

    fit_result = logit.fit(
        # https://stats.stackexchange.com/questions/313426/mle-convergence-errors-with-statespace-sarimax
        # maxiter=200,
    )
    # print(f'fit_result.summary(): {fit_result.summary()}')
    # print(f'fit_result.summary2(): {fit_result.summary2()}')
    fit_result_df = pd.read_html(fit_result.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index().rename(columns={'index': 'bool_column_name'})
    fit_result_df = fit_result_df.rename(columns={'P>|z|': 'rounded_P>|z|'}).merge(fit_result.pvalues.reset_index(name='P>|z|').rename(columns={'index': 'bool_column_name'}))
    if discard_const_row:
        fit_result_df = fit_result_df[fit_result_df['bool_column_name'] != 'const']


    return {
        'fit_result': fit_result,
        'fit_result_df': fit_result_df,
    }

@generic_utils.execute_if_output_doesnt_exist_already
def cached_perform_logistic_regression(
        input_file_path_cds_pairs_df_for_training_csv,
        input_file_path_cds_pairs_df_for_prediction_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        perform_unadjusted_regressions,
        output_file_path_logistic_regression_fit_result_df_csv,
        output_file_path_cds_pairs_df_with_prediction_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    training_cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_for_training_csv, sep='\t', low_memory=False)

    training_y = training_cds_pairs_df[logistic_regression_dependent_var_column_name].astype(int)
    training_X = training_cds_pairs_df[predictor_column_names].astype(int)
    # print(f'len(training_y): {len(training_y)}')
    adjusted_fit_result_info = get_logistic_regression_fit_result_info(training_X, training_y)
    adjusted_fit_result = adjusted_fit_result_info['fit_result']
    print(f'adjusted_fit_result_info: {adjusted_fit_result_info}')

    # print(type(adjusted_fit_result))
    # exit()

    cds_pairs_df_for_prediction = pd.read_csv(input_file_path_cds_pairs_df_for_prediction_csv, sep='\t', low_memory=False)
    all_cds_pairs_X = cds_pairs_df_for_prediction[predictor_column_names].astype(int)
    cds_pairs_df_for_prediction[f'predicted_{logistic_regression_dependent_var_column_name}_probability'] = adjusted_fit_result.predict(
        statsmodels.api.add_constant(all_cds_pairs_X))

    cds_pairs_df_for_prediction.to_csv(output_file_path_cds_pairs_df_with_prediction_csv, sep='\t', index=False)


    adjusted_fit_res_df = adjusted_fit_result_info['fit_result_df']
    adjusted_fit_res_df.rename(columns={x: f'adjusted_{x}' for x in list(adjusted_fit_res_df) if x != 'bool_column_name'}, inplace=True)

    if perform_unadjusted_regressions:
        individual_fit_res_dfs = []
        for logistic_regression_predictor_column_name in predictor_column_names:
            generic_utils.print_and_write_to_log(f'(cached_perform_logistic_regression) '
                                                 f'starting work on logistic_regression_predictor_column_name {logistic_regression_predictor_column_name}')
            curr_training_X = training_X[[logistic_regression_predictor_column_name]]
            curr_individual_fit_res_df = get_logistic_regression_fit_result_info(curr_training_X, training_y)['fit_result_df']
            # print(curr_individual_fit_res_df)
            individual_fit_res_dfs.append(curr_individual_fit_res_df)

        individual_fit_res_df = pd.concat(individual_fit_res_dfs, ignore_index=True)
        # print(individual_fit_res_df)
        fit_result_df = adjusted_fit_res_df.merge(individual_fit_res_df, how='left')
    else:
        fit_result_df = adjusted_fit_res_df


    fit_result_df.to_csv(output_file_path_logistic_regression_fit_result_df_csv, sep='\t', index=False)


def perform_logistic_regression(
        input_file_path_cds_pairs_df_for_training_csv,
        input_file_path_cds_pairs_df_for_prediction_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        perform_unadjusted_regressions,
        output_file_path_logistic_regression_fit_result_df_csv,
        output_file_path_cds_pairs_df_with_prediction_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_perform_logistic_regression(
        input_file_path_cds_pairs_df_for_training_csv=input_file_path_cds_pairs_df_for_training_csv,
        input_file_path_cds_pairs_df_for_prediction_csv=input_file_path_cds_pairs_df_for_prediction_csv,
        predictor_column_names=predictor_column_names,
        logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
        perform_unadjusted_regressions=perform_unadjusted_regressions,
        output_file_path_logistic_regression_fit_result_df_csv=output_file_path_logistic_regression_fit_result_df_csv,
        output_file_path_cds_pairs_df_with_prediction_csv=output_file_path_cds_pairs_df_with_prediction_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=7,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_minimal_cds_pairs_for_logistic_model_assessment(
        input_file_path_cds_pairs_df_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        output_file_path_minimal_cds_pairs_df_csv,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    minimal_cds_pairs_df = cds_pairs_df[predictor_column_names + [logistic_regression_dependent_var_column_name]]
    minimal_cds_pairs_df.to_csv(output_file_path_minimal_cds_pairs_df_csv, sep='\t', index=False)

def write_minimal_cds_pairs_for_logistic_model_assessment(
        input_file_path_cds_pairs_df_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        output_file_path_minimal_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_minimal_cds_pairs_for_logistic_model_assessment(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        predictor_column_names=predictor_column_names,
        logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
        output_file_path_minimal_cds_pairs_df_csv=output_file_path_minimal_cds_pairs_df_csv,
    )

def get_roc_curve_df(test_df, logistic_regression_dependent_var_column_name):
    test_df = test_df.copy()

    predicted_prob_column_name = f'predicted_{logistic_regression_dependent_var_column_name}_probability'
    thresholds = sorted(set(test_df[predicted_prob_column_name]) | {0, 1})
    num_of_positives = test_df[logistic_regression_dependent_var_column_name].sum()
    num_of_negatives = (~test_df[logistic_regression_dependent_var_column_name]).sum()
    flat_dicts = []
    for threshold in thresholds:
        test_df['binarized_prediction'] = test_df[predicted_prob_column_name] >= threshold
        num_of_false_positives = (test_df['binarized_prediction'] & (~test_df[logistic_regression_dependent_var_column_name])).sum()
        fpr = num_of_false_positives / num_of_negatives
        num_of_false_negatives = ((~test_df['binarized_prediction']) & test_df[logistic_regression_dependent_var_column_name]).sum()
        fnr = num_of_false_negatives / num_of_positives
        flat_dicts.append({
            'threshold': threshold,
            'fpr': fpr,
            'fnr': fnr,
        })

    roc_curve_df = pd.DataFrame(flat_dicts)
    return roc_curve_df


@generic_utils.execute_if_output_doesnt_exist_already
def cached_perform_logistic_model_simulations(
        input_file_path_cds_pairs_df_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        test_set_fraction,
        num_of_simulations,
        output_dir_path,
        output_file_path_concat_simulation_roc_curve_dfs_csv,
        output_file_path_concat_simulation_fit_result_dfs_csv,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)

    roc_curve_dfs = []
    fit_result_dfs = []
    for sim_i in range(num_of_simulations):
        generic_utils.print_and_write_to_log(f'(assess_logistic_regression_model) starting simulation {sim_i + 1}/{num_of_simulations}')
        sim_out_dir_path = os.path.join(output_dir_path, f'sim{sim_i}')
        pathlib.Path(sim_out_dir_path).mkdir(parents=True, exist_ok=True)

        # random_state=sim_i for reproducibility
        test_df = cds_pairs_df.sample(frac=test_set_fraction, random_state=sim_i)
        train_df = cds_pairs_df.drop(test_df.index)

        train_df_csv_file_path = os.path.join(sim_out_dir_path, 'train_df.csv')
        test_df_csv_file_path = os.path.join(sim_out_dir_path, 'test_df.csv')

        test_df.to_csv(test_df_csv_file_path, sep='\t', index=False)
        train_df.to_csv(train_df_csv_file_path, sep='\t', index=False)

        fit_result_df_csv_file_path = os.path.join(sim_out_dir_path, 'fit_result_df.csv')
        test_df_with_predictions_csv_file_path = os.path.join(sim_out_dir_path, 'test_df_with_predictions.csv')
        perform_logistic_regression(
            input_file_path_cds_pairs_df_for_training_csv=train_df_csv_file_path,
            input_file_path_cds_pairs_df_for_prediction_csv=test_df_csv_file_path,
            predictor_column_names=predictor_column_names,
            logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
            perform_unadjusted_regressions=False,
            output_file_path_logistic_regression_fit_result_df_csv=fit_result_df_csv_file_path,
            output_file_path_cds_pairs_df_with_prediction_csv=test_df_with_predictions_csv_file_path,
        )
        fit_result_df = pd.read_csv(fit_result_df_csv_file_path, sep='\t', low_memory=False)
        fit_result_df['sim_i'] = sim_i
        fit_result_dfs.append(fit_result_df)

        test_df_with_predictions = pd.read_csv(test_df_with_predictions_csv_file_path, sep='\t', low_memory=False)
        roc_curve_df = get_roc_curve_df(test_df_with_predictions, logistic_regression_dependent_var_column_name)
        roc_curve_df['sim_i'] = sim_i
        roc_curve_dfs.append(roc_curve_df)

    pd.concat(fit_result_dfs, ignore_index=True).to_csv(output_file_path_concat_simulation_fit_result_dfs_csv, sep='\t', index=False)
    pd.concat(roc_curve_dfs, ignore_index=True).to_csv(output_file_path_concat_simulation_roc_curve_dfs_csv, sep='\t', index=False)

def perform_logistic_model_simulations(
        input_file_path_cds_pairs_df_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        test_set_fraction,
        num_of_simulations,
        output_dir_path,
        output_file_path_concat_simulation_roc_curve_dfs_csv,
        output_file_path_concat_simulation_fit_result_dfs_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_perform_logistic_model_simulations(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        predictor_column_names=predictor_column_names,
        logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
        test_set_fraction=test_set_fraction,
        num_of_simulations=num_of_simulations,
        output_dir_path=output_dir_path,
        output_file_path_concat_simulation_roc_curve_dfs_csv=output_file_path_concat_simulation_roc_curve_dfs_csv,
        output_file_path_concat_simulation_fit_result_dfs_csv=output_file_path_concat_simulation_fit_result_dfs_csv,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_assess_logistic_regression_model(
        input_file_path_cds_pairs_df_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        test_set_fraction,
        num_of_simulations,
        predicted_probability_threshold,
        output_file_path_unified_roc_curve_df_csv,
        output_file_path_concat_simulation_fit_result_dfs_csv,
        output_dir_path,
):
    concat_simulation_roc_curve_dfs_csv_file_path = os.path.join(output_dir_path, 'concat_simulation_roc_curve_dfs.csv')
    perform_logistic_model_simulations(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        predictor_column_names=predictor_column_names,
        logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
        test_set_fraction=test_set_fraction,
        num_of_simulations=num_of_simulations,
        output_dir_path=output_dir_path,
        output_file_path_concat_simulation_roc_curve_dfs_csv=concat_simulation_roc_curve_dfs_csv_file_path,
        output_file_path_concat_simulation_fit_result_dfs_csv=output_file_path_concat_simulation_fit_result_dfs_csv,
    )

    concat_simulation_fit_result_dfs = pd.read_csv(output_file_path_concat_simulation_fit_result_dfs_csv, sep='\t', low_memory=False)
    # sanity check:
    for predictor_column_name in predictor_column_names:
        print(f'describe() for {predictor_column_name} adjusted_coef')
        print(concat_simulation_fit_result_dfs[concat_simulation_fit_result_dfs['bool_column_name'] == predictor_column_name]['adjusted_coef'].describe())
        print()

    concat_simulation_roc_curve_dfs = pd.read_csv(concat_simulation_roc_curve_dfs_csv_file_path, sep='\t', low_memory=False)
    thresholds = sorted(set(concat_simulation_roc_curve_dfs['threshold']) | {0, predicted_probability_threshold, 1})
    all_thresholds_simulation_roc_curve_dfs = []
    for sim_i, simulation_roc_curve_df in concat_simulation_roc_curve_dfs.groupby('sim_i'):
        interp_func = scipy.interpolate.interp1d(simulation_roc_curve_df['threshold'], simulation_roc_curve_df['threshold'], kind='previous')
        all_thresholds_simulation_roc_curve_df = pd.DataFrame({'new_threshold': thresholds, 'threshold': interp_func(thresholds)}).merge(simulation_roc_curve_df)
        assert len(all_thresholds_simulation_roc_curve_df) == len(thresholds)

        all_thresholds_simulation_roc_curve_dfs.append(all_thresholds_simulation_roc_curve_df)

    pd.concat(all_thresholds_simulation_roc_curve_dfs, ignore_index=True).to_csv(output_file_path_unified_roc_curve_df_csv, sep='\t', index=False)

def assess_logistic_regression_model(
        input_file_path_cds_pairs_df_csv,
        predictor_column_names,
        logistic_regression_dependent_var_column_name,
        test_set_fraction,
        num_of_simulations,
        predicted_probability_threshold,
        output_file_path_unified_roc_curve_df_csv,
        output_file_path_concat_simulation_fit_result_dfs_csv,
        output_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_assess_logistic_regression_model(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        predictor_column_names=predictor_column_names,
        logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
        test_set_fraction=test_set_fraction,
        num_of_simulations=num_of_simulations,
        predicted_probability_threshold=predicted_probability_threshold,
        output_file_path_unified_roc_curve_df_csv=output_file_path_unified_roc_curve_df_csv,
        output_file_path_concat_simulation_fit_result_dfs_csv=output_file_path_concat_simulation_fit_result_dfs_csv,
        output_dir_path=output_dir_path,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_prediction_column(
        input_file_path_cds_pairs_df_csv,
        logistic_regression_dependent_var_column_name,
        min_predicted_rearrangement_probability,
        output_file_path_extended_cds_pairs_df_csv,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    cds_pairs_df[f'predicted_{logistic_regression_dependent_var_column_name}'] = (
            cds_pairs_df[f'predicted_{logistic_regression_dependent_var_column_name}_probability'] >
            min_predicted_rearrangement_probability
    )

    cds_pairs_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)

def add_prediction_column(
        input_file_path_cds_pairs_df_csv,
        logistic_regression_dependent_var_column_name,
        min_predicted_rearrangement_probability,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_prediction_column(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
        min_predicted_rearrangement_probability=min_predicted_rearrangement_probability,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
    )


def add_adjacent_cds_columns(
        cds_pairs_df,
        cds_df,
):
    cds_pairs_df = cds_pairs_df.copy()
    orig_num_of_cds_pairs = len(cds_pairs_df)
    minimal_cds_df = cds_df[['nuccore_accession', 'index_in_nuccore_cds_features_gb_file', 'start_pos', 'end_pos', 'product', 'strand']]

    assert (cds_pairs_df['repeat1_cds_strand'] != cds_pairs_df['repeat2_cds_strand']).all()
    assert (cds_pairs_df[['repeat1_cds_strand', 'repeat2_cds_strand']].abs() == 1).all().all()

    for repeat_num in (1, 2):
        cds_pairs_df[f'cds_left_to_repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] = cds_pairs_df[
            f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] - 1
        cds_pairs_df[f'cds_right_to_repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] = cds_pairs_df[
            f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] + 1

        cds_pairs_df = cds_pairs_df.merge(minimal_cds_df[['nuccore_accession', 'index_in_nuccore_cds_features_gb_file', 'end_pos', 'product', 'strand']].rename(columns={
            'index_in_nuccore_cds_features_gb_file': f'cds_left_to_repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file',
            'end_pos': f'cds_left_to_repeat{repeat_num}_cds_end_pos',
            'product': f'cds_left_to_repeat{repeat_num}_cds_product',
            'strand': f'cds_left_to_repeat{repeat_num}_cds_strand',
        }), how='left').drop(f'cds_left_to_repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file', axis=1)

        cds_pairs_df = cds_pairs_df.merge(minimal_cds_df[['nuccore_accession', 'index_in_nuccore_cds_features_gb_file', 'start_pos', 'product', 'strand']].rename(columns={
            'index_in_nuccore_cds_features_gb_file': f'cds_right_to_repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file',
            'start_pos': f'cds_right_to_repeat{repeat_num}_cds_start_pos',
            'product': f'cds_right_to_repeat{repeat_num}_cds_product',
            'strand': f'cds_right_to_repeat{repeat_num}_cds_strand',
        }), how='left').drop(f'cds_right_to_repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file', axis=1)

        cds_pairs_df[f'dist_between_repeat{repeat_num}_cds_and_cds_left_to_it'] = (
                cds_pairs_df[f'repeat{repeat_num}_cds_start_pos'] -
                cds_pairs_df[f'cds_left_to_repeat{repeat_num}_cds_end_pos'] - 1
        )
        cds_pairs_df[f'dist_between_repeat{repeat_num}_cds_and_cds_right_to_it'] = (
                cds_pairs_df[f'cds_right_to_repeat{repeat_num}_cds_start_pos'] -
                cds_pairs_df[f'repeat{repeat_num}_cds_end_pos'] - 1
        )

        repeat_cds_is_on_forward_strand_filter = cds_pairs_df[f'repeat{repeat_num}_cds_strand'] == 1

        cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter, f'dist_between_first_ir_pair_repeat{repeat_num}_and_upstream_cds'] = (
            cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter, f'first_ir_pair_left{repeat_num}'] -
            cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter, f'cds_left_to_repeat{repeat_num}_cds_end_pos']
        )
        cds_pairs_df.loc[~repeat_cds_is_on_forward_strand_filter, f'dist_between_first_ir_pair_repeat{repeat_num}_and_upstream_cds'] = (
            cds_pairs_df.loc[~repeat_cds_is_on_forward_strand_filter, f'cds_right_to_repeat{repeat_num}_cds_start_pos'] -
            cds_pairs_df.loc[~repeat_cds_is_on_forward_strand_filter, f'first_ir_pair_right{repeat_num}']
        )

        cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter, f'dist_between_last_ir_pair_repeat{repeat_num}_and_downstream_cds'] = (
            cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter, f'cds_right_to_repeat{repeat_num}_cds_start_pos'] -
            cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter, f'last_ir_pair_right{repeat_num}']
        )
        cds_pairs_df.loc[~repeat_cds_is_on_forward_strand_filter, f'dist_between_last_ir_pair_repeat{repeat_num}_and_downstream_cds'] = (
            cds_pairs_df.loc[~repeat_cds_is_on_forward_strand_filter, f'last_ir_pair_left{repeat_num}'] -
            cds_pairs_df.loc[~repeat_cds_is_on_forward_strand_filter, f'cds_left_to_repeat{repeat_num}_cds_end_pos']
        )

        cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter & (cds_pairs_df[f'cds_right_to_repeat{repeat_num}_cds_strand'] == 1),
                         f'cds_downstream_to_and_potentially_in_operon_with_repeat{repeat_num}_cds_product'] = (
            cds_pairs_df.loc[repeat_cds_is_on_forward_strand_filter & (cds_pairs_df[f'cds_right_to_repeat{repeat_num}_cds_strand'] == 1),
                             f'cds_right_to_repeat{repeat_num}_cds_product'])

        cds_pairs_df.loc[(~repeat_cds_is_on_forward_strand_filter) & (cds_pairs_df[f'cds_left_to_repeat{repeat_num}_cds_strand'] == -1),
                         f'cds_downstream_to_and_potentially_in_operon_with_repeat{repeat_num}_cds_product'] = (
            cds_pairs_df.loc[(~repeat_cds_is_on_forward_strand_filter) & (cds_pairs_df[f'cds_left_to_repeat{repeat_num}_cds_strand'] == -1),
                             f'cds_left_to_repeat{repeat_num}_cds_product'])

    cds_pairs_df['total_dist_between_repeat_cdss_and_surrounding_cdss'] = cds_pairs_df[[
        'dist_between_repeat1_cds_and_cds_left_to_it',
        'dist_between_repeat1_cds_and_cds_right_to_it',
        'dist_between_repeat2_cds_and_cds_left_to_it',
        'dist_between_repeat2_cds_and_cds_right_to_it',
    ]].sum(axis=1)

    for repeat_num in (1, 2):
        other_repeat_num = 3 - repeat_num

        first_ir_pair_curr_repeat_first_position_is_lower_filter = (cds_pairs_df[f'first_ir_pair_repeat{repeat_num}_first_position_in_cds'] <
                                                                    cds_pairs_df[f'first_ir_pair_repeat{other_repeat_num}_first_position_in_cds'])
        cds_pairs_df.loc[first_ir_pair_curr_repeat_first_position_is_lower_filter, 'normalized_dist_between_first_ir_pair_repeat_and_upstream_cds'] = (
            cds_pairs_df.loc[first_ir_pair_curr_repeat_first_position_is_lower_filter, f'dist_between_first_ir_pair_repeat{repeat_num}_and_upstream_cds'] /
            cds_pairs_df.loc[first_ir_pair_curr_repeat_first_position_is_lower_filter, f'dist_between_first_ir_pair_repeat{other_repeat_num}_and_upstream_cds']
        )

        last_ir_pair_curr_repeat_first_rposition_is_lower_filter = (cds_pairs_df[f'last_ir_pair_repeat{repeat_num}_first_rposition_in_cds'] <
                                                                    cds_pairs_df[f'last_ir_pair_repeat{other_repeat_num}_first_rposition_in_cds'])
        cds_pairs_df.loc[last_ir_pair_curr_repeat_first_rposition_is_lower_filter, 'normalized_dist_between_last_ir_pair_repeat_and_downstream_cds'] = (
            cds_pairs_df.loc[last_ir_pair_curr_repeat_first_rposition_is_lower_filter, f'dist_between_last_ir_pair_repeat{repeat_num}_and_downstream_cds'] /
            cds_pairs_df.loc[last_ir_pair_curr_repeat_first_rposition_is_lower_filter, f'dist_between_last_ir_pair_repeat{other_repeat_num}_and_downstream_cds']
        )

        cds_pairs_df[f'first_ir_pair_first_position_and_last_ir_pair_first_rposition_of_repeat{repeat_num}_cds_are_smaller'] = (
            cds_pairs_df[f'first_ir_pair_repeat{repeat_num}_first_position_is_smaller'] &
            cds_pairs_df[f'last_ir_pair_repeat{repeat_num}_first_rposition_is_smaller']
        )
    cds_pairs_df['first_ir_pair_first_position_and_last_ir_pair_first_rposition_of_the_same_repeat_cds_are_smaller'] = cds_pairs_df[[
        'first_ir_pair_first_position_and_last_ir_pair_first_rposition_of_repeat1_cds_are_smaller',
        'first_ir_pair_first_position_and_last_ir_pair_first_rposition_of_repeat2_cds_are_smaller',
    ]].any(axis=1)

    assert len(cds_pairs_df) == orig_num_of_cds_pairs
    return cds_pairs_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_filter_cds_pairs_for_context_analysis(
        input_file_path_cds_pairs_df_csv,
        prediction_column_name,
        output_file_path_filtered_cds_pairs_df_csv,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)

    cds_pairs_df = cds_pairs_df[cds_pairs_df[prediction_column_name]]

    cds_pairs_df.to_csv(output_file_path_filtered_cds_pairs_df_csv, sep='\t', index=False)

def filter_cds_pairs_for_context_analysis(
        input_file_path_cds_pairs_df_csv,
        prediction_column_name,
        output_file_path_filtered_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_filter_cds_pairs_for_context_analysis(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        prediction_column_name=prediction_column_name,
        output_file_path_filtered_cds_pairs_df_csv=output_file_path_filtered_cds_pairs_df_csv,
    )

def add_product_family_column(cds_df, list_of_product_and_product_family, orig_product_column_name, product_family_column_name):
    cds_df = cds_df.copy()
    assert orig_product_column_name in cds_df
    assert product_family_column_name not in cds_df

    product_and_product_family_df = pd.DataFrame(list_of_product_and_product_family, columns=[orig_product_column_name, product_family_column_name])
    assert product_and_product_family_df[orig_product_column_name].is_unique

    cds_df = cds_df.merge(product_and_product_family_df, how='left')
    product_family_not_set_filter = cds_df[product_family_column_name].isna()
    cds_df.loc[product_family_not_set_filter, product_family_column_name] = cds_df.loc[product_family_not_set_filter, orig_product_column_name]

    return cds_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_nearby_cds_product_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        num_of_cds_on_each_side,
        list_of_product_and_product_family,
        output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    cds_df = pd.read_csv(input_file_path_cds_df_csv, sep='\t', low_memory=False)[index_column_names.CDS_INDEX_COLUMN_NAMES + ['product', 'strand']]

    cds_df = add_product_family_column(cds_df, list_of_product_and_product_family, orig_product_column_name='product', product_family_column_name='product_family')
    cds_pairs_df = add_product_family_column(cds_pairs_df, list_of_product_and_product_family,
                                             orig_product_column_name='longer_repeat_cds_product', product_family_column_name='longer_repeat_cds_product_family')

    for nearby_cds_offset in range(1, num_of_cds_on_each_side + 1):
        for upstream_or_downstream in ('upstream', 'downstream'):
            plus_strand_offset = nearby_cds_offset * (1 if (upstream_or_downstream == 'downstream') else -1)
            column_name_to_nearby_cds_column_name = {
                x: f'{nearby_cds_offset}th_{upstream_or_downstream}_cds_{x}'
                for x in (set(cds_df) - {'nuccore_accession'})
            }
            # f'{nearby_cds_offset}th_{upstream_or_downstream}_cds_index_in_nuccore_cds_features_gb_file'
            nearby_cds_index_column_name = column_name_to_nearby_cds_column_name['index_in_nuccore_cds_features_gb_file']
            cds_pairs_df[nearby_cds_index_column_name] = (cds_pairs_df['longer_repeat_cds_index_in_nuccore_cds_features_gb_file'] +
                                                          plus_strand_offset * cds_pairs_df['longer_repeat_cds_strand'])
            # print('set(cds_pairs_df) & set(cds_df.rename(columns=column_name_to_nearby_cds_column_name))')
            # print(set(cds_pairs_df) & set(cds_df.rename(columns=column_name_to_nearby_cds_column_name)))
            cds_pairs_df = cds_pairs_df.merge(cds_df.rename(columns=column_name_to_nearby_cds_column_name), how='left')

    for nearby_cds_offset in range(1, num_of_cds_on_each_side + 1):
        for upstream_or_downstream in ('upstream', 'downstream'):
            cds_pairs_df[f'{nearby_cds_offset}th_{upstream_or_downstream}_cds_is_on_same_strand_as_longer_repeat_cds'] = (
                cds_pairs_df[f'{nearby_cds_offset}th_{upstream_or_downstream}_cds_strand'] == cds_pairs_df['longer_repeat_cds_strand']
            )
            cds_pairs_df[f'{nearby_cds_offset}th_{upstream_or_downstream}_cds_is_on_other_strand_than_longer_repeat_cds'] = (
                cds_pairs_df[f'{nearby_cds_offset}th_{upstream_or_downstream}_cds_strand'] == -cds_pairs_df['longer_repeat_cds_strand']
            )
            nearby_cds_exist_filter = ~cds_pairs_df[f'{nearby_cds_offset}th_{upstream_or_downstream}_cds_strand'].isna()
            cds_pairs_df.loc[nearby_cds_exist_filter, f'max_nearby_cds_offset_of_existing_{upstream_or_downstream}_cds'] = nearby_cds_offset

    cds_pairs_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)

def add_nearby_cds_product_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        num_of_cds_on_each_side,
        list_of_product_and_product_family,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_nearby_cds_product_columns(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        num_of_cds_on_each_side=num_of_cds_on_each_side,
        list_of_product_and_product_family=list_of_product_and_product_family,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )

def get_df_filtered_by_cds_pattern(df, product_family_column_name, cds_pattern, cds_same_strand_column_name, cds_other_strand_column_name):
    df = df.copy()
    assert type(cds_pattern) == tuple
    assert type(cds_pattern[0]) == set
    assert type(cds_pattern[1]) == set
    if 'not' in cds_pattern[0]:
        df = generic_utils.get_df_filtered_by_unallowed_column_values(df, product_family_column_name, cds_pattern[1])
    elif len(cds_pattern[1]) >= 1: # this allows only restricting the cds strand without restricting the product family.
        df = generic_utils.get_df_filtered_by_allowed_column_values(df, product_family_column_name, cds_pattern[1])

    if 'same_strand' in cds_pattern[0]:
        assert cds_same_strand_column_name is not None
        df = df[df[cds_same_strand_column_name]]
    if 'other_strand' in cds_pattern[0]:
        assert cds_other_strand_column_name is not None
        df = df[df[cds_other_strand_column_name]]

    return df

def get_cds_pairs_matching_context_df(cds_pairs_df, cds_context_info):
    print(f'orig num of cds pairs: {len(cds_pairs_df)}')
    cds_pairs_filtered_by_longer_repeat_cds_pattern_df = cds_pairs_df.copy()

    longer_linked_repeat_cds_product_families = cds_context_info['longer_linked_repeat_cds_product_families']
    if longer_linked_repeat_cds_product_families:
        cds_pairs_filtered_by_longer_repeat_cds_pattern_df = generic_utils.get_df_filtered_by_allowed_column_values(
            cds_pairs_filtered_by_longer_repeat_cds_pattern_df, 'longer_repeat_cds_product_family',
            longer_linked_repeat_cds_product_families,
        )
        print(f'len(cds_pairs_filtered_by_longer_repeat_cds_pattern_df) after longer_linked_repeat_cds_product_families filtering: '
              f'{len(cds_pairs_filtered_by_longer_repeat_cds_pattern_df)}')

    filtered_curr_dfs = []
    for cds_context in cds_context_info['cds_contexts']:
        curr_df = cds_pairs_filtered_by_longer_repeat_cds_pattern_df.copy()

        if 'upstream' in cds_context:
            assert type(cds_context['upstream']) == list
            upstream_filtered_curr_dfs = []
            for upstream_cds_context in cds_context['upstream']:
                curr_curr_df = curr_df.copy()
                # print(f'upstream_cds_context: {upstream_cds_context}')
                for cds_dist, curr_upstream_cds_context in upstream_cds_context.items():
                    # print(f'(cds_dist, curr_upstream_cds_context): {(cds_dist, curr_upstream_cds_context)}')
                    curr_curr_df = get_df_filtered_by_cds_pattern(
                        df=curr_curr_df,
                        product_family_column_name=f'{cds_dist}th_upstream_cds_product_family',
                        cds_pattern=curr_upstream_cds_context,
                        cds_same_strand_column_name=f'{cds_dist}th_upstream_cds_is_on_same_strand_as_longer_repeat_cds',
                        cds_other_strand_column_name=f'{cds_dist}th_upstream_cds_is_on_other_strand_than_longer_repeat_cds',
                    )
                upstream_filtered_curr_dfs.append(curr_curr_df)
            curr_df = pd.concat(upstream_filtered_curr_dfs, ignore_index=True).drop_duplicates()
            print(f'len(curr_df) after upstream filtering: {len(curr_df)}')

        if 'downstream' in cds_context:
            assert type(cds_context['downstream']) == list
            downstream_filtered_curr_dfs = []
            for downstream_cds_context in cds_context['downstream']:
                curr_curr_df = curr_df.copy()
                # print(f'downstream_cds_context: {downstream_cds_context}')
                for cds_dist, curr_downstream_cds_context in downstream_cds_context.items():
                    # print(f'(cds_dist, curr_downstream_cds_context): {(cds_dist, curr_downstream_cds_context)}')
                    curr_curr_df = get_df_filtered_by_cds_pattern(
                        df=curr_curr_df,
                        product_family_column_name=f'{cds_dist}th_downstream_cds_product_family',
                        cds_pattern=curr_downstream_cds_context,
                        cds_same_strand_column_name=f'{cds_dist}th_downstream_cds_is_on_same_strand_as_longer_repeat_cds',
                        cds_other_strand_column_name=f'{cds_dist}th_downstream_cds_is_on_other_strand_than_longer_repeat_cds',
                    )
                downstream_filtered_curr_dfs.append(curr_curr_df)
            curr_df = pd.concat(downstream_filtered_curr_dfs, ignore_index=True).drop_duplicates()
            print(f'len(curr_df) after downstream filtering: {len(curr_df)}')

        filtered_curr_dfs.append(curr_df)

    cds_pairs_matching_context_df = pd.concat(filtered_curr_dfs, ignore_index=True).drop_duplicates()

    print(f'final num of cds pairs in context: {len(cds_pairs_matching_context_df)}')
    return cds_pairs_matching_context_df


@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_cds_context_columns(
        input_file_path_cds_pairs_df_csv,
        name_to_cds_context_info,
        conflicting_and_final_cds_context_names,
        output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    orig_cds_pairs_df = cds_pairs_df.copy()
    orig_num_of_cds_pairs = len(orig_cds_pairs_df)

    cds_pairs_df['cds_context_name'] = np.nan
    conflicting_and_final_cds_context_names_df = pd.concat([
        pd.DataFrame(conflicting_and_final_cds_context_names, columns=['cds_context_name1', 'cds_context_name2', 'final_cds_context_name']),
        pd.DataFrame(conflicting_and_final_cds_context_names, columns=['cds_context_name2', 'cds_context_name1', 'final_cds_context_name']),
    ],ignore_index=True)
    assert len(conflicting_and_final_cds_context_names_df) == len(conflicting_and_final_cds_context_names_df.drop_duplicates())

    for name, cds_context_info in name_to_cds_context_info.items():
        print(f'\ncds_context_name: {name}')
        cds_pairs_matching_context_df = get_cds_pairs_matching_context_df(orig_cds_pairs_df, cds_context_info)
        assert 'curr_cds_context_name' not in cds_pairs_df
        cds_pairs_df = cds_pairs_df.merge(cds_pairs_matching_context_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].assign(curr_cds_context_name=name), how='left')
        assert len(cds_pairs_df) == orig_num_of_cds_pairs
        cds_pairs_df.loc[cds_pairs_df['cds_context_name'].isna(), 'cds_context_name'] = cds_pairs_df.loc[cds_pairs_df['cds_context_name'].isna(), 'curr_cds_context_name']

        assert 'final_cds_context_name' not in cds_pairs_df
        cds_pairs_df = cds_pairs_df.merge(conflicting_and_final_cds_context_names_df.rename(
            columns={'cds_context_name1': 'cds_context_name', 'cds_context_name2': 'curr_cds_context_name'}), how='left')
        assert len(cds_pairs_df) == orig_num_of_cds_pairs

        unsolved_conflict_filter = (
            (~cds_pairs_df['cds_context_name'].isna()) &
            (~cds_pairs_df['curr_cds_context_name'].isna()) &
            (cds_pairs_df['cds_context_name'] != cds_pairs_df['curr_cds_context_name']) &
            cds_pairs_df['final_cds_context_name'].isna()
        )
        if unsolved_conflict_filter.sum():
            print(cds_pairs_df[unsolved_conflict_filter][['cds_context_name', 'curr_cds_context_name']].value_counts())
            print(cds_pairs_df[unsolved_conflict_filter][['nuccore_accession', 'repeat1_cds_end_pos']])

        assert not unsolved_conflict_filter.any()
        cds_pairs_df.loc[~cds_pairs_df['final_cds_context_name'].isna(), 'cds_context_name'] = cds_pairs_df.loc[~cds_pairs_df['final_cds_context_name'].isna(),
                                                                                                                'final_cds_context_name']
        cds_pairs_df.drop(['curr_cds_context_name', 'final_cds_context_name'], axis=1, inplace=True)

    cds_pairs_df['cds_context_identified'] = ~(cds_pairs_df['cds_context_name'].isna())

    assert len(orig_cds_pairs_df) == orig_num_of_cds_pairs
    assert len(cds_pairs_df) == orig_num_of_cds_pairs

    assert len(cds_pairs_df) == orig_num_of_cds_pairs
    cds_pairs_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)


def add_cds_context_columns(
        input_file_path_cds_pairs_df_csv,
        name_to_cds_context_info,
        conflicting_and_final_cds_context_names,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_cds_context_columns(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        name_to_cds_context_info=name_to_cds_context_info,
        conflicting_and_final_cds_context_names=conflicting_and_final_cds_context_names,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_merged_padded_cds_index_interval_columns(
        input_file_path_cds_pairs_df_csv,
        cds_index_interval_margin_size,
        output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_cds_pairs = len(cds_pairs_df)

    nuccore_cds_index_dfs = []
    for nuccore_accession, nuccore_curr_df in cds_pairs_df.groupby('nuccore_accession'):
        orig_num_of_cds_pairs_in_nuccore = len(nuccore_curr_df)
        nuccore_cds_index_df = nuccore_curr_df[['repeat1_cds_index_in_nuccore_cds_features_gb_file',
                                                'repeat2_cds_index_in_nuccore_cds_features_gb_file']].copy()
        nuccore_cds_index_df['padded_cds_index_interval_start'] = nuccore_cds_index_df['repeat1_cds_index_in_nuccore_cds_features_gb_file'] - cds_index_interval_margin_size
        nuccore_cds_index_df['padded_cds_index_interval_end'] = nuccore_cds_index_df['repeat2_cds_index_in_nuccore_cds_features_gb_file'] + cds_index_interval_margin_size
        cds_index_padded_intervals = nuccore_cds_index_df[['padded_cds_index_interval_start', 'padded_cds_index_interval_end']].to_records(index=False).tolist()
        padded_interval_to_merged_padded_interval = generic_utils.naive_get_interval_to_merged_interval(cds_index_padded_intervals)

        nuccore_cds_index_df = nuccore_cds_index_df.merge(pd.DataFrame([
            (*padded_interval, *merged_padded_interval)
            for padded_interval, merged_padded_interval in padded_interval_to_merged_padded_interval.items()
        ], columns=['padded_cds_index_interval_start', 'padded_cds_index_interval_end',
                    'merged_padded_cds_index_interval_start', 'merged_padded_cds_index_interval_end']))
        assert len(nuccore_cds_index_df) == orig_num_of_cds_pairs_in_nuccore
        nuccore_cds_index_df['nuccore_accession'] = nuccore_accession
        nuccore_cds_index_dfs.append(nuccore_cds_index_df)

    cds_pairs_with_merged_cds_interval_df = cds_pairs_df.merge(pd.concat(nuccore_cds_index_dfs, ignore_index=False))
    assert len(cds_pairs_with_merged_cds_interval_df) == orig_num_of_cds_pairs

    cds_pairs_with_merged_cds_interval_df = cds_pairs_with_merged_cds_interval_df.merge(
        cds_pairs_with_merged_cds_interval_df[index_column_names.MERGED_PADDED_CDS_INDEX_INTERVAL_INDEX_COLUMN_NAMES].value_counts().reset_index(
        name='num_of_cds_pairs_in_merged_padded_cds_index_interval'))
    assert len(cds_pairs_with_merged_cds_interval_df) == orig_num_of_cds_pairs

    cds_pairs_with_merged_cds_interval_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)

def add_merged_padded_cds_index_interval_columns(
        input_file_path_cds_pairs_df_csv,
        cds_index_interval_margin_size,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_merged_padded_cds_index_interval_columns(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        cds_index_interval_margin_size=cds_index_interval_margin_size,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_taxon_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_and_taxon_scientific_names_df_csv,
        output_file_path_extended_cds_pairs_df_csv,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    nuccore_accession_and_taxon_scientific_names_df = pd.read_csv(input_file_path_nuccore_accession_and_taxon_scientific_names_df_csv, sep='\t', low_memory=False)
    assert set(cds_pairs_df) & set(nuccore_accession_and_taxon_scientific_names_df) == {'nuccore_accession'}

    orig_num_of_cds_pairs = len(cds_pairs_df)
    extended_cds_pairs_df = cds_pairs_df.merge(nuccore_accession_and_taxon_scientific_names_df)
    assert len(extended_cds_pairs_df) == orig_num_of_cds_pairs

    extended_cds_pairs_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)

def add_taxon_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_and_taxon_scientific_names_df_csv,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_taxon_columns(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_and_taxon_scientific_names_df_csv=input_file_path_nuccore_accession_and_taxon_scientific_names_df_csv,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
    )






def perform_cds_enrichment_analysis(search_for_pis_args):
    massive_screening_enrichment_analysis_out_dir_path = search_for_pis_args['enrichment_analysis']['output_dir_path']

    debug___num_of_taxa_to_go_over = search_for_pis_args['debug___num_of_taxa_to_go_over']
    debug___taxon_uids = search_for_pis_args['debug___taxon_uids']

    stage_out_file_name_suffix = massive_screening_stage_1.get_stage_out_file_name_suffix(
        debug___num_of_taxa_to_go_over=debug___num_of_taxa_to_go_over,
        debug___taxon_uids=debug___taxon_uids,
    )

    pathlib.Path(massive_screening_enrichment_analysis_out_dir_path).mkdir(parents=True, exist_ok=True)
    enrichment_analysis_log_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'enrichment_analysis_log.txt')
    logging.basicConfig(filename=enrichment_analysis_log_file_path, level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug(f'---------------starting perform_cds_enrichment_analysis({massive_screening_enrichment_analysis_out_dir_path})---------------')


    nuccore_entries_output_dir_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'primary_nuccore_entries')
    pathlib.Path(nuccore_entries_output_dir_path).mkdir(parents=True, exist_ok=True)

    enrichment_analysis_results_info_pickle_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path,
                                                                     search_for_pis_args['enrichment_analysis']['results_pickle_file_name'])
    enrichment_analysis_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        enrichment_analysis_results_info_pickle_file_path, stage_out_file_name_suffix)

    stage1_out_dir_path = search_for_pis_args['stage1']['output_dir_path']
    stage1_results_info_pickle_file_path = os.path.join(stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])
    stage1_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage1_results_info_pickle_file_path, stage_out_file_name_suffix)

    with open(stage1_results_info_pickle_file_path, 'rb') as f:
        stage1_results_info = pickle.load(f)
    all_cds_df_csv_file_path = stage1_results_info['all_cds_df_csv_file_path']
    nuccore_df_csv_file_path = stage1_results_info['nuccore_df_csv_file_path']
    taxa_df_csv_file_path = stage1_results_info['taxa_df_csv_file_path']
    nuccore_accession_to_nuccore_entry_info_pickle_file_path = stage1_results_info['nuccore_accession_to_nuccore_entry_info_pickle_file_path']
    # with open(nuccore_accession_to_nuccore_entry_info_pickle_file_path, 'rb') as f:
    #     nuccore_accession_to_nuccore_entry_info = pickle.load(f)


    stage5_out_dir_path = search_for_pis_args['stage5']['output_dir_path']
    stage5_results_info_pickle_file_path = os.path.join(stage5_out_dir_path, search_for_pis_args['stage5']['results_pickle_file_name'])
    stage5_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage5_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage5_results_info_pickle_file_path, 'rb') as f:
        stage5_results_info = pickle.load(f)

    pairs_with_highest_confidence_bps_df_csv_file_path = stage5_results_info['pairs_with_highest_confidence_bps_df_csv_file_path']
    extended_merged_cds_pair_region_df_csv_file_path = stage5_results_info['extended_merged_cds_pair_region_df_csv_file_path']

    with generic_utils.timing_context_manager('write_relevant_nuccore_accession_to_nuccore_entry_info'):
        before_clustering_relevant_nuccore_accession_to_nuccore_entry_info_pickle_file_path = os.path.join(
            massive_screening_enrichment_analysis_out_dir_path, 'before_clustering_relevant_nuccore_accession_to_nuccore_entry_info.pickle')
        before_clustering_relevant_nuccore_accession_to_nuccore_entry_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            before_clustering_relevant_nuccore_accession_to_nuccore_entry_info_pickle_file_path, stage_out_file_name_suffix)
        write_relevant_nuccore_accession_to_nuccore_entry_info(
            input_file_path_ir_pairs_df_csv=pairs_with_highest_confidence_bps_df_csv_file_path,
            input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=nuccore_accession_to_nuccore_entry_info_pickle_file_path,
            output_file_path_relevant_nuccore_accession_to_nuccore_entry_info_pickle=(
                before_clustering_relevant_nuccore_accession_to_nuccore_entry_info_pickle_file_path),
        )
    with generic_utils.timing_context_manager('keep_only_cds_of_nuccores_with_any_ir_pairs_etc'):
        before_clustering_relevant_cds_df_csv_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'before_clustering_relevant_cds_df.csv')
        before_clustering_relevant_cds_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            before_clustering_relevant_cds_df_csv_file_path, stage_out_file_name_suffix)
        massive_screening_stage_5.keep_only_cds_of_nuccores_with_any_ir_pairs_etc(
            input_file_path_cds_df_csv=all_cds_df_csv_file_path,
            input_file_path_pairs_df_csv=pairs_with_highest_confidence_bps_df_csv_file_path,
            output_file_path_filtered_cds_df_csv=before_clustering_relevant_cds_df_csv_file_path,
        )
    with generic_utils.timing_context_manager('write_operon_df'):
        operon_df_csv_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'operon_df.csv')
        operon_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            operon_df_csv_file_path, stage_out_file_name_suffix)
        write_operon_df(
            input_file_path_all_cds_df_csv=before_clustering_relevant_cds_df_csv_file_path,
            input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=before_clustering_relevant_nuccore_accession_to_nuccore_entry_info_pickle_file_path,
            output_file_path_operon_df_csv=operon_df_csv_file_path,
            max_dist_between_cds_in_operon=search_for_pis_args['enrichment_analysis']['max_dist_between_cds_in_operon'],
            nuccore_entries_output_dir_path=nuccore_entries_output_dir_path,
        )

    with generic_utils.timing_context_manager('write_ir_pairs_with_operons_df'):
        ir_pairs_with_operons_df_csv_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'ir_pairs_with_operons_df.csv')
        ir_pairs_with_operons_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            ir_pairs_with_operons_df_csv_file_path, stage_out_file_name_suffix)
        write_ir_pairs_with_operons_df(
            input_file_path_pairs_df_csv=pairs_with_highest_confidence_bps_df_csv_file_path,
            input_file_path_operon_df_csv=operon_df_csv_file_path,
            output_file_path_extended_pairs_df=ir_pairs_with_operons_df_csv_file_path,
        )

    with generic_utils.timing_context_manager('add_any_high_confidence_ir_pair_linked_to_cds_pair_column'):
        ir_pairs_with_cds_pair_high_confidence_column_df_csv_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path,
                                                                            'ir_pairs_with_cds_pair_high_confidence_column_df.csv')
        ir_pairs_with_cds_pair_high_confidence_column_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
                ir_pairs_with_cds_pair_high_confidence_column_df_csv_file_path, stage_out_file_name_suffix)
        add_any_high_confidence_ir_pair_linked_to_cds_pair_column(
            input_file_path_ir_pairs_df_csv=ir_pairs_with_operons_df_csv_file_path,
            output_file_path_extended_ir_pairs_df_csv=ir_pairs_with_cds_pair_high_confidence_column_df_csv_file_path,
        )

    with generic_utils.timing_context_manager('write_cds_pairs_df'):
        all_cds_pairs_df_csv_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'all_cds_pairs_df.csv')
        all_cds_pairs_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            all_cds_pairs_df_csv_file_path, stage_out_file_name_suffix)
        write_cds_pairs_df(
            input_file_path_ir_pairs_df_csv=ir_pairs_with_cds_pair_high_confidence_column_df_csv_file_path,
            output_file_path_cds_pairs_df_csv=all_cds_pairs_df_csv_file_path,
        )

    with generic_utils.timing_context_manager('write_repeats_with_margins_to_fasta'):
        repeat_cds_seqs_fasta_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'repeat_cdss.fasta')
        repeat_cds_seqs_fasta_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            repeat_cds_seqs_fasta_file_path, stage_out_file_name_suffix)
        repeat_seq_name_df_csv_file_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path, 'repeat_seq_name_df.csv')
        repeat_seq_name_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            repeat_seq_name_df_csv_file_path, stage_out_file_name_suffix)
        writing_repeat_cdss_to_fasta.write_repeat_cdss_to_fasta(
            input_file_path_cds_pairs_df_csv=all_cds_pairs_df_csv_file_path,
            input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=nuccore_accession_to_nuccore_entry_info_pickle_file_path,
            output_file_path_repeat_cds_seqs_fasta=repeat_cds_seqs_fasta_file_path,
            output_file_path_repeat_cds_seq_name_df_csv=repeat_seq_name_df_csv_file_path,
        )

    with generic_utils.timing_context_manager('min_pairwise_identity_with_cluster_centroid_to_pair_clustering_extended_info'):
        min_pairwise_identity_with_cluster_centroid = search_for_pis_args['enrichment_analysis']['clustering']['min_pairwise_identity_with_cluster_centroid']
        generic_utils.print_and_write_to_log(f'starting clustering with min_pairwise_identity_with_cluster_centroid={min_pairwise_identity_with_cluster_centroid}')
        clustering_out_dir_path = os.path.join(massive_screening_enrichment_analysis_out_dir_path,
                                               f"clustering_with_min_identity{str(min_pairwise_identity_with_cluster_centroid).replace('.', '_')}")
        clustering_out_dir_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            clustering_out_dir_path, stage_out_file_name_suffix)
        pathlib.Path(clustering_out_dir_path).mkdir(parents=True, exist_ok=True)

    if min_pairwise_identity_with_cluster_centroid is not None:
        with generic_utils.timing_context_manager(f'cluster_cds_pairs (min_pairwise_identity_with_cluster_centroid: {min_pairwise_identity_with_cluster_centroid})'):
            pair_clustering_extended_info_pickle_file_path = os.path.join(clustering_out_dir_path, 'pair_clustering_extended_info.pickle')
            cluster_cds_pairs(
                min_pairwise_identity_with_cluster_centroid=min_pairwise_identity_with_cluster_centroid,
                input_file_path_cds_pairs_df_csv=all_cds_pairs_df_csv_file_path,
                input_file_path_repeat_cds_seqs_fasta=repeat_cds_seqs_fasta_file_path,
                input_file_path_repeat_seq_name_df_csv=repeat_seq_name_df_csv_file_path,
                output_file_path_pair_clustering_extended_info_pickle=pair_clustering_extended_info_pickle_file_path,
                pairwise_identity_definition_type=search_for_pis_args['enrichment_analysis']['clustering']['pairwise_identity_definition_type'],
                clustering_out_dir_path=clustering_out_dir_path,
            )
            with open(pair_clustering_extended_info_pickle_file_path, 'rb') as f:
                pair_clustering_extended_info = pickle.load(f)

        cds_pairs_clustered_df_csv_file_path = pair_clustering_extended_info['pairs_with_cluster_indices_df_csv_file_path']
    else:
        cds_pairs_clustered_df_csv_file_path = all_cds_pairs_df_csv_file_path

    if search_for_pis_args['enrichment_analysis']['DEBUG___RANDOM_ANY_HIGH_CONFIDENCE_IR_PAIR_LINKED_TO_CDS_PAIR']:
        clustering_out_dir_path = os.path.join(clustering_out_dir_path, 'DEBUG___random_high_confidence')
        pathlib.Path(clustering_out_dir_path).mkdir(parents=True, exist_ok=True)
        debug__cds_pairs_clustered_df = pd.read_csv(cds_pairs_clustered_df_csv_file_path, sep='\t', low_memory=False)
        np.random.seed(1)
        debug__cds_pairs_clustered_df['any_high_confidence_ir_pair_linked_to_cds_pair'] = pd.Series(
            np.random.randint(0, 2, len(debug__cds_pairs_clustered_df))).astype(bool)
        cds_pairs_clustered_df_csv_file_path = os.path.join(clustering_out_dir_path, 'DEBUG___cds_pairs_clustered_df.csv')
        debug__cds_pairs_clustered_df.to_csv(cds_pairs_clustered_df_csv_file_path, sep='\t', index=False)
    elif search_for_pis_args['enrichment_analysis']['DEBUG___SHUFFLED_PRODUCTS']:
        clustering_out_dir_path = os.path.join(clustering_out_dir_path, 'DEBUG___shuffled_products')
        pathlib.Path(clustering_out_dir_path).mkdir(parents=True, exist_ok=True)
    elif search_for_pis_args['enrichment_analysis']['DEBUG___SHUFFLE_OPERON_ASYMMETRY']:
        clustering_out_dir_path = os.path.join(clustering_out_dir_path, 'DEBUG___shuffled_products')
        pathlib.Path(clustering_out_dir_path).mkdir(parents=True, exist_ok=True)
        debug__cds_pairs_clustered_df = pd.read_csv(cds_pairs_clustered_df_csv_file_path, sep='\t', low_memory=False)
        debug__cds_pairs_clustered_df['operon_asymmetry'] = debug__cds_pairs_clustered_df['operon_asymmetry'].sample(frac=1, random_state=0).values
        cds_pairs_clustered_df_csv_file_path = os.path.join(clustering_out_dir_path, 'DEBUG___cds_pairs_clustered_df.csv')
        debug__cds_pairs_clustered_df.to_csv(cds_pairs_clustered_df_csv_file_path, sep='\t', index=False)

    filtered_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path = os.path.join(
        clustering_out_dir_path, 'filtered_cds_pairs_for_enrichment_analysis_and_logistic_regression_df.csv')
    write_cds_pairs_relevant_for_genomic_architecture_analysis_and_logistic_training(
        input_file_path_cds_pairs_df_csv=cds_pairs_clustered_df_csv_file_path,
        input_file_path_merged_cds_pair_region_df_csv=extended_merged_cds_pair_region_df_csv_file_path,
        min_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds=1,
        output_file_path_filtered_cds_pairs_df_csv=filtered_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path,
    )

    names_of_columns_whose_binarized_versions_are_used_in_logistic_regression = search_for_pis_args['enrichment_analysis'][
        'names_of_columns_whose_binarized_versions_are_used_in_logistic_regression']
    column_name_to_critical_val_and_passed_threshold_column_name_pickle_file_path = os.path.join(
        clustering_out_dir_path, 'column_name_to_critical_val_and_passed_threshold_column_name.pickle')
    write_column_name_to_critical_val_and_passed_threshold_column_name_pickle(
        input_file_path_cds_pairs_df_csv=filtered_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path,
        column_names=names_of_columns_whose_binarized_versions_are_used_in_logistic_regression,
        output_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle=(
            column_name_to_critical_val_and_passed_threshold_column_name_pickle_file_path),
    )

    extended1_all_cds_pairs_df_csv_file_path = os.path.join(clustering_out_dir_path, 'extended1_all_cds_pairs_df.csv')
    add_passed_threshold_columns(
        input_file_path_cds_pairs_df_csv=cds_pairs_clustered_df_csv_file_path,
        input_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle=(
            column_name_to_critical_val_and_passed_threshold_column_name_pickle_file_path),
        output_file_path_extended_cds_pairs_df_csv=extended1_all_cds_pairs_df_csv_file_path,
    )

    extended1_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path = os.path.join(clustering_out_dir_path, 'extended1_cds_pairs_for_enrichment_analysis_and_logistic_regression_df.csv')
    add_passed_threshold_columns(
        input_file_path_cds_pairs_df_csv=filtered_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path,
        input_file_path_column_name_to_critical_val_and_passed_threshold_column_name_pickle=(
            column_name_to_critical_val_and_passed_threshold_column_name_pickle_file_path),
        output_file_path_extended_cds_pairs_df_csv=extended1_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path,
    )

    with open(column_name_to_critical_val_and_passed_threshold_column_name_pickle_file_path, 'rb') as f:
        column_name_to_critical_val_and_passed_threshold_column_name = pickle.load(f)
    names_of_threshold_columns_to_use_in_logistic_regression = sorted(
        x for _, x in column_name_to_critical_val_and_passed_threshold_column_name.values())

    with generic_utils.timing_context_manager('logistic regression'):
        logistic_regression_dependent_var_column_name = search_for_pis_args['enrichment_analysis']['logistic_regression_dependent_var_column_name']
        logistic_regression_fit_result_df_csv_file_path = os.path.join(clustering_out_dir_path, 'logistic_regression_fit_result_df.csv')
        logistic_regression_extended_cds_pairs_df_csv_file_path = os.path.join(clustering_out_dir_path, 'logistic_regression_extended_cds_pairs_df.csv')
        try:
            perform_logistic_regression(
                input_file_path_cds_pairs_df_for_training_csv=extended1_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path,
                input_file_path_cds_pairs_df_for_prediction_csv=extended1_all_cds_pairs_df_csv_file_path,
                predictor_column_names=names_of_threshold_columns_to_use_in_logistic_regression,
                logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
                perform_unadjusted_regressions=True,
                output_file_path_logistic_regression_fit_result_df_csv=logistic_regression_fit_result_df_csv_file_path,
                output_file_path_cds_pairs_df_with_prediction_csv=logistic_regression_extended_cds_pairs_df_csv_file_path,
            )
        except statsmodels.tools.sm_exceptions.PerfectSeparationError as err:
            logistic_regression_converged = False
            print(f'\n\n\n\nlogistic regression failed to converge!\nstr(err):\n{str(err)}\n\n\n')
        else:
            logistic_regression_fit_result_df = pd.read_csv(logistic_regression_fit_result_df_csv_file_path, sep='\t', low_memory=False)
            logistic_regression_converged = True
            print(logistic_regression_fit_result_df)

    generic_utils.print_and_write_to_log('after perform_logistic_regression()')

    predicted_probability_threshold = search_for_pis_args['enrichment_analysis']['min_predicted_rearrangement_probability']

    with generic_utils.timing_context_manager('logistic regression model assessment'):
        logistic_model_assessment_out_dir_path = os.path.join(clustering_out_dir_path, 'logistic_model_assessment')
        pathlib.Path(logistic_model_assessment_out_dir_path).mkdir(parents=True, exist_ok=True)
        cds_pairs_for_logistic_model_assessment_df_csv_file_path = os.path.join(logistic_model_assessment_out_dir_path,
                                                                                'cds_pairs_for_logistic_model_assessment_df.csv')

        write_minimal_cds_pairs_for_logistic_model_assessment(
            input_file_path_cds_pairs_df_csv=extended1_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path,
            predictor_column_names=names_of_threshold_columns_to_use_in_logistic_regression,
            logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
            output_file_path_minimal_cds_pairs_df_csv=cds_pairs_for_logistic_model_assessment_df_csv_file_path,
        )

        unified_roc_curve_df_csv_file_path = os.path.join(logistic_model_assessment_out_dir_path, 'unified_roc_curve_df.csv')
        concat_simulation_fit_result_dfs_csv_file_path = os.path.join(logistic_model_assessment_out_dir_path, 'concat_simulation_fit_result_dfs.csv')
        assess_logistic_regression_model(
            input_file_path_cds_pairs_df_csv=cds_pairs_for_logistic_model_assessment_df_csv_file_path,
            predictor_column_names=names_of_threshold_columns_to_use_in_logistic_regression,
            logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
            test_set_fraction=0.2,
            num_of_simulations=500,
            predicted_probability_threshold=predicted_probability_threshold,
            output_file_path_unified_roc_curve_df_csv=unified_roc_curve_df_csv_file_path,
            output_file_path_concat_simulation_fit_result_dfs_csv=concat_simulation_fit_result_dfs_csv_file_path,
            output_dir_path=logistic_model_assessment_out_dir_path,
        )

    if (not logistic_regression_converged) or (
            search_for_pis_args['enrichment_analysis']['DEBUG___RANDOM_ANY_HIGH_CONFIDENCE_IR_PAIR_LINKED_TO_CDS_PAIR'] or
            search_for_pis_args['enrichment_analysis']['DEBUG___SHUFFLE_OPERON_ASYMMETRY']
    ):
        extended2_all_cds_pairs_df_csv_file_path = None
        repeat_cds_product_fisher_or_g_result_df_csv_file_path = None
        extended3_all_cds_pairs_df_csv_file_path = None
        adjacent_repeat_cds_product_fisher_or_g_result_df_csv_file_path = None
    else:
        extended2_all_cds_pairs_df_csv_file_path = os.path.join(clustering_out_dir_path, 'extended2_all_cds_pairs_df.csv')
        ir_pairs_linked_df_extension.add_prev_and_next_cds_product_columns(
            input_file_path_cds_pairs_df_csv=logistic_regression_extended_cds_pairs_df_csv_file_path,
            input_file_path_cds_df_csv=before_clustering_relevant_cds_df_csv_file_path,
            set_to_nan_prev_or_next_if_it_is_a_repeat_cds=True,
            output_file_path_extended_cds_pairs_df_csv=extended2_all_cds_pairs_df_csv_file_path,
        )

        if search_for_pis_args['enrichment_analysis']['DEBUG___SHUFFLED_PRODUCTS']:
            debug__extended2_all_cds_pairs_df = pd.read_csv(extended2_all_cds_pairs_df_csv_file_path, sep='\t', low_memory=False)
            for i, column_name in enumerate((
                'longer_repeat_cds_product',
                'repeat1_prev_cds_product',
                'repeat1_next_cds_product',
                'repeat2_prev_cds_product',
                'repeat2_next_cds_product',
            )):
                debug__extended2_all_cds_pairs_df[column_name] = debug__extended2_all_cds_pairs_df[column_name].sample(frac=1, random_state=i).values
            extended2_all_cds_pairs_df_csv_file_path = f'{extended2_all_cds_pairs_df_csv_file_path}.DEBUG.csv'
            debug__extended2_all_cds_pairs_df.to_csv(extended2_all_cds_pairs_df_csv_file_path, sep='\t', index=False)

        predicted_probability_threshold_dir_path = os.path.join(clustering_out_dir_path, f'predicted_probability_threshold_{predicted_probability_threshold}')
        pathlib.Path(predicted_probability_threshold_dir_path).mkdir(parents=True, exist_ok=True)

        extended3_all_cds_pairs_df_csv_file_path = os.path.join(predicted_probability_threshold_dir_path, 'extended3_all_cds_pairs_df.csv')
        add_prediction_column(
            input_file_path_cds_pairs_df_csv=extended2_all_cds_pairs_df_csv_file_path,
            logistic_regression_dependent_var_column_name=logistic_regression_dependent_var_column_name,
            min_predicted_rearrangement_probability=predicted_probability_threshold,
            output_file_path_extended_cds_pairs_df_csv=extended3_all_cds_pairs_df_csv_file_path,
        )

        if 1:
            extended3_all_cds_pairs_df = pd.read_csv(extended3_all_cds_pairs_df_csv_file_path, sep='\t', low_memory=False)
            print(extended3_all_cds_pairs_df[f'predicted_{logistic_regression_dependent_var_column_name}'].value_counts())

        test_column_name_for_each_product_comparing_merged_cds_pair_regions = search_for_pis_args[
            'enrichment_analysis']['test_column_name_for_each_product_comparing_merged_cds_pair_regions']
        min_num_of_cds_pairs_with_product_for_enrichment_test = search_for_pis_args['enrichment_analysis'][
            'min_num_of_cds_pairs_with_product_for_enrichment_test']

        extended3_all_cds_pair_representatives_df_csv_file_path = os.path.join(clustering_out_dir_path, 'extended3_all_cds_pair_representatives_df.csv')
        write_cds_pair_representatives_df_for_enrichment_analysis(
            input_file_path_cds_pairs_df_csv=extended3_all_cds_pairs_df_csv_file_path,
            output_file_path_filtered_cds_pairs_df_csv=extended3_all_cds_pair_representatives_df_csv_file_path,
        )
        with generic_utils.timing_context_manager('repeat cdss perform_fisher_or_g_for_each_product_comparing_cds_pairs'):
            repeat_cds_product_fisher_or_g_result_df_csv_file_path = os.path.join(
                predicted_probability_threshold_dir_path, f'repeat_cds_product_tests_for_each_product_result_df__min_num_of_cds_pairs_with_product_'
                                         f'{min_num_of_cds_pairs_with_product_for_enrichment_test}.csv')
            perform_fisher_or_g_for_each_product_comparing_cds_pairs(
                input_file_path_cds_pairs_df_csv=extended3_all_cds_pair_representatives_df_csv_file_path,
                product_column_names=['repeat1_cds_product', 'repeat2_cds_product'],
                min_num_of_cds_pairs_with_product_for_enrichment_test=min_num_of_cds_pairs_with_product_for_enrichment_test,
                test_column_name=test_column_name_for_each_product_comparing_merged_cds_pair_regions,
                output_file_path_repeat_cds_product_fisher_or_g_result_df_csv=repeat_cds_product_fisher_or_g_result_df_csv_file_path,
            )

            repeat_cds_product_fisher_or_g_result_df = pd.read_csv(repeat_cds_product_fisher_or_g_result_df_csv_file_path, sep='\t', low_memory=False)
            print(repeat_cds_product_fisher_or_g_result_df[['product', 'odds_ratio', 'matrix_for_test_true_true', 'corrected_pvalue', 'test_performed']].head(35))

        with generic_utils.timing_context_manager('repeat cdss perform_fisher_or_g_for_each_product_comparing_cds_pairs'):
            adjacent_repeat_cds_product_fisher_or_g_result_df_csv_file_path = os.path.join(
                predicted_probability_threshold_dir_path, f'adjacent_repeat_cds_product_tests_for_each_product_result_df__min_num_of_cds_pairs_with_product_'
                                         f'{min_num_of_cds_pairs_with_product_for_enrichment_test}.csv')
            perform_fisher_or_g_for_each_product_comparing_cds_pairs(
                input_file_path_cds_pairs_df_csv=extended3_all_cds_pair_representatives_df_csv_file_path,
                product_column_names=['repeat1_prev_cds_product', 'repeat1_next_cds_product', 'repeat2_prev_cds_product', 'repeat2_next_cds_product'],
                min_num_of_cds_pairs_with_product_for_enrichment_test=min_num_of_cds_pairs_with_product_for_enrichment_test,
                test_column_name=test_column_name_for_each_product_comparing_merged_cds_pair_regions,
                output_file_path_repeat_cds_product_fisher_or_g_result_df_csv=adjacent_repeat_cds_product_fisher_or_g_result_df_csv_file_path,
            )

            adjacent_repeat_cds_product_fisher_or_g_result_df = pd.read_csv(adjacent_repeat_cds_product_fisher_or_g_result_df_csv_file_path, sep='\t', low_memory=False)
            print(adjacent_repeat_cds_product_fisher_or_g_result_df[['product', 'odds_ratio', 'matrix_for_test_true_true', 'corrected_pvalue', 'test_performed']].head(35))


        if 0:
            # used something similar to this to identify recurring genomic contexts.
            # For these contexts, I searched for similar loci with long-read SRA entries...
            cds_pairs_for_context_analysis_df_csv_file_path = os.path.join(predicted_probability_threshold_dir_path, 'cds_pairs_for_context_analysis_df.csv')
            filter_cds_pairs_for_context_analysis(
                input_file_path_cds_pairs_df_csv=extended3_all_cds_pairs_df_csv_file_path,
                prediction_column_name=f'predicted_{logistic_regression_dependent_var_column_name}',
                output_file_path_filtered_cds_pairs_df_csv=cds_pairs_for_context_analysis_df_csv_file_path,
            )

            cds_relevant_for_context_analysis_df_csv_file_path = os.path.join(predicted_probability_threshold_dir_path, 'cds_relevant_for_context_analysis_df.csv')
            massive_screening_stage_5.keep_only_cds_of_nuccores_with_any_ir_pairs_etc(
                input_file_path_cds_df_csv=before_clustering_relevant_cds_df_csv_file_path,
                input_file_path_pairs_df_csv=cds_pairs_for_context_analysis_df_csv_file_path,
                output_file_path_filtered_cds_df_csv=cds_relevant_for_context_analysis_df_csv_file_path,
            )

            extended_cds_pairs_for_context_analysis_df_csv_file_path = os.path.join(predicted_probability_threshold_dir_path, 'extended_cds_pairs_for_context_analysis_df.csv')
            num_of_cds_on_each_side_for_context_analysis = search_for_pis_args['enrichment_analysis']['num_of_cds_on_each_side_for_context_analysis']
            list_of_product_and_product_family = search_for_pis_args['enrichment_analysis']['list_of_product_and_product_family']
            add_nearby_cds_product_columns(
                input_file_path_cds_pairs_df_csv=cds_pairs_for_context_analysis_df_csv_file_path,
                input_file_path_cds_df_csv=cds_relevant_for_context_analysis_df_csv_file_path,
                num_of_cds_on_each_side=num_of_cds_on_each_side_for_context_analysis,
                list_of_product_and_product_family=list_of_product_and_product_family,
                output_file_path_extended_cds_pairs_df_csv=extended_cds_pairs_for_context_analysis_df_csv_file_path,
            )

            cds_pairs_with_cds_context_df_csv_file_path = os.path.join(predicted_probability_threshold_dir_path, 'cds_pairs_with_cds_context_df.csv')
            add_cds_context_columns(
                input_file_path_cds_pairs_df_csv=extended_cds_pairs_for_context_analysis_df_csv_file_path,
                name_to_cds_context_info=search_for_pis_args['enrichment_analysis']['name_to_cds_context_info'],
                conflicting_and_final_cds_context_names=search_for_pis_args['enrichment_analysis']['conflicting_and_final_cds_context_names'],
                output_file_path_extended_cds_pairs_df_csv=cds_pairs_with_cds_context_df_csv_file_path,
            )


    enrichment_analysis_results_info = {
        'before_clustering_relevant_cds_df_csv_file_path': before_clustering_relevant_cds_df_csv_file_path,

        'all_cds_pairs_df_csv_file_path': all_cds_pairs_df_csv_file_path,

        'pair_clustering_extended_info_pickle_file_path': pair_clustering_extended_info_pickle_file_path,

        'extended1_all_cds_pairs_df_csv_file_path': extended1_all_cds_pairs_df_csv_file_path,
        'extended1_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path': extended1_cds_pairs_for_enrichment_analysis_and_logistic_regression_df_csv_file_path,

        'column_name_to_critical_val_and_passed_threshold_column_name_pickle_file_path': (
            column_name_to_critical_val_and_passed_threshold_column_name_pickle_file_path),

        'logistic_regression_converged': logistic_regression_converged,
        'logistic_regression_fit_result_df_csv_file_path': logistic_regression_fit_result_df_csv_file_path,

        'unified_roc_curve_df_csv_file_path': unified_roc_curve_df_csv_file_path,
        'concat_simulation_fit_result_dfs_csv_file_path': concat_simulation_fit_result_dfs_csv_file_path,

        'extended2_all_cds_pairs_df_csv_file_path': extended2_all_cds_pairs_df_csv_file_path,

        'repeat_cds_product_fisher_or_g_result_df_csv_file_path': repeat_cds_product_fisher_or_g_result_df_csv_file_path,

        'extended3_all_cds_pairs_df_csv_file_path': extended3_all_cds_pairs_df_csv_file_path,
        'adjacent_repeat_cds_product_fisher_or_g_result_df_csv_file_path': adjacent_repeat_cds_product_fisher_or_g_result_df_csv_file_path,


        # 'cds_relevant_for_context_analysis_df_csv_file_path': cds_relevant_for_context_analysis_df_csv_file_path,
        # 'cds_pairs_for_context_analysis_df_csv_file_path': cds_pairs_for_context_analysis_df_csv_file_path,
        # 'extended_cds_pairs_for_context_analysis_df_csv_file_path': extended_cds_pairs_for_context_analysis_df_csv_file_path,
        #
        # 'cds_pairs_with_merged_padded_cds_index_interval_df_csv_file_path': cds_pairs_with_merged_padded_cds_index_interval_df_csv_file_path,

        # 'nuccore_accession_and_taxon_scientific_names_df_csv_file_path': nuccore_accession_and_taxon_scientific_names_df_csv_file_path, # not really a result.
    }
    with open(enrichment_analysis_results_info_pickle_file_path, 'wb') as f:
        pickle.dump(enrichment_analysis_results_info, f, protocol=4)

    return enrichment_analysis_results_info
    # exit()
    #
    # with generic_utils.timing_context_manager('all_cds_df = pd.read_csv()'):
    #     all_cds_df = pd.read_csv(all_cds_df_csv_file_path, sep='\t', dtype={'product': 'string'}, low_memory=False)
    #
    # with generic_utils.timing_context_manager('pairs_df = pd.read_csv()'):
    #     pairs_df = pd.read_csv(filtered_pairs_cluster_representatives_df_csv_file_path, sep='\t', low_memory=False)


def main():
    perform_cds_enrichment_analysis(search_for_pis_args=massive_screening_configuration.SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT)


if __name__ == '__main__':
    main()
