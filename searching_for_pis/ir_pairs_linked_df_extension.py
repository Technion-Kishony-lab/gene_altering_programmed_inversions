import os.path
import pathlib
import pickle

import numpy as np
import pandas as pd

from generic import bio_utils
from generic import blast_interface_and_utils
from generic import generic_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


class AnyCdsPartContainsNonACGT(Exception):
    pass

@generic_utils.execute_if_output_doesnt_exist_already
def cached_blast_to_estimate_repeat_copy_num(
        nuccore_accession,
        input_file_path_nuccore_fasta,
        taxon_blast_db_path,
        taxon_blast_db_caching_file_contents,
        ir_pair,
        min_dist_from_ir_pair_region_for_alignments,
        max_evalue,
        seed_len,
        output_file_path_repeat_copy_num_info_pickle,
        ir_pair_output_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    blasted_repeat1 = ir_pair[0] % 2 == 0
    blasted_repeat = ir_pair[:2] if blasted_repeat1 else ir_pair[2:]
    blasted_repeat_as_str = '_'.join(map(str, blasted_repeat))

    blasted_repeat_fasta_file_path = os.path.join(ir_pair_output_dir_path, f'{blasted_repeat_as_str}.fasta')
    nuccore_entry_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(input_file_path_nuccore_fasta)
    blasted_repeat_seq = bio_utils.get_region_in_chrom_seq(
        chrom_seq=nuccore_entry_seq,
        start_position=blasted_repeat[0],
        end_position=blasted_repeat[1],
        region_name=blasted_repeat_as_str,
    )
    bio_utils.write_records_to_fasta_or_gb_file([blasted_repeat_seq], blasted_repeat_fasta_file_path)

    blast_results_csv_file_path = os.path.join(ir_pair_output_dir_path, f'blast_{blasted_repeat_as_str}_results.csv')

    blast_interface_and_utils.blast_nucleotide(
        query_fasta_file_path=blasted_repeat_fasta_file_path,
        blast_db_path=taxon_blast_db_path,
        blast_results_file_path=blast_results_csv_file_path,
        perform_gapped_alignment=False,
        query_strand_to_search='both',
        max_evalue=max_evalue,
        seed_len=seed_len,
        verbose=False,
    )

    alignments_df = blast_interface_and_utils.read_blast_results_df(blast_results_csv_file_path)

    region_around_ir_pair_start = ir_pair[0] - min_dist_from_ir_pair_region_for_alignments
    region_around_ir_pair_end = ir_pair[-1] + min_dist_from_ir_pair_region_for_alignments

    # discard alignments that overlap region_around_ir_pair.
    orig_num_of_alignments = len(alignments_df)

    # print(f'\n\nir_pair: {ir_pair}')
    # print(alignments_df)
    alignments_df = alignments_df[
        (alignments_df['sseqid'] != nuccore_accession) |
        (alignments_df[['sstart', 'send']].max(axis=1) < region_around_ir_pair_start) |
        (alignments_df[['sstart', 'send']].min(axis=1) > region_around_ir_pair_end)
    ]
    num_of_alignments_overlapping_region_around_ir_pair = orig_num_of_alignments - len(alignments_df)
    # print(alignments_df)

    if 1:
        # this is the estimated copy num of the base (or multiple bases) with the highest number of alignments.
        estimated_copy_num = generic_utils.get_max_num_of_overlapping_intervals(alignments_df[['qstart', 'qend']])
    else:
        # this is the mean estimated copy num, over all bases in the repeat.
        repeat_len = ir_pair[1] - ir_pair[0] + 1
        estimated_copy_num = alignments_df['length'].sum() / repeat_len


    repeat_copy_num_info = {
        'blasted_repeat_fasta_file_path': blasted_repeat_fasta_file_path,
        'blast_results_csv_file_path': blast_results_csv_file_path,

        'blasted_repeat1': blasted_repeat1,
        'num_of_alignments_overlapping_region_around_ir_pair': num_of_alignments_overlapping_region_around_ir_pair,
        'estimated_copy_num': estimated_copy_num,
    }

    with open(output_file_path_repeat_copy_num_info_pickle, 'wb') as f:
        pickle.dump(repeat_copy_num_info, f, protocol=4)

def blast_to_estimate_repeat_copy_num(
        nuccore_accession,
        input_file_path_nuccore_fasta,
        taxon_blast_db_path,
        ir_pair,
        min_dist_from_ir_pair_region_for_alignments,
        max_evalue,
        seed_len,
        output_file_path_repeat_copy_num_info_pickle,
        ir_pair_output_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_blast_to_estimate_repeat_copy_num(
        nuccore_accession=nuccore_accession,
        input_file_path_nuccore_fasta=input_file_path_nuccore_fasta,
        taxon_blast_db_path=taxon_blast_db_path,
        taxon_blast_db_caching_file_contents=generic_utils.read_text_file(blast_interface_and_utils.get_file_used_for_blast_db_caching_path(taxon_blast_db_path)),
        ir_pair=ir_pair,
        min_dist_from_ir_pair_region_for_alignments=min_dist_from_ir_pair_region_for_alignments,
        max_evalue=max_evalue,
        seed_len=seed_len,
        output_file_path_repeat_copy_num_info_pickle=output_file_path_repeat_copy_num_info_pickle,
        ir_pair_output_dir_path=ir_pair_output_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=4,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_find_evidence_for_repeat_overlapping_mobile_element_for_taxon(
        input_file_path_taxon_pairs_df_csv,
        taxon_blast_db_path,
        min_dist_from_ir_pair_region_for_alignments,
        max_evalue,
        seed_len,
        output_file_path_taxon_pairs_with_estimated_copy_num_df_csv,
        output_file_path_taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle,
        nuccore_entries_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    taxon_pairs_df = pd.read_csv(input_file_path_taxon_pairs_df_csv, sep='\t', low_memory=False)

    taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path = {}
    repeat_estimated_copy_num_dicts = []
    taxon_num_of_nuccore_entries = taxon_pairs_df['nuccore_accession'].nunique()
    for j, (nuccore_accession, nuccore_pairs_df) in enumerate(taxon_pairs_df.groupby('nuccore_accession', sort=False)):
        generic_utils.print_and_write_to_log(f'starting work on nuccore {j + 1}/{taxon_num_of_nuccore_entries}: {nuccore_accession}')
        nuccore_entry_fasta_file_path = nuccore_pairs_df['fasta_file_path'].iloc[0]
        nuccore_entry_output_dir_path = os.path.join(nuccore_entries_out_dir_path, nuccore_accession)
        nuccore_entry_ir_pairs_output_dir_path = os.path.join(nuccore_entry_output_dir_path, 'ir_pairs')

        ir_pair_to_repeat_copy_num_info_pickle_file_path = {}
        num_of_repeats_to_estimate_copy_num = len(nuccore_pairs_df)
        for k, (_, pair_df_row) in enumerate(nuccore_pairs_df.iterrows()):
            nuccore_accession = pair_df_row['nuccore_accession']
            index_in_nuccore_ir_pairs_df_csv_file = pair_df_row['index_in_nuccore_ir_pairs_df_csv_file']
            left1 = pair_df_row['left1']
            right1 = pair_df_row['right1']
            left2 = pair_df_row['left2']
            right2 = pair_df_row['right2']
            ir_pair = (left1, right1, left2, right2)
            generic_utils.print_and_write_to_log(f'repeat (to estimate copy num) {k + 1}/{num_of_repeats_to_estimate_copy_num} ({nuccore_accession} {ir_pair})')

            ir_pair_as_str = '_'.join(map(str, ir_pair))
            ir_pair_output_dir_path = os.path.join(nuccore_entry_ir_pairs_output_dir_path, ir_pair_as_str)
            pathlib.Path(ir_pair_output_dir_path).mkdir(parents=True, exist_ok=True)
            repeat_copy_num_info_pickle_file_path = os.path.join(ir_pair_output_dir_path, 'repeat_copy_num_info.pickle')
            ir_pair_to_repeat_copy_num_info_pickle_file_path[ir_pair] = repeat_copy_num_info_pickle_file_path

            blast_to_estimate_repeat_copy_num(
                nuccore_accession=nuccore_accession,
                input_file_path_nuccore_fasta=nuccore_entry_fasta_file_path,
                taxon_blast_db_path=taxon_blast_db_path,
                ir_pair=ir_pair,
                min_dist_from_ir_pair_region_for_alignments=min_dist_from_ir_pair_region_for_alignments,
                max_evalue=max_evalue,
                seed_len=seed_len,
                output_file_path_repeat_copy_num_info_pickle=repeat_copy_num_info_pickle_file_path,
                ir_pair_output_dir_path=ir_pair_output_dir_path,
            )

            with open(repeat_copy_num_info_pickle_file_path, 'rb') as f:
                repeat_copy_num_info = pickle.load(f)
            repeat_estimated_copy_num = repeat_copy_num_info['estimated_copy_num']
            num_of_alignments_overlapping_region_around_ir_pair = repeat_copy_num_info['num_of_alignments_overlapping_region_around_ir_pair']
            # print(f'\nir_pair_as_str: {ir_pair_as_str}')
            # print(f'repeat_estimated_copy_num: {repeat_estimated_copy_num}\n')

            repeat_estimated_copy_num_dict = {
                'nuccore_accession': nuccore_accession,
                'index_in_nuccore_ir_pairs_df_csv_file': index_in_nuccore_ir_pairs_df_csv_file,
                'repeat_estimated_copy_num': repeat_estimated_copy_num,
                'num_of_alignments_overlapping_region_around_ir_pair': num_of_alignments_overlapping_region_around_ir_pair,
            }
            repeat_estimated_copy_num_dicts.append(repeat_estimated_copy_num_dict)
        taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path[nuccore_accession] = ir_pair_to_repeat_copy_num_info_pickle_file_path

    taxon_pairs_with_estimated_copy_num_df = pd.DataFrame(repeat_estimated_copy_num_dicts)
    assert len(taxon_pairs_with_estimated_copy_num_df) == len(taxon_pairs_df)
    assert (len(taxon_pairs_with_estimated_copy_num_df[['nuccore_accession', 'index_in_nuccore_ir_pairs_df_csv_file']]) ==
            len(taxon_pairs_with_estimated_copy_num_df[['nuccore_accession', 'index_in_nuccore_ir_pairs_df_csv_file']].drop_duplicates()))
    taxon_pairs_with_estimated_copy_num_df.to_csv(output_file_path_taxon_pairs_with_estimated_copy_num_df_csv, sep='\t', index=False)

    with open(output_file_path_taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle, 'wb') as f:
        pickle.dump(taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path, f, protocol=4)

def find_evidence_for_repeat_overlapping_mobile_element_for_taxon(
        input_file_path_taxon_pairs_df_csv,
        taxon_blast_db_path,
        min_dist_from_ir_pair_region_for_alignments,
        max_evalue,
        seed_len,
        output_file_path_taxon_pairs_with_estimated_copy_num_df_csv,
        output_file_path_taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle,
        nuccore_entries_out_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_find_evidence_for_repeat_overlapping_mobile_element_for_taxon(
        input_file_path_taxon_pairs_df_csv=input_file_path_taxon_pairs_df_csv,
        taxon_blast_db_path=taxon_blast_db_path,
        min_dist_from_ir_pair_region_for_alignments=min_dist_from_ir_pair_region_for_alignments,
        max_evalue=max_evalue,
        seed_len=seed_len,
        output_file_path_taxon_pairs_with_estimated_copy_num_df_csv=output_file_path_taxon_pairs_with_estimated_copy_num_df_csv,
        output_file_path_taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle=output_file_path_taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle,
        nuccore_entries_out_dir_path=nuccore_entries_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_find_evidence_for_repeat_overlapping_mobile_element(
        input_file_path_ir_pairs_linked_df_csv,
        input_file_path_taxon_uid_to_taxon_info_pickle,
        input_file_path_nuccore_df_csv,
        min_dist_from_ir_pair_region_for_alignments,
        seed_len,
        max_evalue,
        output_file_path_ir_pairs_linked_with_evidence_df_csv,
        output_file_path_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle,
        nuccore_entries_out_dir_path,
        taxa_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    pairs_df = pd.read_csv(input_file_path_ir_pairs_linked_df_csv, sep='\t', low_memory=False)
    orig_pairs_df_column_names = list(pairs_df)
    nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path = {}

    if pairs_df.empty:
        pairs_with_evidence_df = pairs_df
        pairs_with_evidence_df[['repeat_estimated_copy_num', 'num_of_alignments_overlapping_region_around_ir_pair']] = np.nan
    else:
        nuccore_df = pd.read_csv(input_file_path_nuccore_df_csv, sep='\t', low_memory=False)
        with open(input_file_path_taxon_uid_to_taxon_info_pickle, 'rb') as f:
            taxon_uid_to_taxon_info = pickle.load(f)


        pairs_df = pairs_df.merge(nuccore_df, on='nuccore_accession')

        num_of_taxa = pairs_df['taxon_uid'].nunique()
        taxon_uid_to_taxon_pairs_with_estimated_copy_num_df_csv_file_path = {}
        for i, (taxon_uid, taxon_pairs_df) in enumerate(pairs_df.groupby('taxon_uid', sort=False)):
            # START = 1.7e3
            # if i >= START + 100:
            #     exit()
            # if i < START:
            #     continue

            generic_utils.print_and_write_to_log(f'starting work on taxon {i + 1}/{num_of_taxa}: {taxon_uid}')
            taxon_blast_db_path = taxon_uid_to_taxon_info[taxon_uid]['taxon_blast_db_path']
            taxon_out_dir_path = os.path.join(taxa_out_dir_path, str(taxon_uid))
            pathlib.Path(taxon_out_dir_path).mkdir(parents=True, exist_ok=True)
            taxon_pairs_df_csv_file_path = os.path.join(taxon_out_dir_path, 'taxon_pairs_df.csv')
            taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path = os.path.join(
                taxon_out_dir_path, 'taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path.pickle')
            taxon_pairs_df.to_csv(taxon_pairs_df_csv_file_path, sep='\t', index=False)

            taxon_pairs_with_estimated_copy_num_df_csv_file_path = os.path.join(taxon_out_dir_path, 'taxon_pairs_with_estimated_copy_num_df.csv')
            taxon_uid_to_taxon_pairs_with_estimated_copy_num_df_csv_file_path[taxon_uid] = taxon_pairs_with_estimated_copy_num_df_csv_file_path

            find_evidence_for_repeat_overlapping_mobile_element_for_taxon(
                input_file_path_taxon_pairs_df_csv=taxon_pairs_df_csv_file_path,
                taxon_blast_db_path=taxon_blast_db_path,
                min_dist_from_ir_pair_region_for_alignments=min_dist_from_ir_pair_region_for_alignments,
                max_evalue=max_evalue,
                seed_len=seed_len,
                output_file_path_taxon_pairs_with_estimated_copy_num_df_csv=taxon_pairs_with_estimated_copy_num_df_csv_file_path,
                output_file_path_taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle=(
                    taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path),
                nuccore_entries_out_dir_path=nuccore_entries_out_dir_path,
            )

            with open(taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle_file_path, 'rb') as f:
                taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path = pickle.load(f)

            nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path = {**taxon_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path,
                                                                                     **nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path}

        pairs_with_estimated_copy_num_df = pd.concat((pd.read_csv(x, sep='\t') for x
                                                      in taxon_uid_to_taxon_pairs_with_estimated_copy_num_df_csv_file_path.values()), ignore_index=True)
        pairs_with_evidence_df = pairs_df.merge(pairs_with_estimated_copy_num_df)
        assert len(pairs_with_evidence_df) == len(pairs_df)
        assert (len(pairs_with_evidence_df[['nuccore_accession', 'index_in_nuccore_ir_pairs_df_csv_file']]) ==
                len(pairs_with_evidence_df[['nuccore_accession', 'index_in_nuccore_ir_pairs_df_csv_file']].drop_duplicates()))

        # print('\nset(pairs_with_evidence_df) - set(orig_pairs_df_column_names)')
        # print(set(pairs_with_evidence_df) - set(orig_pairs_df_column_names))

    pairs_with_evidence_df[orig_pairs_df_column_names + ['repeat_estimated_copy_num', 'num_of_alignments_overlapping_region_around_ir_pair']].to_csv(
        output_file_path_ir_pairs_linked_with_evidence_df_csv, sep='\t', index=False)

    with open(output_file_path_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle, 'wb') as f:
        pickle.dump(nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path, f, protocol=4)

def find_evidence_for_repeat_overlapping_mobile_element(
        input_file_path_ir_pairs_linked_df_csv,
        input_file_path_taxon_uid_to_taxon_info_pickle,
        input_file_path_nuccore_df_csv,
        min_dist_from_ir_pair_region_for_alignments,
        seed_len,
        max_evalue,
        output_file_path_ir_pairs_linked_with_evidence_df_csv,
        output_file_path_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle,
        nuccore_entries_out_dir_path,
        taxa_out_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_find_evidence_for_repeat_overlapping_mobile_element(
        input_file_path_ir_pairs_linked_df_csv=input_file_path_ir_pairs_linked_df_csv,
        input_file_path_taxon_uid_to_taxon_info_pickle=input_file_path_taxon_uid_to_taxon_info_pickle,
        input_file_path_nuccore_df_csv=input_file_path_nuccore_df_csv,
        min_dist_from_ir_pair_region_for_alignments=min_dist_from_ir_pair_region_for_alignments,
        seed_len=seed_len,
        max_evalue=max_evalue,
        output_file_path_ir_pairs_linked_with_evidence_df_csv=output_file_path_ir_pairs_linked_with_evidence_df_csv,
        output_file_path_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle=(
            output_file_path_nuccore_accession_to_ir_pair_to_repeat_copy_num_info_pickle_file_path_pickle),
        nuccore_entries_out_dir_path=nuccore_entries_out_dir_path,
        taxa_out_dir_path=taxa_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=7,
    )

def get_both_repeats_linked_filter(pairs_df):
    return ~(pairs_df[['repeat1_cds_index_in_nuccore_cds_features_gb_file',
                       'repeat2_cds_index_in_nuccore_cds_features_gb_file']].isna().any(axis=1))

def add_spacer_len_and_repeat_len_and_mismatch_fraction_columns(pairs_df):
    pairs_df = pairs_df.copy()
    pairs_df['spacer_len'] = pairs_df['left2'] - pairs_df['right1'] - 1
    pairs_df['repeat_len'] = pairs_df['right1'] - pairs_df['left1'] + 1
    pairs_df['ir_pair_len'] = pairs_df['right2'] - pairs_df['left1'] + 1
    pairs_df['mismatch_fraction'] = pairs_df['num_of_mismatches'] / pairs_df['repeat_len']
    pairs_df['num_of_matches'] = pairs_df['repeat_len'] - pairs_df['num_of_mismatches']
    return pairs_df

def add_repeat_cds_len_columns(pairs_df):
    pairs_df = pairs_df.copy()
    for repeat_num in (1, 2):
        pairs_df[f'repeat{repeat_num}_cds_len'] = pairs_df[f'repeat{repeat_num}_cds_end_pos'] - pairs_df[f'repeat{repeat_num}_cds_start_pos'] + 1
    for repeat_num in (1, 2):
        pairs_df[f'repeat{repeat_num}_cds_is_shorter'] = pairs_df[f'repeat{repeat_num}_cds_len'] < pairs_df[f'repeat{3 - repeat_num}_cds_len']
    pairs_df['max_repeat_cds_len'] = pairs_df[['repeat1_cds_len', 'repeat2_cds_len']].max(axis=1)
    pairs_df['min_repeat_cds_len'] = pairs_df[['repeat1_cds_len', 'repeat2_cds_len']].min(axis=1)
    return pairs_df

def add_longer_repeat_cds_columns(pairs_df):
    pairs_df = pairs_df.copy()

    assert not pairs_df['repeat2_cds_is_shorter'].isna().any()

    pairs_df.loc[pairs_df['repeat1_cds_is_shorter'], 'longer_repeat_cds_product'] = pairs_df.loc[pairs_df['repeat1_cds_is_shorter'], 'repeat2_cds_product']
    pairs_df.loc[~pairs_df['repeat1_cds_is_shorter'], 'longer_repeat_cds_product'] = pairs_df.loc[~pairs_df['repeat1_cds_is_shorter'], 'repeat1_cds_product']

    pairs_df.loc[pairs_df['repeat1_cds_is_shorter'], 'longer_repeat_cds_index_in_nuccore_cds_features_gb_file'] = pairs_df.loc[
        pairs_df['repeat1_cds_is_shorter'], 'repeat2_cds_index_in_nuccore_cds_features_gb_file']
    pairs_df.loc[~pairs_df['repeat1_cds_is_shorter'], 'longer_repeat_cds_index_in_nuccore_cds_features_gb_file'] = pairs_df.loc[
        ~pairs_df['repeat1_cds_is_shorter'], 'repeat1_cds_index_in_nuccore_cds_features_gb_file']

    pairs_df.loc[pairs_df['repeat1_cds_is_shorter'], 'longer_repeat_cds_strand'] = pairs_df.loc[pairs_df['repeat1_cds_is_shorter'], 'repeat2_cds_strand']
    pairs_df.loc[~pairs_df['repeat1_cds_is_shorter'], 'longer_repeat_cds_strand'] = pairs_df.loc[~pairs_df['repeat1_cds_is_shorter'], 'repeat1_cds_strand']

    return pairs_df

def add_prev_and_next_cds_product_columns_to_cds_df(
        all_cds_df,
        max_num_of_cds_on_each_side=1,
):
    with generic_utils.timing_context_manager('all_cds_df = all_cds_df.copy() (in add_prev_and_next_cds_product_columns_to_cds_df)'):
        all_cds_df = all_cds_df.copy()
    orig_num_of_cds = len(all_cds_df)
    minimal_all_cds_df = all_cds_df[['nuccore_accession', 'index_in_nuccore_cds_features_gb_file', 'product']]
    result_df = all_cds_df.copy()

    for next_or_prev in ('next', 'prev'):
        if 'next' == next_or_prev:
            curr_diff = 1
        else:
            assert 'prev' == next_or_prev
            curr_diff = -1
        for curr_dist_in_cds in range(1, max_num_of_cds_on_each_side + 1):
            curr_diff_in_cds = curr_diff * curr_dist_in_cds
            curr_prefix = '_'.join([next_or_prev] * curr_dist_in_cds)
            new_product_column_name = f'{curr_prefix}_cds_product'
            if new_product_column_name not in result_df:
                new_index_column_name = f'{curr_prefix}_index_in_nuccore_cds_features_gb_file'
                result_df[new_index_column_name] = result_df['index_in_nuccore_cds_features_gb_file'] + curr_diff_in_cds
                with generic_utils.timing_context_manager(f'building {new_product_column_name} (in add_prev_and_next_cds_product_columns_to_cds_df)'):
                    result_df = result_df.merge(
                        minimal_all_cds_df.rename(columns={'index_in_nuccore_cds_features_gb_file': new_index_column_name,
                                                           'product': f'{curr_prefix}_cds_product'}),
                        how='left',
                    ).drop(new_index_column_name, axis=1)
            else:
                print(f'skipping building {new_product_column_name}')

    assert len(result_df) == orig_num_of_cds
    return result_df

def add_is_cds_or_any_near_cds_product_satisfying_predicate_column(
        all_cds_df,
        new_column_name,
        product_series_predicate,
        drop_next_and_prev_cds_product_columns=False,
        min_num_of_cds_on_each_side=0,
        max_num_of_cds_on_each_side=1,
):
    assert 0 <= min_num_of_cds_on_each_side <= max_num_of_cds_on_each_side
    with generic_utils.timing_context_manager('all_cds_df = all_cds_df.copy() (in add_is_cds_or_any_near_cds_product_satisfying_predicate_column)'):
        all_cds_df = all_cds_df.copy()
    orig_num_of_cds = len(all_cds_df)

    all_cds_df = add_prev_and_next_cds_product_columns_to_cds_df(all_cds_df, max_num_of_cds_on_each_side=max_num_of_cds_on_each_side)

    if min_num_of_cds_on_each_side == 0:
        predicate_result_series = product_series_predicate(all_cds_df['product'])
        min_num_of_cds_on_each_side = 1
    else:
        predicate_result_series = pd.Series(np.zeros(len(all_cds_df)), index=all_cds_df.index).astype(bool)


    for next_or_prev in ('next', 'prev'):
        print(f'next_or_prev: {next_or_prev}')
        for curr_dist_in_cds in range(min_num_of_cds_on_each_side, max_num_of_cds_on_each_side + 1):
            print(f'curr_dist_in_cds: {curr_dist_in_cds}')
            curr_prefix = '_'.join([next_or_prev] * curr_dist_in_cds)
            product_column_name = f'{curr_prefix}_cds_product'
            predicate_result_series |= product_series_predicate(all_cds_df[product_column_name])

            if drop_next_and_prev_cds_product_columns:
                all_cds_df.drop(product_column_name, axis=1, inplace=True)

    all_cds_df[new_column_name] = predicate_result_series
    assert len(all_cds_df) == orig_num_of_cds
    return all_cds_df

def add_is_cds_linked_to_any_repeatX_column(all_cds_df, pairs_df, repeat_num, new_column_name):
    all_cds_df = all_cds_df.copy()
    orig_num_of_cds = len(all_cds_df)

    minimal_pairs_df = pairs_df[['nuccore_accession', f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file']].copy()
    minimal_pairs_df[new_column_name] = True
    result_df = all_cds_df.merge(
        minimal_pairs_df.rename(columns={f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file': 'index_in_nuccore_cds_features_gb_file'}),
        how='left',
    ).drop_duplicates()
    result_df[new_column_name] = result_df[new_column_name].fillna(False)

    assert len(result_df) == orig_num_of_cds
    return result_df

def add_is_cds_linked_to_any_repeat_column(cds_df, pairs_df):
    cds_df = cds_df.copy()
    for repeat_num in (1, 2):
        print(f'repeat_num: {repeat_num}')
        cds_df = add_is_cds_linked_to_any_repeatX_column(
            all_cds_df=cds_df,
            pairs_df=pairs_df,
            repeat_num=repeat_num,
            new_column_name=f'is_cds_linked_to_any_repeat{repeat_num}',
        )

    cds_df['is_cds_linked_to_any_repeat'] = cds_df['is_cds_linked_to_any_repeat1'] | cds_df['is_cds_linked_to_any_repeat2']
    return cds_df

def add_prev_and_next_cds_product_columns_to_pairs_df(
        pairs_df,
        cds_df_csv_file_path,
        set_to_nan_prev_or_next_if_it_is_a_repeat_cds,
):
    pairs_df = pairs_df.copy()
    orig_num_of_pairs = len(pairs_df)

    all_cds_df = pd.read_csv(cds_df_csv_file_path, sep='\t', low_memory=False)
    # print(f'len(all_cds_df): {len(all_cds_df)}')
    minimal_all_cds_df = all_cds_df[['nuccore_accession', 'index_in_nuccore_cds_features_gb_file', 'product']].merge(pairs_df['nuccore_accession'].drop_duplicates())
    for repeat_num in (1, 2):
        pairs_df[f'repeat{repeat_num}_next_cds_index_in_nuccore_cds_features_gb_file'] = pairs_df[f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] + 1
        pairs_df[f'repeat{repeat_num}_prev_cds_index_in_nuccore_cds_features_gb_file'] = pairs_df[f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] - 1
        for next_or_prev in ('next', 'prev'):
            generic_utils.print_and_write_to_log(f'(add_prev_and_next_cds_product_columns_to_pairs_df) starting work on repeat{repeat_num}, {next_or_prev}')

            prev_or_next_cds_index_column_name = f'repeat{repeat_num}_{next_or_prev}_cds_index_in_nuccore_cds_features_gb_file'
            prev_or_next_cds_product_column_name = f'repeat{repeat_num}_{next_or_prev}_cds_product'
            pairs_df = pairs_df.merge(minimal_all_cds_df.rename(
                columns={'index_in_nuccore_cds_features_gb_file': prev_or_next_cds_index_column_name,
                         'product': prev_or_next_cds_product_column_name}
            ), how='left')
            if set_to_nan_prev_or_next_if_it_is_a_repeat_cds:
                pairs_df.loc[pairs_df[prev_or_next_cds_index_column_name] == pairs_df[f'repeat{3 - repeat_num}_cds_index_in_nuccore_cds_features_gb_file'],
                             prev_or_next_cds_product_column_name] = np.nan
    # print(f'len(pairs_df): {len(pairs_df)}')
    assert len(pairs_df) == orig_num_of_pairs
    return pairs_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_prev_and_next_cds_product_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        set_to_nan_prev_or_next_if_it_is_a_repeat_cds,
        output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = add_prev_and_next_cds_product_columns_to_pairs_df(
        pairs_df=pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False),
        cds_df_csv_file_path=input_file_path_cds_df_csv,
        set_to_nan_prev_or_next_if_it_is_a_repeat_cds=set_to_nan_prev_or_next_if_it_is_a_repeat_cds,
    )

    cds_pairs_df.to_csv(output_file_path_extended_cds_pairs_df_csv, sep='\t', index=False)


def add_prev_and_next_cds_product_columns(
        input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv,
        set_to_nan_prev_or_next_if_it_is_a_repeat_cds,
        output_file_path_extended_cds_pairs_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_prev_and_next_cds_product_columns(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        set_to_nan_prev_or_next_if_it_is_a_repeat_cds=set_to_nan_prev_or_next_if_it_is_a_repeat_cds,
        output_file_path_extended_cds_pairs_df_csv=output_file_path_extended_cds_pairs_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=14,
    )
