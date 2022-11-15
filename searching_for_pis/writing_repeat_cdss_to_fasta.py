import pickle

import numpy as np
import pandas as pd

from generic import bio_utils
from generic import generic_utils
from searching_for_pis import index_column_names

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_repeat_cdss_to_fasta_internal(
        input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_repeat_cds_seqs_fasta,
        output_file_path_repeat_cds_seq_name_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with open(input_file_path_nuccore_accession_to_nuccore_entry_info_pickle, 'rb') as f:
        nuccore_accession_to_nuccore_entry_info = pickle.load(f)
    pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)\

    repeat_cds_seqs = []
    repeat_cds_seq_name_flat_dicts = []
    num_of_nuccores = pairs_df['nuccore_accession'].nunique()
    repeat_cds_seq_names = set()
    for i, (nuccore_accession, nuccore_pairs_df) in enumerate(pairs_df.groupby('nuccore_accession')):
        generic_utils.print_and_write_to_log(f'(cached_write_repeat_cdss_to_fasta) '
                                             f'starting work on nuccore {i + 1}/{num_of_nuccores} ({nuccore_accession}).')
        nuccore_fasta_file_path = nuccore_accession_to_nuccore_entry_info[nuccore_accession]['fasta_file_path']
        nuccore_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(nuccore_fasta_file_path)
        for _, pair_df_row in nuccore_pairs_df.iterrows():
            repeat_num_to_repeat_cds_region = {
                repeat_num: (pair_df_row[f'repeat{repeat_num}_cds_start_pos'],
                             pair_df_row[f'repeat{repeat_num}_cds_end_pos'])
                for repeat_num in (1, 2)
            }
            for repeat_num in (1, 2):
                index_in_nuccore_cds_features_gb_file = pair_df_row[f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file']
                repeat_cds_seq_name = f'{nuccore_accession}_cds{index_in_nuccore_cds_features_gb_file}'

                strand = pair_df_row[f'repeat{repeat_num}_cds_strand']
                assert abs(strand) == 1
                if strand == -1:
                    repeat_cds_seq_name = f'RC_of_{repeat_cds_seq_name}'

                if repeat_cds_seq_name in repeat_cds_seq_names:
                    continue
                repeat_cds_region = repeat_num_to_repeat_cds_region[repeat_num]
                cds_seq = bio_utils.get_region_in_chrom_seq(nuccore_seq, *repeat_cds_region, region_name=repeat_cds_seq_name)
                if strand == -1:
                    cds_seq = cds_seq.reverse_complement()
                    cds_seq.name = cds_seq.description = cds_seq.id = repeat_cds_seq_name

                repeat_cds_seqs.append(cds_seq)
                repeat_cds_seq_name_flat_dicts.append({
                    'nuccore_accession': nuccore_accession,
                    'index_in_nuccore_cds_features_gb_file': index_in_nuccore_cds_features_gb_file,
                    'cds_seq_name': repeat_cds_seq_name,
                })
                repeat_cds_seq_names.add(repeat_cds_seq_name)

    num_of_cds_seqs = len(repeat_cds_seqs)
    assert repeat_cds_seq_name_flat_dicts
    assert len(repeat_cds_seq_name_flat_dicts) == num_of_cds_seqs
    assert len(repeat_cds_seq_names) == num_of_cds_seqs
    repeat_cds_seq_name_df = pd.DataFrame(repeat_cds_seq_name_flat_dicts).drop_duplicates()
    assert len(repeat_cds_seq_name_df) == num_of_cds_seqs

    print(f'num_of_cds_seqs: {num_of_cds_seqs}')
    print(len(repeat_cds_seq_name_df))

    repeat_cds_seq_name_df.to_csv(output_file_path_repeat_cds_seq_name_df_csv, sep='\t', index=False)

    bio_utils.write_records_to_fasta_or_gb_file(repeat_cds_seqs, output_file_path_repeat_cds_seqs_fasta, file_type='fasta')

def write_repeat_cdss_to_fasta_internal(
        input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_repeat_cds_seqs_fasta,
        output_file_path_repeat_cds_seq_name_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_repeat_cdss_to_fasta_internal(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_repeat_cds_seqs_fasta=output_file_path_repeat_cds_seqs_fasta,
        output_file_path_repeat_cds_seq_name_df_csv=output_file_path_repeat_cds_seq_name_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_repeat_cdss_to_fasta(
        input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_repeat_cds_seqs_fasta,
        output_file_path_repeat_cds_seq_name_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)

    minimal_cds_pairs_df_csv_file_path = f'{input_file_path_cds_pairs_df_csv}.minimal.csv'
    pairs_df[index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES + [
        'repeat1_cds_start_pos',
        'repeat1_cds_end_pos',
        'repeat1_cds_strand',
        'repeat2_cds_start_pos',
        'repeat2_cds_end_pos',
        'repeat2_cds_strand',
    ]].to_csv(minimal_cds_pairs_df_csv_file_path, sep='\t', index=False)
    write_repeat_cdss_to_fasta_internal(
        input_file_path_cds_pairs_df_csv=minimal_cds_pairs_df_csv_file_path,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_repeat_cds_seqs_fasta=output_file_path_repeat_cds_seqs_fasta,
        output_file_path_repeat_cds_seq_name_df_csv=output_file_path_repeat_cds_seq_name_df_csv,
    )

def write_repeat_cdss_to_fasta(
        input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_repeat_cds_seqs_fasta,
        output_file_path_repeat_cds_seq_name_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_repeat_cdss_to_fasta(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=input_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_repeat_cds_seqs_fasta=output_file_path_repeat_cds_seqs_fasta,
        output_file_path_repeat_cds_seq_name_df_csv=output_file_path_repeat_cds_seq_name_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=9,
    )


