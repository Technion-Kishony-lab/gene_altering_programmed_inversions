import collections
import itertools
import logging
import os
import os.path
import pathlib
import pickle

import numpy as np
import pandas as pd
from Bio import SeqIO

from generic import bio_utils
from generic import blast_interface_and_utils
from generic import samtools_and_sam_files_interface
from generic import generic_utils
from generic import mauve_interface_and_utils
from generic import bowtie2_interface
from generic import ncbi_genome_download_interface
from generic import seq_feature_utils
from searching_for_pis import cds_enrichment_analysis
from searching_for_pis import massive_screening_stage_1
from searching_for_pis import massive_screening_configuration
from searching_for_pis import py_repeats_finder

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

DO_STAGE6 = True
# DO_STAGE6 = False

@generic_utils.execute_if_output_doesnt_exist_already
def cached_filter_cds_df_and_add_product_family(
        input_file_path_cds_df_csv,
        cds_region,
        list_of_product_and_product_family,
        output_file_path_filtered_extended_cds_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_df = pd.read_csv(input_file_path_cds_df_csv, sep='\t', low_memory=False)
    cds_df = cds_df[
        (cds_df['start_pos'] >= cds_region[0]) &
        (cds_df['end_pos'] <= cds_region[1])
    ]

    assert (cds_df['start_pos'] == cds_region[0]).any()
    assert (cds_df['end_pos'] == cds_region[1]).any()

    cds_df = cds_enrichment_analysis.add_product_family_column(
        cds_df, list_of_product_and_product_family,
        orig_product_column_name='product', product_family_column_name='product_family',
    )
    cds_df.to_csv(output_file_path_filtered_extended_cds_df_csv, sep='\t', index=False)

def filter_cds_df_and_add_product_family(
        input_file_path_cds_df_csv,
        cds_region,
        list_of_product_and_product_family,
        output_file_path_filtered_extended_cds_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_filter_cds_df_and_add_product_family(
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        cds_region=cds_region,
        list_of_product_and_product_family=list_of_product_and_product_family,
        output_file_path_filtered_extended_cds_df_csv=output_file_path_filtered_extended_cds_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_repeat_cds_homologs_df(
        input_file_path_taxa_df_csv,
        input_file_path_nuccore_fasta,
        repeat_cds_region,
        blast_repeat_cds_to_each_taxon_genome_seed_len,
        blast_repeat_cds_to_each_taxon_genome_max_evalue,
        min_repeat_cds_covered_bases_proportion,
        min_alignment_bases_covered_by_cds_proportion,
        blast_homolog_to_its_margins_seed_len,
        blast_homolog_to_its_margins_max_evalue,
        blast_homolog_to_its_margins_nuccores_out_dir_path,
        homolog_margin_size,
        output_dir_path,
        output_file_path_homologs_df_csv,
        output_file_path_nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle,
        output_file_path_all_repeat_cds_covered_bases_proportions_pickle,
        output_file_path_alignment_bases_covered_by_cds_proportions_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    blast_region_to_taxa_out_dir_path = os.path.join(output_dir_path, 'taxa')

    repeat_cds_len = repeat_cds_region[1] - repeat_cds_region[0] + 1

    nuccore_accessions_of_homolog_nuccores_with_empty_cds_df = set()
    homolog_nuccore_homologs_dfs = []
    list_of_repeat_cds_covered_bases_proportion_series = []
    list_of_alignment_bases_covered_by_cds_proportion_series = []

    taxa_df = pd.read_csv(input_file_path_taxa_df_csv, sep='\t', low_memory=False)
    num_of_taxa = len(taxa_df)
    for i, (_, taxa_df_row) in enumerate(taxa_df.iterrows()):
        # START = 0e3
        # if i >= START + 5000:
        #     exit()
        # if i < START:
        #     continue


        homolog_taxon_uid = taxa_df_row['taxon_uid']
        homolog_taxon_blast_db_path = taxa_df_row['taxon_blast_db_path']

        generic_utils.print_and_write_to_log(f'(cached_write_repeat_cds_homologs_df) '
                                             f'starting to work on other taxon {i + 1}/{num_of_taxa} ({homolog_taxon_uid}).')

        blast_repeat_cds_to_taxon_genome_out_dir_path = os.path.join(blast_region_to_taxa_out_dir_path, str(homolog_taxon_uid))
        pathlib.Path(blast_repeat_cds_to_taxon_genome_out_dir_path).mkdir(parents=True, exist_ok=True)


        blast_repeat_cds_to_taxon_genome_result_df_csv_file_path = os.path.join(
            blast_repeat_cds_to_taxon_genome_out_dir_path,
            f'blast_repeat_cds_to_taxon_genome_result_df__seed_len{blast_repeat_cds_to_each_taxon_genome_seed_len}.csv')
        blast_interface_and_utils.blast_nucleotide(
            query_fasta_file_path=input_file_path_nuccore_fasta,
            region_in_query_sequence=repeat_cds_region,
            blast_db_path=homolog_taxon_blast_db_path,
            blast_results_file_path=blast_repeat_cds_to_taxon_genome_result_df_csv_file_path,
            perform_gapped_alignment=True,
            query_strand_to_search='both',
            max_evalue=blast_repeat_cds_to_each_taxon_genome_max_evalue,
            seed_len=blast_repeat_cds_to_each_taxon_genome_seed_len,
            # verbose=True,
        )
        alignments_df = blast_interface_and_utils.read_blast_results_df(blast_repeat_cds_to_taxon_genome_result_df_csv_file_path)
        if alignments_df.empty:
            continue

        alignments_df['length_in_query'] = alignments_df['qend'] - alignments_df['qstart'] + 1
        alignments_df['repeat_cds_covered_bases_proportion'] = alignments_df['length_in_query'] / repeat_cds_len
        assert (0 < alignments_df['repeat_cds_covered_bases_proportion']).all()
        assert (alignments_df['repeat_cds_covered_bases_proportion'] <= 1).all()
        list_of_repeat_cds_covered_bases_proportion_series.append(alignments_df['repeat_cds_covered_bases_proportion'])

        alignments_df = alignments_df[alignments_df['repeat_cds_covered_bases_proportion'] >= min_repeat_cds_covered_bases_proportion]
        if alignments_df.empty:
            continue

        alignments_df['smax'] = alignments_df[['sstart', 'send']].max(axis=1)
        alignments_df['smin'] = alignments_df[['sstart', 'send']].min(axis=1)
        # print(alignments_df)
        # from the result of this it seems that it isn't worth it to try to gather multiple alignment so that in total they cover a single cds.
        # print(alignments_df['sseqid'].value_counts().value_counts())
        # exit()

        homolog_taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path = taxa_df_row['taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path']
        with open(homolog_taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path, 'rb') as f:
            homolog_taxon_nuccore_accession_to_nuccore_entry_info = pickle.load(f)


        num_of_homolog_nuccores = alignments_df['sseqid'].nunique()
        for i, (homolog_nuccore_accession, homolog_nuccore_alignments_df) in enumerate(alignments_df.groupby('sseqid')):
            generic_utils.print_and_write_to_log(f'(cached_write_repeat_cds_homologs_df) '
                                                 f'starting to work on homolog nuccore {i + 1}/{num_of_homolog_nuccores} ({homolog_nuccore_accession}).')

            homolog_nuccore_entry_info = homolog_taxon_nuccore_accession_to_nuccore_entry_info[homolog_nuccore_accession]
            cds_df_csv_file_path = homolog_nuccore_entry_info['cds_df_csv_file_path']


            if not generic_utils.is_text_file_empty_or_containing_only_whitespaces(cds_df_csv_file_path):
                cds_df = pd.read_csv(cds_df_csv_file_path, sep='\t', low_memory=False)
                assert not cds_df.empty
                cds_and_alignments_df = cds_df.rename(columns={'nuccore_accession': 'sseqid'}).merge(homolog_nuccore_alignments_df)
                cds_and_alignments_df['num_of_overlapping_bases'] = (
                        cds_and_alignments_df[['end_pos', 'smax']].min(axis=1) -
                        cds_and_alignments_df[['start_pos', 'smin']].max(axis=1) + 1
                )
                cds_overlapping_alignments_df = cds_and_alignments_df[cds_and_alignments_df['num_of_overlapping_bases'] > 0].copy()
                cds_overlapping_alignments_df['alignment_bases_covered_by_cds_proportion'] = (
                    cds_overlapping_alignments_df['num_of_overlapping_bases'] /
                    (cds_overlapping_alignments_df['smax'] - cds_overlapping_alignments_df['smin'] + 1)
                )

                homolog_nuccore_homologs_df = cds_overlapping_alignments_df[[
                    # 'smin', 'smax',
                    'qstart', 'qend', 'sstart', 'send',
                    'start_pos', 'end_pos', 'strand', 'product', 'repeat_cds_covered_bases_proportion', 'alignment_bases_covered_by_cds_proportion',
                ]]

                homolog_nuccore_homologs_df = homolog_nuccore_homologs_df.sort_values(
                    ['alignment_bases_covered_by_cds_proportion'], ascending=False,
                ).drop_duplicates(['qstart', 'qend', 'sstart', 'send'], keep='first')


                # filter before adding to list_of_alignment_bases_covered_by_cds_proportion_series, so that we will save one row for each
                # homologous sequence (i.e., alignment).
                list_of_alignment_bases_covered_by_cds_proportion_series.append(homolog_nuccore_homologs_df['alignment_bases_covered_by_cds_proportion'])
                homolog_nuccore_homologs_df = homolog_nuccore_homologs_df[
                    homolog_nuccore_homologs_df['alignment_bases_covered_by_cds_proportion'] >= min_alignment_bases_covered_by_cds_proportion
                ]

                homolog_nuccore_homologs_df = homolog_nuccore_homologs_df.sort_values(
                    ['repeat_cds_covered_bases_proportion', 'alignment_bases_covered_by_cds_proportion'], ascending=False,
                ).drop_duplicates(['start_pos', 'end_pos', 'strand'], keep='first').copy()

                homolog_nuccore_homologs_df['nuccore_accession'] = homolog_nuccore_accession
                homolog_nuccore_homologs_df['taxon_uid'] = homolog_taxon_uid

                homolog_nuccore_len = homolog_nuccore_entry_info['chrom_len']
                homolog_nuccore_homologs_df['any_margin_exceeds_scaffold'] = (
                    ((homolog_nuccore_homologs_df['start_pos'] - homolog_margin_size) < 1) |
                    ((homolog_nuccore_homologs_df['end_pos'] + homolog_margin_size) > homolog_nuccore_len)
                )

                homolog_nuccore_homologs_df_with_ok_margins = homolog_nuccore_homologs_df[~homolog_nuccore_homologs_df['any_margin_exceeds_scaffold']]
                num_of_homolog_nuccore_homologs_with_ok_margins = len(homolog_nuccore_homologs_df_with_ok_margins)
                if num_of_homolog_nuccore_homologs_with_ok_margins > 0:
                    homolog_nuccore_out_dir_path = os.path.join(blast_homolog_to_its_margins_nuccores_out_dir_path, homolog_nuccore_accession)
                    pathlib.Path(homolog_nuccore_out_dir_path).mkdir(parents=True, exist_ok=True)
                    homolog_nuccore_fasta_file_path = homolog_nuccore_entry_info['fasta_file_path']
                    homolog_nuccore_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(homolog_nuccore_fasta_file_path)

                    homolog_min_evalue_of_alignment_to_margin_flat_dicts = []
                    for j, (_, homolog_row_df) in enumerate(homolog_nuccore_homologs_df_with_ok_margins.iterrows()):
                        start_pos = homolog_row_df['start_pos']
                        end_pos = homolog_row_df['end_pos']
                        generic_utils.print_and_write_to_log(f'(cached_write_repeat_cds_homologs_df) '
                                                             f'starting to work on homolog {j + 1}/{num_of_homolog_nuccore_homologs_with_ok_margins} '
                                                             f'{(start_pos, end_pos)}.')


                        left_margin_region = (start_pos - homolog_margin_size, start_pos - 1)
                        right_margin_region = (end_pos + 1, end_pos + homolog_margin_size)
                        homolog_str_repr = f'{start_pos}_{end_pos}'
                        curr_homolog_dir_path = os.path.join(homolog_nuccore_out_dir_path, homolog_str_repr)
                        pathlib.Path(curr_homolog_dir_path).mkdir(parents=True, exist_ok=True)
                        homolog_margins_fasta_file_path = os.path.join(curr_homolog_dir_path, 'margins.fasta')
                        blast_to_homolog_margin_results_csv_file_path = os.path.join(curr_homolog_dir_path, 'blast_to_homolog_margin_results.csv')
                        left_margin_seq_name = f'left_margin_of_homolog_{homolog_str_repr}'
                        right_margin_seq_name = f'right_margin_of_homolog_{homolog_str_repr}'

                        left_margin = bio_utils.get_region_in_chrom_seq(homolog_nuccore_seq, *left_margin_region,
                                                                        region_name=left_margin_seq_name)
                        right_margin = bio_utils.get_region_in_chrom_seq(homolog_nuccore_seq, *right_margin_region,
                                                                         region_name=right_margin_seq_name)
                        bio_utils.write_records_to_fasta_or_gb_file([left_margin, right_margin], homolog_margins_fasta_file_path)
                        blast_interface_and_utils.make_blast_nucleotide_db(homolog_margins_fasta_file_path)

                        blast_interface_and_utils.blast_nucleotide(
                            query_fasta_file_path=homolog_nuccore_fasta_file_path,
                            region_in_query_sequence=(start_pos, end_pos),
                            blast_db_path=homolog_margins_fasta_file_path,
                            blast_results_file_path=blast_to_homolog_margin_results_csv_file_path,
                            perform_gapped_alignment=False,
                            query_strand_to_search='minus',
                            seed_len=blast_homolog_to_its_margins_seed_len,
                            max_evalue=blast_homolog_to_its_margins_max_evalue,
                        )
                        blast_homolog_to_margins_alignments_df = blast_interface_and_utils.read_blast_results_df(blast_to_homolog_margin_results_csv_file_path)
                        curr_homolog_basic_flat_dict = {'start_pos': start_pos, 'end_pos': end_pos}
                        if blast_homolog_to_margins_alignments_df.empty:
                            curr_homolog_flat_dict = {**curr_homolog_basic_flat_dict, 'min_evalue_of_alignment_to_inverted_repeat_in_margin': np.inf}
                        else:
                            min_evalue_alignment_dict = blast_homolog_to_margins_alignments_df.sort_values('evalue', ascending=True).iloc[0].to_dict()
                            if min_evalue_alignment_dict['sseqid'] == left_margin_seq_name:
                                repeat_in_margin_offset = left_margin_region[0] - 1
                            else:
                                assert min_evalue_alignment_dict['sseqid'] == right_margin_seq_name
                                repeat_in_margin_offset = right_margin_region[0] - 1

                            repeat_in_margin_left = min_evalue_alignment_dict['send'] + repeat_in_margin_offset
                            repeat_in_margin_right = min_evalue_alignment_dict['sstart'] + repeat_in_margin_offset
                            assert repeat_in_margin_left < repeat_in_margin_right

                            curr_homolog_flat_dict = {
                                **curr_homolog_basic_flat_dict,
                                'min_evalue_of_alignment_to_inverted_repeat_in_margin': min_evalue_alignment_dict['evalue'],
                                'inverted_repeat_in_homolog_left': min_evalue_alignment_dict['qstart'],
                                'inverted_repeat_in_homolog_right': min_evalue_alignment_dict['qend'],
                                'inverted_repeat_in_margin_left': repeat_in_margin_left,
                                'inverted_repeat_in_margin_right': repeat_in_margin_right,
                            }
                            # print('aoeu\n\n')
                            # print(input_file_path_nuccore_fasta)
                            # print(curr_homolog_flat_dict)
                            # print(homolog_nuccore_accession)
                            # exit()
                        homolog_min_evalue_of_alignment_to_margin_flat_dicts.append(curr_homolog_flat_dict)

                    if homolog_min_evalue_of_alignment_to_margin_flat_dicts:
                        homolog_nuccore_homologs_df = homolog_nuccore_homologs_df.merge(
                            pd.DataFrame(homolog_min_evalue_of_alignment_to_margin_flat_dicts), how='left')
                    # homolog_nuccore_homologs_df = homolog_nuccore_homologs_df.merge(
                    #     (
                    #         pd.DataFrame(homolog_min_evalue_of_alignment_to_margin_flat_dicts) if homolog_min_evalue_of_alignment_to_margin_flat_dicts
                    #         else pd.DataFrame([], columns=['start_pos', 'end_pos', 'min_evalue_of_alignment_to_inverted_repeat_in_margin'])
                    #     ),
                    #     how='left',
                    # )
                homolog_nuccore_homologs_dfs.append(homolog_nuccore_homologs_df)

            else:
                nuccore_accessions_of_homolog_nuccores_with_empty_cds_df.add(homolog_nuccore_accession)

    homologs_df = pd.concat(homolog_nuccore_homologs_dfs, ignore_index=True)
    all_repeat_cds_covered_bases_proportions = list(pd.concat(list_of_repeat_cds_covered_bases_proportion_series, ignore_index=True))
    all_alignment_bases_covered_by_cds_proportions = list(pd.concat(list_of_alignment_bases_covered_by_cds_proportion_series, ignore_index=True))

    with open(output_file_path_nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle, 'wb') as f:
        pickle.dump(nuccore_accessions_of_homolog_nuccores_with_empty_cds_df, f, protocol=4)

    with open(output_file_path_all_repeat_cds_covered_bases_proportions_pickle, 'wb') as f:
        pickle.dump(all_repeat_cds_covered_bases_proportions, f, protocol=4)
    with open(output_file_path_alignment_bases_covered_by_cds_proportions_pickle, 'wb') as f:
        pickle.dump(all_alignment_bases_covered_by_cds_proportions, f, protocol=4)

    homologs_df.to_csv(output_file_path_homologs_df_csv, sep='\t', index=False)

    print(f'nuccore_accessions_of_homolog_nuccores_with_empty_cds_df: {nuccore_accessions_of_homolog_nuccores_with_empty_cds_df}')

def write_repeat_cds_homologs_df(
        input_file_path_taxa_df_csv,
        input_file_path_nuccore_fasta,
        repeat_cds_region,
        blast_repeat_cds_to_each_taxon_genome_seed_len,
        blast_repeat_cds_to_each_taxon_genome_max_evalue,
        min_repeat_cds_covered_bases_proportion,
        min_alignment_bases_covered_by_cds_proportion,
        blast_homolog_to_its_margins_seed_len,
        blast_homolog_to_its_margins_max_evalue,
        blast_homolog_to_its_margins_nuccores_out_dir_path,
        homolog_margin_size,
        output_dir_path,
        output_file_path_homologs_df_csv,
        output_file_path_nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle,
        output_file_path_all_repeat_cds_covered_bases_proportions_pickle,
        output_file_path_alignment_bases_covered_by_cds_proportions_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_repeat_cds_homologs_df(
        input_file_path_taxa_df_csv=input_file_path_taxa_df_csv,
        input_file_path_nuccore_fasta=input_file_path_nuccore_fasta,
        repeat_cds_region=repeat_cds_region,
        blast_repeat_cds_to_each_taxon_genome_seed_len=blast_repeat_cds_to_each_taxon_genome_seed_len,
        blast_repeat_cds_to_each_taxon_genome_max_evalue=blast_repeat_cds_to_each_taxon_genome_max_evalue,
        min_repeat_cds_covered_bases_proportion=min_repeat_cds_covered_bases_proportion,
        min_alignment_bases_covered_by_cds_proportion=min_alignment_bases_covered_by_cds_proportion,
        blast_homolog_to_its_margins_seed_len=blast_homolog_to_its_margins_seed_len,
        blast_homolog_to_its_margins_max_evalue=blast_homolog_to_its_margins_max_evalue,
        blast_homolog_to_its_margins_nuccores_out_dir_path=blast_homolog_to_its_margins_nuccores_out_dir_path,
        homolog_margin_size=homolog_margin_size,
        output_dir_path=output_dir_path,
        output_file_path_homologs_df_csv=output_file_path_homologs_df_csv,
        output_file_path_nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle=output_file_path_nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle,
        output_file_path_all_repeat_cds_covered_bases_proportions_pickle=output_file_path_all_repeat_cds_covered_bases_proportions_pickle,
        output_file_path_alignment_bases_covered_by_cds_proportions_pickle=output_file_path_alignment_bases_covered_by_cds_proportions_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=6,
    )



def get_merged_aligned_regions_in_read(read_alignments_df):
    read_alignments_df = read_alignments_df.copy()
    read_alignments_df['smin'] = read_alignments_df[['sstart', 'send']].min(axis=1)
    read_alignments_df['smax'] = read_alignments_df[['sstart', 'send']].max(axis=1)

    aligned_regions_in_read = list(read_alignments_df[['smin', 'smax']].itertuples(index=False, name=None))
    merged_aligned_regions_in_read = generic_utils.get_merged_intervals(aligned_regions_in_read)
    return merged_aligned_regions_in_read


def get_sseqid_and_num_of_covered_bases_df(alignments_df):
    alignments_df = alignments_df.copy()
    sseqid_and_num_of_covered_bases = []
    for sseqid, read_alignments_df in alignments_df.groupby('sseqid'):
        merged_aligned_regions_in_read = get_merged_aligned_regions_in_read(read_alignments_df)
        num_of_read_bases_covered_by_any_alignment = sum(x[1] - x[0] + 1 for x in merged_aligned_regions_in_read)
        sseqid_and_num_of_covered_bases.append((sseqid, num_of_read_bases_covered_by_any_alignment))

    sseqid_and_num_of_covered_bases_df = pd.DataFrame(sseqid_and_num_of_covered_bases, columns=['sseqid', 'num_of_read_bases_covered_by_any_alignment'])
    return sseqid_and_num_of_covered_bases_df


@generic_utils.execute_if_output_doesnt_exist_already
def cached_blast_alignment_region_to_reads(
        input_file_path_nuccore_fasta,
        input_file_path_sra_entry_fasta,
        min_num_of_read_bases_covered_by_any_alignment,
        alignment_region,
        max_evalue,
        seed_len,
        output_file_path_blast_result_csv,
        output_file_path_filtered_blast_result_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    blast_interface_and_utils.make_blast_nucleotide_db(input_file_path_sra_entry_fasta)

    blast_interface_and_utils.blast_nucleotide(
        query_fasta_file_path=input_file_path_nuccore_fasta,
        region_in_query_sequence=alignment_region,
        blast_db_path=input_file_path_sra_entry_fasta,
        blast_results_file_path=output_file_path_blast_result_csv,
        perform_gapped_alignment=True,
        query_strand_to_search='both',
        max_evalue=max_evalue,
        seed_len=seed_len,
        max_num_of_overlaps_to_report=int(1e8),
        # verbose=True,
    )

    blast_result_df = blast_interface_and_utils.read_blast_results_df(output_file_path_blast_result_csv)
    orig_num_of_alignments = len(blast_result_df)
    print(f'orig_num_of_alignments: {orig_num_of_alignments}')

    read_name_to_read_description = bio_utils.get_chrom_seq_name_to_seq_description_from_fasta_file(input_file_path_sra_entry_fasta)
    read_name_to_length = {
        read_name: int(read_description.rpartition(' length=')[-1])
        for read_name, read_description in read_name_to_read_description.items()
    }
    blast_result_df = blast_result_df.merge(pd.DataFrame(list(read_name_to_length.items()), columns=['sseqid', 'read_length']))
    assert len(blast_result_df) == orig_num_of_alignments

    sseqid_sum_of_alignment_lengths = blast_result_df.groupby(['sseqid', 'read_length'])['length'].sum().reset_index(name='sum_of_alignment_lengths')
    sseqid_with_chance_to_have_high_aligned_read_fraction = sseqid_sum_of_alignment_lengths[
        sseqid_sum_of_alignment_lengths['sum_of_alignment_lengths'] >= min_num_of_read_bases_covered_by_any_alignment
    ]['sseqid']

    orig_num_of_reads_with_alignments = blast_result_df['sseqid'].nunique()
    print(f'orig_num_of_reads_with_alignments: {orig_num_of_reads_with_alignments}')

    blast_result_df = blast_result_df.merge(sseqid_with_chance_to_have_high_aligned_read_fraction)
    num_of_reads_with_alignments_after_naive_aligned_read_fraction_filtering = blast_result_df['sseqid'].nunique()
    print(f'num_of_reads_with_alignments_after_naive_aligned_read_fraction_filtering: {num_of_reads_with_alignments_after_naive_aligned_read_fraction_filtering}')

    sseqid_and_num_of_covered_bases_df = get_sseqid_and_num_of_covered_bases_df(blast_result_df)

    num_of_alignments_before_exact_filtering_by_min_aligned_read_fraction = len(blast_result_df)
    blast_result_df = blast_result_df.merge(sseqid_and_num_of_covered_bases_df)
    assert len(blast_result_df) == num_of_alignments_before_exact_filtering_by_min_aligned_read_fraction

    blast_result_df['aligned_read_fraction'] = blast_result_df['num_of_read_bases_covered_by_any_alignment'] / blast_result_df['read_length']

    blast_result_df = blast_result_df[
        blast_result_df['num_of_read_bases_covered_by_any_alignment'] >= min_num_of_read_bases_covered_by_any_alignment
    ]

    num_of_reads_with_alignments_after_exact_aligned_read_fraction_filtering = blast_result_df['sseqid'].nunique()
    print(f'num_of_reads_with_alignments_after_exact_aligned_read_fraction_filtering: {num_of_reads_with_alignments_after_exact_aligned_read_fraction_filtering}')

    blast_result_df.to_csv(output_file_path_filtered_blast_result_csv, sep='\t', index=False)

def blast_alignment_region_to_reads(
        input_file_path_nuccore_fasta,
        input_file_path_sra_entry_fasta,
        min_num_of_read_bases_covered_by_any_alignment,
        alignment_region,
        max_evalue,
        seed_len,
        output_file_path_blast_result_csv,
        output_file_path_filtered_blast_result_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_blast_alignment_region_to_reads(
        input_file_path_nuccore_fasta=input_file_path_nuccore_fasta,
        input_file_path_sra_entry_fasta=input_file_path_sra_entry_fasta,
        min_num_of_read_bases_covered_by_any_alignment=min_num_of_read_bases_covered_by_any_alignment,
        alignment_region=alignment_region,
        max_evalue=max_evalue,
        seed_len=seed_len,
        output_file_path_blast_result_csv=output_file_path_blast_result_csv,
        output_file_path_filtered_blast_result_csv=output_file_path_filtered_blast_result_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

def discard_irrelevant_alignments(alignments_df, region_outer_and_inner_edges, margin_size):
    return alignments_df[(~(alignments_df['qstart'] > (region_outer_and_inner_edges[-1] + margin_size))) &
                         (~(alignments_df['qend'] < (region_outer_and_inner_edges[0] - margin_size)))]

def are_all_alignments_to_the_same_strand(alignments_df):
    return (alignments_df['sstart'] < alignments_df['send']).all() or (alignments_df['sstart'] > alignments_df['send']).all()

def are_alignments_collinear(alignments_df):
    assert are_all_alignments_to_the_same_strand(alignments_df)
    return alignments_df.sort_values('qstart')['sstart'].is_monotonic

def get_names_of_reads_with_alignments_covering_region_and_at_least_one_margin(
        alignments_df,
        region_outer_and_inner_edges,
        margin_size,
        require_collinear_alignment=False,
):
    # NOTE: cover here means not only covering every base in the region, but also every phosphodiester bond.
    # (that's because i merge intervals and then test whether i have a single merged interval that spans the whole region, which doesn't allow two intervals
    # without distance between them...)

    # NOTE: we also require here that the found alignments cover the region in the ref, and that the read bases that they cover form a single region (without
    # uncovered gaps).

    alignments_df = alignments_df.copy()
    alignments_df = discard_irrelevant_alignments(alignments_df, region_outer_and_inner_edges, margin_size)

    left_margin_start = region_outer_and_inner_edges[0] - margin_size
    right_margin_end = region_outer_and_inner_edges[-1] + margin_size

    # print(alignments_df[alignments_df['sseqid'] == 'ref_with_1000_bases_inserted_in_middle'])
    # raise
    names_of_reads_with_alignments_covering_region_and_at_least_one_margin = set()
    for sseqid, read_alignments_df in alignments_df.groupby('sseqid'):
        aligned_regions_in_ref = list(read_alignments_df[['qstart', 'qend']].itertuples(index=False, name=None))
        merged_aligned_regions_in_ref = generic_utils.get_merged_intervals(aligned_regions_in_ref)
        # print(f'sseqid, merged_aligned_regions_in_ref: {sseqid, merged_aligned_regions_in_ref}')
        for merged_aligned_region_in_ref in merged_aligned_regions_in_ref:
            if (
                    ((merged_aligned_region_in_ref[0] <= left_margin_start) and (merged_aligned_region_in_ref[1] >= (region_outer_and_inner_edges[-2] - 1))) or
                    ((merged_aligned_region_in_ref[0] <= (region_outer_and_inner_edges[1] + 1)) and (merged_aligned_region_in_ref[1] >= right_margin_end))
            ):
                relevant_read_alignments_df = read_alignments_df[
                    (read_alignments_df['qstart'] >= merged_aligned_region_in_ref[0]) &
                    (read_alignments_df['qend'] <= merged_aligned_region_in_ref[1])
                ]
                merged_aligned_regions_in_read = get_merged_aligned_regions_in_read(relevant_read_alignments_df)
                if len(merged_aligned_regions_in_read) > 1:
                    continue
                if require_collinear_alignment and (not are_alignments_collinear(relevant_read_alignments_df)):
                    continue

                names_of_reads_with_alignments_covering_region_and_at_least_one_margin.add(sseqid)
                break

    return names_of_reads_with_alignments_covering_region_and_at_least_one_margin

def get_names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region(
        alignments_df,
        region_outer_and_inner_edges,
        margin_size,
):
    # NOTE: cover here means not only covering every base in the region, but also every phosphodiester bond. see more detailed explanation above.

    alignments_df = alignments_df.copy()
    alignments_df = discard_irrelevant_alignments(alignments_df, region_outer_and_inner_edges, margin_size)
    forward_strand_alignments_df = alignments_df[alignments_df['sstart'] < alignments_df['send']]
    reverse_strand_alignments_df = alignments_df[alignments_df['sstart'] > alignments_df['send']]
    names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region = set()
    for strand_alignments_df in (forward_strand_alignments_df, reverse_strand_alignments_df):
        strand_alignments_df = strand_alignments_df.copy()
        names_of_reads_with_strand_alignments_covering_region = get_names_of_reads_with_alignments_covering_region_and_at_least_one_margin(
            alignments_df=strand_alignments_df,
            region_outer_and_inner_edges=region_outer_and_inner_edges,
            margin_size=margin_size,
            require_collinear_alignment=True
        )
        names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region |= names_of_reads_with_strand_alignments_covering_region

    return names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region

def get_alignments_strictly_contained_in_other_alignments_minimal_df(alignments_df):
    alignments_df = alignments_df.copy()
    assert 'smin' not in alignments_df
    assert 'smax' not in alignments_df
    alignments_df['smin'] = alignments_df[['sstart', 'send']].min(axis=1)
    alignments_df['smax'] = alignments_df[['sstart', 'send']].max(axis=1)
    alignments_strictly_contained_in_other_alignments_df = alignments_df.merge(
        alignments_df[['sseqid', 'qstart', 'qend', 'smin', 'smax']].rename(columns={x: f'other_{x}' for x in ('qstart', 'qend', 'smin', 'smax')}))
    alignments_strictly_contained_in_other_alignments_df = alignments_strictly_contained_in_other_alignments_df[
        (
            (alignments_strictly_contained_in_other_alignments_df['qstart'] >= alignments_strictly_contained_in_other_alignments_df['other_qstart']) &
            (alignments_strictly_contained_in_other_alignments_df['qend'] <= alignments_strictly_contained_in_other_alignments_df['other_qend']) &
            (alignments_strictly_contained_in_other_alignments_df['smin'] >= alignments_strictly_contained_in_other_alignments_df['other_smin']) &
            (alignments_strictly_contained_in_other_alignments_df['smax'] <= alignments_strictly_contained_in_other_alignments_df['other_smax'])
        ) &
        (
            (alignments_strictly_contained_in_other_alignments_df['qstart'] > alignments_strictly_contained_in_other_alignments_df['other_qstart']) |
            (alignments_strictly_contained_in_other_alignments_df['qend'] < alignments_strictly_contained_in_other_alignments_df['other_qend']) |
            (alignments_strictly_contained_in_other_alignments_df['smin'] > alignments_strictly_contained_in_other_alignments_df['other_smin']) |
            (alignments_strictly_contained_in_other_alignments_df['smax'] < alignments_strictly_contained_in_other_alignments_df['other_smax'])
        )
    ]
    return alignments_strictly_contained_in_other_alignments_df[['sseqid', 'qstart', 'qend', 'sstart', 'send']].drop_duplicates()

@generic_utils.execute_if_output_doesnt_exist_already
def cached_process_alignments_to_alignment_region(
        input_file_path_alignments_df_csv,
        ir_pair_outermost_edges,
        min_ir_pair_region_margin_size_for_evidence_read,
        output_file_path_extended_alignments_df_csv,
        output_file_path_ref_variant_evidence_read_name_to_aligned_read_region_pickle,
        output_file_path_potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    alignments_df = pd.read_csv(input_file_path_alignments_df_csv, sep='\t', low_memory=False)
    print(input_file_path_alignments_df_csv)

    assert (alignments_df['qstart'] < alignments_df['qend']).all()
    print(f'len(alignments_df): {len(alignments_df)}')

    alignments_df = alignments_df.merge(get_alignments_strictly_contained_in_other_alignments_minimal_df(alignments_df), how='left', indicator=True)
    alignments_df['alignment_is_strictly_contained_in_another_alignment'] = alignments_df['_merge'] == 'both'
    alignments_df.drop('_merge', axis=1, inplace=True)

    names_of_reads_with_alignments_covering_region_and_at_least_one_margin = get_names_of_reads_with_alignments_covering_region_and_at_least_one_margin(
        alignments_df=alignments_df[~alignments_df['alignment_is_strictly_contained_in_another_alignment']],
        region_outer_and_inner_edges=ir_pair_outermost_edges,
        margin_size=min_ir_pair_region_margin_size_for_evidence_read,
    )
    # raise RuntimeError(f'names_of_reads_with_alignments_covering_region_and_at_least_one_margin: '
    #                    f'{names_of_reads_with_alignments_covering_region_and_at_least_one_margin}')
    if names_of_reads_with_alignments_covering_region_and_at_least_one_margin:
        alignments_df = alignments_df.merge(
            pd.Series(list(names_of_reads_with_alignments_covering_region_and_at_least_one_margin), name='sseqid'), how='left', indicator=True)
        alignments_df['read_alignments_cover_ir_pair_region_with_margins'] = alignments_df['_merge'] == 'both'
        alignments_df.drop('_merge', axis=1, inplace=True)
    else:
        alignments_df['read_alignments_cover_ir_pair_region_with_margins'] = False

    names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region = get_names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region(
        alignments_df=alignments_df[
            alignments_df['read_alignments_cover_ir_pair_region_with_margins'] &
            (~alignments_df['alignment_is_strictly_contained_in_another_alignment'])
        ],
        region_outer_and_inner_edges=ir_pair_outermost_edges,
        margin_size=min_ir_pair_region_margin_size_for_evidence_read,
    )
    if names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region:
        alignments_df = alignments_df.merge(
            pd.Series(list(names_of_reads_with_consecutive_alignments_from_a_single_strand_covering_region), name='sseqid'), how='left', indicator=True)
        alignments_df['consecutive_read_alignments_from_a_single_strand_cover_ir_pair_region_with_margins'] = alignments_df['_merge'] == 'both'
        alignments_df.drop('_merge', axis=1, inplace=True)
    else:
        alignments_df['consecutive_read_alignments_from_a_single_strand_cover_ir_pair_region_with_margins'] = False

    alignments_df['is_read_potentially_from_non_ref_variant'] = (
            alignments_df['read_alignments_cover_ir_pair_region_with_margins'] &
            (~alignments_df['consecutive_read_alignments_from_a_single_strand_cover_ir_pair_region_with_margins'])
    )
    alignments_df.to_csv(output_file_path_extended_alignments_df_csv, sep='\t', index=False)
    # print(alignments_df[alignments_df['sseqid'] == 'fake_ref1'])
    alignments_df_grouped_by_read = alignments_df.groupby(['sseqid', 'read_length'])
    aligned_read_regions_df = alignments_df_grouped_by_read['sstart'].max().reset_index(name='max_sstart').merge(
        alignments_df_grouped_by_read['sstart'].min().reset_index(name='min_sstart')).merge(
        alignments_df_grouped_by_read['send'].max().reset_index(name='max_send')).merge(
        alignments_df_grouped_by_read['send'].min().reset_index(name='min_send'))
    # print(aligned_read_regions_df[['sseqid', 'max_sstart', 'min_sstart', 'max_send', 'min_send', 'read_length']].drop_duplicates())
    aligned_read_regions_df['smin'] = aligned_read_regions_df[['min_sstart', 'min_send']].min(axis=1)
    aligned_read_regions_df['smax'] = aligned_read_regions_df[['max_sstart', 'max_send']].max(axis=1)
    aligned_read_regions_df.drop(['max_sstart', 'min_sstart', 'max_send', 'min_send'], axis=1, inplace=True)

    assert (aligned_read_regions_df['smax'] > aligned_read_regions_df['smin']).all()
    # print(aligned_read_regions_df[['sseqid', 'smax', 'smin', 'read_length']].drop_duplicates())
    assert ((aligned_read_regions_df['smax'] - aligned_read_regions_df['smin'] + 1) <= aligned_read_regions_df['read_length']).all()

    # print("((aligned_read_regions_df['smax'] - aligned_read_regions_df['smin'] + 1) / aligned_read_regions_df['read_length']).describe()")
    print(((aligned_read_regions_df['smax'] - aligned_read_regions_df['smin'] + 1) / aligned_read_regions_df['read_length']).describe())

    ref_variant_evidence_read_name_to_aligned_read_region = {
        row['sseqid']: (row['smin'], row['smax'])
        for _, row in aligned_read_regions_df.merge(
            alignments_df[alignments_df['consecutive_read_alignments_from_a_single_strand_cover_ir_pair_region_with_margins']]['sseqid']).iterrows()
    }
    with open(output_file_path_ref_variant_evidence_read_name_to_aligned_read_region_pickle, 'wb') as f:
        pickle.dump(ref_variant_evidence_read_name_to_aligned_read_region, f, protocol=4)

    potential_non_ref_variant_evidence_read_name_to_aligned_read_region = {
        row['sseqid']: (row['smin'], row['smax'])
        for _, row in aligned_read_regions_df.merge(alignments_df[alignments_df['is_read_potentially_from_non_ref_variant']]['sseqid']).iterrows()
    }
    with open(output_file_path_potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle, 'wb') as f:
        pickle.dump(potential_non_ref_variant_evidence_read_name_to_aligned_read_region, f, protocol=4)


def process_alignments_to_alignment_region(
        input_file_path_alignments_df_csv,
        ir_pair_outermost_edges,
        min_ir_pair_region_margin_size_for_evidence_read,
        output_file_path_extended_alignments_df_csv,
        output_file_path_ref_variant_evidence_read_name_to_aligned_read_region_pickle,
        output_file_path_potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_process_alignments_to_alignment_region(
        input_file_path_alignments_df_csv=input_file_path_alignments_df_csv,
        ir_pair_outermost_edges=ir_pair_outermost_edges,
        min_ir_pair_region_margin_size_for_evidence_read=min_ir_pair_region_margin_size_for_evidence_read,
        output_file_path_extended_alignments_df_csv=output_file_path_extended_alignments_df_csv,
        output_file_path_ref_variant_evidence_read_name_to_aligned_read_region_pickle=output_file_path_ref_variant_evidence_read_name_to_aligned_read_region_pickle,
        output_file_path_potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle=output_file_path_potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=7,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_verify_ir_pairs_according_to_df(
        input_file_path_ir_pairs_df_csv,
        ir_pair_region_with_margins_start,
        ir_pairs,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    ir_pairs_df = pd.read_csv(input_file_path_ir_pairs_df_csv, sep='\t', low_memory=False)
    ir_pairs_df[['left1', 'right1', 'left2', 'right2']] += ir_pair_region_with_margins_start - 1
    # print(ir_pairs_df)
    for ir_pair in ir_pairs:
        left1, right1, left2, right2 = ir_pair
        left1_smaller_than_right1_etc = left1 < right1 < left2 < right2
        if not left1_smaller_than_right1_etc:
            print(f'ir_pair: {ir_pair}')
        assert left1_smaller_than_right1_etc

        both_repeats_are_of_same_len = (right1 - left1) == (right2 - left2)
        if not both_repeats_are_of_same_len:
            print(f'ir_pair: {ir_pair}')
        assert both_repeats_are_of_same_len

        is_ir_pair_in_df_or_could_result_from_merging_ir_pairs_in_the_df = (
                ((ir_pairs_df['left1'] == left1) & (ir_pairs_df['right2'] == right2)).any() and
                ((ir_pairs_df['right1'] == right1) & (ir_pairs_df['left2'] == left2)).any()
        )
        if not is_ir_pair_in_df_or_could_result_from_merging_ir_pairs_in_the_df:
            print(f'ir_pair: {ir_pair}')
            print(ir_pairs_df)
        assert is_ir_pair_in_df_or_could_result_from_merging_ir_pairs_in_the_df

def verify_ir_pairs_according_to_df(
        input_file_path_ir_pairs_df_csv,
        ir_pair_region_with_margins_start,
        ir_pairs,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_verify_ir_pairs_according_to_df(
        input_file_path_ir_pairs_df_csv=input_file_path_ir_pairs_df_csv,
        ir_pair_region_with_margins_start=ir_pair_region_with_margins_start,
        ir_pairs=ir_pairs,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome(
        input_file_path_alignments_df,
        input_file_path_assembly_fasta,
        input_file_path_reads_fasta,
        alignment_region_seq_name,
        alignment_region,
        max_evalue,
        seed_len,
        output_file_path_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle,
        out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    blast_interface_and_utils.make_blast_nucleotide_db(input_file_path_reads_fasta)

    assembly_alignment_to_relevant_reads_df_csv = os.path.join(out_dir_path, 'assembly_alignment_to_relevant_reads_df.csv')
    blast_interface_and_utils.blast_nucleotide(
        query_fasta_file_path=input_file_path_assembly_fasta,
        blast_db_path=input_file_path_reads_fasta,
        blast_results_file_path=assembly_alignment_to_relevant_reads_df_csv,
        perform_gapped_alignment=True,
        query_strand_to_search='both',
        max_evalue=max_evalue,
        seed_len=seed_len,
        max_num_of_overlaps_to_report=int(1e8),
        # verbose=True,
    )

    initial_alignments_df = pd.read_csv(input_file_path_alignments_df, sep='\t', low_memory=False)
    blast_result_df = blast_interface_and_utils.read_blast_results_df(assembly_alignment_to_relevant_reads_df_csv)

    assert alignment_region_seq_name in set(blast_result_df['qseqid'])

    # remove alignments that overlap alignment region.
    assert (blast_result_df['qstart'] < blast_result_df['qend']).all()
    blast_result_df = blast_result_df[
        (blast_result_df['qseqid'] != alignment_region_seq_name) |
        (blast_result_df['qend'] < alignment_region[0]) |
        (blast_result_df['qstart'] > alignment_region[1])
    ]


    sseqid_df = initial_alignments_df[['sseqid', 'num_of_read_bases_covered_by_any_alignment']].drop_duplicates().rename(
        columns={'num_of_read_bases_covered_by_any_alignment': 'num_of_read_bases_covered_by_any_alignment_to_alignment_region'})

    sseqid_sum_of_alignment_lengths = blast_result_df.groupby(['sseqid'])['length'].sum().reset_index(
        name='sum_of_lengthes_of_alignments_not_overlapping_alignment_region')
    sseqid_df = sseqid_df.merge(sseqid_sum_of_alignment_lengths, how='left')

    guaranteed_to_not_have_better_alignments_outside_alignment_region_filter = (
        sseqid_df['sum_of_lengthes_of_alignments_not_overlapping_alignment_region'] <
        sseqid_df['num_of_read_bases_covered_by_any_alignment_to_alignment_region']
    )
    easily_good_reads = set(sseqid_df[guaranteed_to_not_have_better_alignments_outside_alignment_region_filter])
    blast_result_df = blast_result_df.merge(sseqid_df[~guaranteed_to_not_have_better_alignments_outside_alignment_region_filter]['sseqid'])

    sseqid_and_num_of_covered_bases_df = get_sseqid_and_num_of_covered_bases_df(blast_result_df).rename(columns={
        'num_of_read_bases_covered_by_any_alignment': 'num_of_read_bases_covered_by_any_alignment_not_overlapping_alignment_region'})

    sseqid_df = sseqid_df.merge(sseqid_and_num_of_covered_bases_df, how='left')

    filtered_sseqid_df = sseqid_df[
        sseqid_df['num_of_read_bases_covered_by_any_alignment_not_overlapping_alignment_region'].isna() |
        (sseqid_df['num_of_read_bases_covered_by_any_alignment_not_overlapping_alignment_region'] <
         sseqid_df['num_of_read_bases_covered_by_any_alignment_to_alignment_region'])
    ]
    not_easily_but_good_reads = set(filtered_sseqid_df['sseqid'])

    names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome = sorted(easily_good_reads | not_easily_but_good_reads)
    with open(output_file_path_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle, 'wb') as f:
        pickle.dump(names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome, f, protocol=4)

def write_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome(
        input_file_path_alignments_df,
        input_file_path_assembly_fasta,
        input_file_path_reads_fasta,
        alignment_region_seq_name,
        alignment_region,
        max_evalue,
        seed_len,
        output_file_path_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle,
        out_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome(
        input_file_path_alignments_df=input_file_path_alignments_df,
        input_file_path_assembly_fasta=input_file_path_assembly_fasta,
        input_file_path_reads_fasta=input_file_path_reads_fasta,
        alignment_region_seq_name=alignment_region_seq_name,
        alignment_region=alignment_region,
        max_evalue=max_evalue,
        seed_len=seed_len,
        output_file_path_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle=output_file_path_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle,
        out_dir_path=out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=0,
    )

def get_read_names_sorted_by_evidence_quality_as_df(alignments_df):
    alignments_df_grouped_by_read = alignments_df.groupby(['sseqid', 'aligned_read_fraction', 'num_of_read_bases_covered_by_any_alignment'])
    reads_df = alignments_df_grouped_by_read.size().reset_index(name='num_of_read_alignments').merge(
        alignments_df_grouped_by_read['gapopen'].sum().reset_index(name='total_num_of_gapopens')).merge(
        alignments_df_grouped_by_read['mismatch'].sum().reset_index(name='total_num_of_mismatches'))
    reads_df['unaligned_read_fraction'] = 1 - reads_df['aligned_read_fraction']
    reads_df['minus_num_of_read_bases_covered_by_any_alignment'] = -reads_df['num_of_read_bases_covered_by_any_alignment']

    return reads_df.sort_values([
        # 'num_of_read_alignments',
        'minus_num_of_read_bases_covered_by_any_alignment',
        'total_num_of_gapopens',
        'total_num_of_mismatches',
        'unaligned_read_fraction'
    ])[['sseqid']]

def get_variant_ir_pairs_to_evidence_read_names(read_name_to_possible_ir_pairs_used_to_reach_from_ref):
    variant_ir_pairs_to_evidence_read_names = collections.defaultdict(set)
    for read_name, variant_ir_pairs in read_name_to_possible_ir_pairs_used_to_reach_from_ref.items():
        variant_ir_pairs_to_evidence_read_names[frozenset(variant_ir_pairs)].add(read_name)
    variant_ir_pairs_to_evidence_read_names = dict(variant_ir_pairs_to_evidence_read_names) # I don't want a defaultdict moving around.
    return variant_ir_pairs_to_evidence_read_names

def add_is_best_evidence_read_for_variant_column(
        alignments_df,
        reads_info,
        inaccurate_or_not_beautiful_mauve_alignment_read_names,
):
    extended_alignments_df = alignments_df.copy()

    if inaccurate_or_not_beautiful_mauve_alignment_read_names:
        assert '_merge' not in extended_alignments_df
        extended_alignments_df = extended_alignments_df.merge(pd.Series(sorted(inaccurate_or_not_beautiful_mauve_alignment_read_names), name='sseqid'),
                                                              indicator=True, how='left')
        extended_alignments_df = extended_alignments_df[extended_alignments_df['_merge'] == 'left_only'].drop('_merge', axis=1)

    read_names_sorted_by_evidence_quality_as_df = get_read_names_sorted_by_evidence_quality_as_df(extended_alignments_df)

    best_evidence_read_for_ref = read_names_sorted_by_evidence_quality_as_df.merge(
        extended_alignments_df[extended_alignments_df['read_evidence_type'] == 'ref_variant']['sseqid']).iloc[0,0]

    extended_alignments_df['is_best_evidence_read_for_variant'] = False
    extended_alignments_df.loc[extended_alignments_df['sseqid'] == best_evidence_read_for_ref, 'is_best_evidence_read_for_variant'] = True

    if 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref' in reads_info:
        read_name_to_possible_ir_pairs_used_to_reach_from_ref = reads_info['non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref']

        variant_ir_pairs_to_evidence_read_names = get_variant_ir_pairs_to_evidence_read_names(read_name_to_possible_ir_pairs_used_to_reach_from_ref)

        for variant_ir_pairs, evidence_read_names in variant_ir_pairs_to_evidence_read_names.items():
            best_evidence_read_for_variant = read_names_sorted_by_evidence_quality_as_df.merge(
                extended_alignments_df[extended_alignments_df['read_evidence_type'] == 'non_ref_variant']['sseqid'].drop_duplicates()
            ).merge(pd.Series(list(evidence_read_names), name='sseqid')).iloc[0,0]
            extended_alignments_df.loc[extended_alignments_df['sseqid'] == best_evidence_read_for_variant, 'is_best_evidence_read_for_variant'] = True

        assert extended_alignments_df[['sseqid', 'is_best_evidence_read_for_variant']].drop_duplicates()[
                   'is_best_evidence_read_for_variant'].sum() == 1 + len(variant_ir_pairs_to_evidence_read_names)

    if inaccurate_or_not_beautiful_mauve_alignment_read_names:
        assert (set(extended_alignments_df) - set(alignments_df)) == {'is_best_evidence_read_for_variant'}
        extended_alignments_df = alignments_df.merge(extended_alignments_df, how='left')
        extended_alignments_df['is_best_evidence_read_for_variant'].fillna(False, inplace=True)

    assert not extended_alignments_df['is_best_evidence_read_for_variant'].isna().any()
    assert len(extended_alignments_df) == len(alignments_df)
    return extended_alignments_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_run_mauve_for_each_best_evidence_read(
        input_file_path_ir_pair_region_with_margins_fasta,
        input_file_path_potential_evidence_reads_truncated_fasta,
        best_evidence_read_name_to_align_to_plus_strand,
        output_file_path_best_evidence_read_name_to_info_pickle,
        best_evidence_reads_output_dir_path,
):
    best_evidence_read_name_to_info = {}
    for truncated_read_seq in SeqIO.parse(input_file_path_potential_evidence_reads_truncated_fasta, 'fasta'):
        read_name = truncated_read_seq.name
        if read_name in best_evidence_read_name_to_align_to_plus_strand:
            read_output_dir_path = os.path.join(best_evidence_reads_output_dir_path, read_name)
            pathlib.Path(read_output_dir_path).mkdir(parents=True, exist_ok=True)

            align_to_plus_strand = best_evidence_read_name_to_align_to_plus_strand[read_name]
            if align_to_plus_strand:
                truncated_read_fasta_file_path = os.path.join(read_output_dir_path, f'truncated_{read_name}.fasta')
            else:
                truncated_read_fasta_file_path = os.path.join(read_output_dir_path, f'RC_of_truncated_{read_name}.fasta')
                truncated_read_seq = truncated_read_seq.reverse_complement()
                truncated_read_seq.name = truncated_read_seq.description = truncated_read_seq.id = read_name
            bio_utils.write_records_to_fasta_or_gb_file([truncated_read_seq], truncated_read_fasta_file_path)



            mauve_alignment_results_xmfa_file_path = os.path.join(read_output_dir_path, f'truncated_{read_name}_mauve_results.xmfa')
            mauve_alignment_results_backbone_csv_file_path = os.path.join(read_output_dir_path, f'truncated_{read_name}_mauve_result_backbone.csv')
            mauve_interface_and_utils.progressive_mauve(
                input_file_path_seq0_fasta=truncated_read_fasta_file_path,
                input_file_path_seq1_fasta=input_file_path_ir_pair_region_with_margins_fasta,
                assume_input_sequences_are_collinear=False,
                output_file_path_alignment_xmfa=mauve_alignment_results_xmfa_file_path,
                output_file_path_backbone_csv=mauve_alignment_results_backbone_csv_file_path,
            )

            best_evidence_read_name_to_info[read_name] = {
                'truncated_read_fasta_file_path': truncated_read_fasta_file_path,
                'mauve_alignment_results_xmfa_file_path': mauve_alignment_results_xmfa_file_path,
                'mauve_alignment_results_backbone_csv_file_path': mauve_alignment_results_backbone_csv_file_path,
            }

    with open(output_file_path_best_evidence_read_name_to_info_pickle, 'wb') as f:
        pickle.dump(best_evidence_read_name_to_info, f, protocol=4)

def run_mauve_for_each_best_evidence_read(
        input_file_path_ir_pair_region_with_margins_fasta,
        input_file_path_potential_evidence_reads_truncated_fasta,
        best_evidence_read_name_to_align_to_plus_strand,
        output_file_path_best_evidence_read_name_to_info_pickle,
        best_evidence_reads_output_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_run_mauve_for_each_best_evidence_read(
        input_file_path_ir_pair_region_with_margins_fasta=input_file_path_ir_pair_region_with_margins_fasta,
        input_file_path_potential_evidence_reads_truncated_fasta=input_file_path_potential_evidence_reads_truncated_fasta,
        best_evidence_read_name_to_align_to_plus_strand=best_evidence_read_name_to_align_to_plus_strand,
        output_file_path_best_evidence_read_name_to_info_pickle=output_file_path_best_evidence_read_name_to_info_pickle,
        best_evidence_reads_output_dir_path=best_evidence_reads_output_dir_path,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_fasta_file_with_inverted_region(
        input_file_path_fasta,
        region_to_invert,
        new_seq_name,
        output_file_path_fasta,
):
    orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(input_file_path_fasta)

    assert region_to_invert[0] >= 2
    assert region_to_invert[1] <= len(orig_seq) - 1

    new_seq = (
        orig_seq[:(region_to_invert[0] - 1)] +
        orig_seq[(region_to_invert[0] - 1):region_to_invert[1]].reverse_complement() +
        orig_seq[region_to_invert[1]:]
    )
    assert len(new_seq) == len(orig_seq)

    new_seq.name = new_seq.description = new_seq.id = new_seq_name
    bio_utils.write_records_to_fasta_or_gb_file([new_seq], output_file_path_fasta, file_type='fasta')

def write_fasta_file_with_inverted_region(
        input_file_path_fasta,
        region_to_invert,
        new_seq_name,
        output_file_path_fasta,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_fasta_file_with_inverted_region(
        input_file_path_fasta=input_file_path_fasta,
        region_to_invert=region_to_invert,
        new_seq_name=new_seq_name,
        output_file_path_fasta=output_file_path_fasta,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_best_alignment_pair_scores_df(
        input_file_path_sam,
        output_file_path_best_alignments_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with samtools_and_sam_files_interface.horrifying_hack_for_samtools_1_7_libcrypto_error_context_manager():
        alignments_df = samtools_and_sam_files_interface.get_primary_alignments_df_with_int_tags_columns(input_file_path_sam, ['AS', 'YS'])

    alignments_df['pair_total_align_score'] = alignments_df['AS'] + alignments_df['YS']
    minimal_alignments_df = alignments_df[['QNAME', 'pair_total_align_score']].copy()
    assert minimal_alignments_df['QNAME'].is_unique

    assert (minimal_alignments_df['QNAME'].str.endswith('.1') | minimal_alignments_df['QNAME'].str.endswith('.2')).all()
    minimal_alignments_df['QNAME'] = minimal_alignments_df['QNAME'].str.slice(stop=-2)
    assert set(minimal_alignments_df['QNAME'].value_counts()) == {2}
    minimal_alignments_df.drop_duplicates(inplace=True)

    minimal_alignments_df.to_csv(output_file_path_best_alignments_df_csv, sep='\t', index=False)

def write_best_alignment_pair_scores_df(
        input_file_path_sam,
        output_file_path_best_alignments_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_best_alignment_pair_scores_df(
        input_file_path_sam=input_file_path_sam,
        output_file_path_best_alignments_df_csv=output_file_path_best_alignments_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_different_score_alignments_df(
        input_file_path_best_alignments_to_ref_df_csv,
        input_file_path_best_alignments_to_non_ref_df_csv,
        output_file_path_different_score_alignments_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    ref_best_alignments_df = pd.read_csv(input_file_path_best_alignments_to_ref_df_csv, sep='\t', low_memory=False)
    non_ref_best_alignments_df = pd.read_csv(input_file_path_best_alignments_to_non_ref_df_csv, sep='\t', low_memory=False)
    joined_best_alignments_df = ref_best_alignments_df.merge(
        non_ref_best_alignments_df, how='outer', on='QNAME', suffixes=('_ref', '_non_ref'))
    diff_score_alignments_df = joined_best_alignments_df[
        joined_best_alignments_df['pair_total_align_score_ref'] != joined_best_alignments_df['pair_total_align_score_non_ref']].copy()

    diff_score_alignments_df['pair_total_align_score_ref'].fillna(-np.inf, inplace=True)
    diff_score_alignments_df['pair_total_align_score_non_ref'].fillna(-np.inf, inplace=True)
    # this raises pandas.core.common.SettingWithCopyWarning
    # diff_score_alignments_df[['pair_total_align_score_ref', 'pair_total_align_score_non_ref']].fillna(-np.inf, inplace=True)

    diff_score_alignments_df['score_diff'] = diff_score_alignments_df['pair_total_align_score_ref'] - diff_score_alignments_df['pair_total_align_score_non_ref']

    diff_score_alignments_df.to_csv(output_file_path_different_score_alignments_df_csv, sep='\t', index=False)

def write_different_score_alignments_df(
        input_file_path_best_alignments_to_ref_df_csv,
        input_file_path_best_alignments_to_non_ref_df_csv,
        output_file_path_different_score_alignments_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_different_score_alignments_df(
        input_file_path_best_alignments_to_ref_df_csv=input_file_path_best_alignments_to_ref_df_csv,
        input_file_path_best_alignments_to_non_ref_df_csv=input_file_path_best_alignments_to_non_ref_df_csv,
        output_file_path_different_score_alignments_df_csv=output_file_path_different_score_alignments_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

def do_massive_screening_stage6(
        search_for_pis_args,
):
    massive_screening_stage6_out_dir_path = search_for_pis_args['stage6']['output_dir_path']

    debug___num_of_taxa_to_go_over = search_for_pis_args['debug___num_of_taxa_to_go_over']
    debug___taxon_uids = search_for_pis_args['debug___taxon_uids']

    stage_out_file_name_suffix = massive_screening_stage_1.get_stage_out_file_name_suffix(
        debug___num_of_taxa_to_go_over=debug___num_of_taxa_to_go_over,
        debug___taxon_uids=debug___taxon_uids,
    )

    massive_screening_log_file_path = os.path.join(massive_screening_stage6_out_dir_path, 'massive_screening_stage6_log.txt')
    stage6_results_info_pickle_file_path = os.path.join(massive_screening_stage6_out_dir_path, search_for_pis_args['stage6']['results_pickle_file_name'])
    stage6_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage6_results_info_pickle_file_path, stage_out_file_name_suffix)

    if not search_for_pis_args['stage6']['DEBUG___sra_entry_fasta_file_path']:
        stage1_out_dir_path = search_for_pis_args['stage1']['output_dir_path']
        stage1_results_info_pickle_file_path = os.path.join(stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])
        stage1_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            stage1_results_info_pickle_file_path, stage_out_file_name_suffix)

        with open(stage1_results_info_pickle_file_path, 'rb') as f:
            stage1_results_info = pickle.load(f)
        taxa_df_csv_file_path = stage1_results_info['taxa_df_csv_file_path']

    pathlib.Path(massive_screening_stage6_out_dir_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=massive_screening_log_file_path, level=logging.DEBUG, format='%(asctime)s %(message)s')
    generic_utils.print_and_write_to_log(f'---------------starting do_massive_screening_stage6({massive_screening_stage6_out_dir_path})---------------')


    sra_entries_output_dir_path = os.path.join(massive_screening_stage6_out_dir_path, search_for_pis_args['stage6']['sra_entries_dir_name'])
    blast_homolog_to_its_margins_nuccores_out_dir_path = os.path.join(massive_screening_stage6_out_dir_path, 'blast_homolog_to_its_margins__nuccores')
    nuccore_entries_output_dir_path = os.path.join(massive_screening_stage6_out_dir_path, 'nuccore_entries')
    assembly_entries_output_dir_path = os.path.join(massive_screening_stage6_out_dir_path, 'assembly_entries')
    pathlib.Path(sra_entries_output_dir_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(nuccore_entries_output_dir_path).mkdir(parents=True, exist_ok=True)

    cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args = search_for_pis_args['stage6'][
        'cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args']
    sra_accession_to_type_and_sra_file_name = search_for_pis_args['stage6']['sra_accession_to_type_and_sra_file_name']
    sra_accession_to_bioproject_accession = search_for_pis_args['stage6']['sra_accession_to_bioproject_accession']
    min_alignment_region_margin_size = search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_alignment_region_margin_size']
    alignment_region_to_long_reads_max_evalue = search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['max_evalue']
    alignment_region_to_long_reads_seed_len = search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['seed_len']
    min_ir_pair_region_margin_size_for_evidence_read = search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_ir_pair_region_margin_size_for_evidence_read']
    min_num_of_read_bases_covered_by_any_alignment = search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_num_of_read_bases_covered_by_any_alignment']

    align_assembly_to_relevant_long_reads_max_evalue = search_for_pis_args['stage6']['blast_alignment_assembly_to_relevant_long_reads']['max_evalue']
    align_assembly_to_relevant_long_reads_seed_len = search_for_pis_args['stage6']['blast_alignment_assembly_to_relevant_long_reads']['seed_len']

    nuccore_accession_to_assembly_accesion = search_for_pis_args['stage6']['nuccore_accession_to_assembly_accesion']
    nuccore_accession_to_name_in_assembly = search_for_pis_args['stage6']['nuccore_accession_to_name_in_assembly']
    # nuccore_accession_and_ir_pair_region_with_margins_to_special_min_repeat_len = search_for_pis_args[
    #     'stage6']['blast_nuccore_to_find_ir_pairs']['nuccore_accession_and_ir_pair_region_with_margins_to_special_min_repeat_len']

    rna_seq_analysis_inverted_repeat_max_evalue = search_for_pis_args['stage6']['rna_seq_analysis']['inverted_repeat_max_evalue']
    rna_seq_analysis_min_abs_score_diff = search_for_pis_args['stage6']['rna_seq_analysis']['min_abs_score_diff']

    raw_read_alignment_result_dfs = []
    num_of_cds_contexts = len(cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args)
    # print(list(cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args))
    # raise
    homologs_dfs = []
    putative_pi_locus_description_flat_dicts = []
    all_all_repeat_cds_covered_bases_proportions = []
    all_all_alignment_bases_covered_by_cds_proportions = []
    all_all_diff_score_alignments_dfs = []
    rna_seq_summary_flat_dicts = []
    for i, (cds_context_name, nuccore_accession_to_alignment_regions_raw_read_alignment_args) in enumerate(
            sorted(cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args.items())):
        generic_utils.print_and_write_to_log(f'starting to work on CDS context {i + 1}/{num_of_cds_contexts} ({cds_context_name}).')
        # START = 18
        # if i >= START + 2:
        #     exit()
        # if i < START:
        #     continue

        if 0:
            if cds_context_name in (
                'MTase: BREX type 1',
            ):
                continue
        if 0:
            if cds_context_name not in (
                # 'DUF4965',
                # 'MTase: two upstream helicases',
                # 'MTase: Class 1 DISARM-like',
                # 'MTase: Class 1 DISARM', #####
                # 'MTase: upstream PLD&SNF2 helicase', #####
                #
                # 'MTase: downstream PLD&SNF2 helicase',
                # 'MTase: upstream PLD&SNF2 helicase',
                # 'MTase: solitary',
                # 'MTase: upstream DUF1016',
                #
                # 'MTase: BREX type 1', #####
                # 'MTase: BREX type 1, downstream extra short PglX',


                # 'PilV and phage tail collar',

                # 'DUF4393',

                # 'OM receptor: downstream SusD/RagB',

                # 'RM specificity: R-M-S',
                # 'RM specificity: M-invertibleSs-R',
                # 'RM specificity: R-M-DUF1016-S',
                # 'RM specificity: R-M-Fic/DOC-S',
                # 'RM specificity: R-M-RhuM-S',
                #
                # 'phage tail: upstream phage tail protein I and downstream tail fiber assembly',
                # 'phage tail: downstream transporter and endonuclease',
                # 'phage tail: upstream DUF2313 and downstream tail fiber assembly',


            ):
                continue
        for j, (nuccore_accession, alignment_regions_raw_read_alignment_args) in enumerate(sorted(nuccore_accession_to_alignment_regions_raw_read_alignment_args.items())):
            generic_utils.print_and_write_to_log(f'starting to work on nuccore accession {nuccore_accession}.')

            for curr_raw_read_alignment_args in alignment_regions_raw_read_alignment_args:
                if 'describe_in_the_paper' not in curr_raw_read_alignment_args:
                    continue
                nuccore_entry_output_dir_path = os.path.join(nuccore_entries_output_dir_path, nuccore_accession)
                pathlib.Path(nuccore_entry_output_dir_path).mkdir(parents=True, exist_ok=True)
                cds_df_csv_file_path = os.path.join(nuccore_entry_output_dir_path, f'{nuccore_accession}_cds_df.csv')

                if search_for_pis_args['stage6']['DEBUG___nuccore_gb_file_path']:
                    nuccore_gb_file_path = search_for_pis_args['stage6']['DEBUG___nuccore_gb_file_path']
                else:
                    nuccore_gb_file_path = os.path.join(nuccore_entry_output_dir_path, f'{nuccore_accession}.gb')
                    bio_utils.download_nuccore_entry_from_ncbi(
                        nuccore_accession=nuccore_accession,
                        output_file_type='genbank',
                        output_file_path_nuccore_entry=nuccore_gb_file_path,
                    )

                # nuccore_taxon_uid_txt_file_path = os.path.join(nuccore_entry_output_dir_path, f'{nuccore_accession}_taxon_uid.txt')
                # bio_utils.extract_taxon_uid_from_genbank_file(
                #     input_file_path_gb=nuccore_gb_file_path,
                #     output_file_path_taxon_uid=nuccore_taxon_uid_txt_file_path,
                # )
                # taxon_uid = int(generic_utils.read_text_file(nuccore_taxon_uid_txt_file_path))

                if search_for_pis_args['stage6']['DEBUG___nuccore_fasta_file_path']:
                    nuccore_fasta_file_path = search_for_pis_args['stage6']['DEBUG___nuccore_fasta_file_path']
                else:
                    nuccore_fasta_file_path = os.path.join(nuccore_entry_output_dir_path, f'{nuccore_accession}.fasta')
                    bio_utils.download_nuccore_entry_from_ncbi(
                        nuccore_accession=nuccore_accession,
                        output_file_type='fasta',
                        output_file_path_nuccore_entry=nuccore_fasta_file_path,
                    )

                longer_linked_repeat_cds_region_and_protein_id = curr_raw_read_alignment_args['longest_linked_repeat_cds_region_and_protein_id']
                locus_description_for_table_3 = curr_raw_read_alignment_args['locus_description_for_table_3']
                longer_linked_repeat_cds_region, longer_linked_repeat_cds_protein_id = longer_linked_repeat_cds_region_and_protein_id
                longer_linked_repeat_cds_output_dir_path = os.path.join(nuccore_entry_output_dir_path,
                                                                        f'longer_repeat_cds_{longer_linked_repeat_cds_region[0]}_{longer_linked_repeat_cds_region[1]}')
                pathlib.Path(longer_linked_repeat_cds_output_dir_path).mkdir(parents=True, exist_ok=True)

                longer_linked_repeat_cds_homologs_df_csv_file_path = os.path.join(longer_linked_repeat_cds_output_dir_path, 'homologs_df.csv')
                longer_linked_repeat_cds_homologs_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
                    longer_linked_repeat_cds_homologs_df_csv_file_path, stage_out_file_name_suffix)

                nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle_file_path = os.path.join(
                    longer_linked_repeat_cds_output_dir_path, 'nuccore_accessions_of_homolog_nuccores_with_empty_cds_df.pickle')
                nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
                    nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle_file_path, stage_out_file_name_suffix)

                all_repeat_cds_covered_bases_proportions_pickle_file_path = os.path.join(
                    longer_linked_repeat_cds_output_dir_path, 'all_repeat_cds_covered_bases_proportions.pickle')
                all_repeat_cds_covered_bases_proportions_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
                    all_repeat_cds_covered_bases_proportions_pickle_file_path, stage_out_file_name_suffix)

                all_alignment_bases_covered_by_cds_proportions_pickle_file_path = os.path.join(
                    longer_linked_repeat_cds_output_dir_path, 'all_alignment_bases_covered_by_cds_proportions.pickle')
                all_alignment_bases_covered_by_cds_proportions_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
                    all_alignment_bases_covered_by_cds_proportions_pickle_file_path, stage_out_file_name_suffix)

                if not search_for_pis_args['stage6']['DEBUG___sra_entry_fasta_file_path']:
                    write_repeat_cds_homologs_df(
                        input_file_path_taxa_df_csv=taxa_df_csv_file_path,
                        input_file_path_nuccore_fasta=nuccore_fasta_file_path,
                        repeat_cds_region=longer_linked_repeat_cds_region,
                        blast_repeat_cds_to_each_taxon_genome_seed_len=search_for_pis_args['stage6']['blast_longer_linked_repeat_cds_to_each_taxon_genome']['seed_len'],
                        blast_repeat_cds_to_each_taxon_genome_max_evalue=search_for_pis_args['stage6']['blast_longer_linked_repeat_cds_to_each_taxon_genome']['max_evalue'],
                        min_repeat_cds_covered_bases_proportion=search_for_pis_args['stage6']['blast_longer_linked_repeat_cds_to_each_taxon_genome'][
                            'min_repeat_cds_covered_bases_proportion'],
                        min_alignment_bases_covered_by_cds_proportion=search_for_pis_args['stage6']['blast_longer_linked_repeat_cds_to_each_taxon_genome'][
                            'min_alignment_bases_covered_by_cds_proportion'],
                        blast_homolog_to_its_margins_seed_len=search_for_pis_args['stage6']['blast_longer_linked_repeat_cds_homolog_to_its_margins']['seed_len'],
                        blast_homolog_to_its_margins_max_evalue=search_for_pis_args['stage6']['blast_longer_linked_repeat_cds_homolog_to_its_margins']['max_evalue'],
                        homolog_margin_size=search_for_pis_args['stage6']['blast_longer_linked_repeat_cds_homolog_to_its_margins']['margin_size'],
                        output_dir_path=longer_linked_repeat_cds_output_dir_path,
                        output_file_path_homologs_df_csv=longer_linked_repeat_cds_homologs_df_csv_file_path,
                        output_file_path_nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle=(
                            nuccore_accessions_of_homolog_nuccores_with_empty_cds_df_pickle_file_path),
                        output_file_path_all_repeat_cds_covered_bases_proportions_pickle=all_repeat_cds_covered_bases_proportions_pickle_file_path,
                        output_file_path_alignment_bases_covered_by_cds_proportions_pickle=all_alignment_bases_covered_by_cds_proportions_pickle_file_path,
                        blast_homolog_to_its_margins_nuccores_out_dir_path=blast_homolog_to_its_margins_nuccores_out_dir_path,
                    )
                    with open(all_repeat_cds_covered_bases_proportions_pickle_file_path, 'rb') as f:
                        curr_all_repeat_cds_covered_bases_proportions = pickle.load(f)
                    all_all_repeat_cds_covered_bases_proportions.extend(curr_all_repeat_cds_covered_bases_proportions)

                    with open(all_alignment_bases_covered_by_cds_proportions_pickle_file_path, 'rb') as f:
                        curr_all_alignment_bases_covered_by_cds_proportions = pickle.load(f)
                    all_all_alignment_bases_covered_by_cds_proportions.extend(curr_all_alignment_bases_covered_by_cds_proportions)

                    curr_homologs_df = pd.read_csv(longer_linked_repeat_cds_homologs_df_csv_file_path, sep='\t', low_memory=False)

                    curr_homologs_df['cds_context_name'] = cds_context_name
                    curr_homologs_df['longer_linked_repeat_cds_nuccore_accession'] = nuccore_accession
                    curr_homologs_df['longer_linked_repeat_cds_start_pos'] = longer_linked_repeat_cds_region[0]
                    curr_homologs_df['longer_linked_repeat_cds_end_pos'] = longer_linked_repeat_cds_region[1]
                    # print(curr_homologs_df)
                    homologs_dfs.append(curr_homologs_df)
                    # print('\n\ndone write_repeat_cds_homologs_df.\n')
                    # exit()


                ir_pair_region_with_margins = curr_raw_read_alignment_args['ir_pair_region_with_margins']
                nuccore_ir_pair_region_with_margins_output_dir_path = os.path.join(nuccore_entry_output_dir_path,
                                                                                   f'{ir_pair_region_with_margins[0]}_{ir_pair_region_with_margins[1]}')
                pathlib.Path(nuccore_ir_pair_region_with_margins_output_dir_path).mkdir(parents=True, exist_ok=True)
                num_of_filtered_cds_features_file_path = os.path.join(nuccore_ir_pair_region_with_margins_output_dir_path,
                                                                      f'num_of_filtered_cds_features.txt')
                filtered_extended_cds_df_csv_file_path = os.path.join(nuccore_ir_pair_region_with_margins_output_dir_path,
                                                                      f'{nuccore_accession}_filtered_extended_cds_df.csv')

                filtered_nuccore_gb_file_path = f'{nuccore_gb_file_path}.filtered.gb'
                seq_feature_utils.discard_joined_features_with_large_total_dist_between_joined_parts(
                    input_file_path_gb=nuccore_gb_file_path,
                    max_total_dist_between_joined_parts=search_for_pis_args['max_total_dist_between_joined_parts_per_joined_feature'],
                    discard_non_cds=True,
                    output_file_path_filtered_gb=filtered_nuccore_gb_file_path,
                    output_file_path_num_of_filtered_features=num_of_filtered_cds_features_file_path,
                )
                massive_screening_stage_1.write_nuccore_cds_df_csv(
                    nuccore_accession=nuccore_accession,
                    input_file_path_nuccore_cds_features_gb=filtered_nuccore_gb_file_path,
                    output_file_path_nuccore_cds_df_csv=cds_df_csv_file_path,
                )
                presumably_relevant_cds_region = curr_raw_read_alignment_args['presumably_relevant_cds_region']
                filter_cds_df_and_add_product_family(
                    input_file_path_cds_df_csv=cds_df_csv_file_path,
                    cds_region=presumably_relevant_cds_region,
                    list_of_product_and_product_family=search_for_pis_args['enrichment_analysis']['list_of_product_and_product_family'],
                    output_file_path_filtered_extended_cds_df_csv=filtered_extended_cds_df_csv_file_path,
                )

                # print(pd.read_csv(filtered_extended_cds_df_csv_file_path, sep='\t', low_memory=False))
                filtered_extended_cds_df = pd.read_csv(filtered_extended_cds_df_csv_file_path, sep='\t', low_memory=False)

                longer_linked_repeat_cds_strand = filtered_extended_cds_df[
                    (filtered_extended_cds_df['start_pos'] == longer_linked_repeat_cds_region[0]) &
                    (filtered_extended_cds_df['end_pos'] == longer_linked_repeat_cds_region[1])
                ]['strand']
                assert len(longer_linked_repeat_cds_strand) == 1
                longer_linked_repeat_cds_strand = longer_linked_repeat_cds_strand.iloc[0]
                assert abs(longer_linked_repeat_cds_strand) == 1
                longer_linked_repeat_cds_strand = int(longer_linked_repeat_cds_strand)


                assert len(curr_raw_read_alignment_args['sra_accession_to_variants_and_reads_info']) == 1
                variants_and_reads_info = next(iter(curr_raw_read_alignment_args['sra_accession_to_variants_and_reads_info'].values()))
                num_of_identified_variants_including_ref = 1 + (
                    len(get_variant_ir_pairs_to_evidence_read_names(variants_and_reads_info['non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref']))
                    if 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref' in variants_and_reads_info
                    else 0
                )
                putative_pi_locus_description_flat_dict = {
                    'nuccore_accession': nuccore_accession,
                    'longer_linked_repeat_cds_start_pos': longer_linked_repeat_cds_region[0],
                    'longer_linked_repeat_cds_end_pos': longer_linked_repeat_cds_region[1],
                    'longer_linked_repeat_cds_strand': longer_linked_repeat_cds_strand,
                    'longer_linked_repeat_cds_protein_id': longer_linked_repeat_cds_protein_id,
                    'locus_description_for_table_3': locus_description_for_table_3,
                    'presumably_relevant_cds_region_start': presumably_relevant_cds_region[0],
                    'presumably_relevant_cds_region_end': presumably_relevant_cds_region[1],
                    # 'genomic_context': ' ; '.join(filtered_extended_cds_df.sort_values('start_pos', ascending=(longer_linked_repeat_cds_strand == 1))['product']),
                    'target_gene_product_description': curr_raw_read_alignment_args['target_gene_product_description'],
                    'num_of_identified_variants_including_ref': num_of_identified_variants_including_ref,
                }
                if 'presumably_associated_upstream_gene_product_descriptions' in curr_raw_read_alignment_args:
                    putative_pi_locus_description_flat_dict['presumably_associated_upstream_gene_product_descriptions'] = curr_raw_read_alignment_args[
                        'presumably_associated_upstream_gene_product_descriptions']
                if 'presumably_associated_downstream_gene_product_descriptions' in curr_raw_read_alignment_args:
                    putative_pi_locus_description_flat_dict['presumably_associated_downstream_gene_product_descriptions'] = curr_raw_read_alignment_args[
                        'presumably_associated_downstream_gene_product_descriptions']

                putative_pi_locus_description_flat_dicts.append(putative_pi_locus_description_flat_dict)


                assembly_accession = nuccore_accession_to_assembly_accesion[nuccore_accession]

                assembly_entry_output_dir_path = os.path.join(assembly_entries_output_dir_path, assembly_accession)
                pathlib.Path(assembly_entry_output_dir_path).mkdir(parents=True, exist_ok=True)

                if search_for_pis_args['stage6']['DEBUG___assembly_fasta_file_path']:
                    assembly_fasta_file_path = search_for_pis_args['stage6']['DEBUG___assembly_fasta_file_path']
                else:
                    assembly_gb_file_path = os.path.join(assembly_entry_output_dir_path, f'{assembly_accession}.gb')
                    ncbi_genome_download_interface.download_assembly(
                        assembly_accession=assembly_accession,
                        output_file_path_gb=assembly_gb_file_path,
                        output_dir_path=assembly_entry_output_dir_path,
                    )
                    assembly_fasta_file_path = os.path.join(assembly_entry_output_dir_path, f'{assembly_accession}.fasta')
                    bio_utils.convert_gb_file_to_fasta_file(
                        input_file_path_gb=assembly_gb_file_path,
                        output_file_path_fasta=assembly_fasta_file_path,
                    )

                assembly_seq_names_pickle_file_path = os.path.join(assembly_entry_output_dir_path, f'{assembly_accession}_seq_names.pickle')
                bio_utils.write_names_of_fasta_file_seqs(
                    input_file_path_fasta=assembly_fasta_file_path,
                    output_file_path_seq_names_pickle=assembly_seq_names_pickle_file_path,
                )
                with open(assembly_seq_names_pickle_file_path, 'rb') as f:
                    assembly_seq_names = pickle.load(f)


                alignment_region = curr_raw_read_alignment_args['alignment_region']
                alignment_region_output_dir_path = os.path.join(nuccore_entry_output_dir_path, f'{alignment_region[0]}_{alignment_region[1]}')
                pathlib.Path(alignment_region_output_dir_path).mkdir(parents=True, exist_ok=True)


                assert (ir_pair_region_with_margins[0] - min_alignment_region_margin_size) >= alignment_region[0]
                assert (ir_pair_region_with_margins[1] + min_alignment_region_margin_size) <= alignment_region[1]

                ir_pair_region_str_repr = f'{nuccore_accession}_{ir_pair_region_with_margins[0]}_{ir_pair_region_with_margins[1]}'
                ir_pair_region_with_margins_fasta_file_path = os.path.join(alignment_region_output_dir_path, f'{ir_pair_region_str_repr}.fasta')
                bio_utils.write_region_to_fasta_file(
                    input_file_path_fasta=nuccore_fasta_file_path,
                    region=ir_pair_region_with_margins,
                    output_file_path_region_fasta=ir_pair_region_with_margins_fasta_file_path,
                )
                blast_interface_and_utils.make_blast_nucleotide_db(ir_pair_region_with_margins_fasta_file_path)

                seed_len = search_for_pis_args['stage6']['blast_nuccore_to_find_ir_pairs']['seed_len']
                # if (nuccore_accession, ir_pair_region_with_margins) in nuccore_accession_and_ir_pair_region_with_margins_to_special_min_repeat_len:
                #     min_repeat_len = nuccore_accession_and_ir_pair_region_with_margins_to_special_min_repeat_len[(nuccore_accession, ir_pair_region_with_margins)]
                # else:
                min_repeat_len = search_for_pis_args['stage6']['blast_nuccore_to_find_ir_pairs']['min_repeat_len']

                ir_pairs_df_csv_file_path = os.path.join(alignment_region_output_dir_path, f'{ir_pair_region_str_repr}_ir_pairs_df.csv')
                py_repeats_finder.find_imperfect_repeats_pairs(
                    input_file_path_fasta=ir_pair_region_with_margins_fasta_file_path,
                    seed_len=seed_len,
                    min_repeat_len=min_repeat_len,
                    max_spacer_len=np.inf,
                    min_spacer_len=1,
                    inverted_or_direct_or_both='inverted',
                    output_file_path_imperfect_repeats_pairs_csv=ir_pairs_df_csv_file_path,
                    max_evalue=search_for_pis_args['stage6']['blast_nuccore_to_find_ir_pairs']['max_evalue'],
                )

                for sra_accession, reads_info in sorted(
                        curr_raw_read_alignment_args['sra_accession_to_variants_and_reads_info'].items()):
                    if 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref' in reads_info:
                        possible_ir_pairs_used_to_reach_from_ref = set(itertools.chain.from_iterable(
                                reads_info['non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref'].values()))
                        ir_pair_with_evidence_outermost_edges = (
                            *(min(x[y] for x in possible_ir_pairs_used_to_reach_from_ref) for y in (0,1)),
                            *(max(x[y] for x in possible_ir_pairs_used_to_reach_from_ref) for y in (2,3)),
                        )
                    else:
                        ir_pairs_df = pd.read_csv(ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
                        ir_pairs_df[['left1', 'right1', 'left2', 'right2']] += ir_pair_region_with_margins[0] - 1
                        ir_pair_with_evidence_outermost_edges = (
                            ir_pairs_df['left1'].min(),
                            ir_pairs_df['right1'].min(),
                            ir_pairs_df['left2'].max(),
                            ir_pairs_df['right2'].max(),
                        )

                    assert (ir_pair_with_evidence_outermost_edges[2] - ir_pair_with_evidence_outermost_edges[1]) >= min_ir_pair_region_margin_size_for_evidence_read
                    ir_pair_with_evidence_outermost_edges = tuple(int(x) for x in ir_pair_with_evidence_outermost_edges)
                    # raise RuntimeError(str(ir_pair_with_evidence_outermost_edges))
                    # if sra_accession != 'SRR488166':
                    # if sra_accession != 'SRR5817722':
                    #     continue
                    generic_utils.print_and_write_to_log(f'starting to work on sra_accession {sra_accession}')

                    alignment_region_sra_output_dir_path = os.path.join(alignment_region_output_dir_path, sra_accession)
                    pathlib.Path(alignment_region_sra_output_dir_path).mkdir(parents=True, exist_ok=True)
                    if search_for_pis_args['stage6']['DEBUG___sra_entry_fasta_file_path']:
                        sra_entry_fasta_file_path = search_for_pis_args['stage6']['DEBUG___sra_entry_fasta_file_path']
                    else:
                        sra_entry_type = sra_accession_to_type_and_sra_file_name[sra_accession][0]
                        downloaded_sra_file_name = sra_accession_to_type_and_sra_file_name[sra_accession][1]
                        downloaded_sra_file_path = os.path.join(sra_entries_output_dir_path, downloaded_sra_file_name)
                        assert os.path.isfile(downloaded_sra_file_path)
                        assert sra_entry_type == 'long_reads'

                        sra_entry_fasta_file_path = f'{downloaded_sra_file_path}_pass.fasta'
                        bio_utils.extract_sra_file_to_fasta_with_predetermined_path(
                            input_file_path_sra=downloaded_sra_file_path,
                            output_file_path_fasta=sra_entry_fasta_file_path,
                        )

                    alignment_region_str_repr = f'{nuccore_accession}_{alignment_region[0]}_{alignment_region[1]}'
                    blast_result_out_file_path = os.path.join(alignment_region_sra_output_dir_path, f'{alignment_region_str_repr}_blast_result.csv')
                    filtered_blast_result_out_file_path = os.path.join(alignment_region_sra_output_dir_path, f'{alignment_region_str_repr}_filtered_blast_result.csv')

                    blast_alignment_region_to_reads(
                        input_file_path_nuccore_fasta=nuccore_fasta_file_path,
                        input_file_path_sra_entry_fasta=sra_entry_fasta_file_path,
                        min_num_of_read_bases_covered_by_any_alignment=min_num_of_read_bases_covered_by_any_alignment,
                        alignment_region=alignment_region,
                        max_evalue=alignment_region_to_long_reads_max_evalue,
                        seed_len=alignment_region_to_long_reads_seed_len,
                        output_file_path_blast_result_csv=blast_result_out_file_path,
                        output_file_path_filtered_blast_result_csv=filtered_blast_result_out_file_path,
                    )
                    # df = blast_interface_and_utils.read_blast_results_df(blast_result_out_file_path)
                    # # df = pd.read_csv(filtered_blast_result_out_file_path, sep='\t', low_memory=False)
                    # print("df[df['sseqid'].str.contains('SRR14725125.1.1143613')]")
                    # print(df[df['sseqid'].str.contains('SRR14725125.1.1143613')])

                    ref_variant_evidence_read_name_to_aligned_read_region_pickle_file_path = os.path.join(
                        alignment_region_sra_output_dir_path, 'ref_variant_evidence_read_name_to_aligned_read_region.pickle')
                    potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle_file_path = os.path.join(
                        alignment_region_sra_output_dir_path, 'potential_non_ref_variant_evidence_read_name_to_aligned_read_region.pickle')
                    extended_filtered_blast_result_out_file_path = os.path.join(
                        alignment_region_sra_output_dir_path, f'{alignment_region_str_repr}_extended_filtered_blast_result.csv')
                    process_alignments_to_alignment_region(
                        input_file_path_alignments_df_csv=filtered_blast_result_out_file_path,
                        ir_pair_outermost_edges=ir_pair_with_evidence_outermost_edges,
                        min_ir_pair_region_margin_size_for_evidence_read=min_ir_pair_region_margin_size_for_evidence_read,
                        output_file_path_extended_alignments_df_csv=extended_filtered_blast_result_out_file_path,
                        output_file_path_ref_variant_evidence_read_name_to_aligned_read_region_pickle=(
                            ref_variant_evidence_read_name_to_aligned_read_region_pickle_file_path),
                        output_file_path_potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle=(
                            potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle_file_path),
                    )
                    with open(ref_variant_evidence_read_name_to_aligned_read_region_pickle_file_path, 'rb') as f:
                        ref_variant_evidence_read_name_to_aligned_read_region = pickle.load(f)

                    with open(potential_non_ref_variant_evidence_read_name_to_aligned_read_region_pickle_file_path, 'rb') as f:
                        potential_non_ref_variant_evidence_read_name_to_aligned_read_region = pickle.load(f)
                    print(f'len(potential_non_ref_variant_evidence_read_name_to_aligned_read_region): '
                          f'{len(potential_non_ref_variant_evidence_read_name_to_aligned_read_region)}')

                    assert {'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref', 'not_non_ref_variant_read_name_to_anomaly_description',
                            'inaccurate_or_not_beautiful_mauve_alignment',
                            'complex_variant_ir_pairs_to_variant_regions_and_types'} >= set(
                        reads_info)

                    if 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref' in reads_info:
                        hardcoded_non_ref_variant_evidence_read_names = set(
                            reads_info['non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref'])
                        assert not (set(ref_variant_evidence_read_name_to_aligned_read_region) & hardcoded_non_ref_variant_evidence_read_names)

                        verify_ir_pairs_according_to_df(
                            input_file_path_ir_pairs_df_csv=ir_pairs_df_csv_file_path,
                            ir_pair_region_with_margins_start=ir_pair_region_with_margins[0],
                            ir_pairs=possible_ir_pairs_used_to_reach_from_ref,
                        )
                    else:
                        hardcoded_non_ref_variant_evidence_read_names = set()

                    print(f'len(hardcoded_non_ref_variant_evidence_read_names) before whole genome alignment filtering: {len(hardcoded_non_ref_variant_evidence_read_names)}')
                    print(f'len(ref_variant_evidence_read_names) before whole genome alignment filtering: {len(ref_variant_evidence_read_name_to_aligned_read_region)}')

                    potential_evidence_read_name_to_aligned_read_region = {
                        **ref_variant_evidence_read_name_to_aligned_read_region,
                        **potential_non_ref_variant_evidence_read_name_to_aligned_read_region,
                    }
                    # print(potential_evidence_read_name_to_aligned_read_region)
                    # exit()
                    final_blast_result_df = pd.read_csv(extended_filtered_blast_result_out_file_path, sep='\t', low_memory=False)
                    if curr_raw_read_alignment_args['align_to_whole_genome_to_verify_evidence_reads']:
                        potential_evidence_reads_truncated_fasta_file_path = os.path.join(alignment_region_sra_output_dir_path,
                                                                                          f'potential_evidence_reads_truncated.fasta')
                        # if sra_accession == 'SRR11812841':
                        #     print(potential_evidence_read_name_to_aligned_read_region)
                        #     exit()
                        bio_utils.filter_fasta_file(
                            input_file_path_fasta=sra_entry_fasta_file_path,
                            name_of_read_to_read_region_to_keep=potential_evidence_read_name_to_aligned_read_region,
                            output_file_path_fasta_with_only_specified_reads=potential_evidence_reads_truncated_fasta_file_path,
                        )

                        if nuccore_accession in assembly_seq_names:
                            alignment_region_seq_name_in_assembly = nuccore_accession
                        else:
                            if nuccore_accession in nuccore_accession_to_name_in_assembly:
                                alignment_region_seq_name_in_assembly = nuccore_accession_to_name_in_assembly[nuccore_accession]
                            elif f'NZ_{nuccore_accession}' in assembly_seq_names:
                                alignment_region_seq_name_in_assembly = f'NZ_{nuccore_accession}'
                            else:
                                print(assembly_seq_names)
                        assert alignment_region_seq_name_in_assembly in assembly_seq_names

                        names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle_file_path = os.path.join(
                            alignment_region_sra_output_dir_path, f'names_of_reads_without_a_better_alignment.pickle')
                        # print((ir_pair_with_evidence_outermost_edges[1:3]))
                        # print((ir_pair_with_evidence_outermost_edges))
                        # exit()
                        write_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome(
                            input_file_path_alignments_df=filtered_blast_result_out_file_path,
                            input_file_path_assembly_fasta=assembly_fasta_file_path,
                            input_file_path_reads_fasta=potential_evidence_reads_truncated_fasta_file_path,
                            alignment_region_seq_name=alignment_region_seq_name_in_assembly,
                            alignment_region=(ir_pair_with_evidence_outermost_edges[1:3]),
                            max_evalue=align_assembly_to_relevant_long_reads_max_evalue,
                            seed_len=align_assembly_to_relevant_long_reads_seed_len,
                            output_file_path_names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle=(
                                names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle_file_path),
                            out_dir_path=alignment_region_sra_output_dir_path,
                        )
                        print(f'alignment_region_sra_output_dir_path: {alignment_region_sra_output_dir_path}')
                        with open(names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome_pickle_file_path, 'rb') as f:
                            names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome = pickle.load(f)

                        ref_variant_evidence_read_names = (set(names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome) &
                                                           set(ref_variant_evidence_read_name_to_aligned_read_region))
                        non_ref_variant_evidence_read_names = (set(names_of_reads_without_a_better_alignment_to_somewhere_else_in_the_genome) &
                                                               set(potential_non_ref_variant_evidence_read_name_to_aligned_read_region))

                        unidentified_hardcoded_non_ref_variant_read_names = hardcoded_non_ref_variant_evidence_read_names - non_ref_variant_evidence_read_names
                        if unidentified_hardcoded_non_ref_variant_read_names:
                            raise RuntimeError(f'unidentified_hardcoded_non_ref_variant_read_names: {unidentified_hardcoded_non_ref_variant_read_names}')

                        hardcoded_not_non_ref_read_names = set(
                            reads_info.get('not_non_ref_variant_read_name_to_anomaly_description', set()))
                        hardcoded_as_both_not_non_ref_and_non_ref_read_names = hardcoded_not_non_ref_read_names & hardcoded_non_ref_variant_evidence_read_names
                        if hardcoded_as_both_not_non_ref_and_non_ref_read_names:
                            raise RuntimeError(f'hardcoded_as_both_not_non_ref_and_non_ref_read_names: {hardcoded_as_both_not_non_ref_and_non_ref_read_names}')

                        unidentified_hardcoded_not_non_ref_variant_read_names = hardcoded_not_non_ref_read_names - non_ref_variant_evidence_read_names
                        if unidentified_hardcoded_not_non_ref_variant_read_names:
                            raise RuntimeError(f'unidentified_hardcoded_not_non_ref_variant_read_names: {unidentified_hardcoded_not_non_ref_variant_read_names}')

                        potential_non_ref_but_not_hardcoded_read_names = (non_ref_variant_evidence_read_names -
                                                                          hardcoded_non_ref_variant_evidence_read_names - hardcoded_not_non_ref_read_names)
                        if potential_non_ref_but_not_hardcoded_read_names:
                            raise RuntimeError(f'potential_non_ref_but_not_hardcoded_read_names: {potential_non_ref_but_not_hardcoded_read_names}')

                        ref_variant_evidence_read_names = sorted(ref_variant_evidence_read_names)
                        non_ref_variant_evidence_read_names = sorted(non_ref_variant_evidence_read_names)

                        print(f'len(non_ref_variant_evidence_read_names) after whole genome alignment filtering: {len(non_ref_variant_evidence_read_names)}')
                        print(f'len(ref_variant_evidence_read_names) after whole genome alignment filtering: {len(ref_variant_evidence_read_names)}')


                        final_blast_result_df['read_evidence_type'] = np.nan
                        if ref_variant_evidence_read_names:
                            final_blast_result_df = final_blast_result_df.merge(pd.Series(ref_variant_evidence_read_names, name='sseqid'),
                                                                                how='left', indicator=True)
                            ref_variant_filter = final_blast_result_df['_merge'] == 'both'
                            assert final_blast_result_df[ref_variant_filter]['read_evidence_type'].isna().all()
                            final_blast_result_df.loc[ref_variant_filter, 'read_evidence_type'] = 'ref_variant'
                            final_blast_result_df.drop('_merge', axis=1, inplace=True)

                        if non_ref_variant_evidence_read_names:
                            final_blast_result_df = final_blast_result_df.merge(pd.Series(non_ref_variant_evidence_read_names, name='sseqid'),
                                                                                how='left', indicator=True)
                            non_ref_variant_filter = final_blast_result_df['_merge'] == 'both'
                            assert final_blast_result_df[non_ref_variant_filter]['read_evidence_type'].isna().all()
                            final_blast_result_df.loc[non_ref_variant_filter, 'read_evidence_type'] = 'non_ref_variant'
                            final_blast_result_df.drop('_merge', axis=1, inplace=True)

                        print(final_blast_result_df['read_evidence_type'].value_counts(dropna=False))

                        if 'inaccurate_or_not_beautiful_mauve_alignment' in reads_info:
                            hardcoded_inaccurate_or_not_beautiful_mauve_alignment_read_names = (
                                reads_info['inaccurate_or_not_beautiful_mauve_alignment'])
                        else:
                            hardcoded_inaccurate_or_not_beautiful_mauve_alignment_read_names = set()

                        final_blast_result_df = add_is_best_evidence_read_for_variant_column(
                            final_blast_result_df,
                            reads_info,
                            hardcoded_inaccurate_or_not_beautiful_mauve_alignment_read_names,
                        )

                        best_evidence_reads_output_dir_path = os.path.join(alignment_region_sra_output_dir_path, 'best_evidence_reads')
                        pathlib.Path(best_evidence_reads_output_dir_path).mkdir(parents=True, exist_ok=True)
                        best_evidence_read_name_to_info_pickle_file_path = os.path.join(best_evidence_reads_output_dir_path, 'best_evidence_read_name_to_info.pickle')

                        best_evidence_read_names = set(final_blast_result_df[final_blast_result_df['is_best_evidence_read_for_variant']]['sseqid'])
                        best_evidence_read_alignments_df = final_blast_result_df.merge(pd.Series(list(best_evidence_read_names), name='sseqid'))
                        best_evidence_read_alignments_df['alignment_to_subject_plus_strand'] = (best_evidence_read_alignments_df['sstart'] <
                                                                                                best_evidence_read_alignments_df['send'])
                        best_evidence_read_alignment_length_sums_df = (
                            best_evidence_read_alignments_df[best_evidence_read_alignments_df['alignment_to_subject_plus_strand']].groupby('sseqid')['length'].sum().reset_index(
                                name='sum_of_lengths_of_alignments_to_subject_plus_strand').merge(
                            best_evidence_read_alignments_df[~best_evidence_read_alignments_df['alignment_to_subject_plus_strand']].groupby('sseqid')['length'].sum().reset_index(
                                name='sum_of_lengths_of_alignments_to_subject_minus_strand'), how='outer')
                        )
                        best_evidence_read_alignment_length_sums_df['sum_of_lengths_of_alignments_to_subject_plus_strand'].fillna(0, inplace=True)
                        best_evidence_read_alignment_length_sums_df['sum_of_lengths_of_alignments_to_subject_minus_strand'].fillna(0, inplace=True)

                        # this is not guaranteed, but I assume this because the probability seems pretty high that it would be true for my few cases.
                        assert (best_evidence_read_alignment_length_sums_df['sum_of_lengths_of_alignments_to_subject_plus_strand'] !=
                                best_evidence_read_alignment_length_sums_df['sum_of_lengths_of_alignments_to_subject_minus_strand']).all()
                        # best_evidence_read_alignments

                        assert best_evidence_read_alignment_length_sums_df['sseqid'].is_unique
                        # if sra_accession == 'ERR1055234':
                        #     print("(best_evidence_read_alignment_length_sums_df['sseqid'] == 'ERR1055234.1.139400.1').any()")
                        #     print(best_evidence_read_alignment_length_sums_df[
                        #               best_evidence_read_alignment_length_sums_df['sseqid'] == 'ERR1055234.1.139400.1'
                        #           ][['sum_of_lengths_of_alignments_to_subject_plus_strand', 'sum_of_lengths_of_alignments_to_subject_minus_strand']])
                        #     exit()
                        best_evidence_read_name_to_align_to_plus_strand = {
                            row['sseqid']: (
                                (
                                    (row['sum_of_lengths_of_alignments_to_subject_plus_strand'] > row['sum_of_lengths_of_alignments_to_subject_minus_strand']) ^
                                    (longer_linked_repeat_cds_strand == -1)
                                )
                                if (row['sseqid'] != 'ERR1055234.1.139400.1')
                                else False
                            )
                            for _, row in best_evidence_read_alignment_length_sums_df.iterrows()
                        }

                        # print(f'best_evidence_reads_output_dir_path: {best_evidence_reads_output_dir_path}')
                        run_mauve_for_each_best_evidence_read(
                            input_file_path_ir_pair_region_with_margins_fasta=ir_pair_region_with_margins_fasta_file_path,
                            input_file_path_potential_evidence_reads_truncated_fasta=potential_evidence_reads_truncated_fasta_file_path,
                            best_evidence_read_name_to_align_to_plus_strand=best_evidence_read_name_to_align_to_plus_strand,
                            output_file_path_best_evidence_read_name_to_info_pickle=best_evidence_read_name_to_info_pickle_file_path,
                            best_evidence_reads_output_dir_path=best_evidence_reads_output_dir_path,
                        )
                        num_of_non_ref_variant_evidence_reads = len(non_ref_variant_evidence_read_names)
                        num_of_ref_variant_evidence_reads = len(ref_variant_evidence_read_names)
                    else:
                        best_evidence_read_name_to_info_pickle_file_path = None
                        # because nothing was validated - so use -1 to remind me that something is missing.
                        num_of_non_ref_variant_evidence_reads = -1
                        num_of_ref_variant_evidence_reads = -1

                    # DEBUG CODE:
                    assert num_of_non_ref_variant_evidence_reads > 0
                    assert num_of_ref_variant_evidence_reads > 0


                    # just for convenience:
                    final_blast_result_df['ir_pairs_df_csv_file_path'] = ir_pairs_df_csv_file_path
                    final_blast_result_df['ir_pair_region_with_margins_start'] = ir_pair_region_with_margins[0]
                    final_blast_result_df['ir_pair_region_with_margins_end'] = ir_pair_region_with_margins[1]
                    final_blast_result_df['longer_linked_repeat_cds_start'] = longer_linked_repeat_cds_region[0]
                    final_blast_result_df['longer_linked_repeat_cds_end'] = longer_linked_repeat_cds_region[1]
                    final_blast_result_df['sra_accession'] = sra_accession
                    final_blast_result_df['nuccore_accession'] = nuccore_accession
                    final_blast_result_df['nuccore_gb_file_path'] = nuccore_gb_file_path
                    final_blast_result_df['alignment_region_start'] = alignment_region[0]
                    final_blast_result_df['alignment_region_end'] = alignment_region[1]
                    final_blast_result_df['presumably_relevant_cds_region_start'] = presumably_relevant_cds_region[0]
                    final_blast_result_df['presumably_relevant_cds_region_end'] = presumably_relevant_cds_region[1]
                    final_blast_result_df['filtered_extended_cds_df_csv_file_path'] = filtered_extended_cds_df_csv_file_path
                    final_blast_result_df['num_of_ref_variant_evidence_reads'] = num_of_ref_variant_evidence_reads
                    final_blast_result_df['num_of_non_ref_variant_evidence_reads'] = num_of_non_ref_variant_evidence_reads
                    final_blast_result_df['best_evidence_read_name_to_info_pickle_file_path'] = best_evidence_read_name_to_info_pickle_file_path
                    final_blast_result_df['cds_context_name'] = cds_context_name
                    final_blast_result_df['longer_linked_repeat_cds_strand'] = longer_linked_repeat_cds_strand

                    final_blast_result_df['describe_in_the_paper'] = False
                    if 'describe_in_the_paper' in curr_raw_read_alignment_args:
                        final_blast_result_df['describe_in_the_paper'] = curr_raw_read_alignment_args['describe_in_the_paper']

                    raw_read_alignment_result_dfs.append(final_blast_result_df)


                rna_seq_sra_accessions = curr_raw_read_alignment_args.get('rna_seq_sra_accessions', set())
                if rna_seq_sra_accessions:
                    all_ir_pairs_df = pd.read_csv(ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
                    all_ir_pairs_df[['left1', 'right1', 'left2', 'right2']] += ir_pair_region_with_margins[0] - 1
                    ir_pairs_to_test = sorted(
                        all_ir_pairs_df[all_ir_pairs_df['evalue'] <
                                        rna_seq_analysis_inverted_repeat_max_evalue][['left1', 'right1', 'left2', 'right2']].itertuples(index=False, name=None))

                for rna_seq_sra_accession in sorted(rna_seq_sra_accessions):
                    bioproject_accession = sra_accession_to_bioproject_accession[rna_seq_sra_accession]
                    sra_entry_type = sra_accession_to_type_and_sra_file_name[rna_seq_sra_accession][0]
                    downloaded_sra_file_name = sra_accession_to_type_and_sra_file_name[rna_seq_sra_accession][1]
                    downloaded_sra_file_path = os.path.join(sra_entries_output_dir_path, downloaded_sra_file_name)
                    assert os.path.isfile(downloaded_sra_file_path)
                    assert sra_entry_type == 'paired_rna_seq'

                    sra_entry_fastq1_file_path = f'{downloaded_sra_file_path}_pass_1.fastq'
                    sra_entry_fastq2_file_path = f'{downloaded_sra_file_path}_pass_2.fastq'
                    # sra_entry_fastq_orphan_file_path = f'{downloaded_sra_file_path}_pass_orphan.fastq'

                    bio_utils.extract_paired_end_sra_file_to_fastq_with_predetermined_paths(
                        input_file_path_sra=downloaded_sra_file_path,
                        output_file_path_fastq1=sra_entry_fastq1_file_path,
                        output_file_path_fastq2=sra_entry_fastq2_file_path,
                        # output_file_path_fastq_orphan=sra_entry_fastq_orphan_file_path,
                    )

                    bowtie2_interface.bowtie2_build_index(nuccore_fasta_file_path)

                    all_alignment_dir_path = os.path.join(sra_entries_output_dir_path, rna_seq_sra_accession, 'rna_seq_alignments')
                    pathlib.Path(all_alignment_dir_path).mkdir(parents=True, exist_ok=True)
                    best_alignments_to_ref_df_csv_file_path = os.path.join(all_alignment_dir_path, f'best_alignments_to_{nuccore_accession}.csv')

                    diff_score_alignments_dfs = []
                    for ir_pair in ir_pairs_to_test:
                        print(f'ir_pair: {ir_pair}')
                        expected_invertible_region = (ir_pair[1] + 1, ir_pair[2] - 1)
                        non_ref_variant_desc = f'{nuccore_accession}_inv_{expected_invertible_region[0]}_{expected_invertible_region[1]}'
                        non_ref_variant_fasta_file_path = os.path.join(nuccore_entry_output_dir_path, f'{non_ref_variant_desc}.fasta')
                        best_alignments_to_non_ref_variant_df_csv_file_path = os.path.join(all_alignment_dir_path,
                                                                                           f'best_alignments_to_{non_ref_variant_desc}.csv')
                        write_fasta_file_with_inverted_region(
                            input_file_path_fasta=nuccore_fasta_file_path,
                            region_to_invert=expected_invertible_region,
                            new_seq_name=non_ref_variant_desc,
                            output_file_path_fasta=non_ref_variant_fasta_file_path,
                        )
                        bowtie2_interface.bowtie2_build_index(non_ref_variant_fasta_file_path)

                        for ref_seq_desc, ref_seq_fasta_file_path, best_alignments_df_csv_file_path in (
                                (nuccore_accession, nuccore_fasta_file_path, best_alignments_to_ref_df_csv_file_path),
                                (non_ref_variant_desc, non_ref_variant_fasta_file_path, best_alignments_to_non_ref_variant_df_csv_file_path),
                        ):
                            alignment_sam_file_path = os.path.join(all_alignment_dir_path, f'alignment_to_{ref_seq_desc}.sam')
                            alignment_err_file_path = os.path.join(all_alignment_dir_path, f'alignment_to_{ref_seq_desc}_stderr.txt')

                            # print(alignment_sam_file_path)
                            # print(alignment_err_file_path)
                            bowtie2_interface.bowtie2_align_paired_reads(
                                input_file_path_reference_fasta=ref_seq_fasta_file_path,
                                input_file_path_raw_reads_file1=sra_entry_fastq1_file_path,
                                input_file_path_raw_reads_file2=sra_entry_fastq2_file_path,
                                output_file_path_sam=alignment_sam_file_path,
                                output_file_path_stderr=alignment_err_file_path,
                                end_to_end_mode=True,
                                # max_insert_size=rna_seq_analysis_bowtie2_max_insert_size,
                                require_both_reads_to_be_aligned=True,
                                require_paired_alignments=True,
                                report_all_alignments=False,
                            )

                            write_best_alignment_pair_scores_df(
                                input_file_path_sam=alignment_sam_file_path,
                                output_file_path_best_alignments_df_csv=best_alignments_df_csv_file_path,
                            )

                        different_score_alignments_df_csv_file_path = os.path.join(
                            all_alignment_dir_path, f'different_score_alignments_df_ref_and_inv_{expected_invertible_region[0]}_{expected_invertible_region[1]}.csv')
                        write_different_score_alignments_df(
                            input_file_path_best_alignments_to_ref_df_csv=best_alignments_to_ref_df_csv_file_path,
                            input_file_path_best_alignments_to_non_ref_df_csv=best_alignments_to_non_ref_variant_df_csv_file_path,
                            output_file_path_different_score_alignments_df_csv=different_score_alignments_df_csv_file_path,
                        )

                        diff_score_alignments_df = pd.read_csv(different_score_alignments_df_csv_file_path, sep='\t', low_memory=False)

                        rna_seq_min_max_score = -np.inf

                        diff_score_alignments_df['max_score'] = diff_score_alignments_df[['pair_total_align_score_ref', 'pair_total_align_score_non_ref']].max(axis=1)
                        diff_score_alignments_df = diff_score_alignments_df[diff_score_alignments_df['max_score'] >= rna_seq_min_max_score].copy()
                        diff_score_alignments_df['big_abs_score_diff'] = diff_score_alignments_df['score_diff'].abs() >= rna_seq_analysis_min_abs_score_diff
                        assert (diff_score_alignments_df['score_diff'] != 0).all()
                        diff_score_alignments_df['is_read_matching_ref_better'] = diff_score_alignments_df['score_diff'] > 0
                        diff_score_alignments_df['in_silico_inverted_region'] = f'{expected_invertible_region[0]}-{expected_invertible_region[1]}'
                        diff_score_alignments_df['sra_accession'] = rna_seq_sra_accession
                        diff_score_alignments_df['bioproject_accession'] = bioproject_accession
                        diff_score_alignments_df['different_score_alignments_df_csv_file_path'] = different_score_alignments_df_csv_file_path

                        read_name_prefix_len = len(downloaded_sra_file_name) + 1
                        assert ((diff_score_alignments_df['QNAME'].str.rfind('.') + 1) == read_name_prefix_len).all()
                        diff_score_alignments_df['short_read_name'] = diff_score_alignments_df['QNAME'].str.slice(start=read_name_prefix_len)
                        diff_score_alignments_df.rename(columns={'QNAME': 'read_name'}, inplace=True)
                        diff_score_alignments_dfs.append(diff_score_alignments_df)

                    all_diff_score_alignments_df = pd.concat(diff_score_alignments_dfs, ignore_index=True)

                    # print(all_diff_score_alignments_df)
                    all_diff_score_alignments_df['cds_context_name'] = cds_context_name
                    all_diff_score_alignments_df['longer_linked_repeat_cds_nuccore_accession'] = nuccore_accession
                    all_diff_score_alignments_df['longer_linked_repeat_cds_start_pos'] = longer_linked_repeat_cds_region[0]
                    all_diff_score_alignments_df['longer_linked_repeat_cds_end_pos'] = longer_linked_repeat_cds_region[1]
                    all_all_diff_score_alignments_dfs.append(all_diff_score_alignments_df)

                    rna_seq_summary_flat_dicts.extend([
                        {
                            'cds_context_name': cds_context_name,
                            'longer_linked_repeat_cds_nuccore_accession': nuccore_accession,
                            'longer_linked_repeat_cds_start_pos': longer_linked_repeat_cds_region[0],
                            'longer_linked_repeat_cds_end_pos': longer_linked_repeat_cds_region[1],
                            'sra_accession': rna_seq_sra_accession,
                            'bioproject_accession': bioproject_accession,
                            'in_silico_inverted_region': in_silico_inverted_region,
                            'num_of_reads_matching_ref_better': group_df[group_df['is_read_matching_ref_better'] & group_df['big_abs_score_diff']]['read_name'].nunique(),
                            'num_of_reads_matching_non_ref_better': group_df[(~group_df['is_read_matching_ref_better']) & group_df['big_abs_score_diff']]['read_name'].nunique(),
                        }
                        for in_silico_inverted_region, group_df in all_diff_score_alignments_df.groupby('in_silico_inverted_region')
                    ])



    all_homologs_df_csv_file_path = os.path.join(massive_screening_stage6_out_dir_path, 'all_homologs_df.csv')
    if homologs_dfs:
        all_homologs_df = pd.concat(homologs_dfs, ignore_index=True)

        print("all_homologs_df['min_evalue_of_alignment_to_inverted_repeat_in_margin'].isna().value_counts()")
        print(all_homologs_df['min_evalue_of_alignment_to_inverted_repeat_in_margin'].isna().value_counts())
        all_homologs_df = all_homologs_df[~(all_homologs_df['min_evalue_of_alignment_to_inverted_repeat_in_margin'].isna())]

        max_evalue_to_consider_as_potential_programmed_inversion = search_for_pis_args[
            'stage6']['blast_longer_linked_repeat_cds_homolog_to_its_margins']['max_evalue_to_consider_as_potential_programmed_inversion']
        all_homologs_df['is_homolog_potentially_modified_by_pi'] = (
            all_homologs_df['min_evalue_of_alignment_to_inverted_repeat_in_margin'] <=
            max_evalue_to_consider_as_potential_programmed_inversion
        )
        all_homologs_df.to_csv(all_homologs_df_csv_file_path, sep='\t', index=False)

        print(all_homologs_df[['longer_linked_repeat_cds_nuccore_accession', 'longer_linked_repeat_cds_start_pos',
                               'is_homolog_potentially_modified_by_pi']].value_counts())

    else:
        generic_utils.write_empty_file(all_homologs_df_csv_file_path)

    all_all_repeat_cds_covered_bases_proportions_pickle_file_path = os.path.join(massive_screening_stage6_out_dir_path,
                                                                                 'all_all_repeat_cds_covered_bases_proportions.pickle')
    with open(all_all_repeat_cds_covered_bases_proportions_pickle_file_path, 'wb') as f:
        pickle.dump(all_all_repeat_cds_covered_bases_proportions, f, protocol=4)

    all_all_alignment_bases_covered_by_cds_proportions_pickle_file_path = os.path.join(massive_screening_stage6_out_dir_path,
                                                                                 'all_all_alignment_bases_covered_by_cds_proportions.pickle')
    with open(all_all_alignment_bases_covered_by_cds_proportions_pickle_file_path, 'wb') as f:
        pickle.dump(all_all_alignment_bases_covered_by_cds_proportions, f, protocol=4)

    putative_pi_locus_descriptions_df = pd.DataFrame(putative_pi_locus_description_flat_dicts)
    putative_pi_locus_descriptions_df_csv_file_path = os.path.join(massive_screening_stage6_out_dir_path, 'putative_pi_locus_descriptions_df.csv')
    putative_pi_locus_descriptions_df.to_csv(putative_pi_locus_descriptions_df_csv_file_path, sep='\t', index=False)


    all_raw_read_alignment_results_df = pd.concat(raw_read_alignment_result_dfs, ignore_index=True)
    # all_raw_read_alignment_results_df.sort_values(['product', 'nuccore_accession', 'left1', 'right1', 'left2', 'right2'], inplace=True)
    # print(all_raw_read_alignment_results_df)
    all_raw_read_alignment_results_df_csv_file_path = os.path.join(massive_screening_stage6_out_dir_path, 'all_raw_read_alignment_results_df.csv')
    all_raw_read_alignment_results_df.to_csv(all_raw_read_alignment_results_df_csv_file_path, sep='\t', index=False)

    if all_all_diff_score_alignments_dfs:
        all_all_diff_score_alignments_df = pd.concat(all_all_diff_score_alignments_dfs, ignore_index=True)
        all_all_diff_score_alignments_df_csv_file_path = os.path.join(massive_screening_stage6_out_dir_path, 'all_all_diff_score_alignments_df.csv')
        all_all_diff_score_alignments_df.to_csv(all_all_diff_score_alignments_df_csv_file_path, sep='\t', index=False)
    else:
        all_all_diff_score_alignments_df_csv_file_path = None

    if rna_seq_summary_flat_dicts:
        rna_seq_summary_df = pd.DataFrame(rna_seq_summary_flat_dicts)
        rna_seq_summary_df_csv_file_path = os.path.join(massive_screening_stage6_out_dir_path, 'rna_seq_summary_df.csv')
        print(f'rna_seq_summary_df:\n{rna_seq_summary_df}')
        rna_seq_summary_df.to_csv(rna_seq_summary_df_csv_file_path, sep='\t', index=False)
        print('CDS contexts with positive rna seq evidence:')
        print(
            rna_seq_summary_df[
                (rna_seq_summary_df['num_of_reads_matching_ref_better'] > 0) &
                (rna_seq_summary_df['num_of_reads_matching_non_ref_better'] > 0)
            ]['cds_context_name'].drop_duplicates()
        )
    else:
        rna_seq_summary_df_csv_file_path = None

    stage6_results_info = {
        'all_raw_read_alignment_results_df_csv_file_path': all_raw_read_alignment_results_df_csv_file_path,
        'all_homologs_df_csv_file_path': all_homologs_df_csv_file_path,
        'all_all_repeat_cds_covered_bases_proportions_pickle_file_path': all_all_repeat_cds_covered_bases_proportions_pickle_file_path,
        'all_all_alignment_bases_covered_by_cds_proportions_pickle_file_path': all_all_alignment_bases_covered_by_cds_proportions_pickle_file_path,
        'putative_pi_locus_descriptions_df_csv_file_path': putative_pi_locus_descriptions_df_csv_file_path,
        'all_all_diff_score_alignments_df_csv_file_path': all_all_diff_score_alignments_df_csv_file_path,
        'rna_seq_summary_df_csv_file_path': rna_seq_summary_df_csv_file_path,
    }

    with open(stage6_results_info_pickle_file_path, 'wb') as f:
        pickle.dump(stage6_results_info, f, protocol=4)

    return stage6_results_info

def main():
    with generic_utils.timing_context_manager('massive_screening_stage_6.py'):
        if DO_STAGE6:
            do_massive_screening_stage6(
                search_for_pis_args=massive_screening_configuration.SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT,
            )
            print('done stage6')

        print('\n')

if __name__ == '__main__':
    main()
