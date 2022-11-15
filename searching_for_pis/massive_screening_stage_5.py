import logging
import os
import os.path
import pathlib
import pickle

import numpy as np
import pandas as pd

from generic import bio_utils
from generic import blast_interface_and_utils
from generic import generic_utils
from searching_for_pis import index_column_names
from searching_for_pis import massive_screening_stage_1
from searching_for_pis import massive_screening_stage_3
from searching_for_pis import massive_screening_configuration
from searching_for_pis import other_nuccore_entries_evidence_for_pi

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

DO_STAGE5 = True
# DO_STAGE5 = False


@generic_utils.execute_if_output_doesnt_exist_already
def cached_keep_only_cds_of_nuccores_with_any_ir_pairs_etc(
        input_file_path_cds_df_csv,
        input_file_path_pairs_df_csv,
        output_file_path_filtered_cds_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_df = pd.read_csv(input_file_path_cds_df_csv, sep='\t', low_memory=False)
    pairs_df = pd.read_csv(input_file_path_pairs_df_csv, sep='\t', low_memory=False)
    relevant_nuccore_accessions_series = pairs_df['nuccore_accession'].drop_duplicates()
    relevant_cds_df = cds_df.merge(relevant_nuccore_accessions_series)

    relevant_cds_df.to_csv(output_file_path_filtered_cds_df_csv, sep='\t', index=False)

def keep_only_cds_of_nuccores_with_any_ir_pairs_etc(
        input_file_path_cds_df_csv,
        input_file_path_pairs_df_csv,
        output_file_path_filtered_cds_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_keep_only_cds_of_nuccores_with_any_ir_pairs_etc(
        input_file_path_cds_df_csv=input_file_path_cds_df_csv,
        input_file_path_pairs_df_csv=input_file_path_pairs_df_csv,
        output_file_path_filtered_cds_df_csv=output_file_path_filtered_cds_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_accession_to_minimal_nuccore_entry_info_for_each_relevant_taxon(
        input_file_path_relevant_taxon_uids_pickle,
        input_file_path_nuccore_df_csv,
        output_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle,
        taxa_output_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with generic_utils.timing_context_manager('cached_write_nuccore_accession_to_minimal_nuccore_entry_info_for_each_relevant_taxon'):
        with open(input_file_path_relevant_taxon_uids_pickle, 'rb') as f:
            relevant_taxon_uids = pickle.load(f)

        assert relevant_taxon_uids
        minimal_nuccore_df = pd.read_csv(input_file_path_nuccore_df_csv, sep='\t', low_memory=False)[['taxon_uid', 'nuccore_accession', 'chrom_len',
                                                                                                      'fasta_file_path']].merge(pd.Series(relevant_taxon_uids,
                                                                                                                                          name='taxon_uid'))
        taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path = {}
        for taxon_uid, taxon_minimal_nuccore_df in minimal_nuccore_df.groupby('taxon_uid'):
            nuccore_accession_to_minimal_nuccore_entry_info = {
                row['nuccore_accession']: {
                    'fasta_file_path': row['fasta_file_path'],
                    'chrom_len': row['chrom_len'],
                }
                for _, row in taxon_minimal_nuccore_df.iterrows()
            }
            taxon_output_dir_path = os.path.join(taxa_output_dir_path, str(taxon_uid))
            pathlib.Path(taxon_output_dir_path).mkdir(parents=True, exist_ok=True)
            nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path = os.path.join(taxon_output_dir_path, 'nuccore_accession_to_minimal_nuccore_entry_info.pickle')
            with open(nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path, 'wb') as f:
                pickle.dump(nuccore_accession_to_minimal_nuccore_entry_info, f, protocol=4)
            taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path[taxon_uid] = nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path

        with open(output_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle, 'wb') as f:
            pickle.dump(taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path, f, protocol=4)

def write_nuccore_accession_to_minimal_nuccore_entry_info_for_each_relevant_taxon(
        input_file_path_relevant_taxon_uids_pickle,
        input_file_path_nuccore_df_csv,
        output_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle,
        taxa_output_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_accession_to_minimal_nuccore_entry_info_for_each_relevant_taxon(
        input_file_path_relevant_taxon_uids_pickle=input_file_path_relevant_taxon_uids_pickle,
        input_file_path_nuccore_df_csv=input_file_path_nuccore_df_csv,
        output_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle=(
            output_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle),
        taxa_output_dir_path=taxa_output_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_download_taxon_wgs_nuccore_entries_and_make_blast_db(
        input_file_path_taxon_wgs_nuccore_entries_info_pickle,
        output_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle,
        downloaded_wgs_nuccore_entries_dir_path,
        taxon_out_dir_path,
):
    with open(input_file_path_taxon_wgs_nuccore_entries_info_pickle, 'rb') as f:
        taxon_wgs_nuccore_entries_info = pickle.load(f)

    taxon_wgs_nuccore_accession_to_nuccore_entry_len = taxon_wgs_nuccore_entries_info['nuccore_accession_to_nuccore_entry_len']
    if taxon_wgs_nuccore_accession_to_nuccore_entry_len is None:
        taxon_wgs_nuccore_accessions = []
    else:
        taxon_wgs_nuccore_accessions = sorted(taxon_wgs_nuccore_accession_to_nuccore_entry_len)

    wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path = {}
    num_of_taxon_wgs_nuccore_accessions = len(taxon_wgs_nuccore_accessions)
    for i, wgs_nuccore_accession in enumerate(taxon_wgs_nuccore_accessions):
        generic_utils.print_and_write_to_log(f'(cached_download_taxon_wgs_nuccore_entries_and_make_blast_db) starting work on wgs nuccore '
                                             f'{i + 1}/{num_of_taxon_wgs_nuccore_accessions} ({wgs_nuccore_accession}).')

        wgs_nuccore_accession_dir_path = os.path.join(downloaded_wgs_nuccore_entries_dir_path, wgs_nuccore_accession)
        wgs_nuccore_entry_fasta_file_path = os.path.join(wgs_nuccore_accession_dir_path, f'{wgs_nuccore_accession}.fasta')

        expected_wgs_nuccore_len = taxon_wgs_nuccore_accession_to_nuccore_entry_len[wgs_nuccore_accession]
        should_download_wgs_nuccore = (not os.path.isfile(wgs_nuccore_entry_fasta_file_path)) or (
                bio_utils.get_chrom_len_from_single_chrom_fasta_file(wgs_nuccore_entry_fasta_file_path) != expected_wgs_nuccore_len)

        if should_download_wgs_nuccore:
            pathlib.Path(wgs_nuccore_accession_dir_path).mkdir(parents=True, exist_ok=True)
            bio_utils.download_nuccore_entry_from_ncbi(
                nuccore_accession=wgs_nuccore_accession,
                output_file_type='fasta',
                output_file_path_nuccore_entry=wgs_nuccore_entry_fasta_file_path,
            )

            assert bio_utils.get_chrom_len_from_single_chrom_fasta_file(wgs_nuccore_entry_fasta_file_path) == expected_wgs_nuccore_len

        assert wgs_nuccore_accession not in wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path
        wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path[wgs_nuccore_accession] = wgs_nuccore_entry_fasta_file_path

    if wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path:
        taxon_wgs_blast_db_path = os.path.join(taxon_out_dir_path, 'taxon_wgs_blast_db')
        blast_interface_and_utils.make_blast_nucleotide_db_for_multiple_fasta_files(sorted(wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path.values()),
                                                                                    taxon_wgs_blast_db_path)
    else:
        print(f'no wgs nuccore entries (that passed the thresholds) for curr taxon.')
        taxon_wgs_blast_db_path = None

    downloaded_taxon_wgs_nuccore_entries_info = {
        'taxon_wgs_blast_db_path': taxon_wgs_blast_db_path,
        'wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path': wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path,
    }

    with open(output_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle, 'wb') as f:
            pickle.dump(downloaded_taxon_wgs_nuccore_entries_info, f, protocol=4)

def download_taxon_wgs_nuccore_entries_and_make_blast_db(
        input_file_path_taxon_wgs_nuccore_entries_info_pickle,
        output_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle,
        downloaded_wgs_nuccore_entries_dir_path,
        taxon_out_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_download_taxon_wgs_nuccore_entries_and_make_blast_db(
        input_file_path_taxon_wgs_nuccore_entries_info_pickle=input_file_path_taxon_wgs_nuccore_entries_info_pickle,
        output_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle=output_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle,
        downloaded_wgs_nuccore_entries_dir_path=downloaded_wgs_nuccore_entries_dir_path,
        taxon_out_dir_path=taxon_out_dir_path,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_download_wgs_and_find_other_nuccore_entries_evidence(
        input_file_path_pairs_linked_df_csv,
        input_file_path_nuccore_df_csv,
        input_file_path_taxa_potential_evidence_for_pi_info_pickle,
        input_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle,
        input_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path,
        local_blast_nt_database_update_log_for_caching_only,
        min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion,
        output_file_path_taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle,
        output_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle,
        output_file_path_merged_cds_pair_region_df_csv,
        output_file_path_region_in_other_nuccore_df_csv,
        output_file_path_potential_breakpoint_df_csv,
        taxa_output_dir_path,
        downloaded_wgs_nuccore_entries_dir_path,
        nuccore_entries_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        merged_cds_pair_region_margin_size,
        blast_margins_and_identify_regions_in_other_nuccores_args,
        local_blast_nt_database_path,
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
        input_file_path_debug_local_blast_database_fasta,
        debug_other_nuccore_accession_to_fasta_file_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    pairs_linked_df = massive_screening_stage_3.get_pairs_linked_df_with_taxon_uid(
        pairs_linked_df_csv_file_path=input_file_path_pairs_linked_df_csv,
        nuccore_df_csv_file_path=input_file_path_nuccore_df_csv,
    )

    with open(input_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle, 'rb') as f:
        taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path = pickle.load(f)

    with open(input_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path, 'rb') as f:
        species_taxon_uid_to_more_taxon_uids_of_the_same_species = pickle.load(f)

    num_of_taxa = pairs_linked_df['taxon_uid'].nunique()

    with open(input_file_path_taxa_potential_evidence_for_pi_info_pickle, 'rb') as f:
        taxa_potential_evidence_for_pi_info = pickle.load(f)

    taxon_uid_to_taxon_potential_evidence_for_pi_info = taxa_potential_evidence_for_pi_info['taxon_uid_to_taxon_potential_evidence_for_pi_info']

    taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path = {}
    taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info = {}
    nuccore_region_in_other_nuccore_dfs = []
    nuccore_potential_breakpoint_dfs = []
    nuccore_merged_ir_pair_dfs = []
    for i, (taxon_uid, taxon_pairs_df) in enumerate(pairs_linked_df.groupby('taxon_uid', sort=False)):
        # START = 0e3
        # if i >= START + 2e3:
        #     exit()
        # if i < START:
        #     continue

        # if taxon_uid not in (195, 210, 213, 256, 258, 271, 287, 296, 300, 305, 316, 375, 470, 471, 476, 486, 562, 585, 648, 715, 731, 733, 817, 818, 821, 823, 1061, 1254, 1264, 1307, 1313, 1328, 1337, 1351, 1390, 1396, 1399, 1582, 1588, 1590, 1602, 1622, 1680, 1717, 1736, 1750, 1768, 1790, 1888, 2115, 13373, 28095, 28111, 28116, 28129, 28137, 28139, 28251, 28450, 28454, 28901, 28903, 29378, 29448, 29459, 29488, 29497, 33050, 33905, 34085, 36746, 37326, 40216, 42235, 43767, 43768, 43992, 44250, 46228, 46503, 46506, 47678, 47850, 48296, 52771, 53344, 53417, 60519, 65700, 68213, 68280, 68892, 75985, 76758, 76862, 84112, 85698, 85831, 87883, 95486, 120213, 120577, 123899, 136468, 161895, 162426, 166486, 172042, 185949, 194702, 198618, 198620, 204039, 204516, 208479, 208962, 212663, 214856, 223967, 227945, 239498, 239935, 246787, 257708, 270374, 285676, 291112, 292800, 305977, 310298, 310514, 321661, 328812, 329854, 333367, 338188, 357276, 380021, 387661, 460384, 469591, 480418, 501571, 515393, 544645, 571256, 574930, 626929, 658457, 671267, 674529, 683124, 744515, 930166, 1064539, 1082704, 1089444, 1124743, 1124835, 1148157, 1176259, 1249999, 1288410, 1355477, 1511761, 1530123, 1552123, 1608996, 1618207, 1785128, 1841857, 1843235, 2009329, 2044587, 2109915, 2219225, 2496117, 2508296, 2529843, 2650158, 2681861, 2697515, 2743473, 2759943, 2777781):
        # if taxon_uid != 77608:
        # if taxon_uid not in {2606628, 478807, 2315694, 195, 210, 213, 256, 258, 271, 287, 296, 300, 305, 316, 375, 470, 471, 476, 486, 562, 585, 648, 715}:
        #     continue

        generic_utils.print_and_write_to_log(f'(cached_download_wgs_and_find_other_nuccore_entries_evidence) '
                                             f'starting work on taxon {i + 1}/{num_of_taxa} ({taxon_uid}).')
        taxon_wgs_nuccore_entries_info_pickle_file_path = taxon_uid_to_taxon_potential_evidence_for_pi_info[taxon_uid]['taxon_wgs_nuccore_entries_info_pickle_file_path']
        taxon_out_dir_path = os.path.join(taxa_output_dir_path, str(taxon_uid))
        pathlib.Path(taxon_out_dir_path).mkdir(parents=True, exist_ok=True)
        downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path = os.path.join(taxon_out_dir_path, 'downloaded_taxon_wgs_nuccore_entries_info.pickle')
        print('taxon_out_dir_path')
        print(taxon_out_dir_path)
        download_taxon_wgs_nuccore_entries_and_make_blast_db(
            input_file_path_taxon_wgs_nuccore_entries_info_pickle=taxon_wgs_nuccore_entries_info_pickle_file_path,
            output_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle=downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path,
            downloaded_wgs_nuccore_entries_dir_path=downloaded_wgs_nuccore_entries_dir_path,
            taxon_out_dir_path=taxon_out_dir_path,
        )
        # continue
        taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path[taxon_uid] = downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path
        # with open(downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path, 'rb') as f:
        #     downloaded_taxon_wgs_nuccore_entries_info = pickle.load(f)

        nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path = taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path[taxon_uid]
        with open(nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path, 'rb') as f:
            nuccore_accession_to_minimal_nuccore_entry_info = pickle.load(f)

        taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path = (
            taxon_uid_to_taxon_potential_evidence_for_pi_info[taxon_uid]['taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path'])
        more_taxon_uids_of_the_same_species = species_taxon_uid_to_more_taxon_uids_of_the_same_species[taxon_uid]

        nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info = {}
        num_of_nuccore_entries = taxon_pairs_df['nuccore_accession'].nunique()
        for j, (nuccore_accession, nuccore_pairs_df) in enumerate(taxon_pairs_df.groupby('nuccore_accession', sort=False)):
            generic_utils.print_and_write_to_log(f'starting work on nuccore {j + 1}/{num_of_nuccore_entries} ({nuccore_accession}).')

            minimal_nuccore_entry_info = nuccore_accession_to_minimal_nuccore_entry_info[nuccore_accession]

            other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info, result_df_dict = (
                other_nuccore_entries_evidence_for_pi.find_other_nuccore_entries_evidence_for_pis_in_nuccore_entry(
                    taxon_uid=taxon_uid,
                    nuccore_accession=nuccore_accession,
                    nuccore_pairs_df=nuccore_pairs_df,
                    minimal_nuccore_entry_info=minimal_nuccore_entry_info,
                    taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path=taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path,
                    more_taxon_uids_of_the_same_species_for_blasting_local_nt=more_taxon_uids_of_the_same_species,
                    downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path=downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path,
                    nuccore_entries_output_dir_path=nuccore_entries_output_dir_path,
                    other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path=other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
                    local_blast_nt_database_update_log_for_caching_only=local_blast_nt_database_update_log_for_caching_only,
                    min_mauve_total_match_proportion=min_mauve_total_match_proportion,
                    min_min_sub_alignment_min_match_proportion=min_min_sub_alignment_min_match_proportion,
                    merged_cds_pair_region_margin_size=merged_cds_pair_region_margin_size,
                    blast_margins_and_identify_regions_in_other_nuccores_args=blast_margins_and_identify_regions_in_other_nuccores_args,
                    local_blast_nt_database_path=local_blast_nt_database_path,
                    max_num_of_non_identical_regions_in_other_nuccores_to_analyze=max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
                    input_file_path_debug_local_blast_database_fasta=input_file_path_debug_local_blast_database_fasta,
                    debug_other_nuccore_accession_to_fasta_file_path=debug_other_nuccore_accession_to_fasta_file_path,
                )
            )
            if result_df_dict['merged_cds_pair_region_df'] is not None:
                nuccore_merged_ir_pair_dfs.append(result_df_dict['merged_cds_pair_region_df'])
            if result_df_dict['nuccore_region_in_other_nuccore_df'] is not None:
                nuccore_region_in_other_nuccore_dfs.append(result_df_dict['nuccore_region_in_other_nuccore_df'])
            if result_df_dict['nuccore_potential_breakpoint_df'] is not None:
                nuccore_potential_breakpoint_dfs.append(result_df_dict['nuccore_potential_breakpoint_df'])

            nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info[nuccore_accession] = (
                other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info)

        taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info[taxon_uid] = (
            nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info)


    with open(output_file_path_taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle, 'wb') as f:
        pickle.dump(taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path, f, protocol=4)

    with open(output_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle, 'wb') as f:
        pickle.dump(taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info, f, protocol=4)

    generic_utils.print_and_write_to_log('starting to build merged_cds_pair_region_df')
    if nuccore_merged_ir_pair_dfs:
        merged_cds_pair_region_df = pd.concat(nuccore_merged_ir_pair_dfs, ignore_index=True)
        merged_cds_pair_region_df.to_csv(output_file_path_merged_cds_pair_region_df_csv, sep='\t', index=False)
        print(f'len(merged_cds_pair_region_df): {len(merged_cds_pair_region_df)}')
        del merged_cds_pair_region_df # because it might be pretty big.
    else:
        print(f'len(merged_cds_pair_region_df): 0')
        generic_utils.write_empty_file(output_file_path_merged_cds_pair_region_df_csv)

    generic_utils.print_and_write_to_log('starting to build region_in_other_nuccore_df')
    if nuccore_region_in_other_nuccore_dfs:
        region_in_other_nuccore_df = pd.concat(nuccore_region_in_other_nuccore_dfs, ignore_index=True)
        region_in_other_nuccore_df.to_csv(output_file_path_region_in_other_nuccore_df_csv, sep='\t', index=False)
        print(f'len(region_in_other_nuccore_df): {len(region_in_other_nuccore_df)}')
        del region_in_other_nuccore_df # because it might be pretty big.
    else:
        print(f'len(region_in_other_nuccore_df): 0')
        generic_utils.write_empty_file(output_file_path_region_in_other_nuccore_df_csv)

    generic_utils.print_and_write_to_log('starting to build potential_breakpoint_df')
    if nuccore_potential_breakpoint_dfs:
        potential_breakpoint_df = pd.concat(nuccore_potential_breakpoint_dfs, ignore_index=True)
        potential_breakpoint_df = potential_breakpoint_df.reset_index().rename(columns={'index': 'index_in_potential_breakpoint_df_csv_file'})
        potential_breakpoint_df.to_csv(output_file_path_potential_breakpoint_df_csv, sep='\t', index=False)
        print(f'len(potential_breakpoint_df): {len(potential_breakpoint_df)}')
    else:
        print(f'len(potential_breakpoint_df): 0')
        generic_utils.write_empty_file(output_file_path_potential_breakpoint_df_csv)

def download_wgs_and_find_other_nuccore_entries_evidence(
        input_file_path_pairs_linked_df_csv,
        input_file_path_nuccore_df_csv,
        input_file_path_taxa_potential_evidence_for_pi_info_pickle,
        input_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle,
        input_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path,
        local_blast_nt_database_update_log_for_caching_only,
        min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion,
        output_file_path_taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle,
        output_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle,
        output_file_path_merged_cds_pair_region_df_csv,
        output_file_path_region_in_other_nuccore_df_csv,
        output_file_path_potential_breakpoint_df_csv,
        taxa_output_dir_path,
        downloaded_wgs_nuccore_entries_dir_path,
        nuccore_entries_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        merged_cds_pair_region_margin_size,
        blast_margins_and_identify_regions_in_other_nuccores_args,
        local_blast_nt_database_path,
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
        input_file_path_debug_local_blast_database_fasta,
        debug_other_nuccore_accession_to_fasta_file_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_download_wgs_and_find_other_nuccore_entries_evidence(
        input_file_path_pairs_linked_df_csv=input_file_path_pairs_linked_df_csv,
        input_file_path_nuccore_df_csv=input_file_path_nuccore_df_csv,
        input_file_path_taxa_potential_evidence_for_pi_info_pickle=input_file_path_taxa_potential_evidence_for_pi_info_pickle,
        input_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle=(
            input_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle),
        input_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path=(
            input_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path),
        local_blast_nt_database_update_log_for_caching_only=local_blast_nt_database_update_log_for_caching_only,
        min_mauve_total_match_proportion=min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion=min_min_sub_alignment_min_match_proportion,
        output_file_path_taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle=(
            output_file_path_taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle),
        output_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle=(
            output_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle),
        output_file_path_merged_cds_pair_region_df_csv=output_file_path_merged_cds_pair_region_df_csv,
        output_file_path_region_in_other_nuccore_df_csv=output_file_path_region_in_other_nuccore_df_csv,
        output_file_path_potential_breakpoint_df_csv=output_file_path_potential_breakpoint_df_csv,
        taxa_output_dir_path=taxa_output_dir_path,
        downloaded_wgs_nuccore_entries_dir_path=downloaded_wgs_nuccore_entries_dir_path,
        nuccore_entries_output_dir_path=nuccore_entries_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path=other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        merged_cds_pair_region_margin_size=merged_cds_pair_region_margin_size,
        blast_margins_and_identify_regions_in_other_nuccores_args=blast_margins_and_identify_regions_in_other_nuccores_args,
        local_blast_nt_database_path=local_blast_nt_database_path,
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze=max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
        input_file_path_debug_local_blast_database_fasta=input_file_path_debug_local_blast_database_fasta,
        debug_other_nuccore_accession_to_fasta_file_path=debug_other_nuccore_accession_to_fasta_file_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=20,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_lens_of_spanning_regions_ratios_df(
        input_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle,
        output_file_path_lens_of_spanning_regions_ratios_df_csv,
):
    with open(input_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle, 'rb') as f:
        taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info = pickle.load(f)

    flat_dicts = []
    num_of_taxa = len(taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info)
    for i, (taxon_uid, nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info) in enumerate(
            taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info.items()):
        generic_utils.print_and_write_to_log(f'(cached_write_lens_of_spanning_regions_ratios_df) starting work on taxon {i + 1}/{num_of_taxa} ({taxon_uid}).')

        for nuccore_accession, other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info in (
                nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info.items()):
            for merged_cds_pair_region, result_file_paths in other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info['merged_cds_pair_region_to_result_file_paths'].items():
                merged_cds_pair_region_evidence_for_pi_info_pickle_file_path = result_file_paths['merged_cds_pair_region_evidence_for_pi_info_pickle_file_path']
                with open(merged_cds_pair_region_evidence_for_pi_info_pickle_file_path, 'rb') as f:
                    merged_cds_pair_region_evidence_for_pi_info = pickle.load(f)

                if merged_cds_pair_region_evidence_for_pi_info['blast_target_to_regions_in_other_nuccores_preliminary_info']:
                    for blast_target, regions_in_other_nuccores_preliminary_info in (
                            merged_cds_pair_region_evidence_for_pi_info['blast_target_to_regions_in_other_nuccores_preliminary_info'].items()):
                        flat_dicts.extend(
                            {
                                'taxon_uid': taxon_uid,
                                'nuccore_accession': nuccore_accession,
                                'merged_cds_pair_region_start': merged_cds_pair_region[0],
                                'merged_cds_pair_region_end': merged_cds_pair_region[1],
                                'other_nuccore_location': blast_target,
                                'lens_of_spanning_regions_ratio': x,
                            }
                            for x in regions_in_other_nuccores_preliminary_info['lens_of_spanning_regions_ratios']
                        )
    lens_of_spanning_regions_ratios_df = pd.DataFrame(flat_dicts)

    lens_of_spanning_regions_ratios_df.to_csv(output_file_path_lens_of_spanning_regions_ratios_df_csv, sep='\t', index=False)

def write_lens_of_spanning_regions_ratios_df(
        input_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle,
        output_file_path_lens_of_spanning_regions_ratios_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_lens_of_spanning_regions_ratios_df(
        input_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle=(
            input_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle),
        output_file_path_lens_of_spanning_regions_ratios_df_csv=output_file_path_lens_of_spanning_regions_ratios_df_csv,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds_column(
        input_file_path_merged_cds_pair_region_df_csv,
        input_file_path_region_in_other_nuccore_df_csv,
        output_file_path_extended_merged_cds_pair_region_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    if generic_utils.is_file_empty(input_file_path_region_in_other_nuccore_df_csv):
        generic_utils.write_empty_file(output_file_path_extended_merged_cds_pair_region_df_csv)
    else:
        region_in_other_nuccore_df = pd.read_csv(input_file_path_region_in_other_nuccore_df_csv, sep='\t', low_memory=False)

        assert not region_in_other_nuccore_df['satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds'].isna().any()

        merged_cds_pair_region_df = pd.read_csv(input_file_path_merged_cds_pair_region_df_csv, sep='\t', low_memory=False)
        extended_merged_cds_pair_region_df = merged_cds_pair_region_df.merge(
            region_in_other_nuccore_df.groupby(index_column_names.MERGED_CDS_PAIR_REGION_INDEX_COLUMN_NAMES)[
                'satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds'].sum().reset_index(
                name='num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds'),
            how='left',
        )
        extended_merged_cds_pair_region_df[
            'num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds'].fillna(0, inplace=True)

        # print(extended_merged_cds_pair_region_df[
        #           'num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds'].value_counts())

        extended_merged_cds_pair_region_df.to_csv(output_file_path_extended_merged_cds_pair_region_df_csv, sep='\t', index=False)

def add_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds_column(
        input_file_path_merged_cds_pair_region_df_csv,
        input_file_path_region_in_other_nuccore_df_csv,
        output_file_path_extended_merged_cds_pair_region_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds_column(
        input_file_path_merged_cds_pair_region_df_csv=input_file_path_merged_cds_pair_region_df_csv,
        input_file_path_region_in_other_nuccore_df_csv=input_file_path_region_in_other_nuccore_df_csv,
        output_file_path_extended_merged_cds_pair_region_df_csv=output_file_path_extended_merged_cds_pair_region_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_discard_long_potential_breakpoint_containing_intervals(
        input_file_path_potential_breakpoint_df_csv,
        max_breakpoint_containing_interval_len,
        output_file_path_filtered_potential_breakpoint_df_csv,
):
    potential_breakpoint_df = pd.read_csv(input_file_path_potential_breakpoint_df_csv, sep='\t', low_memory=False)

    filtered_potential_breakpoint_df = potential_breakpoint_df[
        potential_breakpoint_df['potential_breakpoint_containing_interval_end'] - potential_breakpoint_df['potential_breakpoint_containing_interval_start'] + 1 <=
        max_breakpoint_containing_interval_len
    ]

    filtered_potential_breakpoint_df.to_csv(output_file_path_filtered_potential_breakpoint_df_csv, sep='\t', index=False)

def discard_long_potential_breakpoint_containing_intervals(
        input_file_path_potential_breakpoint_df_csv,
        max_breakpoint_containing_interval_len,
        output_file_path_filtered_potential_breakpoint_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_discard_long_potential_breakpoint_containing_intervals(
        input_file_path_potential_breakpoint_df_csv=input_file_path_potential_breakpoint_df_csv,
        max_breakpoint_containing_interval_len=max_breakpoint_containing_interval_len,
        output_file_path_filtered_potential_breakpoint_df_csv=output_file_path_filtered_potential_breakpoint_df_csv,
    )

def discard_potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments(
        potential_breakpoint_df,
        min_matching_interval_between_intervals_coverage_by_mauve_alignments,
):
    orig_num_of_potential_breakpoints = len(potential_breakpoint_df)

    filtered_potential_breakpoint_df = potential_breakpoint_df[
        (potential_breakpoint_df['bases_of_interval_between_intervals_in_region_in_other_nuccore_overlapping_mauve_alignments_fraction'].isna()) |
        (potential_breakpoint_df['bases_of_interval_between_intervals_in_region_in_other_nuccore_overlapping_mauve_alignments_fraction'] >=
         min_matching_interval_between_intervals_coverage_by_mauve_alignments)
    ].copy()

    num_of_potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments = (
            orig_num_of_potential_breakpoints - len(filtered_potential_breakpoint_df))
    potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments_proportion = (
            num_of_potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments /
            orig_num_of_potential_breakpoints)

    discard_potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments_info = {
        'num_of_potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments': (
            num_of_potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments),
        'potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments_proportion': (
            potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments_proportion),
    }
    return (
        filtered_potential_breakpoint_df,
        discard_potential_breakpoints_with_low_matching_interval_between_intervals_coverage_by_mauve_alignments_info,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_max_diff_between_total_and_interval_near_breakpoint_match_proportion_column(
        input_file_path_potential_breakpoint_df_csv,
        input_file_path_region_in_other_nuccore_df_csv,
        output_file_path_extended_potential_breakpoint_df_csv,
):
    potential_breakpoint_df = pd.read_csv(input_file_path_potential_breakpoint_df_csv, sep='\t', low_memory=False)
    orig_num_of_potential_breakpoints = len(potential_breakpoint_df)
    minimal_region_in_other_nuccore_df = pd.read_csv(input_file_path_region_in_other_nuccore_df_csv, sep='\t', low_memory=False)[[
        'nuccore_accession',
        'merged_cds_pair_region_start',
        'merged_cds_pair_region_end',
        'other_nuccore_accession',
        'region_in_other_nuccore_start',
        'region_in_other_nuccore_end',
        'mauve_total_match_proportion',
    ]]
    potential_breakpoint_df = potential_breakpoint_df.merge(minimal_region_in_other_nuccore_df)
    min_interval_near_potential_breakpoint_containing_interval_min_match_proportion_column = potential_breakpoint_df[[
        'interval_left_to_potential_breakpoint_containing_interval_min_match_proportion',
        'interval_right_to_potential_breakpoint_containing_interval_min_match_proportion',
    ]].min(axis=1)
    potential_breakpoint_df['max_diff_between_total_and_interval_near_breakpoint_match_proportion'] = (
        potential_breakpoint_df['mauve_total_match_proportion'] - min_interval_near_potential_breakpoint_containing_interval_min_match_proportion_column)
    potential_breakpoint_df.drop('mauve_total_match_proportion', axis=1, inplace=True)

    assert len(potential_breakpoint_df) == orig_num_of_potential_breakpoints
    potential_breakpoint_df.to_csv(output_file_path_extended_potential_breakpoint_df_csv, sep='\t', index=False)

def add_max_diff_between_total_and_interval_near_breakpoint_match_proportion_column(
        input_file_path_potential_breakpoint_df_csv,
        input_file_path_region_in_other_nuccore_df_csv,
        output_file_path_extended_potential_breakpoint_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_max_diff_between_total_and_interval_near_breakpoint_match_proportion_column(
        input_file_path_potential_breakpoint_df_csv=input_file_path_potential_breakpoint_df_csv,
        input_file_path_region_in_other_nuccore_df_csv=input_file_path_region_in_other_nuccore_df_csv,
        output_file_path_extended_potential_breakpoint_df_csv=output_file_path_extended_potential_breakpoint_df_csv,
    )

def discard_potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion(
        potential_breakpoint_df,
        min_min_interval_near_breakpoint_match_proportion,
):
    orig_num_of_potential_breakpoints = len(potential_breakpoint_df)

    min_interval_near_potential_breakpoint_containing_interval_min_match_proportion_column = potential_breakpoint_df[[
        'interval_left_to_potential_breakpoint_containing_interval_min_match_proportion',
        'interval_right_to_potential_breakpoint_containing_interval_min_match_proportion',
    ]].min(axis=1)
    # print('min_interval_near_potential_breakpoint_containing_interval_min_match_proportion_column')
    # print(min_interval_near_potential_breakpoint_containing_interval_min_match_proportion_column)

    filtered_potential_breakpoint_df = potential_breakpoint_df[min_interval_near_potential_breakpoint_containing_interval_min_match_proportion_column >=
                                                               min_min_interval_near_breakpoint_match_proportion].copy()

    num_of_potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion = (
            orig_num_of_potential_breakpoints - len(filtered_potential_breakpoint_df))
    potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion_proportion = (
            num_of_potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion /
            orig_num_of_potential_breakpoints)

    discard_potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion_info = {
        'num_of_potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion': (
            num_of_potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion),
        'potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion_proportion': (
            potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion_proportion),
    }
    return (
        filtered_potential_breakpoint_df,
        discard_potential_breakpoints_with_low_min_interval_near_breakpoint_match_proportion_info,
    )

def write_bps_whose_region_in_other_nuccore_contains_only_good_bps_and_at_least_2_bps_df(
        orig_bp_df,
        good_bp_df,
        regions_in_other_nuccores_that_contain_only_good_bps_and_at_least_2_bps_df_csv_file_path,
        bps_whose_region_in_other_nuccore_contains_only_good_bps_and_at_least_2_bps_df_csv_file_path,
):
    assert len(good_bp_df) == len(good_bp_df['index_in_potential_breakpoint_df_csv_file'].drop_duplicates())
    bp_status_df = orig_bp_df[index_column_names.REGION_IN_OTHER_NUCCORE_INDEX_COLUMN_NAMES + ['index_in_potential_breakpoint_df_csv_file']].merge(
        good_bp_df['index_in_potential_breakpoint_df_csv_file'],
        how='left', indicator=True,
    )
    assert len(bp_status_df) == len(bp_status_df['index_in_potential_breakpoint_df_csv_file'].drop_duplicates())

    assert ((bp_status_df['_merge'] == 'left_only') | (bp_status_df['_merge'] == 'both')).all()
    bp_status_df['passed_all_thresholds'] = bp_status_df['_merge'] == 'both'
    bp_status_df.drop('_merge', axis=1, inplace=True)

    grouped_bp_status_df = bp_status_df.groupby(index_column_names.REGION_IN_OTHER_NUCCORE_INDEX_COLUMN_NAMES)
    regions_in_other_nuccores_that_contains_only_good_bps_and_at_least_2_bps_df = grouped_bp_status_df['passed_all_thresholds'].all().reset_index(
        name='all_bps_passed_all_thresholds').merge(grouped_bp_status_df.size().reset_index(name='num_of_bps'))

    regions_in_other_nuccores_that_contain_only_good_bps_and_at_least_2_bps_df = regions_in_other_nuccores_that_contains_only_good_bps_and_at_least_2_bps_df[
        regions_in_other_nuccores_that_contains_only_good_bps_and_at_least_2_bps_df['all_bps_passed_all_thresholds'] &
        (regions_in_other_nuccores_that_contains_only_good_bps_and_at_least_2_bps_df['num_of_bps'] >= 2)
    ].drop(['all_bps_passed_all_thresholds', 'num_of_bps'], axis=1)
    regions_in_other_nuccores_that_contain_only_good_bps_and_at_least_2_bps_df.to_csv(
        regions_in_other_nuccores_that_contain_only_good_bps_and_at_least_2_bps_df_csv_file_path, sep='\t', index=False)

    bps_whose_region_in_other_nuccore_contains_only_good_bps_and_at_least_2_bps_df = good_bp_df.merge(
        regions_in_other_nuccores_that_contain_only_good_bps_and_at_least_2_bps_df)
    bps_whose_region_in_other_nuccore_contains_only_good_bps_and_at_least_2_bps_df.to_csv(
        bps_whose_region_in_other_nuccore_contains_only_good_bps_and_at_least_2_bps_df_csv_file_path, sep='\t', index=False)

@generic_utils.execute_if_output_doesnt_exist_already
def cached_add_merged_cds_pair_region_to_pairs_df(
        input_file_path_pairs_df_csv,
        input_file_path_merged_cds_pair_region_df_csv,
        output_file_path_extended_pairs_df_csv,
        output_file_path_add_merged_cds_pair_region_to_pairs_df_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    orig_pairs_df = pd.read_csv(input_file_path_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_pairs = len(orig_pairs_df)
    minimal_merged_ir_pair_df = pd.read_csv(input_file_path_merged_cds_pair_region_df_csv, sep='\t', low_memory=False)[[
        'nuccore_accession',
        'merged_cds_pair_region_start',
        'merged_cds_pair_region_end',
    ]]
    assert not (set(orig_pairs_df) & {'merged_cds_pair_region_start', 'merged_cds_pair_region_end'})
    pairs_df = orig_pairs_df.merge(minimal_merged_ir_pair_df)
    pairs_df = pairs_df[(pairs_df['left1'] >= pairs_df['merged_cds_pair_region_start']) &
                        (pairs_df['right2'] <= pairs_df['merged_cds_pair_region_end'])]

    # pairs_df = orig_pairs_df.merge(pairs_df, how='left') # 220401: no idea why this was needed.
    assert len(pairs_df) == orig_num_of_pairs
    pairs_df.to_csv(output_file_path_extended_pairs_df_csv, sep='\t', index=False)

    num_of_pairs_without_merged_cds_pair_region = orig_num_of_pairs - len(pairs_df)
    pairs_without_merged_cds_pair_region_proportion = num_of_pairs_without_merged_cds_pair_region / orig_num_of_pairs

    add_merged_cds_pair_region_to_pairs_df_info = {
        'num_of_pairs_without_merged_cds_pair_region': num_of_pairs_without_merged_cds_pair_region,
        'pairs_without_merged_cds_pair_region_proportion': pairs_without_merged_cds_pair_region_proportion,
    }

    with open(output_file_path_add_merged_cds_pair_region_to_pairs_df_info_pickle, 'wb') as f:
        pickle.dump(add_merged_cds_pair_region_to_pairs_df_info, f, protocol=4)


def add_merged_cds_pair_region_to_pairs_df(
        input_file_path_pairs_df_csv,
        input_file_path_merged_cds_pair_region_df_csv,
        output_file_path_extended_pairs_df_csv,
        output_file_path_add_merged_cds_pair_region_to_pairs_df_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_add_merged_cds_pair_region_to_pairs_df(
        input_file_path_pairs_df_csv=input_file_path_pairs_df_csv,
        input_file_path_merged_cds_pair_region_df_csv=input_file_path_merged_cds_pair_region_df_csv,
        output_file_path_extended_pairs_df_csv=output_file_path_extended_pairs_df_csv,
        output_file_path_add_merged_cds_pair_region_to_pairs_df_info_pickle=output_file_path_add_merged_cds_pair_region_to_pairs_df_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_repeat_linked_to_breakpoint_df(
        input_file_path_pairs_df_csv,
        input_file_path_potential_breakpoint_df_csv,
        output_file_path_repeat_with_dist_from_potential_breakpoints_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    potential_breakpoint_df = pd.read_csv(input_file_path_potential_breakpoint_df_csv, sep='\t', low_memory=False)
    potential_breakpoint_df['potential_breakpoint_containing_interval_middle'] = (potential_breakpoint_df['potential_breakpoint_containing_interval_start'] +
                                                                                  potential_breakpoint_df['potential_breakpoint_containing_interval_end']) / 2
    potential_breakpoint_df['dist_from_potential_breakpoint_containing_interval_middle_to_edge'] = (
        potential_breakpoint_df['potential_breakpoint_containing_interval_end'] -
        potential_breakpoint_df['potential_breakpoint_containing_interval_start']
    ) / 2

    minimal_potential_breakpoint_df = potential_breakpoint_df[[
        'nuccore_accession',
        'index_in_potential_breakpoint_df_csv_file',
        'merged_cds_pair_region_start',
        'merged_cds_pair_region_end',
        'potential_breakpoint_containing_interval_middle',
        'dist_from_potential_breakpoint_containing_interval_middle_to_edge',
    ]]

    pairs_df = pd.read_csv(input_file_path_pairs_df_csv, sep='\t', low_memory=False)

    minimal_pairs_df = pairs_df[[
        'nuccore_accession',
        'index_in_nuccore_ir_pairs_df_csv_file',
        'merged_cds_pair_region_start',
        'merged_cds_pair_region_end',
        'left1',
        'right1',
        'left2',
        'right2',
    ]]
    minimal_repeats_df = pd.concat([
        minimal_pairs_df[['nuccore_accession', 'merged_cds_pair_region_start', 'merged_cds_pair_region_end',
                          f'left{repeat_num}', f'right{repeat_num}',
                          'index_in_nuccore_ir_pairs_df_csv_file']].assign(repeat_num=repeat_num).rename(columns={f'left{repeat_num}': 'left',
                                                                                                                  f'right{repeat_num}': 'right'})
        for repeat_num in (1,2)
    ], ignore_index=True)
    minimal_repeats_df['dist_from_repeat_middle_to_edge'] = (minimal_repeats_df['right'] - minimal_repeats_df['left']) / 2
    minimal_repeats_df['repeat_middle'] = (minimal_repeats_df['left'] + minimal_repeats_df['right']) / 2
    minimal_repeats_df.drop(['left', 'right'], axis=1, inplace=True)

    joined_df = minimal_repeats_df.merge(minimal_potential_breakpoint_df, on=['nuccore_accession', 'merged_cds_pair_region_start', 'merged_cds_pair_region_end'])
    joined_df['dist_between_potential_breakpoint_containing_interval_and_repeat'] = (
            joined_df['potential_breakpoint_containing_interval_middle'] - joined_df['repeat_middle']
    ).abs() - joined_df['dist_from_repeat_middle_to_edge'] - joined_df['dist_from_potential_breakpoint_containing_interval_middle_to_edge']
    joined_df.drop(['potential_breakpoint_containing_interval_middle', 'repeat_middle',
                    'dist_from_repeat_middle_to_edge', 'dist_from_potential_breakpoint_containing_interval_middle_to_edge'], axis=1, inplace=True)
    joined_df.to_csv(output_file_path_repeat_with_dist_from_potential_breakpoints_df_csv, sep='\t', index=False)

def write_repeat_linked_to_breakpoint_df(
        input_file_path_pairs_df_csv,
        input_file_path_potential_breakpoint_df_csv,
        output_file_path_repeat_with_dist_from_potential_breakpoints_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_repeat_linked_to_breakpoint_df(
        input_file_path_pairs_df_csv=input_file_path_pairs_df_csv,
        input_file_path_potential_breakpoint_df_csv=input_file_path_potential_breakpoint_df_csv,
        output_file_path_repeat_with_dist_from_potential_breakpoints_df_csv=output_file_path_repeat_with_dist_from_potential_breakpoints_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

def get_minimal_ir_pairs_linked_to_breakpoint_pairs_df(
        potential_breakpoint_df,
        repeat_with_dist_from_potential_breakpoints_df,
):
    minimal_potential_breakpoint_with_overlapping_repeat_df = potential_breakpoint_df.merge(repeat_with_dist_from_potential_breakpoints_df)[[
        'nuccore_accession',
        'merged_cds_pair_region_start',
        'merged_cds_pair_region_end',
        'index_in_potential_breakpoint_df_csv_file',
        'index_in_nuccore_ir_pairs_df_csv_file',
        'repeat_num',
        'dist_between_potential_breakpoint_containing_interval_and_repeat',

        'other_nuccore_accession',
        'region_in_other_nuccore_start',
        'region_in_other_nuccore_end',
    ]]

    minimal_potential_breakpoint_df_per_repeat_num = [
        minimal_potential_breakpoint_with_overlapping_repeat_df[
            minimal_potential_breakpoint_with_overlapping_repeat_df['repeat_num'] == repeat_num
        ].drop('repeat_num', axis=1).rename(
            columns={
                'index_in_potential_breakpoint_df_csv_file': f'bp_roughly_overlapping_repeat{repeat_num}_index_in_potential_breakpoint_df_csv_file',
                'dist_between_potential_breakpoint_containing_interval_and_repeat': f'dist_between_potential_breakpoint_containing_interval_and_repeat{repeat_num}',
            })
        for repeat_num in (1, 2)
    ]
    minimal_ir_pairs_linked_to_breakpoint_pairs_df = minimal_potential_breakpoint_df_per_repeat_num[0].merge(minimal_potential_breakpoint_df_per_repeat_num[1])
    minimal_ir_pairs_linked_to_breakpoint_pairs_df = minimal_ir_pairs_linked_to_breakpoint_pairs_df[
        minimal_ir_pairs_linked_to_breakpoint_pairs_df['bp_roughly_overlapping_repeat1_index_in_potential_breakpoint_df_csv_file'] !=
        minimal_ir_pairs_linked_to_breakpoint_pairs_df['bp_roughly_overlapping_repeat2_index_in_potential_breakpoint_df_csv_file']
    ]
    minimal_ir_pairs_linked_to_breakpoint_pairs_df['max_dist_between_potential_breakpoint_containing_interval_and_repeat'] = (
        minimal_ir_pairs_linked_to_breakpoint_pairs_df[['dist_between_potential_breakpoint_containing_interval_and_repeat1',
                                                              'dist_between_potential_breakpoint_containing_interval_and_repeat2']].max(axis=1)
    )

    return minimal_ir_pairs_linked_to_breakpoint_pairs_df


def write_pairs_with_highest_confidence_bps_df(
        minimal_ir_pairs_linked_to_breakpoint_pairs_df,
        pairs_with_merged_cds_pair_region_df_csv_file_path,
        region_in_other_nuccore_df_csv_file_path,
        pairs_with_highest_confidence_bps_df_csv_file_path,
):
    region_in_other_nuccore_df = pd.read_csv(region_in_other_nuccore_df_csv_file_path, sep='\t', low_memory=False)

    ir_pairs_and_overlapping_breakpoint_pairs_df = minimal_ir_pairs_linked_to_breakpoint_pairs_df.merge(
        region_in_other_nuccore_df[index_column_names.REGION_IN_OTHER_NUCCORE_INDEX_COLUMN_NAMES + ['mauve_total_match_proportion']]
    )

    # sort by ['mauve_total_match_proportion'] + region_in_other_nuccore_index_column_names_without_merged_cds_pair_region_column_names in order to prevent a case in which we
    # have two high confidence pairs that are overlapping or something annoying like that, with the same region in other nuccore providing the high confidence. by sorting this
    # way, we make sure that both high confidence pairs would have the same region_in_other_nuccore_with_highest_confidence_bps, such that in fig1c we will drop duplicates and
    # have only a single data point contribute to the histogram.
    filtered_pairs_with_highest_confidence_breakpoints_df = ir_pairs_and_overlapping_breakpoint_pairs_df.sort_values(
        ['mauve_total_match_proportion'], ascending=False,
    ).drop_duplicates(subset=['nuccore_accession', 'index_in_nuccore_ir_pairs_df_csv_file'], keep='first').rename(columns={
        'other_nuccore_accession': 'other_nuccore_accession_with_highest_confidence_bps',
        'region_in_other_nuccore_start': 'region_in_other_nuccore_start_with_highest_confidence_bps',
        'region_in_other_nuccore_end': 'region_in_other_nuccore_end_with_highest_confidence_bps',
        'mauve_total_match_proportion': 'mauve_total_match_proportion_of_region_in_other_nuccore_with_highest_confidence_bps',
        'bp_roughly_overlapping_repeat1_index_in_potential_breakpoint_df_csv_file': 'highest_confidence_bp_roughly_overlapping_repeat1_index_in_potential_breakpoint_df_csv_file',
        'bp_roughly_overlapping_repeat2_index_in_potential_breakpoint_df_csv_file': 'highest_confidence_bp_roughly_overlapping_repeat2_index_in_potential_breakpoint_df_csv_file',
        'dist_between_potential_breakpoint_containing_interval_and_repeat': 'highest_confidence_bp_containing_interval_dist_from_roughly_overlapping_repeat',
    })

    pairs_with_merged_cds_pair_region_df = pd.read_csv(pairs_with_merged_cds_pair_region_df_csv_file_path, sep='\t', low_memory=False)
    pairs_with_highest_confidence_bps_df = pairs_with_merged_cds_pair_region_df.merge(filtered_pairs_with_highest_confidence_breakpoints_df, how='left', indicator=True)
    pairs_with_highest_confidence_bps_df.loc[pairs_with_highest_confidence_bps_df['_merge'] == 'left_only', 'high_confidence_bp_for_both_repeats'] = False
    pairs_with_highest_confidence_bps_df.loc[pairs_with_highest_confidence_bps_df['_merge'] == 'both', 'high_confidence_bp_for_both_repeats'] = True
    pairs_with_highest_confidence_bps_df.drop('_merge', axis=1, inplace=True)

    assert len(pairs_with_highest_confidence_bps_df) == len(pairs_with_merged_cds_pair_region_df)
    assert len(pairs_with_highest_confidence_bps_df) == len(pairs_with_highest_confidence_bps_df[['nuccore_accession',
                                                                                                  'index_in_nuccore_ir_pairs_df_csv_file']].drop_duplicates())

    pairs_with_highest_confidence_bps_df.to_csv(pairs_with_highest_confidence_bps_df_csv_file_path, sep='\t', index=False)

def do_massive_screening_stage5(
        search_for_pis_args,
):
    massive_screening_stage5_out_dir_path = search_for_pis_args['stage5']['output_dir_path']

    debug___num_of_taxa_to_go_over = search_for_pis_args['debug___num_of_taxa_to_go_over']
    debug___taxon_uids = search_for_pis_args['debug___taxon_uids']

    stage_out_file_name_suffix = massive_screening_stage_1.get_stage_out_file_name_suffix(
        debug___num_of_taxa_to_go_over=debug___num_of_taxa_to_go_over,
        debug___taxon_uids=debug___taxon_uids,
    )

    massive_screening_log_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'massive_screening_stage5_log.txt')
    stage5_results_info_pickle_file_path = os.path.join(massive_screening_stage5_out_dir_path, search_for_pis_args['stage5']['results_pickle_file_name'])
    stage5_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage5_results_info_pickle_file_path, stage_out_file_name_suffix)

    pathlib.Path(massive_screening_stage5_out_dir_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=massive_screening_log_file_path, level=logging.DEBUG, format='%(asctime)s %(message)s')
    generic_utils.print_and_write_to_log(f'---------------starting do_massive_screening_stage5({massive_screening_stage5_out_dir_path})---------------')

    other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path = os.path.join(
        massive_screening_stage5_out_dir_path,
        search_for_pis_args['stage5']['other_nuccore_entries_extracted_from_local_nt_blast_db_dir_name'],
    )
    downloaded_wgs_nuccore_entries_dir_path = os.path.join(massive_screening_stage5_out_dir_path, 'downloaded_wgs_nuccore_entries')
    nuccore_entries_output_dir_path = os.path.join(massive_screening_stage5_out_dir_path, 'primary_nuccore_entries')
    taxa_output_dir_path = os.path.join(massive_screening_stage5_out_dir_path, 'taxa')
    pathlib.Path(other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(downloaded_wgs_nuccore_entries_dir_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(nuccore_entries_output_dir_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(taxa_output_dir_path).mkdir(parents=True, exist_ok=True)

    massive_screening_stage4_out_dir_path = search_for_pis_args['stage4']['output_dir_path']
    stage4_results_info_pickle_file_path = os.path.join(massive_screening_stage4_out_dir_path, search_for_pis_args['stage4']['results_pickle_file_name'])
    stage4_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage4_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage4_results_info_pickle_file_path, 'rb') as f:
        stage4_results_info = pickle.load(f)

    stage3_out_dir_path = search_for_pis_args['stage3']['output_dir_path']
    stage3_results_info_pickle_file_path = os.path.join(stage3_out_dir_path, search_for_pis_args['stage3']['results_pickle_file_name'])
    stage3_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage3_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage3_results_info_pickle_file_path, 'rb') as f:
        stage3_results_info = pickle.load(f)
    pairs_linked_df_csv_file_path = stage3_results_info[
        'pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path']

    stage1_results_info_pickle_file_path = os.path.join(search_for_pis_args['stage1']['output_dir_path'], search_for_pis_args['stage1']['results_pickle_file_name'])
    stage1_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage1_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage1_results_info_pickle_file_path, 'rb') as f:
        stage1_results_info = pickle.load(f)

    relevant_taxon_uids_pickle_file_path = stage4_results_info['relevant_taxon_uids_pickle_file_path']
    species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path = stage4_results_info[
        'species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path']
    nuccore_df_csv_file_path = stage1_results_info['nuccore_df_csv_file_path']
    # all_cds_df_csv_file_path = stage1_results_info['all_cds_df_csv_file_path']
    # nuccore_accession_to_nuccore_entry_info_pickle_file_path = stage1_results_info['nuccore_accession_to_nuccore_entry_info_pickle_file_path']

    taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle_file_path = os.path.join(
        massive_screening_stage5_out_dir_path, 'taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path.pickle')
    taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle_file_path, stage_out_file_name_suffix)

    write_nuccore_accession_to_minimal_nuccore_entry_info_for_each_relevant_taxon(
        input_file_path_relevant_taxon_uids_pickle=relevant_taxon_uids_pickle_file_path,
        input_file_path_nuccore_df_csv=nuccore_df_csv_file_path,
        output_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle=(
            taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle_file_path),
        taxa_output_dir_path=taxa_output_dir_path,
    )

    taxa_potential_evidence_for_pi_info_pickle_file_path = stage4_results_info['taxa_potential_evidence_for_pi_info_pickle_file_path']

    taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle_file_path = os.path.join(
        massive_screening_stage5_out_dir_path, 'taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path.pickle')
    taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle_file_path, stage_out_file_name_suffix)

    taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle_file_path = os.path.join(
        massive_screening_stage5_out_dir_path, 'taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info.pickle')
    taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle_file_path, stage_out_file_name_suffix)

    merged_cds_pair_region_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'merged_cds_pair_region_df.csv')
    merged_cds_pair_region_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        merged_cds_pair_region_df_csv_file_path, stage_out_file_name_suffix)

    region_in_other_nuccore_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'region_in_other_nuccore_df.csv')
    region_in_other_nuccore_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        region_in_other_nuccore_df_csv_file_path, stage_out_file_name_suffix)

    potential_breakpoint_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'potential_breakpoint_df.csv')
    potential_breakpoint_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        potential_breakpoint_df_csv_file_path, stage_out_file_name_suffix)

    download_wgs_and_find_other_nuccore_entries_evidence(
        input_file_path_pairs_linked_df_csv=pairs_linked_df_csv_file_path,
        input_file_path_nuccore_df_csv=nuccore_df_csv_file_path,
        input_file_path_taxa_potential_evidence_for_pi_info_pickle=taxa_potential_evidence_for_pi_info_pickle_file_path,
        input_file_path_taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle=(
            taxon_uid_to_nuccore_accession_to_minimal_nuccore_entry_info_pickle_file_path_pickle_file_path),
        input_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path=(
            species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path),
        local_blast_nt_database_update_log_for_caching_only=generic_utils.read_text_file(
            search_for_pis_args['local_blast_nt_database_update_log_file_path']),
        min_mauve_total_match_proportion=search_for_pis_args['stage5']['min_mauve_total_match_proportion'],
        min_min_sub_alignment_min_match_proportion=search_for_pis_args['stage5']['min_min_sub_alignment_min_match_proportion'],
        merged_cds_pair_region_margin_size=search_for_pis_args['stage5']['merged_cds_pair_region_margin_size'],
        blast_margins_and_identify_regions_in_other_nuccores_args=search_for_pis_args['stage5']['blast_margins_and_identify_regions_in_other_nuccores'],
        local_blast_nt_database_path=search_for_pis_args['local_blast_nt_database_path'],
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze=search_for_pis_args['stage5'][
            'max_num_of_non_identical_regions_in_other_nuccores_to_analyze_per_merged_cds_pair_region'],
        output_file_path_taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle=(
            taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle_file_path),
        output_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle=(
            taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle_file_path),
        output_file_path_merged_cds_pair_region_df_csv=merged_cds_pair_region_df_csv_file_path,
        output_file_path_region_in_other_nuccore_df_csv=region_in_other_nuccore_df_csv_file_path,
        output_file_path_potential_breakpoint_df_csv=potential_breakpoint_df_csv_file_path,
        taxa_output_dir_path=taxa_output_dir_path,
        downloaded_wgs_nuccore_entries_dir_path=downloaded_wgs_nuccore_entries_dir_path,
        nuccore_entries_output_dir_path=nuccore_entries_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path=other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        input_file_path_debug_local_blast_database_fasta=search_for_pis_args['debug_local_blast_database_path'],
        debug_other_nuccore_accession_to_fasta_file_path=search_for_pis_args['debug_other_nuccore_accession_to_fasta_file_path'],
    )

    lens_of_spanning_regions_ratios_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'lens_of_spanning_regions_ratios_df.csv')
    lens_of_spanning_regions_ratios_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        lens_of_spanning_regions_ratios_df_csv_file_path, stage_out_file_name_suffix)
    write_lens_of_spanning_regions_ratios_df(
        input_file_path_taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle=(
            taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle_file_path),
        output_file_path_lens_of_spanning_regions_ratios_df_csv=lens_of_spanning_regions_ratios_df_csv_file_path,
    )

    extended_merged_cds_pair_region_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'extended_merged_cds_pair_region_df.csv')
    extended_merged_cds_pair_region_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        extended_merged_cds_pair_region_df_csv_file_path, stage_out_file_name_suffix)
    add_num_of_regions_in_other_nuccores_satisfying_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds_column(
        input_file_path_merged_cds_pair_region_df_csv=merged_cds_pair_region_df_csv_file_path,
        input_file_path_region_in_other_nuccore_df_csv=region_in_other_nuccore_df_csv_file_path,
        output_file_path_extended_merged_cds_pair_region_df_csv=extended_merged_cds_pair_region_df_csv_file_path,
    )

    if generic_utils.is_file_empty(potential_breakpoint_df_csv_file_path):
        potential_breakpoint_df = None
    else:
        potential_breakpoint_df = pd.read_csv(potential_breakpoint_df_csv_file_path, sep='\t', low_memory=False)

    bp_out_dir_path = os.path.join(massive_screening_stage5_out_dir_path, 'breakpoints')
    bp_out_dir_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(bp_out_dir_path, stage_out_file_name_suffix)
    pathlib.Path(bp_out_dir_path).mkdir(parents=True, exist_ok=True)

    pairs_with_merged_cds_pair_region_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'pairs_with_merged_cds_pair_region_df.csv')
    pairs_with_merged_cds_pair_region_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        pairs_with_merged_cds_pair_region_df_csv_file_path, stage_out_file_name_suffix)
    add_merged_cds_pair_region_to_pairs_df_info_pickle_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'add_merged_cds_pair_region_to_pairs_df_info.pickle')
    add_merged_cds_pair_region_to_pairs_df_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        add_merged_cds_pair_region_to_pairs_df_info_pickle_file_path, stage_out_file_name_suffix)
    if generic_utils.is_file_empty(merged_cds_pair_region_df_csv_file_path):
        add_merged_cds_pair_region_to_pairs_df_info = None
    else:
        add_merged_cds_pair_region_to_pairs_df(
            input_file_path_pairs_df_csv=pairs_linked_df_csv_file_path,
            input_file_path_merged_cds_pair_region_df_csv=merged_cds_pair_region_df_csv_file_path,
            output_file_path_extended_pairs_df_csv=pairs_with_merged_cds_pair_region_df_csv_file_path,
            output_file_path_add_merged_cds_pair_region_to_pairs_df_info_pickle=add_merged_cds_pair_region_to_pairs_df_info_pickle_file_path,
        )
        with open(add_merged_cds_pair_region_to_pairs_df_info_pickle_file_path, 'rb') as f:
            add_merged_cds_pair_region_to_pairs_df_info = pickle.load(f)
        print(f'\nadd_merged_cds_pair_region_to_pairs_df_info:\n'
                  f'{add_merged_cds_pair_region_to_pairs_df_info}\n')

    if potential_breakpoint_df is not None:
        potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path = os.path.join(
            massive_screening_stage5_out_dir_path, 'potential_breakpoint_after_discarding_long_breakpoint_intervals_df.csv')
        potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path, stage_out_file_name_suffix)
        discard_long_potential_breakpoint_containing_intervals(
            input_file_path_potential_breakpoint_df_csv=potential_breakpoint_df_csv_file_path,
            max_breakpoint_containing_interval_len=search_for_pis_args['stage5']['max_breakpoint_containing_interval_len'],
            output_file_path_filtered_potential_breakpoint_df_csv=potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path,
        )
        potential_breakpoint_after_discarding_long_breakpoint_intervals_df = pd.read_csv(
            potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path, sep='\t', low_memory=False)

        # print('potential_breakpoint_after_discarding_long_breakpoint_intervals_df')
        # print(potential_breakpoint_after_discarding_long_breakpoint_intervals_df)

        repeat_with_dist_from_potential_breakpoints_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path,
                                                                                    'repeat_with_dist_from_potential_breakpoints_df.csv')
        repeat_with_dist_from_potential_breakpoints_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            repeat_with_dist_from_potential_breakpoints_df_csv_file_path, stage_out_file_name_suffix)
        write_repeat_linked_to_breakpoint_df(
            input_file_path_pairs_df_csv=pairs_with_merged_cds_pair_region_df_csv_file_path,
            input_file_path_potential_breakpoint_df_csv=potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path,
            output_file_path_repeat_with_dist_from_potential_breakpoints_df_csv=repeat_with_dist_from_potential_breakpoints_df_csv_file_path,
        )

        repeat_with_dist_from_potential_breakpoints_df = pd.read_csv(repeat_with_dist_from_potential_breakpoints_df_csv_file_path, sep='\t', low_memory=False)
        # print(repeat_roughly_overlapping_potential_breakpoints_df.head())

        minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path,
                                                                                    'minimal_ir_pairs_linked_to_breakpoint_pairs_df.csv')
        minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path, stage_out_file_name_suffix)

        minimal_ir_pairs_linked_to_breakpoint_pairs_df = get_minimal_ir_pairs_linked_to_breakpoint_pairs_df(
            potential_breakpoint_df=potential_breakpoint_after_discarding_long_breakpoint_intervals_df,
            repeat_with_dist_from_potential_breakpoints_df=repeat_with_dist_from_potential_breakpoints_df,
        )
        minimal_ir_pairs_linked_to_breakpoint_pairs_df.to_csv(minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path, sep='\t', index=False)

        max_max_dist_between_potential_breakpoint_containing_interval_and_repeat = search_for_pis_args[
            'stage5']['max_max_dist_between_potential_breakpoint_containing_interval_and_repeat']
        filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df = minimal_ir_pairs_linked_to_breakpoint_pairs_df[
            minimal_ir_pairs_linked_to_breakpoint_pairs_df['max_dist_between_potential_breakpoint_containing_interval_and_repeat'] <=
            max_max_dist_between_potential_breakpoint_containing_interval_and_repeat
        ]
        filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path,
                                                                                    'filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df.csv')
        filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path, stage_out_file_name_suffix)
        filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df.to_csv(filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path,
                                                                       sep='\t', index=False)

        pairs_with_highest_confidence_bps_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path, 'pairs_with_highest_confidence_bps_df.csv')
        pairs_with_highest_confidence_bps_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            pairs_with_highest_confidence_bps_df_csv_file_path, stage_out_file_name_suffix)
        write_pairs_with_highest_confidence_bps_df(
            minimal_ir_pairs_linked_to_breakpoint_pairs_df=filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df,
            pairs_with_merged_cds_pair_region_df_csv_file_path=pairs_with_merged_cds_pair_region_df_csv_file_path,
            region_in_other_nuccore_df_csv_file_path=region_in_other_nuccore_df_csv_file_path,
            pairs_with_highest_confidence_bps_df_csv_file_path=pairs_with_highest_confidence_bps_df_csv_file_path,
        )

        pairs_with_highest_confidence_bps_df = pd.read_csv(pairs_with_highest_confidence_bps_df_csv_file_path, sep='\t', low_memory=False)
        print("pairs_with_highest_confidence_bps_df['high_confidence_bp_for_both_repeats'].value_counts():")
        print(pairs_with_highest_confidence_bps_df['high_confidence_bp_for_both_repeats'].value_counts())
        print("len(pairs_with_highest_confidence_bps_df[pairs_with_highest_confidence_bps_df['high_confidence_bp_for_both_repeats']][['nuccore_accession', 'repeat1_cds_index_in_nuccore_cds_features_gb_file', 'repeat2_cds_index_in_nuccore_cds_features_gb_file']].drop_duplicates()):")
        print(len(pairs_with_highest_confidence_bps_df[pairs_with_highest_confidence_bps_df['high_confidence_bp_for_both_repeats']][['nuccore_accession', 'repeat1_cds_index_in_nuccore_cds_features_gb_file', 'repeat2_cds_index_in_nuccore_cds_features_gb_file']].drop_duplicates()))

        filtered_pairs_with_highest_confidence_bps_df = pairs_with_highest_confidence_bps_df[
            pairs_with_highest_confidence_bps_df['high_confidence_bp_for_both_repeats']]
        # print(f'\nfiltered_pairs_with_highest_confidence_bps_df: {filtered_pairs_with_highest_confidence_bps_df}\n')

        filtered_pairs_with_highest_confidence_bps_df_csv_file_path = os.path.join(massive_screening_stage5_out_dir_path,
                                                                                   'filtered_pairs_with_highest_confidence_bps_df.csv')
        filtered_pairs_with_highest_confidence_bps_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            filtered_pairs_with_highest_confidence_bps_df_csv_file_path, stage_out_file_name_suffix)
        filtered_pairs_with_highest_confidence_bps_df.to_csv(filtered_pairs_with_highest_confidence_bps_df_csv_file_path, sep='\t', index=False)

        num_of_pairs_with_high_confidence_bps = pairs_with_highest_confidence_bps_df['high_confidence_bp_for_both_repeats'].sum()
        print(f'\nnum_of_pairs_with_high_confidence_bps: {num_of_pairs_with_high_confidence_bps}\n')

        assert len(pairs_with_highest_confidence_bps_df) == len(pairs_with_highest_confidence_bps_df[['nuccore_accession',
                                                                                                                  'index_in_nuccore_ir_pairs_df_csv_file']].drop_duplicates())
        merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df = pairs_with_highest_confidence_bps_df[
            pairs_with_highest_confidence_bps_df['high_confidence_bp_for_both_repeats']
        ][[
            'nuccore_accession',
            'merged_cds_pair_region_start',
            'merged_cds_pair_region_end',
        ]].value_counts().reset_index(name='num_of_pairs_with_high_confidence_bps').sort_values('num_of_pairs_with_high_confidence_bps', ascending=False)
        # print(f'\nmerged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df:\n{merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df}\n')

        merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df_csv_file_path = os.path.join(
            massive_screening_stage5_out_dir_path, 'merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df.csv')
        merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
            merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df_csv_file_path, stage_out_file_name_suffix)
        merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df.to_csv(merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df_csv_file_path,
                                                                                   sep='\t', index=False)

        if 0:
            with generic_utils.timing_context_manager('print taxon uids with any high_confidence_bp_for_both_repeats'):
                nuccore_df = pd.read_csv(nuccore_df_csv_file_path, sep='\t', low_memory=False)
                # print('taxon uids with any high_confidence_bp_for_both_repeats:')
                # # print(sorted(filtered_pairs_with_highest_confidence_bps_df.merge(nuccore_df[['nuccore_accession', 'taxon_uid']])['taxon_uid'].drop_duplicates()))
                # print(filtered_pairs_with_highest_confidence_bps_df.merge(nuccore_df[['nuccore_accession', 'taxon_uid']])[['nuccore_accession', 'taxon_uid']].drop_duplicates())
                print('taxon uids with any potential_breakpoint_df:')
                print(sorted(potential_breakpoint_df.merge(nuccore_df[['nuccore_accession', 'taxon_uid']])['taxon_uid'].drop_duplicates()))
                # 220422 print result:
                # [213, 250, 256, 258, 476, 562, 715, 779, 817, 818, 821, 823, 1160, 1264, 1307, 1313, 1339, 1351, 1512, 1582, 1588, 1590, 1622, 1624, 1888, 2115, 2130, 13373, 28111, 28116, 28129, 28137, 28139, 28251, 28450, 28454, 28901, 29459, 35623, 40216, 42235, 42862, 45361, 46503, 47678, 53344, 53417, 59737, 60519, 68892, 71451, 75985, 84112, 85831, 110321, 114090, 120577, 161895, 162426, 166486, 204038, 204039, 208479, 208962, 214856, 239935, 246787, 291112, 310298, 310300, 310514, 328812, 328813, 329854, 338188, 357276, 371601, 387661, 398555, 446660, 469591, 481722, 501571, 544645, 574930, 626929, 671267, 674529, 683124, 712710, 744515, 938155, 1064539, 1082704, 1089444, 1124835, 1249999, 1355477, 1511761, 1618207, 1778540, 1796635, 1805473, 1841857, 1843235, 2044587, 2507160, 2599607, 2650158, 2777781, 2836161, 2854757, 2854759, 2854763]
                # print(sorted(breakpoint_df.merge(nuccore_df[['nuccore_accession', 'taxon_uid']])['taxon_uid'].drop_duplicates()))
        cds_pairs_of_filtered_pairs_with_highest_confidence_bps_df = filtered_pairs_with_highest_confidence_bps_df[
            index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES].drop_duplicates()
        print(f'len(cds_pairs_of_filtered_pairs_with_highest_confidence_bps_df): {len(cds_pairs_of_filtered_pairs_with_highest_confidence_bps_df)}')

    else:
        minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path = None
        filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path = None
        pairs_with_highest_confidence_bps_df_csv_file_path = None
        merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df_csv_file_path = None
        num_of_pairs_with_high_confidence_bps = 0
        potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path = None
        filtered_pairs_with_highest_confidence_bps_df_csv_file_path = None
        repeat_with_dist_from_potential_breakpoints_df_csv_file_path = None



    stage5_results_info = {
        'taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle_file_path': (
            taxon_uid_to_nuccore_accession_to_other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info_pickle_file_path),
        'taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle_file_path': (
            taxon_uid_to_downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path_pickle_file_path),

        'add_merged_cds_pair_region_to_pairs_df_info': add_merged_cds_pair_region_to_pairs_df_info,
        'num_of_pairs_with_high_confidence_bps': num_of_pairs_with_high_confidence_bps,

        # 'merged_cds_pair_region_df_csv_file_path': merged_cds_pair_region_df_csv_file_path,
        'extended_merged_cds_pair_region_df_csv_file_path': extended_merged_cds_pair_region_df_csv_file_path,
        # 'pairs_with_merged_cds_pair_region_df_csv_file_path': pairs_with_merged_cds_pair_region_df_csv_file_path,

        'region_in_other_nuccore_df_csv_file_path': region_in_other_nuccore_df_csv_file_path,

        'potential_breakpoint_df_csv_file_path': potential_breakpoint_df_csv_file_path,
        'potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path': (
            potential_breakpoint_after_discarding_long_breakpoint_intervals_df_csv_file_path),
        'minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path': minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path,
        'filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path': filtered_minimal_ir_pairs_linked_to_breakpoint_pairs_df_csv_file_path,
        'pairs_with_highest_confidence_bps_df_csv_file_path': pairs_with_highest_confidence_bps_df_csv_file_path,
        'filtered_pairs_with_highest_confidence_bps_df_csv_file_path': filtered_pairs_with_highest_confidence_bps_df_csv_file_path,
        'merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df_csv_file_path': (
            merged_cds_pair_region_with_num_of_pairs_with_high_confidence_bps_df_csv_file_path),
        'lens_of_spanning_regions_ratios_df_csv_file_path': lens_of_spanning_regions_ratios_df_csv_file_path,
        'repeat_with_dist_from_potential_breakpoints_df_csv_file_path': repeat_with_dist_from_potential_breakpoints_df_csv_file_path,
    }

    print(f'\nstage5_results_info:\n{stage5_results_info}\n')

    with open(stage5_results_info_pickle_file_path, 'wb') as f:
        pickle.dump(stage5_results_info, f, protocol=4)

    return stage5_results_info



def main():
    with generic_utils.timing_context_manager('massive_screening_stage_5.py'):

        if DO_STAGE5:
            do_massive_screening_stage5(
                search_for_pis_args=massive_screening_configuration.SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT,
            )

        print('\n')

if __name__ == '__main__':
    main()
