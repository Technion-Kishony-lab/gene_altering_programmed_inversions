import logging
import pathlib
import os
import os.path
import pickle
import random
import shutil
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from generic import bio_utils
from generic import blast_interface_and_utils
from generic import find_best_assemblies_accessions
from generic import generic_utils
from generic import ncbi_genome_download_interface
from generic import seq_feature_utils
from searching_for_pis import massive_screening_configuration

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

DO_STAGE1 = True
# DO_STAGE1 = False


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_entry_info(
        nuccore_entry_output_dir_path,
        input_file_path_gbff,
        taxon_uid,
        max_total_dist_between_joined_parts_per_joined_feature,
        nuccore_accession,
        output_file_path_nuccore_entry_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    fasta_file_path = os.path.join(nuccore_entry_output_dir_path, f'sequence.fasta')
    chrom_len_file_path = os.path.join(nuccore_entry_output_dir_path, f'len.txt')
    cds_features_gb_file_path = os.path.join(nuccore_entry_output_dir_path, f'cds_features.gb')
    num_of_filtered_cds_features_file_path = os.path.join(nuccore_entry_output_dir_path, f'num_of_filtered_cds_features.txt')

    if os.path.isfile(fasta_file_path) and os.path.isfile(chrom_len_file_path):
        generic_utils.print_and_write_to_log(f'skipping writing fasta and chrom_len files as we already did this earlier.')
    else:
        gb_record = bio_utils.get_gb_record(input_file_path_gbff)
        chrom_len = len(gb_record.seq)
        assert isinstance(chrom_len, int)
        generic_utils.write_text_file(chrom_len_file_path, str(int(chrom_len)))

        bio_utils.write_records_to_fasta_or_gb_file(gb_record, fasta_file_path, 'fasta')

    blast_interface_and_utils.make_blast_nucleotide_db(fasta_file_path)
    seq_feature_utils.discard_joined_features_with_large_total_dist_between_joined_parts(
        input_file_path_gb=input_file_path_gbff,
        max_total_dist_between_joined_parts=max_total_dist_between_joined_parts_per_joined_feature,
        discard_non_cds=True,
        output_file_path_filtered_gb=cds_features_gb_file_path,
        output_file_path_num_of_filtered_features=num_of_filtered_cds_features_file_path,
    )
    num_of_filtered_cds_features = int(generic_utils.read_text_file(num_of_filtered_cds_features_file_path))

    chrom_len = int(generic_utils.read_text_file(chrom_len_file_path))
    cds_df_csv_file_path = f'{cds_features_gb_file_path}.csv'

    nuccore_entry_info = {
        'nuccore_accession': nuccore_accession, # just for convenience
        'taxon_uid': taxon_uid,
        'gbff_file_path': input_file_path_gbff,
        'num_of_filtered_cds_features': num_of_filtered_cds_features,
        'cds_features_gb_file_path': cds_features_gb_file_path,
        'cds_df_csv_file_path': cds_df_csv_file_path,
        'fasta_file_path': fasta_file_path,
        'chrom_len': chrom_len,
    }

    with open(output_file_path_nuccore_entry_info_pickle, 'wb') as f:
        pickle.dump(nuccore_entry_info, f, protocol=4)

def write_nuccore_entry_info(
        nuccore_entry_output_dir_path,
        input_file_path_gbff,
        taxon_uid,
        max_total_dist_between_joined_parts_per_joined_feature,
        nuccore_accession,
        output_file_path_nuccore_entry_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_entry_info(
        nuccore_entry_output_dir_path=nuccore_entry_output_dir_path,
        input_file_path_gbff=input_file_path_gbff,
        taxon_uid=taxon_uid,
        max_total_dist_between_joined_parts_per_joined_feature=max_total_dist_between_joined_parts_per_joined_feature,
        nuccore_accession=nuccore_accession,
        output_file_path_nuccore_entry_info_pickle=output_file_path_nuccore_entry_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_cds_df_csv(
        nuccore_accession,
        input_file_path_nuccore_cds_features_gb,
        output_file_path_nuccore_cds_df_csv,
):
    cds_features = seq_feature_utils.get_cds_seq_features(input_file_path_nuccore_cds_features_gb)
    # num_of_cds_features = len(cds_features)
    cds_flat_dicts = []
    for index_in_nuccore_cds_features_gb_file, cds_feature in enumerate(cds_features):
        start_pos = seq_feature_utils.get_feature_start_pos(cds_feature)
        end_pos = seq_feature_utils.get_feature_end_pos(cds_feature)
        strand = cds_feature.location.strand
        product = seq_feature_utils.get_product_qualifier(cds_feature)

        # prev_cds_i = index_in_nuccore_cds_features_gb_file - 1
        # if prev_cds_i >= 0:
        #     prev_cds = cds_features[prev_cds_i]
        #     assert seq_feature_utils.get_feature_start_pos(prev_cds) <= start_pos
        #     prev_cds_strand = prev_cds.location.strand
        #     dist_from_prev_cds = start_pos - seq_feature_utils.get_feature_end_pos(prev_cds) - 1
        # else:
        #     prev_cds_strand = np.nan
        #     dist_from_prev_cds = np.nan
        #
        # next_cds_i = index_in_nuccore_cds_features_gb_file + 1
        # if next_cds_i < num_of_cds_features:
        #     next_cds = cds_features[next_cds_i]
        #     assert seq_feature_utils.get_feature_start_pos(next_cds) >= start_pos
        #     next_cds_strand = next_cds.location.strand
        #     dist_from_next_cds = seq_feature_utils.get_feature_start_pos(next_cds) - end_pos - 1
        # else:
        #     next_cds_strand = np.nan
        #     dist_from_next_cds = np.nan

        cds_flat_dict = {
            'nuccore_accession': nuccore_accession,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'strand': strand,
            'product': product,
            'index_in_nuccore_cds_features_gb_file': index_in_nuccore_cds_features_gb_file,
        }
        cds_flat_dicts.append(cds_flat_dict)
    nuccore_cds_df = pd.DataFrame(cds_flat_dicts)

    nuccore_cds_df.to_csv(output_file_path_nuccore_cds_df_csv, sep='\t', index=False)

def write_nuccore_cds_df_csv(
        nuccore_accession,
        input_file_path_nuccore_cds_features_gb,
        output_file_path_nuccore_cds_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_cds_df_csv(
        nuccore_accession=nuccore_accession,
        input_file_path_nuccore_cds_features_gb=input_file_path_nuccore_cds_features_gb,
        output_file_path_nuccore_cds_df_csv=output_file_path_nuccore_cds_df_csv,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_all_cds_df_csv(
        input_file_path_nuccore_df_csv,
        output_file_path_all_cds_df_csv,
):
    nuccore_df = pd.read_csv(input_file_path_nuccore_df_csv, sep='\t', low_memory=False)
    num_of_nuccore_entries = len(nuccore_df)
    nuccore_cds_df_csv_file_paths = []
    for i, (_, row) in enumerate(nuccore_df.iterrows()):
        num_of_filtered_cds_features = row['num_of_filtered_cds_features']
        if num_of_filtered_cds_features > 0:
            nuccore_accession = row['nuccore_accession']
            generic_utils.print_and_write_to_log(f'starting work on nuccore {i + 1}/{num_of_nuccore_entries}: {nuccore_accession}')
            nuccore_cds_features_gb_file_path = row['cds_features_gb_file_path']
            nuccore_cds_df_csv_file_path = row['cds_df_csv_file_path']
            write_nuccore_cds_df_csv(
                nuccore_accession=nuccore_accession,
                input_file_path_nuccore_cds_features_gb=nuccore_cds_features_gb_file_path,
                output_file_path_nuccore_cds_df_csv=nuccore_cds_df_csv_file_path,
            )
            nuccore_cds_df_csv_file_paths.append(nuccore_cds_df_csv_file_path)

    cds_df = pd.concat((pd.read_csv(x, sep='\t') for x in nuccore_cds_df_csv_file_paths), ignore_index=True)
    cds_df.to_csv(output_file_path_all_cds_df_csv, sep='\t', index=False)


def write_all_cds_df_csv(
        input_file_path_nuccore_df_csv,
        output_file_path_all_cds_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_all_cds_df_csv(
        input_file_path_nuccore_df_csv=input_file_path_nuccore_df_csv,
        output_file_path_all_cds_df_csv=output_file_path_all_cds_df_csv,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_best_assembly_accessions_pickle(
        input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        taxon_uid,
        output_file_path_best_assembly_accessions_pickle,
):
    with open(input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle, 'rb') as f:
        species_taxon_uid_to_best_assembly_accessions = pickle.load(f)

    best_assembly_accessions = species_taxon_uid_to_best_assembly_accessions[taxon_uid]
    with open(output_file_path_best_assembly_accessions_pickle, 'wb') as f:
        pickle.dump(best_assembly_accessions, f, protocol=4)

def write_best_assembly_accessions_pickle(
        input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        taxon_uid,
        output_file_path_best_assembly_accessions_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_best_assembly_accessions_pickle(
        input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle=input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        taxon_uid=taxon_uid,
        output_file_path_best_assembly_accessions_pickle=output_file_path_best_assembly_accessions_pickle,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_taxon_info(
        taxon_uid,
        primary_assembly_accession,
        taxon_dir_path,
        input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        max_total_dist_between_joined_parts_per_joined_feature,
        output_file_path_taxon_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    best_assembly_accessions_pickle_file_path = os.path.join(taxon_dir_path, 'best_assembly_accessions.pickle')
    write_best_assembly_accessions_pickle(
        input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle=input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        taxon_uid=taxon_uid,
        output_file_path_best_assembly_accessions_pickle=best_assembly_accessions_pickle_file_path,
    )

    taxon_lineage_pickle_file_path = os.path.join(taxon_dir_path, 'taxon_lineage.pickle')
    bio_utils.write_taxon_lineage(
        taxon_uid=taxon_uid,
        output_file_path_taxon_lineage_pickle=taxon_lineage_pickle_file_path,
    )
    with open(taxon_lineage_pickle_file_path, 'rb') as f:
        taxon_lineage_info = pickle.load(f)
    lineage_rank_to_scientific_name = bio_utils.get_rank_to_scientific_name_from_lineage_info(taxon_lineage_info)

    downloaded_assemblies_dir_path = os.path.join(taxon_dir_path, 'downloaded_assemblies')
    pathlib.Path(downloaded_assemblies_dir_path).mkdir(parents=True, exist_ok=True)

    primary_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle_file_path = os.path.join(
        downloaded_assemblies_dir_path, 'primary_assembly_accession_to_nuccore_accession_to_gbff_file_path.pickle')

    with generic_utils.timing_context_manager('download_bacteria_assemblies_and_ungz_and_get_assembly_accession_to_nuccore_accession_to_gbff_file_path'):
        ncbi_genome_download_interface.download_bacteria_assemblies_and_ungz_and_get_assembly_accession_to_nuccore_accession_to_gbff_file_path(
            assemblies_accessions=[primary_assembly_accession],
            output_dir_path=downloaded_assemblies_dir_path,
            output_file_path_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle=primary_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle_file_path,
        )
    with open(primary_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle_file_path, 'rb') as f:
        primary_assembly_accession_to_nuccore_accession_to_gbff_file_path = pickle.load(f)

    # print(primary_assembly_accession_to_nuccore_accession_to_gbff_file_path)
    assert len(primary_assembly_accession_to_nuccore_accession_to_gbff_file_path) == 1
    nuccore_accession_to_gbff_file_path = next(iter(primary_assembly_accession_to_nuccore_accession_to_gbff_file_path.values()))

    primary_assembly_nuccore_accessions = set(nuccore_accession_to_gbff_file_path)
    primary_assembly_nuccore_accessions_pickle_file_path = os.path.join(taxon_dir_path, 'primary_assembly_nuccore_accessions.pickle')
    with open(primary_assembly_nuccore_accessions_pickle_file_path, 'wb') as f:
        pickle.dump(primary_assembly_nuccore_accessions, f, protocol=4)

    num_of_primary_assembly_nuccore_accessions = len(nuccore_accession_to_gbff_file_path)
    taxon_nuccore_accession_to_nuccore_entry_info = {}
    curr_taxon_fasta_file_paths = []
    for j, (nuccore_accession, gbff_file_path) in enumerate(sorted(nuccore_accession_to_gbff_file_path.items())):
        generic_utils.print_and_write_to_log(f'starting work on nuccore {j + 1}/{num_of_primary_assembly_nuccore_accessions}: {nuccore_accession}')

        nuccore_entry_output_dir_path = os.path.join(taxon_dir_path, 'primary_assembly_nuccores', nuccore_accession)
        pathlib.Path(nuccore_entry_output_dir_path).mkdir(parents=True, exist_ok=True)
        nuccore_entry_info_pickle_file_path = os.path.join(nuccore_entry_output_dir_path, 'nuccore_entry_info.pickle')

        write_nuccore_entry_info(
            nuccore_entry_output_dir_path=nuccore_entry_output_dir_path,
            input_file_path_gbff=gbff_file_path,
            taxon_uid=taxon_uid,
            max_total_dist_between_joined_parts_per_joined_feature=max_total_dist_between_joined_parts_per_joined_feature,
            nuccore_accession=nuccore_accession, # just for convenience
            output_file_path_nuccore_entry_info_pickle=nuccore_entry_info_pickle_file_path,
        )
        with open(nuccore_entry_info_pickle_file_path, 'rb') as f:
            nuccore_entry_info = pickle.load(f)

        assert 'nuccore_accession' in nuccore_entry_info # important for the creation of nuccore_df later.
        taxon_nuccore_accession_to_nuccore_entry_info[nuccore_accession] = nuccore_entry_info
        curr_taxon_fasta_file_paths.append(nuccore_entry_info['fasta_file_path'])

    nuccore_entry_lens = [taxon_nuccore_accession_to_nuccore_entry_info[x]['chrom_len'] for x in primary_assembly_nuccore_accessions]
    primary_assembly_nuccore_total_len = sum(nuccore_entry_lens)
    primary_assembly_nuccore_median_len = np.median(nuccore_entry_lens)
    num_of_nuccore_entries = len(primary_assembly_nuccore_accessions)
    assert len(curr_taxon_fasta_file_paths) == len(set(curr_taxon_fasta_file_paths))
    taxon_blast_db_path = os.path.join(taxon_dir_path, 'taxon_blast_db')
    blast_interface_and_utils.make_blast_nucleotide_db_for_multiple_fasta_files(sorted(curr_taxon_fasta_file_paths), taxon_blast_db_path)

    taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path = os.path.join(taxon_dir_path, 'taxon_nuccore_accession_to_nuccore_entry_info.pickle')
    with open(taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path, 'wb') as f:
        pickle.dump(taxon_nuccore_accession_to_nuccore_entry_info, f, protocol=4)

    taxon_info = {
        'taxon_dir_path': taxon_dir_path,
        'best_assembly_accessions_pickle_file_path': best_assembly_accessions_pickle_file_path,
        'primary_assembly_accession': primary_assembly_accession,
        'taxon_lineage_pickle_file_path': taxon_lineage_pickle_file_path,
        'lineage_rank_to_scientific_name': lineage_rank_to_scientific_name,
        'taxon_blast_db_path': taxon_blast_db_path,
        'num_of_nuccore_entries': num_of_nuccore_entries,
        'primary_assembly_nuccore_total_len': primary_assembly_nuccore_total_len,
        'primary_assembly_nuccore_median_len': primary_assembly_nuccore_median_len,
        'primary_assembly_nuccore_accessions_pickle_file_path': primary_assembly_nuccore_accessions_pickle_file_path,
        'taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path': taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path,
    }

    with open(output_file_path_taxon_info_pickle, 'wb') as f:
        pickle.dump(taxon_info, f, protocol=4)

def write_taxon_info(
        taxon_uid,
        primary_assembly_accession,
        taxon_dir_path,
        input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        max_total_dist_between_joined_parts_per_joined_feature,
        output_file_path_taxon_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_taxon_info(
        taxon_uid=taxon_uid,
        primary_assembly_accession=primary_assembly_accession,
        taxon_dir_path=taxon_dir_path,
        input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle=input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        max_total_dist_between_joined_parts_per_joined_feature=max_total_dist_between_joined_parts_per_joined_feature,
        output_file_path_taxon_info_pickle=output_file_path_taxon_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_accession_to_nuccore_entry_info(
        input_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with open(input_file_path_taxon_uid_to_taxon_info_pickle, 'rb') as f:
        taxon_uid_to_taxon_info = pickle.load(f)

    nuccore_accession_to_nuccore_entry_info = {}
    for taxon_info in taxon_uid_to_taxon_info.values():
        if taxon_info is not None:
            with open(taxon_info['taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path'], 'rb') as f:
                taxon_nuccore_accession_to_nuccore_entry_info = pickle.load(f)
            nuccore_accession_to_nuccore_entry_info = {**taxon_nuccore_accession_to_nuccore_entry_info, **nuccore_accession_to_nuccore_entry_info}

    with open(output_file_path_nuccore_accession_to_nuccore_entry_info_pickle, 'wb') as f:
        pickle.dump(nuccore_accession_to_nuccore_entry_info, f, protocol=4)

def write_nuccore_accession_to_nuccore_entry_info(
        input_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_accession_to_nuccore_entry_info(
        input_file_path_taxon_uid_to_taxon_info_pickle=input_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle=output_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_taxa_df(
        input_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_taxa_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with open(input_file_path_taxon_uid_to_taxon_info_pickle, 'rb') as f:
        taxon_uid_to_taxon_info = pickle.load(f)

    taxon_info_flat_dicts = []
    for taxon_uid, taxon_info in taxon_uid_to_taxon_info.items():
        if taxon_info is not None:
            flat_dict = {'taxon_uid': taxon_uid}
            for k1, v1 in taxon_info.items():
                assert k1 not in flat_dict

                if isinstance(v1, dict):
                    assert k1 == 'lineage_rank_to_scientific_name'
                    for k2, v2 in v1.items():
                        new_key = f'taxon_{k2}'
                        assert new_key not in flat_dict
                        flat_dict[new_key] = v2
                else:
                    flat_dict[k1] = v1
            taxon_info_flat_dicts.append(flat_dict)

    taxa_df = pd.DataFrame(taxon_info_flat_dicts)
    taxa_df.to_csv(output_file_path_taxa_df_csv, sep='\t', index=False)

def write_taxa_df(
        input_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_taxa_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_taxa_df(
        input_file_path_taxon_uid_to_taxon_info_pickle=input_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_taxa_df_csv=output_file_path_taxa_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

def get_stage_out_file_name_suffix(debug___num_of_taxa_to_go_over=None, debug___taxon_uids=None):
    if debug___num_of_taxa_to_go_over is not None:
        assert debug___taxon_uids is None
        stage_out_file_name_suffix = f'_{debug___num_of_taxa_to_go_over}'
    elif debug___taxon_uids is not None:
        assert debug___num_of_taxa_to_go_over is None
        stage_out_file_name_suffix = f'_debug_{len(debug___taxon_uids)}'
    else:
        stage_out_file_name_suffix = ''
    return stage_out_file_name_suffix

def do_massive_screening_stage1(search_for_pis_args):
    massive_screening_stage1_out_dir_path = search_for_pis_args['stage1']['output_dir_path']

    debug___taxon_uid_to_forced_best_assembly_accession = search_for_pis_args['debug___taxon_uid_to_forced_best_assembly_accession']
    debug___num_of_taxa_to_go_over = search_for_pis_args['debug___num_of_taxa_to_go_over']
    debug___taxon_uids = search_for_pis_args['debug___taxon_uids']

    stage_out_file_name_suffix = get_stage_out_file_name_suffix(
        debug___num_of_taxa_to_go_over=debug___num_of_taxa_to_go_over,
        debug___taxon_uids=debug___taxon_uids,
    )

    stage1_results_info_pickle_file_path = os.path.join(massive_screening_stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])
    stage1_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage1_results_info_pickle_file_path, stage_out_file_name_suffix)
    massive_screening_log_file_path = os.path.join(massive_screening_stage1_out_dir_path, 'massive_screening_stage1_log.txt')

    pathlib.Path(massive_screening_stage1_out_dir_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=massive_screening_log_file_path, level=logging.DEBUG, format='%(asctime)s %(message)s')
    generic_utils.print_and_write_to_log(f'---------------starting do_massive_screening_stage1({massive_screening_stage1_out_dir_path})---------------')


    refseq_bacteria_assembly_summary_file_path = os.path.join(massive_screening_stage1_out_dir_path, 'refseq_bacteria_assembly_summary.txt')
    if (not search_for_pis_args['stage1']['use_cached_refseq_bacteria_assembly_summary_file']) or (not os.path.isfile(refseq_bacteria_assembly_summary_file_path)):
        with urllib.request.urlopen(search_for_pis_args['stage1']['refseq_bacteria_assembly_summary_file_url']) as url_f:
            with open(refseq_bacteria_assembly_summary_file_path, 'wb') as f:
                shutil.copyfileobj(url_f, f)

    allowed_assembly_levels_dir_name = '__'.join(search_for_pis_args['stage1']['allowed_assembly_level_values_sorted_by_preference']).replace(' ', '_')
    allowed_assembly_levels_dir_path = os.path.join(massive_screening_stage1_out_dir_path, allowed_assembly_levels_dir_name)
    pathlib.Path(allowed_assembly_levels_dir_path).mkdir(parents=True, exist_ok=True)

    species_taxon_uid_to_best_assembly_accessions_pickle_file_path = os.path.join(allowed_assembly_levels_dir_path, f'species_taxon_uid_to_best_assembly_accessions.pickle')
    find_best_assemblies_accessions.write_species_taxon_uid_to_best_assembly_accessions_pickle(
        input_file_path_refseq_assembly_summary=refseq_bacteria_assembly_summary_file_path,
        allowed_assembly_level_values_sorted_by_preference=search_for_pis_args['stage1']['allowed_assembly_level_values_sorted_by_preference'],
        max_num_of_assemblies_per_species=1,
        output_file_path_species_taxon_uid_to_best_assembly_accessions_pickle=species_taxon_uid_to_best_assembly_accessions_pickle_file_path,
        debug___taxon_uid_to_forced_best_assembly_accession=debug___taxon_uid_to_forced_best_assembly_accession,
    )
    with open(species_taxon_uid_to_best_assembly_accessions_pickle_file_path, 'rb') as f:
        species_taxon_uid_to_best_assembly_accessions = pickle.load(f)

    species_taxa_dir_path = os.path.join(massive_screening_stage1_out_dir_path, 'species_taxa')
    pathlib.Path(species_taxa_dir_path).mkdir(parents=True, exist_ok=True)

    # print(f'species_taxon_uid_to_best_assembly_accessions:\n{species_taxon_uid_to_best_assembly_accessions}')
    # exit()
    species_taxon_uid_to_primary_assembly_accession = {
        species_taxon_uid: best_assemblies_accessions[0]
        for species_taxon_uid, best_assemblies_accessions in species_taxon_uid_to_best_assembly_accessions.items()
    }
    print(f'len(species_taxon_uid_to_primary_assembly_accession): {len(species_taxon_uid_to_primary_assembly_accession)}')

    # print('species_taxon_uid_to_primary_assembly_accession')
    # print(species_taxon_uid_to_primary_assembly_accession)
    if debug___taxon_uids:
        species_taxon_uid_to_primary_assembly_accession = {
            taxon_uid: species_taxon_uid_to_primary_assembly_accession[taxon_uid]
            for taxon_uid in debug___taxon_uids
        }


    # species_taxon_uid_to_primary_assembly_accession = {
    #     L_RHAMNOSUS_TAXON_UID: species_taxon_uid_to_primary_assembly_accession[L_RHAMNOSUS_TAXON_UID],
    #     # PNEUMOCOCCUS_TAXON_UID: species_taxon_uid_to_primary_assembly_accession[PNEUMOCOCCUS_TAXON_UID],
    #     # E_COLI_TAXON_UID: species_taxon_uid_to_primary_assembly_accession[E_COLI_TAXON_UID],
    #     # 139: species_taxon_uid_to_primary_assembly_accession[139],
    # }



    num_of_taxa = len(species_taxon_uid_to_primary_assembly_accession)
    taxa_for_which_download_of_primary_assembly_nuccore_entries_failed_uids = set()

    taxon_uid_to_taxon_info = {}

    random.seed(0) # for reproducibility
    shuffled_species_taxon_uid_and_primary_assembly_accession = sorted(species_taxon_uid_to_primary_assembly_accession.items())
    random.shuffle(shuffled_species_taxon_uid_and_primary_assembly_accession)

    for i, (taxon_uid, primary_assembly_accession) in enumerate(shuffled_species_taxon_uid_and_primary_assembly_accession):
        # START = 30364
        # START = 25364
        # START = 20364
        # START = 15364
        # START = 10364
        # START = 5364
        # START = 3000
        # START = 0
        # START = 8000
        # START = 13000
        # START = 18000
        # START = 23000
        # START = 28000
        # START = 33000
        # # if i >= START + 5e3:
        # START = 34800
        # if i >= START + 500:
        #     exit()
        # if i < START:
        #     continue

        taxon_dir_path = os.path.join(species_taxa_dir_path, str(taxon_uid))
        pathlib.Path(taxon_dir_path).mkdir(parents=True, exist_ok=True)

        if (debug___num_of_taxa_to_go_over is not None) and (i >= debug___num_of_taxa_to_go_over):
            taxon_info = None
            # taxon_info = {
            #     'taxon_dir_path': taxon_dir_path,
            #     'best_assembly_accessions_pickle_file_path': None,
            #     'taxon_lineage_pickle_file_path': None,
            #     'primary_assembly_accession': primary_assembly_accession,
            #     'lineage_rank_to_scientific_name': None,
            #     'taxon_blast_db_path': None,
            #     'num_of_nuccore_entries': None,
            #     'primary_assembly_nuccore_total_len': None,
            #     'primary_assembly_nuccore_median_len': None,
            #     'primary_assembly_nuccore_accessions_pickle_file_path': None,
            #     'taxon_nuccore_accession_to_nuccore_entry_info_pickle_file_path': None,
            # }
        else:
            # if taxon_uid != 195064:
            #     continue
            # if i < 24105:
            #     continue
            generic_utils.print_and_write_to_log(f'starting work on taxon {i + 1}/{num_of_taxa}: {taxon_uid}')

            taxon_info_pickle_file_path = os.path.join(taxon_dir_path, 'taxon_info.pickle')
            try:
                write_taxon_info(
                    taxon_uid=taxon_uid,
                    primary_assembly_accession=primary_assembly_accession,
                    taxon_dir_path=taxon_dir_path,
                    input_file_path_species_taxon_uid_to_best_assembly_accessions_pickle=species_taxon_uid_to_best_assembly_accessions_pickle_file_path,
                    max_total_dist_between_joined_parts_per_joined_feature=search_for_pis_args['max_total_dist_between_joined_parts_per_joined_feature'],
                    output_file_path_taxon_info_pickle=taxon_info_pickle_file_path,
                )
            except ncbi_genome_download_interface.NcbiGenomeDownloadSubprocessError:
                generic_utils.print_and_write_to_log(f'failed to download nuccore entries of primary assembly {primary_assembly_accession} (for taxon {taxon_uid}).')
                taxa_for_which_download_of_primary_assembly_nuccore_entries_failed_uids.add(taxon_uid)
                continue

            with open(taxon_info_pickle_file_path, 'rb') as f:
                taxon_info = pickle.load(f)

        taxon_uid_to_taxon_info[taxon_uid] = taxon_info

    taxon_uid_to_taxon_info_pickle_file_path = os.path.join(massive_screening_stage1_out_dir_path, 'taxon_uid_to_taxon_info.pickle')
    taxon_uid_to_taxon_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        taxon_uid_to_taxon_info_pickle_file_path, stage_out_file_name_suffix)
    with open(taxon_uid_to_taxon_info_pickle_file_path, 'wb') as f:
        pickle.dump(taxon_uid_to_taxon_info, f, protocol=4)

    taxa_df_csv_file_path = os.path.join(massive_screening_stage1_out_dir_path, 'taxa_df.csv')
    taxa_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        taxa_df_csv_file_path, stage_out_file_name_suffix)
    write_taxa_df(
        input_file_path_taxon_uid_to_taxon_info_pickle=taxon_uid_to_taxon_info_pickle_file_path,
        output_file_path_taxa_df_csv=taxa_df_csv_file_path,
    )

    nuccore_accession_to_nuccore_entry_info_pickle_file_path = os.path.join(massive_screening_stage1_out_dir_path, 'nuccore_accession_to_nuccore_entry_info.pickle')
    nuccore_accession_to_nuccore_entry_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        nuccore_accession_to_nuccore_entry_info_pickle_file_path, stage_out_file_name_suffix)
    write_nuccore_accession_to_nuccore_entry_info(
        input_file_path_taxon_uid_to_taxon_info_pickle=taxon_uid_to_taxon_info_pickle_file_path,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle=nuccore_accession_to_nuccore_entry_info_pickle_file_path,
    )

    with open(nuccore_accession_to_nuccore_entry_info_pickle_file_path, 'rb') as f:
        nuccore_accession_to_nuccore_entry_info = pickle.load(f)

    nuccore_df = pd.DataFrame(nuccore_accession_to_nuccore_entry_info.values())
    nuccore_df_csv_file_path = os.path.join(massive_screening_stage1_out_dir_path, 'nuccore_df.csv')
    nuccore_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        nuccore_df_csv_file_path, stage_out_file_name_suffix)
    nuccore_df.to_csv(nuccore_df_csv_file_path, sep='\t', index=False)

    all_cds_df_csv_file_path = os.path.join(massive_screening_stage1_out_dir_path, 'all_cds_df.csv')
    all_cds_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        all_cds_df_csv_file_path, stage_out_file_name_suffix)
    write_all_cds_df_csv(
        input_file_path_nuccore_df_csv=nuccore_df_csv_file_path,
        output_file_path_all_cds_df_csv=all_cds_df_csv_file_path,
    )

    taxa_df = pd.read_csv(taxa_df_csv_file_path, sep='\t', low_memory=False)
    num_of_taxa = len(taxa_df)
    print(f'num_of_taxa: {num_of_taxa}')

    all_cds_df = pd.read_csv(all_cds_df_csv_file_path, sep='\t', low_memory=False)
    num_of_all_cds = len(all_cds_df)
    print(f'num_of_all_cds: {num_of_all_cds}')

    stage1_results_info = {
        'massive_screening_log_file_path': massive_screening_log_file_path,
        'allowed_assembly_levels_dir_path': allowed_assembly_levels_dir_path,
        'species_taxa_dir_path': species_taxa_dir_path,
        'taxon_uid_to_taxon_info_pickle_file_path': taxon_uid_to_taxon_info_pickle_file_path,
        'nuccore_accession_to_nuccore_entry_info_pickle_file_path': nuccore_accession_to_nuccore_entry_info_pickle_file_path,
        'taxa_for_which_download_of_primary_assembly_nuccore_entries_failed_uids': taxa_for_which_download_of_primary_assembly_nuccore_entries_failed_uids,
        'taxa_df_csv_file_path': taxa_df_csv_file_path,
        'nuccore_df_csv_file_path': nuccore_df_csv_file_path,
        'all_cds_df_csv_file_path': all_cds_df_csv_file_path,
        'refseq_bacteria_assembly_summary_file_path': refseq_bacteria_assembly_summary_file_path,

        'num_of_taxa': num_of_taxa,
        'num_of_all_cds': num_of_all_cds,
    }
    with open(stage1_results_info_pickle_file_path, 'wb') as f:
        pickle.dump(stage1_results_info, f, protocol=4)

    return stage1_results_info



def main():
    if DO_STAGE1:
        with generic_utils.timing_context_manager('do_massive_screening_stage1'):
            do_massive_screening_stage1(
                search_for_pis_args=massive_screening_configuration.SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT,
            )


if __name__ == '__main__':
    main()
