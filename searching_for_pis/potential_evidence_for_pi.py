import os
import pathlib
import pickle

import pandas as pd

from generic import blast_interface_and_utils
from generic import generic_utils
from generic import taxon_wgs_nuccore_entries


def estimate_potential_evidence_for_pi_for_each_taxon(
        taxon_uids,
        taxon_uid_to_taxon_info,
        taxa_out_dir_path,
        species_taxon_uid_to_more_taxon_uids_of_the_same_species,
        output_file_path_taxa_potential_evidence_for_pi_info_pickle,
        output_file_path_taxa_potential_evidence_for_pi_df_csv,
        search_for_pis_args,
):
    taxon_uid_to_taxon_potential_evidence_for_pi_info = {}
    num_of_taxa = len(taxon_uids)
    for i, taxon_uid in enumerate(taxon_uids):
        # START = 1.9e3
        # if i >= START + 0.1e3:
        #     exit()
        # if i < START:
        #     continue

        taxon_primary_assembly_nuccore_total_len = taxon_uid_to_taxon_info[taxon_uid]['primary_assembly_nuccore_total_len']

        generic_utils.print_and_write_to_log(f'taxon {i + 1}/{num_of_taxa} ({taxon_uid}). (estimate_potential_evidence_for_pi_for_each_taxon)')

        taxon_out_dir_path = os.path.join(taxa_out_dir_path, str(taxon_uid))
        pathlib.Path(taxon_out_dir_path).mkdir(parents=True, exist_ok=True)
        more_taxon_uids_of_the_same_species = species_taxon_uid_to_more_taxon_uids_of_the_same_species[taxon_uid]
        taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path = os.path.join(taxon_out_dir_path, 'local_blast_nt_database_nuccore_entries_info.pickle')
        if more_taxon_uids_of_the_same_species is None:
            with open(taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path, 'wb') as f:
                pickle.dump(None, f, protocol=4)
            taxon_local_blast_nt_database_nuccore_entries_info = None
        else:
            blast_interface_and_utils.build_taxon_local_blast_nt_database_nuccore_entries_info(
                local_blast_db_path=search_for_pis_args['local_blast_nt_database_path'],
                local_blast_nt_database_update_log_for_caching_only=generic_utils.read_text_file(search_for_pis_args['local_blast_nt_database_update_log_file_path']),
                taxon_uids=([taxon_uid] + more_taxon_uids_of_the_same_species),
                taxon_primary_assembly_nuccore_total_len=taxon_primary_assembly_nuccore_total_len,
                output_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle=taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path,
            )
            with open(taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path, 'rb') as f:
                taxon_local_blast_nt_database_nuccore_entries_info = pickle.load(f)

        if taxon_local_blast_nt_database_nuccore_entries_info is None:
            estimated_num_of_genomes_in_local_blast_nt_database = None
        else:
            estimated_num_of_genomes_in_local_blast_nt_database = taxon_local_blast_nt_database_nuccore_entries_info['estimated_num_of_genomes_in_nt']

        taxon_wgs_nuccore_entries_info_pickle_file_path = os.path.join(taxon_out_dir_path, 'wgs_nuccore_entries_info.pickle')
        taxon_wgs_nuccore_entries.write_taxon_wgs_nuccore_entries_info(
            taxon_uid=taxon_uid,
            taxon_primary_assembly_nuccore_total_len=taxon_primary_assembly_nuccore_total_len,
            min_wgs_nuccore_entry_len=search_for_pis_args['stage4']['min_wgs_nuccore_entry_len'],
            max_num_of_wgs_nuccore_entries=search_for_pis_args['stage4']['max_num_of_wgs_nuccore_entries_per_taxon_for_entrez_query'],
            taxon_out_dir_path=taxon_out_dir_path,
            output_file_path_taxon_wgs_nuccore_entries_info_pickle=taxon_wgs_nuccore_entries_info_pickle_file_path,
        )

        with open(taxon_wgs_nuccore_entries_info_pickle_file_path, 'rb') as f:
            taxon_wgs_nuccore_entries_info = pickle.load(f)
        estimated_num_of_genomes_in_wgs_entries = taxon_wgs_nuccore_entries_info['estimated_num_of_genomes_in_wgs']

        taxon_potential_evidence_for_pi_info = {
            'estimated_num_of_genomes_in_local_blast_nt_database': estimated_num_of_genomes_in_local_blast_nt_database,
            'estimated_num_of_genomes_in_wgs_entries': estimated_num_of_genomes_in_wgs_entries,
            'taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path': taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path,
            'taxon_wgs_nuccore_entries_info_pickle_file_path': taxon_wgs_nuccore_entries_info_pickle_file_path,
        }
        taxon_uid_to_taxon_potential_evidence_for_pi_info[taxon_uid] = taxon_potential_evidence_for_pi_info

    num_of_taxa_for_which_search_for_wgs_nuccore_entries_failed = sum((x['estimated_num_of_genomes_in_wgs_entries'] is None) for x in
                                                                      taxon_uid_to_taxon_potential_evidence_for_pi_info.values())
    num_of_taxa_for_which_search_for_local_blast_nuccore_entries_failed = sum((x['estimated_num_of_genomes_in_local_blast_nt_database'] is None) for x in
                                                                              taxon_uid_to_taxon_potential_evidence_for_pi_info.values())
    print(f'num_of_taxa_for_which_search_for_wgs_nuccore_entries_failed: {num_of_taxa_for_which_search_for_wgs_nuccore_entries_failed}')
    print(f'num_of_taxa_for_which_search_for_local_blast_nuccore_entries_failed: {num_of_taxa_for_which_search_for_local_blast_nuccore_entries_failed}')

    print(f'taxa_for_which_search_for_local_blast_nuccore_entries_failed:')
    print([taxon_uid for taxon_uid, x in taxon_uid_to_taxon_potential_evidence_for_pi_info.items()
           if x['estimated_num_of_genomes_in_local_blast_nt_database'] is None])

    flat_dicts = []
    for taxon_uid, taxon_potential_evidence_for_pi_info in taxon_uid_to_taxon_potential_evidence_for_pi_info.items():
        flat_dict = taxon_potential_evidence_for_pi_info.copy()
        flat_dict['taxon_uid'] = taxon_uid
        flat_dicts.append(flat_dict)
    taxa_potential_evidence_for_pi_df = pd.DataFrame(flat_dicts)
    taxa_potential_evidence_for_pi_df.sort_values(['estimated_num_of_genomes_in_local_blast_nt_database', 'taxon_uid'], ascending=False, inplace=True)
    taxa_potential_evidence_for_pi_df.to_csv(output_file_path_taxa_potential_evidence_for_pi_df_csv, sep='\t', index=False)

    taxa_potential_evidence_for_pi_info = {
        'taxon_uid_to_taxon_potential_evidence_for_pi_info': taxon_uid_to_taxon_potential_evidence_for_pi_info,
        'num_of_taxa_for_which_search_for_wgs_nuccore_entries_failed': num_of_taxa_for_which_search_for_wgs_nuccore_entries_failed,
        'num_of_taxa_for_which_search_for_local_blast_nuccore_entries_failed': num_of_taxa_for_which_search_for_local_blast_nuccore_entries_failed,
    }

    with open(output_file_path_taxa_potential_evidence_for_pi_info_pickle, 'wb') as f:
        pickle.dump(taxa_potential_evidence_for_pi_info, f, protocol=4)


    # taxon_uid_to_entries_in_blast_db_info
    # taxon_uid_to_wgs_entries_info


