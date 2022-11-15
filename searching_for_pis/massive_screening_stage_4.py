import logging
import os
import pathlib
import pickle

import numpy as np
import pandas as pd

from generic import blast_interface_and_utils
from generic import generic_utils
from searching_for_pis import massive_screening_stage_1
from searching_for_pis import massive_screening_stage_3
from searching_for_pis import massive_screening_configuration
from searching_for_pis import potential_evidence_for_pi

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

DO_STAGE4 = True
# DO_STAGE4 = False

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_relevant_taxon_uids(
        input_file_path_pairs_linked_df_csv,
        input_file_path_nuccore_df_csv,
        output_file_path_relevant_taxon_uids_pickle,
):
    pairs_linked_df = massive_screening_stage_3.get_pairs_linked_df_with_taxon_uid(
        pairs_linked_df_csv_file_path=input_file_path_pairs_linked_df_csv,
        nuccore_df_csv_file_path=input_file_path_nuccore_df_csv,
    )
    relevant_taxon_uids = sorted(int(x) for x in pairs_linked_df['taxon_uid'].unique())
    with open(output_file_path_relevant_taxon_uids_pickle, 'wb') as f:
        pickle.dump(relevant_taxon_uids, f, protocol=4)

def write_relevant_taxon_uids(
        input_file_path_pairs_linked_df_csv,
        input_file_path_nuccore_df_csv,
        output_file_path_relevant_taxon_uids_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_relevant_taxon_uids(
        input_file_path_pairs_linked_df_csv=input_file_path_pairs_linked_df_csv,
        input_file_path_nuccore_df_csv=input_file_path_nuccore_df_csv,
        output_file_path_relevant_taxon_uids_pickle=output_file_path_relevant_taxon_uids_pickle,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_species_taxon_uid_to_more_taxon_uids_of_the_same_species(
        input_file_path_relevant_taxon_uids_pickle,
        max_num_of_taxon_uids_of_the_same_species,
        taxa_out_dir_path,
        output_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with open(input_file_path_relevant_taxon_uids_pickle, 'rb') as f:
        relevant_taxon_uids = pickle.load(f)

    species_taxon_uid_to_more_taxon_uids_of_the_same_species = {}
    num_of_taxa = len(relevant_taxon_uids)
    for i, species_taxon_uid in enumerate(relevant_taxon_uids):
        # START = 26.7e3
        # if i >= START + 200:
        #     exit()
        # if i < START:
        #     continue

        generic_utils.print_and_write_to_log(f'taxon {i + 1}/{num_of_taxa} ({species_taxon_uid}) (cached_write_species_taxon_uid_to_more_taxon_uids_of_the_same_species).')

        taxon_dir_path = os.path.join(taxa_out_dir_path, str(species_taxon_uid))
        pathlib.Path(taxon_dir_path).mkdir(parents=True, exist_ok=True)
        uids_of_taxa_contained_in_species_pickle_file_path = os.path.join(taxon_dir_path, 'uids_of_taxa_contained_in_species.pickle')

        blast_interface_and_utils.write_uids_of_contained_taxa_according_to_blast_get_species_taxids(
            species_taxon_uid=species_taxon_uid,
            output_file_path_uids_of_contained_taxa_pickle=uids_of_taxa_contained_in_species_pickle_file_path,
        )
        with open(uids_of_taxa_contained_in_species_pickle_file_path, 'rb') as f:
            uids_of_taxa_contained_in_species = pickle.load(f)

        if uids_of_taxa_contained_in_species is None:
            species_taxon_uid_to_more_taxon_uids_of_the_same_species[species_taxon_uid] = None
        else:
            # print('species_taxon_uid, uids_of_taxa_contained_in_species')
            # print(species_taxon_uid, uids_of_taxa_contained_in_species)
            assert species_taxon_uid in uids_of_taxa_contained_in_species

            uids_of_taxa_contained_in_species.remove(species_taxon_uid)
            more_taxon_uids_of_the_same_species = uids_of_taxa_contained_in_species[:(max_num_of_taxon_uids_of_the_same_species - 1)]
            species_taxon_uid_to_more_taxon_uids_of_the_same_species[species_taxon_uid] = more_taxon_uids_of_the_same_species

            # if species_taxon_uid in uids_of_taxa_contained_in_species:
            #     uids_of_taxa_contained_in_species.remove(species_taxon_uid)
            #     more_taxon_uids_of_the_same_species = uids_of_taxa_contained_in_species[:(max_num_of_taxon_uids_of_the_same_species - 1)]
            # else:
            #     more_taxon_uids_of_the_same_species = uids_of_taxa_contained_in_species[:(max_num_of_taxon_uids_of_the_same_species - 1)]

    with open(output_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle, 'wb') as f:
        pickle.dump(species_taxon_uid_to_more_taxon_uids_of_the_same_species, f, protocol=4)

def write_species_taxon_uid_to_more_taxon_uids_of_the_same_species(
        input_file_path_relevant_taxon_uids_pickle,
        max_num_of_taxon_uids_of_the_same_species,
        taxa_out_dir_path,
        output_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_species_taxon_uid_to_more_taxon_uids_of_the_same_species(
        input_file_path_relevant_taxon_uids_pickle=input_file_path_relevant_taxon_uids_pickle,
        max_num_of_taxon_uids_of_the_same_species=max_num_of_taxon_uids_of_the_same_species,
        taxa_out_dir_path=taxa_out_dir_path,
        output_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle=(
            output_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle),
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )

def do_massive_screening_stage4(
        search_for_pis_args,
):
    massive_screening_stage4_out_dir_path = search_for_pis_args['stage4']['output_dir_path']

    debug___num_of_taxa_to_go_over = search_for_pis_args['debug___num_of_taxa_to_go_over']
    debug___taxon_uids = search_for_pis_args['debug___taxon_uids']

    stage_out_file_name_suffix = massive_screening_stage_1.get_stage_out_file_name_suffix(
        debug___num_of_taxa_to_go_over=debug___num_of_taxa_to_go_over,
        debug___taxon_uids=debug___taxon_uids,
    )

    stage4_results_info_pickle_file_path = os.path.join(massive_screening_stage4_out_dir_path, search_for_pis_args['stage4']['results_pickle_file_name'])
    stage4_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage4_results_info_pickle_file_path, stage_out_file_name_suffix)
    massive_screening_log_file_path = os.path.join(massive_screening_stage4_out_dir_path, 'massive_screening_stage4_log.txt')

    pathlib.Path(massive_screening_stage4_out_dir_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=massive_screening_log_file_path, level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug(f'---------------starting do_massive_screening_stage4({massive_screening_stage4_out_dir_path})---------------')

    massive_screening_stage1_out_dir_path = search_for_pis_args['stage1']['output_dir_path']
    stage1_results_info_pickle_file_path = os.path.join(massive_screening_stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])
    stage1_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage1_results_info_pickle_file_path, stage_out_file_name_suffix)
    with open(stage1_results_info_pickle_file_path, 'rb') as f:
        stage1_results_info = pickle.load(f)
    taxon_uid_to_taxon_info_pickle_file_path = stage1_results_info['taxon_uid_to_taxon_info_pickle_file_path']
    nuccore_df_csv_file_path = stage1_results_info['nuccore_df_csv_file_path']

    with open(taxon_uid_to_taxon_info_pickle_file_path, 'rb') as f:
        taxon_uid_to_taxon_info = pickle.load(f)

    stage3_out_dir_path = search_for_pis_args['stage3']['output_dir_path']
    stage3_results_info_pickle_file_path = os.path.join(stage3_out_dir_path, search_for_pis_args['stage3']['results_pickle_file_name'])
    stage3_results_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        stage3_results_info_pickle_file_path, stage_out_file_name_suffix)


    with open(stage3_results_info_pickle_file_path, 'rb') as f:
        stage3_results_info = pickle.load(f)
    pairs_linked_df_csv_file_path = stage3_results_info[
        'pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path']

    taxa_out_dir_path = os.path.join(massive_screening_stage4_out_dir_path, 'taxa')
    pathlib.Path(taxa_out_dir_path).mkdir(parents=True, exist_ok=True)


    relevant_taxon_uids_pickle_file_path = os.path.join(massive_screening_stage4_out_dir_path, 'relevant_taxon_uids.pickle')
    relevant_taxon_uids_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        relevant_taxon_uids_pickle_file_path, stage_out_file_name_suffix)
    write_relevant_taxon_uids(
        input_file_path_pairs_linked_df_csv=pairs_linked_df_csv_file_path,
        input_file_path_nuccore_df_csv=nuccore_df_csv_file_path,
        output_file_path_relevant_taxon_uids_pickle=relevant_taxon_uids_pickle_file_path,
    )
    with open(relevant_taxon_uids_pickle_file_path, 'rb') as f:
        relevant_taxon_uids = pickle.load(f)

    species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path = os.path.join(
        massive_screening_stage4_out_dir_path, 'species_taxon_uid_to_more_taxon_uids_of_the_same_species.pickle')
    species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path, stage_out_file_name_suffix)
    write_species_taxon_uid_to_more_taxon_uids_of_the_same_species(
        input_file_path_relevant_taxon_uids_pickle=relevant_taxon_uids_pickle_file_path,
        max_num_of_taxon_uids_of_the_same_species=search_for_pis_args['stage4']['max_num_of_taxon_uids_to_search_local_nt_per_taxon'],
        taxa_out_dir_path=taxa_out_dir_path,
        output_file_path_species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle=(
            species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path),
    )
    with open(species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path, 'rb') as f:
        species_taxon_uid_to_more_taxon_uids_of_the_same_species = pickle.load(f)

    taxa_potential_evidence_for_pi_info_pickle_file_path = os.path.join(massive_screening_stage4_out_dir_path, 'taxa_potential_evidence_for_pi_info.pickle')
    taxa_potential_evidence_for_pi_info_pickle_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        taxa_potential_evidence_for_pi_info_pickle_file_path, stage_out_file_name_suffix)
    taxa_potential_evidence_for_pi_df_csv_file_path = os.path.join(massive_screening_stage4_out_dir_path, 'taxa_potential_evidence_for_pi_df.csv')
    taxa_potential_evidence_for_pi_df_csv_file_path = generic_utils.add_suffix_to_file_name_while_keeping_extension(
        taxa_potential_evidence_for_pi_df_csv_file_path, stage_out_file_name_suffix)
    potential_evidence_for_pi.estimate_potential_evidence_for_pi_for_each_taxon(
        taxon_uids=relevant_taxon_uids,
        taxon_uid_to_taxon_info=taxon_uid_to_taxon_info,
        taxa_out_dir_path=taxa_out_dir_path,
        species_taxon_uid_to_more_taxon_uids_of_the_same_species=species_taxon_uid_to_more_taxon_uids_of_the_same_species,
        output_file_path_taxa_potential_evidence_for_pi_info_pickle=taxa_potential_evidence_for_pi_info_pickle_file_path,
        output_file_path_taxa_potential_evidence_for_pi_df_csv=taxa_potential_evidence_for_pi_df_csv_file_path,
        search_for_pis_args=search_for_pis_args,
    )

    stage4_results_info = {
        'taxa_potential_evidence_for_pi_info_pickle_file_path': taxa_potential_evidence_for_pi_info_pickle_file_path,
        'taxa_potential_evidence_for_pi_df_csv_file_path': taxa_potential_evidence_for_pi_df_csv_file_path,
        'relevant_taxon_uids_pickle_file_path': relevant_taxon_uids_pickle_file_path,
        'species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path': species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path,
    }
    with open(stage4_results_info_pickle_file_path, 'wb') as f:
        pickle.dump(stage4_results_info, f, protocol=4)

    print(f'created {stage4_results_info_pickle_file_path} successfully')

    return stage4_results_info


def main():
    with generic_utils.timing_context_manager('massive_screening_stage_4.py'):
        if DO_STAGE4:
            do_massive_screening_stage4(
                search_for_pis_args=massive_screening_configuration.SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT,
            )

        print('\n')

if __name__ == '__main__':
    main()
