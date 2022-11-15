import os
import os.path
import pathlib
import pickle

import numpy as np
import pandas as pd

from generic import generic_utils
from searching_for_pis import find_ir_pairs_in_nuccore_entry

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


def find_all_ir_pairs(
        stage1_results_info,
        output_dir_path,
        search_for_pis_args,
        debug___num_of_nuccore_entries_to_go_over,
):
    with generic_utils.timing_context_manager('find_all_ir_pairs'):
        nuccore_accession_to_nuccore_entry_info_pickle_file_path = stage1_results_info['nuccore_accession_to_nuccore_entry_info_pickle_file_path']
        with open(nuccore_accession_to_nuccore_entry_info_pickle_file_path, 'rb') as f:
            nuccore_accession_to_nuccore_entry_info = pickle.load(f)
        orig_num_of_nuccore_entries = len(nuccore_accession_to_nuccore_entry_info)
        nuccore_accession_to_ir_pairs_info = {}
        filtered_nuccore_accession_to_ir_pairs_info = {}
        total_len_of_nuccores_that_were_searched = 0
        total_num_of_ir_pairs = 0
        total_num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs = 0

        nuccore_df_csv_file_path = stage1_results_info['nuccore_df_csv_file_path']
        nuccore_df = pd.read_csv(nuccore_df_csv_file_path, sep='\t', low_memory=False)
        assert len(nuccore_df) == orig_num_of_nuccore_entries
        num_of_nuccores_skipped_due_to_zero_filtered_cds_features = (nuccore_df['num_of_filtered_cds_features'] == 0).sum()
        accessions_of_nuccore_entries_to_search = nuccore_df[nuccore_df['num_of_filtered_cds_features'] > 0]['nuccore_accession']
        num_of_nuccore_entries_to_search = len(accessions_of_nuccore_entries_to_search)
        assert num_of_nuccore_entries_to_search + num_of_nuccores_skipped_due_to_zero_filtered_cds_features == orig_num_of_nuccore_entries

        # ########### debug ############
        # nuccore_accession_to_nuccore_entry_info = {
        #     'NZ_WLZY01000016.1': nuccore_accession_to_nuccore_entry_info['NZ_WLZY01000016.1']}
        # ########### debug ############


        for i, nuccore_accession in enumerate(sorted(accessions_of_nuccore_entries_to_search)):
            if debug___num_of_nuccore_entries_to_go_over is not None:
                assert i <= debug___num_of_nuccore_entries_to_go_over
                if i == debug___num_of_nuccore_entries_to_go_over:
                    break

            # 2775185
            # 75184
            # START = 0
            # START = 2510e3
            # # if i >= START + 75184:
            # if i >= START + 100e3:
            # if i >= START + 1e3:
            #     exit()
            # if i < START:
            #     continue

            nuccore_entry_info = nuccore_accession_to_nuccore_entry_info[nuccore_accession]

            generic_utils.print_and_write_to_log(f'nuccore_accession: {nuccore_accession}   (nuccore {i + 1}/{num_of_nuccore_entries_to_search} (find_all_ir_pairs))')

            nuccore_entry_ir_pairs_root_dir_path = os.path.join(output_dir_path, 'nuccore_accessions', nuccore_accession)
            pathlib.Path(nuccore_entry_ir_pairs_root_dir_path).mkdir(parents=True, exist_ok=True)

            ir_pairs_info = find_ir_pairs_in_nuccore_entry.get_nuccore_entry_ir_pairs_info(
                nuccore_entry_info=nuccore_entry_info,
                nuccore_entry_ir_pairs_root_dir_path=nuccore_entry_ir_pairs_root_dir_path,
                search_for_pis_args=search_for_pis_args,
                nuccore_uid=nuccore_accession,
            )
            nuccore_accession_to_ir_pairs_info[nuccore_accession] = ir_pairs_info
            total_len_of_nuccores_that_were_searched += ir_pairs_info['nuccore_total_num_of_base_pairs_that_were_searched']
            nuccore_num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs = ir_pairs_info[
                'ir_pairs_minimal_info']['num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs']
            nuccore_total_num_of_ir_pairs = ir_pairs_info['ir_pairs_minimal_info']['num_of_filtered_ir_pairs']
            if nuccore_total_num_of_ir_pairs > 0:
                total_num_of_ir_pairs += nuccore_total_num_of_ir_pairs
                total_num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs += nuccore_num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs
                filtered_nuccore_accession_to_ir_pairs_info[nuccore_accession] = ir_pairs_info

        ir_pairs_per_base_pair = (total_num_of_ir_pairs / total_len_of_nuccores_that_were_searched) if total_len_of_nuccores_that_were_searched else None


        all_ir_pairs_info = {
            'num_of_nuccores_skipped_due_to_zero_filtered_cds_features': num_of_nuccores_skipped_due_to_zero_filtered_cds_features,
            'num_of_nuccore_entries_to_search': num_of_nuccore_entries_to_search,
            'total_len_of_nuccores_that_were_searched': total_len_of_nuccores_that_were_searched,
            'total_num_of_ir_pairs': total_num_of_ir_pairs,
            'total_num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs': total_num_of_ir_pairs_discarded_due_to_being_fully_contained_in_other_ir_pairs,
            'ir_pairs_per_base_pair': ir_pairs_per_base_pair,
        }
        print(f'all_ir_pairs_info: {all_ir_pairs_info}')

        print(f'len(nuccore_accession_to_ir_pairs_info): {len(nuccore_accession_to_ir_pairs_info)}')
        print(f'len(filtered_nuccore_accession_to_ir_pairs_info): {len(filtered_nuccore_accession_to_ir_pairs_info)}')
        all_ir_pairs_extended_info = {
            'all_ir_pairs_info': all_ir_pairs_info,
            'nuccore_accession_to_ir_pairs_info': nuccore_accession_to_ir_pairs_info,
            'filtered_nuccore_accession_to_ir_pairs_info': filtered_nuccore_accession_to_ir_pairs_info,
        }

        return all_ir_pairs_extended_info
