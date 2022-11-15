import contextlib
import os
import os.path
import unittest
import numpy as np
import pandas as pd
import pathlib
import pickle

from searching_for_pis import cds_enrichment_analysis
from searching_for_pis import find_ir_pairs_in_nuccore_entry
from searching_for_pis import ir_pairs_linkage
from searching_for_pis import massive_screening_stage_1
from searching_for_pis import massive_screening_stage_2
from searching_for_pis import massive_screening_stage_3
from searching_for_pis import massive_screening_stage_5
from searching_for_pis import massive_screening_stage_6
from searching_for_pis import writing_repeat_cdss_to_fasta
from generic import bio_utils
from generic import generic_utils
from generic import blast_interface_and_utils

TEST_STAGE2 = True
# TEST_STAGE2 = False
TEST_STAGE3 = True
# TEST_STAGE3 = False
TEST_STAGE5 = True
# TEST_STAGE5 = False
TEST_STAGE6 = True
# TEST_STAGE6 = False
TEST_CLUSTER_CDS_PAIRS = True
# TEST_CLUSTER_CDS_PAIRS = False
TEST_GET_CDS_PAIRS = True
# TEST_GET_CDS_PAIRS = False


# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)





TESTS_OUT_DIR_NAME = 'temp_test_output'
TESTS_INPUTS_DIR_PATH = os.path.join('searching_for_pis', 'tests_inputs')
TEMP_TEST_OUT_CSV_FILE_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'TEMP_TEST_out.POSITIONS.csv')
TEMP_TEST_ANOTHER_OUT_CSV_FILE_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'TEMP_TEST_another_out.POSITIONS.csv')
TEMP_TEST_ALL_REPEATS_PAIRS_CSV_FILE_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'TEMP_TEST_all_repeats_pairs.csv')
TEMP_TEST_REPEATS_PAIRS_ACCORDING_TO_OBSERVED_BREAKPOINTS_CSV_FILE_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'TEMP_TEST_repeats_pairs_according_to_breakpoints.csv')
TEMP_TEST_REPEATS_PAIRS_FOR_WHICH_BREAKPOINTS_WERE_DETECTED_CSV_FILE_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'TEMP_TEST_repeats_pairs_with_breakpoints.csv')
TEMP_TEST_REPEATS_PAIRS_DIR_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'TEMP_TEST_repeats_pairs')
TEMP_TEST_OBSERVED_BREAKPOINTS_REPEATS_PAIRS_INFO_DICT_PICKLE_FILE_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'TEMP_TEST_repeats_pairs_according_to_breakpoints_info_dict.pickle')
TEMP_TEST_NPZ_FILE_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'temp_test.npz')
STAGE1_OUTPUT_FOR_RHAMNOSUS_AND_PNEUMOCOCCUS_DIR_PATH = os.path.join(TESTS_INPUTS_DIR_PATH, 'stage1_output_only_for_rhamnosus_and_pneumococcus')
MINIMALIZED_STAGE1_OUTPUT_FOR_RHAMNOSUS_AND_PNEUMOCOCCUS_DIR_PATH = os.path.join(TESTS_INPUTS_DIR_PATH, 'minimalized_stage1_output_only_for_rhamnosus_and_pneumococcus')
STAGE2_OUTPUT_FOR_RHAMNOSUS_AND_PNEUMOCOCCUS_DIR_PATH = os.path.join(TESTS_INPUTS_DIR_PATH, 'stage2_output_only_for_rhamnosus_and_pneumococcus')
STAGE3_OUTPUT_FOR_RHAMNOSUS_AND_PNEUMOCOCCUS_DIR_PATH = os.path.join(TESTS_INPUTS_DIR_PATH, 'stage3_output_only_for_rhamnosus_and_pneumococcus')
FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH = 'fake_output_of_massive_screening'
TEST_OUT_POTENTIAL_MISES_ROOT_DIR_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'potential_mises_test_out')
TEST_OUT_POTENTIAL_MIS_DIR_PATH = os.path.join(TESTS_OUT_DIR_NAME, 'potential_mis_dir')

L_RHAMNOSUS_TAXON_UID = 47715
PNEUMOCOCCUS_TAXON_UID = 1313
E_COLI_TAXON_UID = 562

def write_fasta_with_inversions(
        orig_seq_fasta_file_path,
        ordered_regions_to_invert,
        seq_after_inversions_fasta_file_path,
        seq_name,
):
    seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(orig_seq_fasta_file_path)
    for region_to_invert in ordered_regions_to_invert:
        seq = (
            bio_utils.get_region_in_chrom_seq(seq, 1, region_to_invert[0] - 1) +
            bio_utils.get_region_in_chrom_seq(seq, *region_to_invert).reverse_complement() +
            bio_utils.get_region_in_chrom_seq(seq, region_to_invert[1] + 1, len(seq))
        )
    seq.name = seq.description = seq.id = seq_name
    bio_utils.write_records_to_fasta_or_gb_file([seq], seq_after_inversions_fasta_file_path, 'fasta')


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_truncated_rhamnosus_stuff(
        input_file_path_truncated_rhamnosus_gb,
        output_file_path_truncated_rhamnosus_cds_df_csv,
        output_file_path_truncated_rhamnosus_fasta,
        output_file_path_truncated_rhamnosus_ir_pairs_df_csv,
        output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv,
        output_file_path_truncated_rhamnosus_ir_pairs_minimal_info_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_nuccore_df_csv,
        output_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta,
        truncated_rhamnosus_fake_taxon_uid,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    gb_record = bio_utils.get_gb_record(input_file_path_truncated_rhamnosus_gb)
    nuccore_accession = gb_record.id
    massive_screening_stage_1.write_nuccore_cds_df_csv(
        nuccore_accession=nuccore_accession,
        input_file_path_nuccore_cds_features_gb=input_file_path_truncated_rhamnosus_gb,
        output_file_path_nuccore_cds_df_csv=output_file_path_truncated_rhamnosus_cds_df_csv,
    )


    bio_utils.write_records_to_fasta_or_gb_file(gb_record, output_file_path_truncated_rhamnosus_fasta, 'fasta')
    with contextlib.redirect_stdout(None):
        blast_interface_and_utils.make_blast_nucleotide_db(output_file_path_truncated_rhamnosus_fasta)

    intermediate_ir_pairs_df_csv_file_path = f'{output_file_path_truncated_rhamnosus_ir_pairs_df_csv}.intermediate.csv'

    find_ir_pairs_in_nuccore_entry.write_nuccore_entry_ir_pairs(
        input_file_path_fasta=output_file_path_truncated_rhamnosus_fasta,
        seed_len=20,
        min_repeat_len=20,
        max_spacer_len=int(15e3),
        min_spacer_len=1,
        max_evalue=1000,
        inverted_or_direct_or_both='inverted',
        output_file_path_ir_pairs_csv=intermediate_ir_pairs_df_csv_file_path,
        output_file_path_ir_pairs_minimal_info_pickle=output_file_path_truncated_rhamnosus_ir_pairs_minimal_info_pickle,
    )

    intermediate_ir_pairs_df = pd.read_csv(intermediate_ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
    ir_pairs_df = intermediate_ir_pairs_df[['left1', 'right1', 'left2', 'right2', 'mismatch']].reset_index().rename(
        columns={'index': 'index_in_nuccore_ir_pairs_df_csv_file',
                 'mismatch': 'num_of_mismatches'})
    ir_pairs_df['nuccore_accession'] = nuccore_accession

    ir_pairs_df.to_csv(output_file_path_truncated_rhamnosus_ir_pairs_df_csv, sep='\t', index=False)

    ir_pairs_linkage.write_ir_pairs_linked_df_csv(
        input_file_path_all_cds_df_csv=output_file_path_truncated_rhamnosus_cds_df_csv,
        input_file_path_all_ir_pairs_df_csv=output_file_path_truncated_rhamnosus_ir_pairs_df_csv,
        output_file_path_ir_pairs_linked_df_csv=output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv,
        output_file_path_ir_pair_linkage_info_pickle=f'{output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv}.linkage_info.pickle',
        debug___num_of_nuccore_entries_to_go_over=None,
    )
    # truncated_rhamnosus_linked_ir_pairs_df = pd.read_csv(output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv, sep='\t', low_memory=False)
    # print('truncated_rhamnosus_linked_ir_pairs_df')
    # print(truncated_rhamnosus_linked_ir_pairs_df)
    # exit()

    write_fasta_with_inversions(
        orig_seq_fasta_file_path=output_file_path_truncated_rhamnosus_fasta,
        ordered_regions_to_invert=[(8636 + 20, 14942 - 20)],
        seq_after_inversions_fasta_file_path=output_file_path_truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta,
        seq_name='fake_other_genome',
    )
    with contextlib.redirect_stdout(None):
        blast_interface_and_utils.make_blast_nucleotide_db(output_file_path_truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta, verbose=False)


    nuccore_len = len(gb_record.seq)
    nuccore_entry_info = {
        'nuccore_accession': nuccore_accession, # just for convenience
        'taxon_uid': truncated_rhamnosus_fake_taxon_uid,
        'gbff_file_path': input_file_path_truncated_rhamnosus_gb,
        'num_of_filtered_cds_features': len(gb_record.features),
        'cds_features_gb_file_path': input_file_path_truncated_rhamnosus_gb,
        'cds_df_csv_file_path': output_file_path_truncated_rhamnosus_cds_df_csv,
        'fasta_file_path': output_file_path_truncated_rhamnosus_fasta,
        'chrom_len': nuccore_len,
    }
    with open(output_file_path_nuccore_accession_to_nuccore_entry_info_pickle, 'wb') as f:
        pickle.dump({nuccore_accession: nuccore_entry_info}, f, protocol=4)

    taxon_info = {
        'taxon_blast_db_path': output_file_path_truncated_rhamnosus_fasta,
        'primary_assembly_nuccore_total_len': nuccore_len,
    }
    with open(output_file_path_taxon_uid_to_taxon_info_pickle, 'wb') as f:
        pickle.dump({truncated_rhamnosus_fake_taxon_uid: taxon_info}, f, protocol=4)


    pd.DataFrame([nuccore_entry_info]).to_csv(output_file_path_nuccore_df_csv, sep='\t', index=False)


def write_truncated_rhamnosus_stuff(
        input_file_path_truncated_rhamnosus_gb,
        output_file_path_truncated_rhamnosus_cds_df_csv,
        output_file_path_truncated_rhamnosus_fasta,
        output_file_path_truncated_rhamnosus_ir_pairs_df_csv,
        output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv,
        output_file_path_truncated_rhamnosus_ir_pairs_minimal_info_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_nuccore_df_csv,
        output_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta,
        truncated_rhamnosus_fake_taxon_uid,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_truncated_rhamnosus_stuff(
        input_file_path_truncated_rhamnosus_gb=input_file_path_truncated_rhamnosus_gb,
        output_file_path_truncated_rhamnosus_cds_df_csv=output_file_path_truncated_rhamnosus_cds_df_csv,
        output_file_path_truncated_rhamnosus_fasta=output_file_path_truncated_rhamnosus_fasta,
        output_file_path_truncated_rhamnosus_ir_pairs_df_csv=output_file_path_truncated_rhamnosus_ir_pairs_df_csv,
        output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv=output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv,
        output_file_path_truncated_rhamnosus_ir_pairs_minimal_info_pickle=output_file_path_truncated_rhamnosus_ir_pairs_minimal_info_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle=output_file_path_nuccore_accession_to_nuccore_entry_info_pickle,
        output_file_path_nuccore_df_csv=output_file_path_nuccore_df_csv,
        output_file_path_taxon_uid_to_taxon_info_pickle=output_file_path_taxon_uid_to_taxon_info_pickle,
        output_file_path_truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta=(
            output_file_path_truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta),
        truncated_rhamnosus_fake_taxon_uid=truncated_rhamnosus_fake_taxon_uid,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=4,
    )

truncated_rhamnosus_gb_file_path = os.path.join(TESTS_INPUTS_DIR_PATH, 'truncated_NC_013198.1.gb')
truncated_rhamnosus_out_dir_path = os.path.join(TESTS_OUT_DIR_NAME, 'truncated_rhamnosus_temp_out')
pathlib.Path(truncated_rhamnosus_out_dir_path).mkdir(parents=True, exist_ok=True)
truncated_rhamnosus_cds_df_csv_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'truncated_rhamnosus_cds_df.csv')
truncated_rhamnosus_fasta_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'truncated_rhamnosus.fasta')
truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path = os.path.join(
    truncated_rhamnosus_out_dir_path, 'truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair.fasta')
truncated_rhamnosus_ir_pairs_df_csv_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'truncated_rhamnosus_ir_pairs_df.csv')
truncated_rhamnosus_linked_ir_pairs_df_csv_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'truncated_rhamnosus_linked_ir_pairs_df.csv')
truncated_rhamnosus_ir_pairs_minimal_info_pickle_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'truncated_rhamnosus_ir_pairs_minimal_info.pickle')
nuccore_accession_to_nuccore_entry_info_pickle_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'nuccore_accession_to_nuccore_entry_info.pickle')
nuccore_df_csv_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'nuccore_df.csv')
taxon_uid_to_taxon_info_pickle_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'taxon_uid_to_taxon_info.pickle')
rhamnosus_fake_sra_fasta_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'fake_sra.fasta')
truncated_rhamnosus_fake_taxon_uid = 13371337
dummy_empty_file_path = os.path.join(truncated_rhamnosus_out_dir_path, 'dummy_empty')
generic_utils.write_empty_file(dummy_empty_file_path)

with contextlib.redirect_stdout(None):
    write_truncated_rhamnosus_stuff(
        input_file_path_truncated_rhamnosus_gb=truncated_rhamnosus_gb_file_path,
        output_file_path_truncated_rhamnosus_cds_df_csv=truncated_rhamnosus_cds_df_csv_file_path,
        output_file_path_truncated_rhamnosus_fasta=truncated_rhamnosus_fasta_file_path,
        output_file_path_truncated_rhamnosus_ir_pairs_df_csv=truncated_rhamnosus_ir_pairs_df_csv_file_path,
        output_file_path_truncated_rhamnosus_linked_ir_pairs_df_csv=truncated_rhamnosus_linked_ir_pairs_df_csv_file_path,
        output_file_path_truncated_rhamnosus_ir_pairs_minimal_info_pickle=truncated_rhamnosus_ir_pairs_minimal_info_pickle_file_path,
        output_file_path_nuccore_accession_to_nuccore_entry_info_pickle=nuccore_accession_to_nuccore_entry_info_pickle_file_path,
        output_file_path_nuccore_df_csv=nuccore_df_csv_file_path,
        output_file_path_taxon_uid_to_taxon_info_pickle=taxon_uid_to_taxon_info_pickle_file_path,
        output_file_path_truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta=(
            truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path),
        truncated_rhamnosus_fake_taxon_uid=truncated_rhamnosus_fake_taxon_uid,
    )

ir_pairs_df = pd.read_csv(truncated_rhamnosus_ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
# print(ir_pairs_df)
# print(cds_df)
# exit()


# class TestMassiveScreening(TestSearchForMises):
class TestMassiveScreening(generic_utils.MyTestCase):
    def get_set_of_cds_pairs_in_df(self, cds_pairs_df):
        self.assertEqual(cds_pairs_df['nuccore_accession'].nunique(), 1)
        orig_num_of_cds_pairs = len(cds_pairs_df)
        cds_pairs = set(cds_pairs_df[['repeat1_cds_start_pos', 'repeat1_cds_end_pos',
                                    'repeat2_cds_start_pos', 'repeat2_cds_end_pos']].to_records(index=False).tolist())
        self.assertEqual(len(cds_pairs), orig_num_of_cds_pairs)
        return cds_pairs


    def assert_df_contains_cds_pairs(self, cds_pairs_df, expected_cds_pairs, allow_more_cds_pairs):
        self.assertIsInstance(expected_cds_pairs, set)
        if cds_pairs_df.empty:
            self.assertEqual(len(expected_cds_pairs), 0)
            return
        cds_pairs = self.get_set_of_cds_pairs_in_df(cds_pairs_df)

        if allow_more_cds_pairs:
            self.assertLessEqual(expected_cds_pairs, cds_pairs)
        else:
            self.assertEqual(expected_cds_pairs, cds_pairs)

    def get_set_of_ir_pairs_in_df(self, ir_pairs_df):
        self.assertEqual(ir_pairs_df['nuccore_accession'].nunique(), 1)
        orig_num_of_ir_pairs = len(ir_pairs_df)
        ir_pairs = set(ir_pairs_df[['left1', 'right1', 'left2', 'right2']].to_records(index=False).tolist())
        self.assertEqual(len(ir_pairs), orig_num_of_ir_pairs)
        return ir_pairs


    def assert_df_contains_ir_pairs(self, ir_pairs_df, expected_ir_pairs, allow_more_ir_pairs):
        self.assertIsInstance(expected_ir_pairs, set)
        if ir_pairs_df.empty:
            self.assertEqual(len(expected_ir_pairs), 0)
            return
        ir_pairs = self.get_set_of_ir_pairs_in_df(ir_pairs_df)

        if allow_more_ir_pairs:
            self.assertLessEqual(expected_ir_pairs, ir_pairs)
        else:
            self.assertEqual(expected_ir_pairs, ir_pairs)

    def assert_df_csv_contains_ir_pairs(self, ir_pairs_df_csv_file_path, expected_ir_pairs, allow_more_ir_pairs):
        ir_pairs_df = pd.read_csv(ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
        self.assert_df_contains_ir_pairs(ir_pairs_df, expected_ir_pairs, allow_more_ir_pairs)

    def assert_df_contains_linked_ir_pairs(self, ir_pairs_df, expected_ir_pairs, allow_more_ir_pairs):
        self.assertIsInstance(expected_ir_pairs, set)
        if ir_pairs_df.empty:
            self.assertEqual(len(expected_ir_pairs), 0)
            return
        ir_pairs = self.get_set_of_linked_ir_pairs_in_df(ir_pairs_df)

        if allow_more_ir_pairs:
            self.assertLessEqual(expected_ir_pairs, ir_pairs)
        else:
            self.assertEqual(expected_ir_pairs, ir_pairs)

    def get_set_of_linked_ir_pairs_in_df(self, ir_pairs_df):
        self.assertEqual(ir_pairs_df['nuccore_accession'].nunique(), 1)
        orig_num_of_ir_pairs = len(ir_pairs_df)
        linked_ir_pairs = set(ir_pairs_df[['left1', 'right1', 'left2', 'right2',
                                    'repeat1_cds_start_pos', 'repeat1_cds_end_pos',
                                    'repeat2_cds_start_pos', 'repeat2_cds_end_pos']].to_records(index=False).tolist())
        self.assertEqual(len(linked_ir_pairs), orig_num_of_ir_pairs)
        return linked_ir_pairs

    def assert_df_csv_contains_linked_ir_pairs(self, ir_pairs_df_csv_file_path, expected_ir_pairs, allow_more_ir_pairs):
        ir_pairs_df = pd.read_csv(ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
        self.assert_df_contains_linked_ir_pairs(ir_pairs_df, expected_ir_pairs, allow_more_ir_pairs)

    def test_discard_ir_pairs_fully_contained_in_other_ir_pairs(self):
        filtered_pairs_df = find_ir_pairs_in_nuccore_entry.discard_ir_pairs_fully_contained_in_other_ir_pairs(
            pd.DataFrame([
                ('', 100, 110, 200, 210),
                ('', 100, 111, 200, 211),
            ], columns=['nuccore_accession', 'left1', 'right1', 'left2', 'right2'])
        )
        self.assert_df_contains_ir_pairs(filtered_pairs_df, {(100, 111, 200, 211)}, allow_more_ir_pairs=False)

        filtered_pairs_df = find_ir_pairs_in_nuccore_entry.discard_ir_pairs_fully_contained_in_other_ir_pairs(
            pd.DataFrame([
                ('', 100, 110, 200, 210),
                ('', 101, 111, 201, 211),
            ], columns=['nuccore_accession', 'left1', 'right1', 'left2', 'right2'])
        )
        self.assert_df_contains_ir_pairs(filtered_pairs_df, {(100, 110, 200, 210), (101, 111, 201, 211)}, allow_more_ir_pairs=False)

    if TEST_STAGE2:
        def test_massive_screening_stage2(self):
            generic_utils.rmtree_silent(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH)
            pathlib.Path(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH).mkdir(parents=True, exist_ok=True)
            fake_stage1_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s1_out')
            pathlib.Path(fake_stage1_out_dir_path).mkdir(parents=True, exist_ok=True)
            fake_stage2_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s2_out')

            search_for_pis_args = {
                'debug___num_of_taxa_to_go_over': None,
                'debug___num_of_nuccore_entries_to_go_over': None,
                'debug___taxon_uids': None,
                'stage1': {
                    'output_dir_path': fake_stage1_out_dir_path,
                    'results_pickle_file_name': 'stage1_results_info.pickle',
                },
                'stage2': {
                    'output_dir_path': fake_stage2_out_dir_path,
                    'results_pickle_file_name': 'stage2_results_info.pickle',
                    'repeat_pairs': {
                        'seed_len': 20,
                        'min_repeat_len': 20,
                        'max_spacer_len': int(15e3),
                        'min_spacer_len': 1,
                        'max_evalue': 1000,
                    },
                },
            }

            fake_stage1_results_pickle_file_path = os.path.join(fake_stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])

            fake_stage1_results_info = {
                'nuccore_df_csv_file_path': nuccore_df_csv_file_path,
                'nuccore_accession_to_nuccore_entry_info_pickle_file_path': nuccore_accession_to_nuccore_entry_info_pickle_file_path,
            }
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)

            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                    (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage2']['repeat_pairs']['min_repeat_len'] = 32
            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                    (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage2']['repeat_pairs']['min_repeat_len'] = 33
            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                },
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage2']['repeat_pairs']['min_repeat_len'] = 20
            search_for_pis_args['stage2']['repeat_pairs']['min_spacer_len'] = 13225 - 10407 - 1
            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                    (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage2']['repeat_pairs']['min_spacer_len'] = 13225 - 10407
            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                },
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage2']['repeat_pairs']['min_spacer_len'] = 1
            search_for_pis_args['stage2']['repeat_pairs']['max_spacer_len'] = 14818 - 8760 - 1
            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                    (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage2']['repeat_pairs']['max_spacer_len'] = 14818 - 8760 - 2
            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )
            search_for_pis_args['stage2']['repeat_pairs']['max_spacer_len'] = int(15e3)

            nuccore_df = pd.read_csv(nuccore_df_csv_file_path, sep='\t', low_memory=False)
            assert len(nuccore_df) == 1
            nuccore_df['num_of_filtered_cds_features'] = 1
            single_cds_nuccore_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'single_cds_nuccore_df.csv')
            nuccore_df.to_csv(single_cds_nuccore_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['nuccore_df_csv_file_path'] = single_cds_nuccore_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)

            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                    (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )

            nuccore_df = pd.read_csv(nuccore_df_csv_file_path, sep='\t', low_memory=False)
            assert len(nuccore_df) == 1
            nuccore_df['num_of_filtered_cds_features'] = 0
            no_cds_nuccore_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'no_cds_nuccore_df.csv')
            nuccore_df.to_csv(no_cds_nuccore_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['nuccore_df_csv_file_path'] = no_cds_nuccore_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage2_results_info = massive_screening_stage_2.do_massive_screening_stage2(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage2_results_info['all_ir_pairs_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

    if TEST_STAGE3:
        def test_massive_screening_stage3(self):
            generic_utils.rmtree_silent(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH)
            pathlib.Path(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH).mkdir(parents=True, exist_ok=True)
            fake_stage1_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s1_out')
            pathlib.Path(fake_stage1_out_dir_path).mkdir(parents=True, exist_ok=True)
            fake_stage2_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s2_out')
            pathlib.Path(fake_stage2_out_dir_path).mkdir(parents=True, exist_ok=True)
            fake_stage3_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s3_out')

            search_for_pis_args = {
                'debug___num_of_taxa_to_go_over': None,
                'debug___num_of_nuccore_entries_to_go_over': None,
                'debug___taxon_uids': None,
                'stage1': {
                    'output_dir_path': fake_stage1_out_dir_path,
                    'results_pickle_file_name': 'stage1_results_info.pickle',
                },
                'stage2': {
                    'output_dir_path': fake_stage2_out_dir_path,
                    'results_pickle_file_name': 'stage2_results_info.pickle',
                },
                'stage3': {
                    'output_dir_path': fake_stage3_out_dir_path,
                    'results_pickle_file_name': 'stage3_results_info.pickle',

                    'min_repeat_len': 20,

                    'min_max_estimated_copy_num_to_classify_as_mobile_element': 3,
                    'blast_repeat_to_its_taxon_genome_to_find_copy_num': {
                        'min_dist_from_ir_pair_region_for_alignments': int(4e3),
                        'max_evalue': 1e-4,
                        'seed_len': 15,
                    },
                },
            }

            fake_stage1_results_pickle_file_path = os.path.join(fake_stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])
            fake_stage1_results_info = {
                # 'species_taxa_dir_path': species_taxa_dir_path,
                # 'taxa_for_which_download_of_primary_assembly_nuccore_entries_failed_uids': taxa_for_which_download_of_primary_assembly_nuccore_entries_failed_uids,
                # 'taxa_df_csv_file_path': taxa_df_csv_file_path,
                'taxon_uid_to_taxon_info_pickle_file_path': taxon_uid_to_taxon_info_pickle_file_path,
                'nuccore_df_csv_file_path': nuccore_df_csv_file_path,
                'nuccore_accession_to_nuccore_entry_info_pickle_file_path': nuccore_accession_to_nuccore_entry_info_pickle_file_path,
                'all_cds_df_csv_file_path': truncated_rhamnosus_cds_df_csv_file_path,
            }
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)

            fake_stage2_results_pickle_file_path = os.path.join(fake_stage2_out_dir_path, search_for_pis_args['stage2']['results_pickle_file_name'])
            fake_stage2_results_info = {
                'all_ir_pairs_df_csv_file_path': truncated_rhamnosus_ir_pairs_df_csv_file_path,
            }
            with open(fake_stage2_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage2_results_info, f, protocol=4)

            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8636, 10858, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8636, 10858, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8636, 10858, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8637, 10858, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8637, 10858, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8637, 10858, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(10858, 8761, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 10833, 11915, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8637, 8761, 11973, 15527),
                    (10376, 10407, 13225, 13256, 10833, 11915, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(10858, 8760, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 10833, 11915, 11973, 15527),
                    (10376, 10407, 13225, 13256, 10833, 11915, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['end_pos'].replace(10858, 8761, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 10833, 11915, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8636, 8761, 11973, 15527),
                    (10376, 10407, 13225, 13256, 10833, 11915, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['end_pos'].replace(10858, 8760, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 10833, 11915, 11973, 15527),
                    (10376, 10407, 13225, 13256, 10833, 11915, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8635, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8635, 10858, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8635, 10858, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8635, 10858, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8761, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8761, 10858, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8761, 10858, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8761, 10858, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['end_pos'].replace(10858, 10407, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8636, 10407, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8636, 10407, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8636, 10407, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['end_pos'].replace(10858, 10406, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8636, 10406, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8636, 10406, 11973, 15527),
                    (10376, 10407, 13225, 13256, 10833, 11915, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(15527, 14942, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8637, 10858, 11973, 14942),
                    (8636, 8760, 14818, 14942, 8637, 10858, 11973, 14942),
                    (10376, 10407, 13225, 13256, 8637, 10858, 11973, 14942),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(15527, 14941, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8637, 10858, 11973, 14941),
                    (10376, 10407, 13225, 13256, 8637, 10858, 11973, 14941),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df.loc[cds_df['start_pos'] == 8636, 'strand'] = -1
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df.loc[cds_df['start_pos'] == 8636, 'strand'] = -1
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df.loc[cds_df['start_pos'] == 8636, 'strand'] = -1
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(10858, 10407, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df.loc[cds_df['start_pos'] == 8636, 'strand'] = -1
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(10858, 10406, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (10376, 10407, 13225, 13256, 10833, 11915, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df.loc[cds_df['start_pos'] == 11973, 'strand'] = 1
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df.loc[cds_df['start_pos'] == 11973, 'strand'] = 1
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (8636, 8760, 14818, 14942, 6046, 8544, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )
            fake_stage1_results_info['all_cds_df_csv_file_path'] = truncated_rhamnosus_cds_df_csv_file_path



            seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            # TTGTTCACTCC AAGCTAT TCGGTTGGGGTCATTCGTG A TCTTGTGGAT GATATTCGTGAAGATGATTTTGACGTGGACAAGGGCGGTCAGGTTGAGATTATTGGGTGGCTGTATCAGTATTACAA
            modified_truncated_rhamnosus_fasta_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_truncated_rhamnosus.fasta')
            seq_to_replace_end_with = 'AGGAAGCTATTCGGTTGGGGTCATTCGTGAAGA' + 'AGCGTTGGGGTCATTCGTGATCTTGTGGATCTA' + 'CACATCTTGTGGATGATATTCGTGAAGATGTAA'
            orig_seq_len = len(seq)
            seq = seq[:-len(seq_to_replace_end_with)] + seq_to_replace_end_with
            assert len(seq) == orig_seq_len
            bio_utils.write_records_to_fasta_or_gb_file([seq], modified_truncated_rhamnosus_fasta_file_path, 'fasta')
            with contextlib.redirect_stdout(None):
                blast_interface_and_utils.make_blast_nucleotide_db(modified_truncated_rhamnosus_fasta_file_path)
            modified_taxon_uid_to_taxon_info_pickle_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_taxon_uid_to_taxon_info.pickle')
            with open(modified_taxon_uid_to_taxon_info_pickle_file_path, 'wb') as f:
                pickle.dump({truncated_rhamnosus_fake_taxon_uid: {'taxon_blast_db_path': modified_truncated_rhamnosus_fasta_file_path}}, f, protocol=4)
            fake_stage1_results_info['taxon_uid_to_taxon_info_pickle_file_path'] = modified_taxon_uid_to_taxon_info_pickle_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage3']['min_max_estimated_copy_num_to_classify_as_mobile_element'] = 4
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8636, 10858, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8636, 10858, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8636, 10858, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )
            search_for_pis_args['stage3']['min_max_estimated_copy_num_to_classify_as_mobile_element'] = 3

            search_for_pis_args['stage3']['blast_repeat_to_its_taxon_genome_to_find_copy_num']['min_dist_from_ir_pair_region_for_alignments'] = 21905 - 14942
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8636, 10858, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8636, 10858, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8636, 10858, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )
            search_for_pis_args['stage3']['blast_repeat_to_its_taxon_genome_to_find_copy_num']['min_dist_from_ir_pair_region_for_alignments'] = 21905 - 14942 - 1
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )
            search_for_pis_args['stage3']['blast_repeat_to_its_taxon_genome_to_find_copy_num']['min_dist_from_ir_pair_region_for_alignments'] = int(4e3)

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

            search_for_pis_args['stage3']['min_max_estimated_copy_num_to_classify_as_mobile_element'] = 4
            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(8544, 8760, inplace=True)
            cds_df.loc[cds_df['start_pos'] == 6046, 'strand'] = 1
            cds_df['end_pos'].replace(15527, 14942, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8637, 10858, 11973, 14942),
                    (8636, 8760, 14818, 14942, 6046, 8760, 11973, 14942),
                    (10376, 10407, 13225, 13256, 8637, 10858, 11973, 14942),
                },
                allow_more_ir_pairs=False,
            )
            search_for_pis_args['stage3']['min_max_estimated_copy_num_to_classify_as_mobile_element'] = 3
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

            cds_df = pd.read_csv(truncated_rhamnosus_cds_df_csv_file_path, sep='\t', low_memory=False)
            cds_df['start_pos'].replace(8636, 8637, inplace=True)
            cds_df['end_pos'].replace(8544, 8760, inplace=True)
            cds_df.loc[cds_df['start_pos'] == 6046, 'strand'] = 1
            cds_df['end_pos'].replace(15527, 14941, inplace=True)
            modified_cds_df_csv_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_cds_df.csv')
            cds_df.to_csv(modified_cds_df_csv_file_path, sep='\t', index=False)
            fake_stage1_results_info['all_cds_df_csv_file_path'] = modified_cds_df_csv_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8637, 10858, 11973, 14941),
                    (10376, 10407, 13225, 13256, 8637, 10858, 11973, 14941),
                },
                allow_more_ir_pairs=False,
            )
            fake_stage1_results_info['all_cds_df_csv_file_path'] = truncated_rhamnosus_cds_df_csv_file_path

            seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            # TTGTTCACTCC AAGCTAT TCGGTTGGGGTCATTCGTG A TCTTGTGGAT GATATTCGTGAAGATGATTTTGACGTGGACAAGGGCGGTCAGGTTGAGATTATTGGGTGGCTGTATCAGTATTACAA
            modified_truncated_rhamnosus_fasta_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_truncated_rhamnosus.fasta')
            seq_to_replace_end_with = 'AGGAAGCTATTCGGTTGGGGTCATTCGTGAAGA' + 'AGCGTTGGGGTCATTCGTGATCTTGTGGATCTA' + 'CACTTCTTGTGGATGATATTCGTGAAGATGTAA'
            orig_seq_len = len(seq)
            seq = seq[:-len(seq_to_replace_end_with)] + seq_to_replace_end_with
            assert len(seq) == orig_seq_len
            bio_utils.write_records_to_fasta_or_gb_file([seq], modified_truncated_rhamnosus_fasta_file_path, 'fasta')
            with contextlib.redirect_stdout(None):
                blast_interface_and_utils.make_blast_nucleotide_db(modified_truncated_rhamnosus_fasta_file_path)
            modified_taxon_uid_to_taxon_info_pickle_file_path = os.path.join(fake_stage1_out_dir_path, 'modified_taxon_uid_to_taxon_info.pickle')
            with open(modified_taxon_uid_to_taxon_info_pickle_file_path, 'wb') as f:
                pickle.dump({truncated_rhamnosus_fake_taxon_uid: {'taxon_blast_db_path': modified_truncated_rhamnosus_fasta_file_path}}, f, protocol=4)
            fake_stage1_results_info['taxon_uid_to_taxon_info_pickle_file_path'] = modified_taxon_uid_to_taxon_info_pickle_file_path
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                {
                    (9995, 10324, 13308, 13637, 8636, 10858, 11973, 15527),
                    (8636, 8760, 14818, 14942, 8636, 10858, 11973, 15527),
                    (10376, 10407, 13225, 13256, 8636, 10858, 11973, 15527),
                },
                allow_more_ir_pairs=False,
            )
            search_for_pis_args['stage3']['min_max_estimated_copy_num_to_classify_as_mobile_element'] = 2
            with contextlib.redirect_stdout(None):
                test_stage3_results_info = massive_screening_stage_3.do_massive_screening_stage3(search_for_pis_args)
            self.assert_df_csv_contains_linked_ir_pairs(
                test_stage3_results_info['pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path'],
                set(),
                allow_more_ir_pairs=False,
            )

    if TEST_STAGE5:
        def test_massive_screening_stage5(self):
            generic_utils.rmtree_silent(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH)
            pathlib.Path(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH).mkdir(parents=True, exist_ok=True)
            fake_stage1_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s1_out')
            pathlib.Path(fake_stage1_out_dir_path).mkdir(parents=True, exist_ok=True)
            fake_stage3_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s3_out')
            pathlib.Path(fake_stage3_out_dir_path).mkdir(parents=True, exist_ok=True)
            fake_stage4_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s4_out')
            pathlib.Path(fake_stage4_out_dir_path).mkdir(parents=True, exist_ok=True)
            fake_stage5_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s5_out')
            pathlib.Path(fake_stage5_out_dir_path).mkdir(parents=True, exist_ok=True)

            search_for_pis_args = {
                'debug___num_of_taxa_to_go_over': None,
                'debug___num_of_nuccore_entries_to_go_over': None,
                'debug___taxon_uids': None,
                'local_blast_nt_database_update_log_file_path': dummy_empty_file_path,
                'local_blast_nt_database_path': None,

                'stage1': {
                    'output_dir_path': fake_stage1_out_dir_path,
                    'results_pickle_file_name': 'stage1_results_info.pickle',
                },
                'stage3': {
                    'output_dir_path': fake_stage3_out_dir_path,
                    'results_pickle_file_name': 'stage3_results_info.pickle',
                },
                'stage4': {
                    'output_dir_path': fake_stage4_out_dir_path,
                    'results_pickle_file_name': 'stage4_results_info.pickle',
                },
                'stage5': {
                    'output_dir_path': fake_stage5_out_dir_path,
                    'results_pickle_file_name': 'stage5_results_info.pickle',
                    'other_nuccore_entries_extracted_from_local_nt_blast_db_dir_name': 'other_nuccore_entries_extracted_from_local_nt_blast_db',

                    'merged_cds_pair_region_margin_size': 200,

                    'blast_margins_and_identify_regions_in_other_nuccores': {
                        'max_evalue': 1e-5,
                        'seed_len': 20,
                        'num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate': 20,
                        'max_dist_between_lens_of_spanning_regions_ratio_and_1': 0.05,
                    },
                    'max_num_of_non_identical_regions_in_other_nuccores_to_analyze_per_merged_cds_pair_region': 100,

                    'min_mauve_total_match_proportion': 0.95,
                    'min_min_sub_alignment_min_match_proportion': 0.95,
                    'max_breakpoint_containing_interval_len': 10,
                    'max_max_dist_between_potential_breakpoint_containing_interval_and_repeat': 10,
                }
            }

            fake_stage1_results_pickle_file_path = os.path.join(fake_stage1_out_dir_path, search_for_pis_args['stage1']['results_pickle_file_name'])
            fake_stage1_results_info = {'nuccore_df_csv_file_path': nuccore_df_csv_file_path}
            with open(fake_stage1_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage1_results_info, f, protocol=4)

            fake_stage3_results_pickle_file_path = os.path.join(fake_stage3_out_dir_path, search_for_pis_args['stage3']['results_pickle_file_name'])
            fake_stage3_results_info = {'pairs_after_discarding_ir_pairs_linked_to_cds_containing_repeats_of_ir_pairs_with_high_repeat_estimated_copy_num_df_csv_file_path':
                                            truncated_rhamnosus_linked_ir_pairs_df_csv_file_path}
            with open(fake_stage3_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage3_results_info, f, protocol=4)

            relevant_taxon_uids_pickle_file_path = os.path.join(fake_stage4_out_dir_path, 'relevant_taxon_uids.pickle')
            with open(relevant_taxon_uids_pickle_file_path, 'wb') as f:
                pickle.dump([truncated_rhamnosus_fake_taxon_uid], f, protocol=4)

            species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path = os.path.join(
                fake_stage4_out_dir_path, 'species_taxon_uid_to_more_taxon_uids_of_the_same_species.pickle')
            with open(species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path, 'wb') as f:
                pickle.dump({truncated_rhamnosus_fake_taxon_uid: []}, f, protocol=4)

            taxon_wgs_nuccore_entries_info_pickle_file_path = os.path.join(fake_stage4_out_dir_path, 'taxon_wgs_nuccore_entries_info.pickle')
            with open(taxon_wgs_nuccore_entries_info_pickle_file_path, 'wb') as f:
                pickle.dump({'nuccore_accession_to_nuccore_entry_len': None}, f, protocol=4)

            taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path = os.path.join(
                fake_stage4_out_dir_path, 'taxon_local_blast_nt_database_nuccore_entries_info.pickle')
            with open(taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path, 'wb') as f:
                pickle.dump(None, f, protocol=4)

            taxa_potential_evidence_for_pi_info_pickle_file_path = os.path.join(fake_stage4_out_dir_path, 'taxa_potential_evidence_for_pi_info.pickle')
            with open(taxa_potential_evidence_for_pi_info_pickle_file_path, 'wb') as f:
                pickle.dump({
                    'taxon_uid_to_taxon_potential_evidence_for_pi_info': {
                        truncated_rhamnosus_fake_taxon_uid: {
                            'taxon_wgs_nuccore_entries_info_pickle_file_path': taxon_wgs_nuccore_entries_info_pickle_file_path,
                            'taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path': taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path,
                        },
                    },
                }, f, protocol=4)

            fake_stage4_results_pickle_file_path = os.path.join(fake_stage4_out_dir_path, search_for_pis_args['stage4']['results_pickle_file_name'])
            fake_stage4_results_info = {
                'relevant_taxon_uids_pickle_file_path': relevant_taxon_uids_pickle_file_path,
                'species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path': species_taxon_uid_to_more_taxon_uids_of_the_same_species_pickle_file_path,
                'taxa_potential_evidence_for_pi_info_pickle_file_path': taxa_potential_evidence_for_pi_info_pickle_file_path,
            }
            with open(fake_stage4_results_pickle_file_path, 'wb') as f:
                pickle.dump(fake_stage4_results_info, f, protocol=4)

            search_for_pis_args['debug_local_blast_database_path'] = truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path
            search_for_pis_args['debug_other_nuccore_accession_to_fasta_file_path'] = {
                'fake_other_genome': truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path}

            if 1:
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )

                search_for_pis_args['stage5']['merged_cds_pair_region_margin_size'] = 22000 - 15527
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )

                search_for_pis_args['stage5']['merged_cds_pair_region_margin_size'] = 22000 - 15527 + 1
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assertIsNone(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'])
                search_for_pis_args['stage5']['merged_cds_pair_region_margin_size'] = 200

            temp_fake_other_genome_fasta_file_path = os.path.join(fake_stage5_out_dir_path, 'temp_fake_other_genome.fasta')

            orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            seq_with_inversion = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path)
            ref_variant_with_margins = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 15527 + 200)
            non_ref_variant_with_small_margins = bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 100, 15527 + 100)
            fake_seq = ref_variant_with_margins + non_ref_variant_with_small_margins
            fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
            bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
            with contextlib.redirect_stdout(None):
                blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)

            search_for_pis_args['debug_local_blast_database_path'] = temp_fake_other_genome_fasta_file_path
            search_for_pis_args['debug_other_nuccore_accession_to_fasta_file_path'] = {'fake_other_genome': temp_fake_other_genome_fasta_file_path}

            if 1:
                search_for_pis_args['stage5']['blast_margins_and_identify_regions_in_other_nuccores']['num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate'] = 2
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )
                search_for_pis_args['stage5']['blast_margins_and_identify_regions_in_other_nuccores']['num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate'] = 1
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assertIsNone(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'])
                search_for_pis_args['stage5']['blast_margins_and_identify_regions_in_other_nuccores']['num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate'] = 20


                seq_with_inversion = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 8636 - 6) +
                    'A' * 10 +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 5, len(seq_with_inversion))
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                # search_for_pis_args['stage5']['max_breakpoint_containing_interval_len'] = 20
                search_for_pis_args['stage5']['blast_margins_and_identify_regions_in_other_nuccores'][
                    'max_dist_between_lens_of_spanning_regions_ratio_and_1'] = 10 / (15527 - 8636 + 1 + 400) + 0.0001
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )
                search_for_pis_args['stage5']['blast_margins_and_identify_regions_in_other_nuccores'][
                    'max_dist_between_lens_of_spanning_regions_ratio_and_1'] = 10 / (15527 - 8636 + 1 + 400) - 0.0001
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assertIsNone(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'])
                search_for_pis_args['stage5']['blast_margins_and_identify_regions_in_other_nuccores'][
                    'max_dist_between_lens_of_spanning_regions_ratio_and_1'] = 0.05

                seq_with_inversion = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 10636) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 8636 - 175) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 10637, 15527 + 200) +
                    'A' * (15527 + 175 - (8636 - 175) - 1 - (15527 + 200 - 10637 + 1) + 27) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 15527 + 175, 15527 + 200)
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )

                seq_with_inversion = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 10636) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 8636 - 175) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 10637, 15527 + 200) +
                    'A' * (15527 + 175 - (8636 - 175) - 1 - (15527 + 200 - 10637 + 1) + 25) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 15527 + 175, 15527 + 200)
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assertIsNone(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'])

                search_for_pis_args['stage5']['max_num_of_non_identical_regions_in_other_nuccores_to_analyze_per_merged_cds_pair_region'] = 2
                orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
                seq_with_inversion = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 10000) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 10005, 15527 + 200) +
                    'A' * 100 +
                    bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 10000) +
                    bio_utils.get_region_in_chrom_seq(orig_seq, 10002, 15527 + 200)
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )
                search_for_pis_args['stage5']['max_num_of_non_identical_regions_in_other_nuccores_to_analyze_per_merged_cds_pair_region'] = 1
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assertIsNone(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'])
                search_for_pis_args['stage5']['max_num_of_non_identical_regions_in_other_nuccores_to_analyze_per_merged_cds_pair_region'] = 100

            if 1:

                normal_region_with_margin_len = (15527 + 200 - (8636 - 200) + 1)
                search_for_pis_args['stage5']['min_mauve_total_match_proportion'] = normal_region_with_margin_len / (normal_region_with_margin_len + 100) - 0.001
                seq_with_inversion = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 10000) +
                    'A' * 100 +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 10001, 15527 + 200)
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )
                search_for_pis_args['stage5']['min_mauve_total_match_proportion'] = normal_region_with_margin_len / (normal_region_with_margin_len + 100) + 0.001
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assertIsNone(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'])
                search_for_pis_args['stage5']['min_mauve_total_match_proportion'] = 0.95

            if 1:
                seq_with_inversion = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_with_inversion_overlapping_outermost_ir_pair_fasta_file_path)
                inverted_seq_in_middle = bio_utils.get_region_in_chrom_seq(seq_with_inversion, 11501, 12000).reverse_complement()
                inverted_seq_in_middle_len = len(inverted_seq_in_middle)
                search_for_pis_args['stage5']['min_min_sub_alignment_min_match_proportion'] = 1 - (1 / inverted_seq_in_middle_len) - 0.001
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 8636 - 200, 11500) +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1, 199) +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 200, 200).reverse_complement() +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 201, inverted_seq_in_middle_len) +
                    bio_utils.get_region_in_chrom_seq(seq_with_inversion, 12001, 15527 + 200)
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )
                search_for_pis_args['stage5']['min_min_sub_alignment_min_match_proportion'] = 1 - (1 / inverted_seq_in_middle_len) + 0.001
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assertIsNone(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'])
                search_for_pis_args['stage5']['min_min_sub_alignment_min_match_proportion'] = 0.95


            orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            inverted_seq_in_middle = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 20, 14942 - 20).reverse_complement()
            inverted_seq_in_middle_len = len(inverted_seq_in_middle)
            fake_seq = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 8636 + 19) +
                bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1, 1000) +
                bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1001, inverted_seq_in_middle_len) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 19, 15527 + 200)
            )
            fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
            bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
            with contextlib.redirect_stdout(None):
                blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
            with contextlib.redirect_stdout(None):
                test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                {
                    # (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                    # (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )

            if 1:
                # for some reason, mauve leaves some bps unaligned (where there are multiple alignments, instead of choosing one it chooses none?)
                # therefore, i have to set a higher max_breakpoint_containing_interval_len.
                search_for_pis_args['stage5']['max_breakpoint_containing_interval_len'] = 100
                orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
                inverted_seq_in_middle = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 20, 14942 - 20).reverse_complement()
                inverted_seq_in_middle_len = len(inverted_seq_in_middle)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 8636 + 19) +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1, 1000).reverse_complement() +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1001, inverted_seq_in_middle_len) +
                    bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 19, 15527 + 200)
                )
                # print(bio_utils.seq_record_to_str(fake_seq))
                # print()
                # print()
                # print(bio_utils.seq_record_to_str(orig_seq))
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )
                search_for_pis_args['stage5']['max_breakpoint_containing_interval_len'] = 10

            if 1:
                orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
                inverted_seq_in_middle = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 20, 14942 - 20).reverse_complement()
                inverted_seq_in_middle_len = len(inverted_seq_in_middle)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 8636 + 19) +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1, 1000).reverse_complement() +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1001, inverted_seq_in_middle_len).reverse_complement() +
                    bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 19, 15527 + 200)
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )

                # for some reason, mauve leaves some bps unaligned (where there are multiple alignments, instead of choosing one it chooses none?)
                # therefore, i have to set a higher max_breakpoint_containing_interval_len.
                search_for_pis_args['stage5']['max_breakpoint_containing_interval_len'] = 100
                orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
                inverted_seq_in_middle = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 20, 14942 - 20).reverse_complement()
                inverted_seq_in_middle_len = len(inverted_seq_in_middle)
                fake_seq = (
                    bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 8636 + 19) +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1, 1000) +
                    bio_utils.get_region_in_chrom_seq(inverted_seq_in_middle, 1001, inverted_seq_in_middle_len).reverse_complement() +
                    bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 19, 15527 + 200)
                )
                fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
                bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
                with contextlib.redirect_stdout(None):
                    blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
                with contextlib.redirect_stdout(None):
                    test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
                self.assert_df_csv_contains_ir_pairs(
                    test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                    {
                        # (9995, 10324, 13308, 13637),
                        (8636, 8760, 14818, 14942),
                        # (10376, 10407, 13225, 13256),
                    },
                    allow_more_ir_pairs=False,
                )
                search_for_pis_args['stage5']['max_breakpoint_containing_interval_len'] = 10


            orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            seq_in_middle = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 20, 14942 - 20)
            seq_in_middle_len = len(seq_in_middle)
            fake_seq = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 8636 + 19) +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 1, 1000).reverse_complement() +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 1001, 2000) +
                # bio_utils.get_region_in_chrom_seq(seq_in_middle, 2001, seq_in_middle_len).reverse_complement() +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 2001, seq_in_middle_len) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 19, 15527 + 200)
            )
            fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
            bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
            with contextlib.redirect_stdout(None):
                blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
            with contextlib.redirect_stdout(None):
                test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
            self.assertTrue(pd.read_csv(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'], sep='\t', low_memory=False).empty)

            orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            seq_in_middle = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 20, 14942 - 20)
            seq_in_middle_len = len(seq_in_middle)
            fake_seq = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 8636 + 19) +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 1, 1000) +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 1001, 2000) +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 2001, seq_in_middle_len).reverse_complement() +
                bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 19, 15527 + 200)
            )
            fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
            bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
            with contextlib.redirect_stdout(None):
                blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
            with contextlib.redirect_stdout(None):
                test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
            self.assertTrue(pd.read_csv(test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'], sep='\t', low_memory=False).empty)

            orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            seq_in_middle = bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 20, 14942 - 20)
            seq_in_middle_len = len(seq_in_middle)
            fake_seq = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 8636 - 200, 8636 + 19) +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 1, 1000).reverse_complement() +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 1001, 2000) +
                bio_utils.get_region_in_chrom_seq(seq_in_middle, 2001, seq_in_middle_len).reverse_complement() +
                bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 19, 15527 + 200)
            )
            fake_seq.name = fake_seq.description = fake_seq.id = 'fake_other_genome'
            bio_utils.write_records_to_fasta_or_gb_file([fake_seq], temp_fake_other_genome_fasta_file_path, 'fasta')
            with contextlib.redirect_stdout(None):
                blast_interface_and_utils.make_blast_nucleotide_db(temp_fake_other_genome_fasta_file_path)
            with contextlib.redirect_stdout(None):
                test_stage5_results_info = massive_screening_stage_5.do_massive_screening_stage5(search_for_pis_args)
            self.assert_df_csv_contains_ir_pairs(
                test_stage5_results_info['filtered_pairs_with_highest_confidence_bps_df_csv_file_path'],
                {
                    # (9995, 10324, 13308, 13637),
                    (8636, 8760, 14818, 14942),
                    # (10376, 10407, 13225, 13256),
                },
                allow_more_ir_pairs=False,
            )

    if TEST_CLUSTER_CDS_PAIRS:
        def test_cluster_cds_pairs(self):
            generic_utils.rmtree_silent(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH)
            pathlib.Path(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH).mkdir(parents=True, exist_ok=True)

            fake_genome_fasta_file_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_genome.fasta')
            orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            fake_seq = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 1, 20000) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 1, 8615) + 'A' * 1000 +
                bio_utils.get_region_in_chrom_seq(orig_seq, 9616, 20000) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 1, 20000) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 1, 8615) + 'G' * 1000 +
                bio_utils.get_region_in_chrom_seq(orig_seq, 9616, 20000)
            )
            assert len(fake_seq) == 20000 * 4
            fake_seq.name = fake_seq.description = fake_seq.id = 'fake_genome'
            bio_utils.write_records_to_fasta_or_gb_file([fake_seq], fake_genome_fasta_file_path, 'fasta')

            fake_nuccore_accession_to_nuccore_entry_info_pickle_file_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH,
                                                                                         'fake_nuccore_accession_to_nuccore_entry_info.pickle')
            with open(fake_nuccore_accession_to_nuccore_entry_info_pickle_file_path, 'wb') as f:
                pickle.dump({'FAKE_NC_013198.1': {'fasta_file_path': fake_genome_fasta_file_path,}}, f, protocol=4)

            fake_ir_pairs_df_csv_file_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_ir_pairs_df.csv')
            fake_cds_pairs_df_csv_file_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_cds_pairs_df.csv')
            ir_pairs_df = pd.read_csv(truncated_rhamnosus_linked_ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
            ir_pairs_df['merged_cds_pair_region_start'] = 8636
            ir_pairs_df['merged_cds_pair_region_end'] = 15527
            ir_pairs_df['repeat1_cds_operon_start'] = 8636
            ir_pairs_df['repeat1_cds_operon_end'] = 10858
            ir_pairs_df['repeat2_cds_operon_start'] = 11973
            ir_pairs_df['repeat2_cds_operon_end'] = 15527
            ir_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'] = False

            ir_pairs_df2 = ir_pairs_df.copy()
            ir_pairs_df2[['left1', 'right1', 'left2', 'right2']] += 20000
            ir_pairs_df2['index_in_nuccore_ir_pairs_df_csv_file'] += 3
            ir_pairs_df2['merged_cds_pair_region_start'] += 20000
            ir_pairs_df2['merged_cds_pair_region_end'] += 20000
            for repeat_num in (1, 2):
                ir_pairs_df2[f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] += 3
                ir_pairs_df2[f'repeat{repeat_num}_cds_start_pos'] += 20000
                ir_pairs_df2[f'repeat{repeat_num}_cds_end_pos'] += 20000
                ir_pairs_df2[f'repeat{repeat_num}_cds_operon_start'] += 20000
                ir_pairs_df2[f'repeat{repeat_num}_cds_operon_end'] += 20000

            ir_pairs_df3 = ir_pairs_df2.copy()
            ir_pairs_df3[['left1', 'right1', 'left2', 'right2']] += 20000
            ir_pairs_df3['index_in_nuccore_ir_pairs_df_csv_file'] += 3
            ir_pairs_df3['merged_cds_pair_region_start'] += 20000
            ir_pairs_df3['merged_cds_pair_region_end'] += 20000
            for repeat_num in (1, 2):
                ir_pairs_df3[f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] += 3
                ir_pairs_df3[f'repeat{repeat_num}_cds_start_pos'] += 20000
                ir_pairs_df3[f'repeat{repeat_num}_cds_end_pos'] += 20000
                ir_pairs_df3[f'repeat{repeat_num}_cds_operon_start'] += 20000
                ir_pairs_df3[f'repeat{repeat_num}_cds_operon_end'] += 20000

            ir_pairs_df4 = ir_pairs_df3.copy()
            ir_pairs_df4[['left1', 'right1', 'left2', 'right2']] += 20000
            ir_pairs_df4['index_in_nuccore_ir_pairs_df_csv_file'] += 3
            ir_pairs_df4['merged_cds_pair_region_start'] += 20000
            ir_pairs_df4['merged_cds_pair_region_end'] += 20000
            for repeat_num in (1, 2):
                ir_pairs_df4[f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'] += 3
                ir_pairs_df4[f'repeat{repeat_num}_cds_start_pos'] += 20000
                ir_pairs_df4[f'repeat{repeat_num}_cds_end_pos'] += 20000
                ir_pairs_df4[f'repeat{repeat_num}_cds_operon_start'] += 20000
                ir_pairs_df4[f'repeat{repeat_num}_cds_operon_end'] += 20000

            all_ir_pairs_df = pd.concat(
                [
                    ir_pairs_df,
                    ir_pairs_df2,
                    ir_pairs_df3,
                    ir_pairs_df4,
                ],
                ignore_index=True,
            )

            all_ir_pairs_df.to_csv(fake_ir_pairs_df_csv_file_path, sep='\t', index=False)

            with contextlib.redirect_stdout(None):
                cds_enrichment_analysis.write_cds_pairs_df(
                    input_file_path_ir_pairs_df_csv=fake_ir_pairs_df_csv_file_path,
                    output_file_path_cds_pairs_df_csv=fake_cds_pairs_df_csv_file_path,
                )


            repeat_cds_seqs_fasta_file_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'repeats_with_margins.fasta')
            repeat_seq_name_df_csv_file_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH,
                                                                               'repeat_seq_name_df.csv')

            with contextlib.redirect_stdout(None):
                writing_repeat_cdss_to_fasta.write_repeat_cdss_to_fasta(
                    input_file_path_cds_pairs_df_csv=fake_cds_pairs_df_csv_file_path,
                    input_file_path_nuccore_accession_to_nuccore_entry_info_pickle=fake_nuccore_accession_to_nuccore_entry_info_pickle_file_path,
                    output_file_path_repeat_cds_seqs_fasta=repeat_cds_seqs_fasta_file_path,
                    output_file_path_repeat_cds_seq_name_df_csv=repeat_seq_name_df_csv_file_path,
                )

            clustering_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'clustering')
            pathlib.Path(clustering_out_dir_path).mkdir(parents=True, exist_ok=True)
            pair_clustering_extended_info_pickle_file_path = os.path.join(clustering_out_dir_path, 'pair_clustering_extended_info.pickle')
            with contextlib.redirect_stdout(None):
                cds_enrichment_analysis.cluster_cds_pairs(
                    min_pairwise_identity_with_cluster_centroid=0.98,
                    # min_pairwise_identity_with_cluster_centroid=1,
                    input_file_path_cds_pairs_df_csv=fake_cds_pairs_df_csv_file_path,
                    input_file_path_repeat_cds_seqs_fasta=repeat_cds_seqs_fasta_file_path,
                    input_file_path_repeat_seq_name_df_csv=repeat_seq_name_df_csv_file_path,
                    output_file_path_pair_clustering_extended_info_pickle=pair_clustering_extended_info_pickle_file_path,
                    pairwise_identity_definition_type='matching_columns_divided_by_alignment_length_such_that_terminal_gaps_are_penalized',
                    clustering_out_dir_path=clustering_out_dir_path,
                )
            with open(pair_clustering_extended_info_pickle_file_path, 'rb') as f:
                pair_clustering_extended_info = pickle.load(f)
            pairs_with_cluster_indices_df_csv = pd.read_csv(pair_clustering_extended_info['pairs_with_cluster_indices_df_csv_file_path'], sep='\t', low_memory=False)[
                [
                    'repeat1_cds_start_pos',
                    'repeat1_cds_end_pos',
                    'repeat2_cds_start_pos',
                    'repeat2_cds_end_pos',
                    'cds_pair_cluster_index',
                ]
            ]
            # print(pairs_with_cluster_indices_df_csv)
            self.assertEqual(
                pairs_with_cluster_indices_df_csv[(pairs_with_cluster_indices_df_csv['repeat1_cds_start_pos'] == 8636) |
                                                  (pairs_with_cluster_indices_df_csv['repeat1_cds_start_pos'] == 48636)]['cds_pair_cluster_index'].nunique(),
                1,
            )
            self.assertEqual(
                pairs_with_cluster_indices_df_csv[pairs_with_cluster_indices_df_csv['repeat1_cds_start_pos'] != 8636]['cds_pair_cluster_index'].nunique(),
                3,
            )

    if TEST_GET_CDS_PAIRS:
        def test_get_cds_pairs(self):
            ir_pairs_df = pd.read_csv(truncated_rhamnosus_linked_ir_pairs_df_csv_file_path, sep='\t', low_memory=False)
            ir_pairs_df['merged_cds_pair_region_start'] = 8636
            ir_pairs_df['merged_cds_pair_region_end'] = 15527
            # the recombinase gene is at 2160833:2161915.
            ir_pairs_df['repeat1_cds_operon_start'] = 8636
            ir_pairs_df['repeat1_cds_operon_end'] = 11915
            ir_pairs_df['repeat2_cds_operon_start'] = 11973
            ir_pairs_df['repeat2_cds_operon_end'] = 15527
            ir_pairs_df['any_high_confidence_ir_pair_linked_to_cds_pair'] = False
            cds_pairs_df = cds_enrichment_analysis.get_cds_pairs_df(ir_pairs_df)
            self.assert_df_contains_cds_pairs(cds_pairs_df, {(8636, 10858, 11973, 15527)}, allow_more_cds_pairs=False)
            # (9995, 10324, 13308, 13637, 8636, 10858, 11973, 15527),
            # (8636, 8760, 14818, 14942, 8636, 10858, 11973, 15527),
            # (10376, 10407, 13225, 13256, 8636, 10858, 11973, 15527),
            self.assertEqual(cds_pairs_df['operon_asymmetry'].iloc[0], 1)
            self.assertEqual(cds_pairs_df['operon_spacer_len'].iloc[0], 11973 - 11915 - 1)
            self.assertEqual(cds_pairs_df['max_repeat_len'].iloc[0], 10324 - 9995 + 1)
            alpha_sum = 15527 - 14942
            gamma_sum = 11915 - 10407 + 13225 - 11973
            self.assertEqual(cds_pairs_df['operon_closest_repeat_position_orientation_matching'].iloc[0],
                             alpha_sum / (gamma_sum + alpha_sum))

            assert 'repeat_len' not in ir_pairs_df
            ir_pairs_df.loc[ir_pairs_df['left1'] == 9995, 'right2'] = 13638
            ir_pairs_df.loc[ir_pairs_df['left1'] == 9995, 'left1'] = 9994
            cds_pairs_df = cds_enrichment_analysis.get_cds_pairs_df(ir_pairs_df)
            self.assert_df_contains_cds_pairs(cds_pairs_df, {(8636, 10858, 11973, 15527)}, allow_more_cds_pairs=False)
            self.assertEqual(cds_pairs_df['operon_asymmetry'].iloc[0], 1)
            self.assertEqual(cds_pairs_df['operon_spacer_len'].iloc[0], 11973 - 11915 - 1)
            self.assertEqual(cds_pairs_df['max_repeat_len'].iloc[0], 10324 - 9994 + 1)
            self.assertEqual(cds_pairs_df['operon_closest_repeat_position_orientation_matching'].iloc[0],
                             alpha_sum / (gamma_sum + alpha_sum))

            ir_pairs_df.loc[ir_pairs_df['left1'] == 8636, 'right2'] = 14941
            ir_pairs_df.loc[ir_pairs_df['left1'] == 8636, 'left1'] = 8637
            cds_pairs_df = cds_enrichment_analysis.get_cds_pairs_df(ir_pairs_df)
            self.assert_df_contains_cds_pairs(cds_pairs_df, {(8636, 10858, 11973, 15527)}, allow_more_cds_pairs=False)
            self.assertAlmostEqual(cds_pairs_df['operon_asymmetry'].iloc[0], 1 - 1 / (15527 - 14942), places=3)
            alpha_sum = 15527 - 14941 + 1
            gamma_sum = 11915 - 10407 + 13225 - 11973
            self.assertEqual(cds_pairs_df['operon_closest_repeat_position_orientation_matching'].iloc[0],
                             alpha_sum / (gamma_sum + alpha_sum))

            ir_pairs_df.loc[ir_pairs_df['left1'] == 8637, 'right2'] = 14943
            ir_pairs_df.loc[ir_pairs_df['left1'] == 8637, 'left1'] = 8635
            cds_pairs_df = cds_enrichment_analysis.get_cds_pairs_df(ir_pairs_df)
            self.assert_df_contains_cds_pairs(cds_pairs_df, {(8636, 10858, 11973, 15527)}, allow_more_cds_pairs=False)
            self.assertEqual(cds_pairs_df['operon_asymmetry'].iloc[0], 1)
            alpha_sum = 15527 - 14943
            gamma_sum = 11915 - 10407 + 13225 - 11973
            self.assertEqual(cds_pairs_df['operon_closest_repeat_position_orientation_matching'].iloc[0],
                             alpha_sum / (gamma_sum + alpha_sum))

            ir_pairs_df['repeat1_cds_operon_start'] = 8000
            ir_pairs_df['repeat1_cds_operon_end'] = 11000
            ir_pairs_df['repeat2_cds_operon_start'] = 11500
            ir_pairs_df['repeat2_cds_operon_end'] = 18000
            cds_pairs_df = cds_enrichment_analysis.get_cds_pairs_df(ir_pairs_df)
            self.assert_df_contains_cds_pairs(cds_pairs_df, {(8636, 10858, 11973, 15527)}, allow_more_cds_pairs=False)
            self.assertEqual(cds_pairs_df['operon_asymmetry'].iloc[0], 1 - 635 / (18000 - 14943))
            alpha_sum = 635 + 18000 - 14943
            gamma_sum = 11000 - 10407 + 13225 - 11500
            self.assertEqual(cds_pairs_df['operon_closest_repeat_position_orientation_matching'].iloc[0],
                             alpha_sum / (gamma_sum + alpha_sum))

    if TEST_STAGE6:
        def assert_raw_read_alignment_results_df_contains_evidence_reads(self, all_raw_read_alignment_results_df_csv_file_path, list_of_read_name_and_evidence_type):
            all_raw_read_alignment_results_df = pd.read_csv(all_raw_read_alignment_results_df_csv_file_path, sep='\t', low_memory=False)
            all_raw_read_alignment_results_df = all_raw_read_alignment_results_df[~all_raw_read_alignment_results_df['read_evidence_type'].isna()]
            self.assertEqual(
                set(all_raw_read_alignment_results_df[['sseqid', 'read_evidence_type']].drop_duplicates().to_records(index=False).tolist()),
                set(list_of_read_name_and_evidence_type),
            )

        def add_length_to_each_seq_description_and_write_to_fasta(self, seqs, output_fasta_file_path):
            seqs_with_new_desc = []
            for seq in seqs:
                curr_seq = seq[:]
                curr_seq.description = f'length={len(curr_seq)}'
                seqs_with_new_desc.append(curr_seq)
                # print(seqs_with_new_desc)
            bio_utils.write_records_to_fasta_or_gb_file(seqs_with_new_desc, output_fasta_file_path, 'fasta')


        def test_massive_screening_stage6(self):
            generic_utils.rmtree_silent(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH)
            pathlib.Path(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH).mkdir(parents=True, exist_ok=True)
            fake_stage6_out_dir_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'fake_s6_out')
            pathlib.Path(fake_stage6_out_dir_path).mkdir(parents=True, exist_ok=True)

            curr_rhamnosus_fake_sra_fasta_file_path = os.path.join(FAKE_OUTPUT_OF_MASSIVE_SCREENING_DIR_PATH, 'curr_rhamnosus_fake_sra.fasta')

            search_for_pis_args = {
                'debug___num_of_taxa_to_go_over': None,
                'debug___num_of_nuccore_entries_to_go_over': None,
                'debug___taxon_uids': None,
                'max_total_dist_between_joined_parts_per_joined_feature': 100,

                'enrichment_analysis': {'list_of_product_and_product_family': []},
                'stage6': {
                    'output_dir_path': fake_stage6_out_dir_path,
                    'results_pickle_file_name': 'test_stage6_results_info.pickle',
                    'sra_entries_dir_name': 'dummy',

                    'DEBUG___sra_entry_fasta_file_path': curr_rhamnosus_fake_sra_fasta_file_path,
                    'DEBUG___nuccore_fasta_file_path': truncated_rhamnosus_fasta_file_path,
                    'DEBUG___nuccore_gb_file_path': truncated_rhamnosus_gb_file_path,
                    'DEBUG___assembly_fasta_file_path': truncated_rhamnosus_fasta_file_path,

                    'blast_nuccore_to_find_ir_pairs': {
                        'seed_len': 7,
                        'min_repeat_len': 14,
                        'max_evalue': 1000,
                    },

                    'blast_alignment_region_to_long_reads': {
                        'min_ir_pair_region_margin_size_for_evidence_read': 500,
                        'min_num_of_read_bases_covered_by_any_alignment': int(1e3),
                        'min_alignment_region_margin_size': int(4e3),
                        'seed_len': 8,
                        'max_evalue': 1e-4,
                    },
                    'blast_alignment_assembly_to_relevant_long_reads': {
                        'seed_len': 8,
                        'max_evalue': 1e-4,
                    },


                    'cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args': {
                        'MTase: BREX type 1': {
                            'FAKE_NC_013198.1': [
                                {
                                    'target_gene_product_description': 'BREX type 1 PglX',
                                    'locus_description_for_table_3': 'BREX type 1',
                                    'longest_linked_repeat_cds_region_and_protein_id': ((11973, 15527), 'WP_015765014.1'),
                                    'ir_pair_region_with_margins': (8000, 16000),
                                    'presumably_relevant_cds_region': (6046, 19193),
                                    'alignment_region': (3000, 20000),
                                    'sra_accession_to_variants_and_reads_info': {
                                        'SRR9952487': {
                                            'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                                # 'SRR9952487.1.101097.1': {(10721, 10749, 12886, 12914)}, # inversion
                                                # 'SRR9952487.1.101241.1': {(8636, 8760, 14818, 14942), (9995, 10407, 13225, 13637)}, # a beautiful cassette switch
                                                # 'SRR9952487.1.102154.1': {(8636, 8760, 14818, 14942), (10721, 10749, 12886, 12914)}, # a beautiful cassette switch
                                                # 'SRR9952487.1.107988.1': {(9995, 10407, 13225, 13637), (10721, 10749, 12886, 12914)}, # a beautiful cassette switch
                                            },
                                            # ref reads:
                                            # 'SRR9952487.1.22524.1'
                                            # 'SRR9952487.1.26980.1'
                                            # 'SRR9952487.1.3643.1'
                                            # reads that shouldn't align:
                                            # 'fake_read1'
                                            # 'fake_read2'
                                        },
                                    },
                                    'align_to_whole_genome_to_verify_evidence_reads': True,
                                    'describe_in_the_paper': True,
                                },
                            ],
                        },
                    },
                    'nuccore_accession_to_assembly_accesion': {'FAKE_NC_013198.1': 'GCF_000026505.1'},
                    'nuccore_accession_to_name_in_assembly': {},
                    'sra_accession_to_type_and_sra_file_name': {'SRR9952487': ('long_reads', 'SRR9952487.1')},
                }
            }

            orig_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(truncated_rhamnosus_fasta_file_path)
            seq_with_8636_inversion_starting_at_1 = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 1, 8636 + 30) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 8636 + 31, 14942 - 30).reverse_complement() +
                bio_utils.get_region_in_chrom_seq(orig_seq, 14942 - 29, 19000)
            )
            seq_with_8636_inversion = bio_utils.get_region_in_chrom_seq(seq_with_8636_inversion_starting_at_1, 3700, len(seq_with_8636_inversion_starting_at_1))
            seq_with_9995_inversion = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 3500, 9995 + 100) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 9995 + 101, 13637 - 100).reverse_complement() +
                bio_utils.get_region_in_chrom_seq(orig_seq, 13637 - 99, 19000)
            )
            seq_with_9995_inversion_and_some_more_unaligned = (
                bio_utils.get_region_in_chrom_seq(seq_with_9995_inversion, 1, len(seq_with_9995_inversion)).reverse_complement() +
                bio_utils.get_random_dna_str(500, random_seed=2341)
            )

            read1_seq = seq_with_8636_inversion[:]
            read1_seq.name = read1_seq.description = read1_seq.id = 'fake_inversion_8636'
            read2_seq = seq_with_9995_inversion[:]
            read2_seq.name = read2_seq.description = read2_seq.id = 'fake_inversion_9995'
            read3_seq = seq_with_9995_inversion_and_some_more_unaligned[:]
            read3_seq.name = read3_seq.description = read3_seq.id = 'fake_inversion_9995_and_some_more_unaligned'
            read4_seq = bio_utils.get_region_in_chrom_seq(orig_seq, 4200, 18200).reverse_complement()
            read4_seq.name = read4_seq.description = read4_seq.id = 'fake_ref1'
            read5_seq = bio_utils.get_region_in_chrom_seq(orig_seq, 4300, 18600)
            read5_seq.name = read5_seq.description = read5_seq.id = 'fake_ref2'
            read6_seq = bio_utils.str_to_seq_record(bio_utils.get_random_dna_str(5000, random_seed=1153))
            read6_seq.name = read6_seq.description = read6_seq.id = 'random_dna'
            read7_seq = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 1, 10500) +
                bio_utils.str_to_seq_record(bio_utils.get_random_dna_str(100, random_seed=5731756)) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 10600, 18600)
            )
            read7_seq.name = read7_seq.description = read7_seq.id = 'ref_with_100_bases_in_middle_overwritten'
            read8_seq = (
                bio_utils.get_region_in_chrom_seq(orig_seq, 1, 10500) +
                bio_utils.str_to_seq_record(bio_utils.get_random_dna_str(1000, random_seed=4135)) +
                bio_utils.get_region_in_chrom_seq(orig_seq, 10500, 18600)
            )
            read8_seq.name = read8_seq.description = read8_seq.id = 'ref_with_1000_bases_inserted_in_middle'

            self.add_length_to_each_seq_description_and_write_to_fasta(
                [read1_seq, read2_seq, read3_seq, read4_seq, read5_seq, read6_seq, read7_seq, read8_seq],
                curr_rhamnosus_fake_sra_fasta_file_path,
            )

            search_for_pis_args['stage6']['cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args'][
                'MTase: BREX type 1']['FAKE_NC_013198.1'][0][
                    'sra_accession_to_variants_and_reads_info']['SRR9952487']['non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref'] = {
                'fake_inversion_8636': {(8636, 8760, 14818, 14942)},
                'fake_inversion_9995': {(9995, 10407, 13225, 13637)},
                'fake_inversion_9995_and_some_more_unaligned': {(9995, 10407, 13225, 13637)},
            }

            with contextlib.redirect_stdout(None):
                test_stage6_results_info = massive_screening_stage_6.do_massive_screening_stage6(search_for_pis_args)
            # all_raw_read_alignment_results_df = pd.read_csv(test_stage6_results_info['all_raw_read_alignment_results_df_csv_file_path'], sep='\t', low_memory=False)
            # print(all_raw_read_alignment_results_df[['sseqid', 'num_of_read_bases_covered_by_any_alignment']].drop_duplicates())

            # print(all_raw_read_alignment_results_df.head(2))
            self.assert_raw_read_alignment_results_df_contains_evidence_reads(
                test_stage6_results_info['all_raw_read_alignment_results_df_csv_file_path'],
                {
                    ('fake_inversion_8636', 'non_ref_variant'),
                    ('fake_inversion_9995', 'non_ref_variant'),
                    ('fake_inversion_9995_and_some_more_unaligned', 'non_ref_variant'),
                    ('fake_ref1', 'ref_variant'),
                    ('fake_ref2', 'ref_variant'),
                },
            )

            search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_num_of_read_bases_covered_by_any_alignment'] = 14001
            with contextlib.redirect_stdout(None):
                test_stage6_results_info = massive_screening_stage_6.do_massive_screening_stage6(search_for_pis_args)
            self.assert_raw_read_alignment_results_df_contains_evidence_reads(
                test_stage6_results_info['all_raw_read_alignment_results_df_csv_file_path'],
                {
                    ('fake_inversion_8636', 'non_ref_variant'),
                    ('fake_inversion_9995', 'non_ref_variant'),
                    ('fake_inversion_9995_and_some_more_unaligned', 'non_ref_variant'),
                    ('fake_ref1', 'ref_variant'),
                    ('fake_ref2', 'ref_variant'),
                },
            )
            search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_num_of_read_bases_covered_by_any_alignment'] = 14002
            with contextlib.redirect_stdout(None):
                test_stage6_results_info = massive_screening_stage_6.do_massive_screening_stage6(search_for_pis_args)
            self.assert_raw_read_alignment_results_df_contains_evidence_reads(
                test_stage6_results_info['all_raw_read_alignment_results_df_csv_file_path'],
                {
                    ('fake_inversion_8636', 'non_ref_variant'),
                    ('fake_inversion_9995', 'non_ref_variant'),
                    ('fake_inversion_9995_and_some_more_unaligned', 'non_ref_variant'),
                    # ('fake_ref1', 'ref_variant'),
                    ('fake_ref2', 'ref_variant'),
                },
            )
            search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_num_of_read_bases_covered_by_any_alignment'] = 2000

            # inverted repeat region: (8761, 14817)
            # inverted repeat region: 8636, 8760, 14818, 14942
            search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_ir_pair_region_margin_size_for_evidence_read'] = 8636 - 4300
            with contextlib.redirect_stdout(None):
                test_stage6_results_info = massive_screening_stage_6.do_massive_screening_stage6(search_for_pis_args)
            self.assert_raw_read_alignment_results_df_contains_evidence_reads(
                test_stage6_results_info['all_raw_read_alignment_results_df_csv_file_path'],
                {
                    ('fake_inversion_8636', 'non_ref_variant'),
                    ('fake_inversion_9995', 'non_ref_variant'),
                    ('fake_inversion_9995_and_some_more_unaligned', 'non_ref_variant'),
                    ('fake_ref1', 'ref_variant'),
                    ('fake_ref2', 'ref_variant'),
                },
            )
            search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_ir_pair_region_margin_size_for_evidence_read'] = 8636 - 4300 + 1
            with contextlib.redirect_stdout(None):
                test_stage6_results_info = massive_screening_stage_6.do_massive_screening_stage6(search_for_pis_args)
            self.assert_raw_read_alignment_results_df_contains_evidence_reads(
                test_stage6_results_info['all_raw_read_alignment_results_df_csv_file_path'],
                {
                    ('fake_inversion_8636', 'non_ref_variant'),
                    ('fake_inversion_9995', 'non_ref_variant'),
                    ('fake_inversion_9995_and_some_more_unaligned', 'non_ref_variant'),
                    ('fake_ref1', 'ref_variant'),
                    # ('fake_ref2', 'ref_variant'),
                },
            )
            search_for_pis_args['stage6']['blast_alignment_region_to_long_reads']['min_ir_pair_region_margin_size_for_evidence_read'] = 500

if __name__ == '__main__':
    unittest.main()
