import os.path
import pathlib
import pickle
import subprocess

import numpy as np
import pandas as pd

from generic import bio_utils
from generic import blast_interface_and_utils
from generic import generic_utils
from generic import mauve_interface_and_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

class TooManyNBasesInPotentialMisRegionMargins(RuntimeError):
    pass

class BlastMarginsSubprocessFailed(subprocess.SubprocessError):
    pass


L_RHAMNOSUS_TAXON_UID = '47715'
PNEUMOCOCCUS_TAXON_UID = '1313'
E_COLI_TAXON_UID = '562'

L_RHAMNOSUS_GENOME_UID = '913'
PNEUMOCOCCUS_GENOME_UID = '176'
E_COLI_GENOME_UID = '167'

L_RHAMNOSUS_ASSEMBLY_UID = '45428'
PNEUMOCOCCUS_ASSEMBLY_UID = '683411'
E_COLI_ASSEMBLY_UID = '1755381'

L_RHAMNOSUS_NUCCORE_UID = '258506995'  # another uid which is not of the same nuccore entry - 1905645664
L_RHAMNOSUS_PRIMARY_NUCCORE_CHROM_LEN = 3010111

OUT_DIR_PATH = 'find_potential_mis_variants_in_other_nuccore_entries_output'
L_RHAMNOSUS_OTHER_NUCCORE_ENTRIES_FASTA_FILES_DIR_PATH = os.path.join(OUT_DIR_PATH, 'l_rhamnosus_other_nuccore_entries_fasta_files')
L_RHAMNOSUS_OUT_DIR_PATH = os.path.join(OUT_DIR_PATH, 'l_rhamnosus_output')
L_RHAMNOSUS_PRIMARY_NUCCORE_FASTA_FILE_PATH = os.path.join(OUT_DIR_PATH, 'l_rhamnosus_primary_nuccore.fasta')
L_RHAMNOSUS_POTENTIAL_MIS_VARIANTS_IN_OTHER_NUCCORE_ENTRIES_INFO = os.path.join(OUT_DIR_PATH, 'l_rhamnosus_potential_mis_variants_in_other_nuccore_entries_info.pickle')



# pathlib.Path(L_RHAMNOSUS_OUT_DIR_PATH).mkdir(parents=True, exist_ok=True)
# pathlib.Path(L_RHAMNOSUS_OTHER_NUCCORE_ENTRIES_FASTA_FILES_DIR_PATH).mkdir(parents=True, exist_ok=True)

def get_potential_breakpoint_containing_interval_info(
        curr_interval,
        next_interval,
        interval_in_nuccore_to_relevant_mauve_alignment_info,
        matching_interval_start_to_index,
):
    # potential_breakpoint_containing_interval_len = bio_utils.get_num_of_phosphodiester_bonds_in_region(potential_breakpoint_containing_interval)

    potential_breakpoint_containing_interval = (curr_interval[1] + 0.5, next_interval[0] - 0.5)

    curr_interval_relevant_mauve_alignment_info = interval_in_nuccore_to_relevant_mauve_alignment_info[curr_interval]
    curr_matching_interval = curr_interval_relevant_mauve_alignment_info['interval_in_region_in_other_nuccore']
    curr_matching_interval_is_on_forward_strand = curr_interval_relevant_mauve_alignment_info['is_other_nuccore_interval_on_forward_strand']

    next_interval_relevant_mauve_alignment_info = interval_in_nuccore_to_relevant_mauve_alignment_info[next_interval]
    next_matching_interval = next_interval_relevant_mauve_alignment_info['interval_in_region_in_other_nuccore']
    next_matching_interval_is_on_forward_strand = next_interval_relevant_mauve_alignment_info['is_other_nuccore_interval_on_forward_strand']

    are_matching_intervals_on_different_strands = (curr_matching_interval_is_on_forward_strand != next_matching_interval_is_on_forward_strand)
    are_matching_intervals_consecutive_in_the_expected_order = (matching_interval_start_to_index[curr_matching_interval[0]] ==
                                                                (matching_interval_start_to_index[next_matching_interval[0]] - 1))
    interval_left_to_potential_breakpoint_containing_interval_len = curr_interval[1] - curr_interval[0] + 1
    interval_right_to_potential_breakpoint_containing_interval_len = next_interval[1] - next_interval[0] + 1

    potential_breakpoint_containing_interval_info = {
        'potential_breakpoint_containing_interval_start': potential_breakpoint_containing_interval[0],
        'potential_breakpoint_containing_interval_end': potential_breakpoint_containing_interval[1],
        'are_matching_intervals_on_different_strands': are_matching_intervals_on_different_strands,
        'are_matching_intervals_consecutive_in_the_expected_order': are_matching_intervals_consecutive_in_the_expected_order,

        # 'interval_left_to_potential_breakpoint_containing_interval' can be inferred from 'potential_breakpoint_containing_interval_start' and
        # 'interval_left_to_potential_breakpoint_containing_interval_len'.
        'interval_left_to_potential_breakpoint_containing_interval_len': interval_left_to_potential_breakpoint_containing_interval_len,
        # 'interval_right_to_potential_breakpoint_containing_interval' can be inferred from 'potential_breakpoint_containing_interval_end' and
        # 'interval_right_to_potential_breakpoint_containing_interval_len'.
        'interval_right_to_potential_breakpoint_containing_interval_len': interval_right_to_potential_breakpoint_containing_interval_len,

        'interval_left_to_potential_breakpoint_containing_interval_min_match_proportion': curr_interval_relevant_mauve_alignment_info['min_match_proportion'],
        'interval_right_to_potential_breakpoint_containing_interval_min_match_proportion': next_interval_relevant_mauve_alignment_info['min_match_proportion'],
    }

    return potential_breakpoint_containing_interval_info


def is_potential_breakpoint_containing_interval(potential_breakpoint_containing_interval_info):
    return (
        potential_breakpoint_containing_interval_info['are_matching_intervals_on_different_strands'] or
        (~potential_breakpoint_containing_interval_info['are_matching_intervals_consecutive_in_the_expected_order'])
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_region_in_other_nuccore_potential_breakpoint_df(
        interval_in_nuccore_to_relevant_mauve_alignment_info,
        output_file_path_region_in_other_nuccore_potential_breakpoint_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    sorted_intervals_in_merged_cds_pair_region_with_margins = sorted(interval_in_nuccore_to_relevant_mauve_alignment_info)
    sorted_matching_interval_starts = sorted(x['interval_in_region_in_other_nuccore'][0] for x in interval_in_nuccore_to_relevant_mauve_alignment_info.values())
    matching_interval_start_to_index = {x: i for i, x in enumerate(sorted_matching_interval_starts)}

    print('sorted_intervals_in_merged_cds_pair_region_with_margins')
    print(sorted_intervals_in_merged_cds_pair_region_with_margins)
    print('sorted_matching_interval_starts')
    print(sorted_matching_interval_starts)
    print('interval_in_nuccore_to_relevant_mauve_alignment_info')
    print(interval_in_nuccore_to_relevant_mauve_alignment_info)

    potential_breakpoint_containing_interval_infos = []

    for curr_interval, next_interval in zip(sorted_intervals_in_merged_cds_pair_region_with_margins[:-1], sorted_intervals_in_merged_cds_pair_region_with_margins[1:]):
        potential_breakpoint_containing_interval_info = get_potential_breakpoint_containing_interval_info(
            curr_interval=curr_interval,
            next_interval=next_interval,
            interval_in_nuccore_to_relevant_mauve_alignment_info=interval_in_nuccore_to_relevant_mauve_alignment_info,
            matching_interval_start_to_index=matching_interval_start_to_index,
        )
        if is_potential_breakpoint_containing_interval(potential_breakpoint_containing_interval_info):
            potential_breakpoint_containing_interval_infos.append(potential_breakpoint_containing_interval_info)

    if potential_breakpoint_containing_interval_infos:
        potential_breakpoint_df = pd.DataFrame(potential_breakpoint_containing_interval_infos)
        assert potential_breakpoint_df['potential_breakpoint_containing_interval_start'].is_unique
        assert potential_breakpoint_df['potential_breakpoint_containing_interval_end'].is_unique
        potential_breakpoint_df.to_csv(output_file_path_region_in_other_nuccore_potential_breakpoint_df_csv, sep='\t', index=False)
    else:
        generic_utils.write_empty_file(output_file_path_region_in_other_nuccore_potential_breakpoint_df_csv)


def write_region_in_other_nuccore_potential_breakpoint_df(
        interval_in_nuccore_to_relevant_mauve_alignment_info,
        output_file_path_region_in_other_nuccore_potential_breakpoint_df_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_region_in_other_nuccore_potential_breakpoint_df(
        interval_in_nuccore_to_relevant_mauve_alignment_info=interval_in_nuccore_to_relevant_mauve_alignment_info,
        output_file_path_region_in_other_nuccore_potential_breakpoint_df_csv=output_file_path_region_in_other_nuccore_potential_breakpoint_df_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info(
        input_file_path_mauve_alignment_results_xmfa,
        output_file_path_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info = mauve_interface_and_utils.get_region_in_seq1_to_mauve_alignment_info(
        input_file_path_mauve_alignment_results_xmfa)
    with open(output_file_path_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle, 'wb') as f:
        pickle.dump(interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info, f, protocol=4)

def write_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info(
        input_file_path_mauve_alignment_results_xmfa,
        output_file_path_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info(
        input_file_path_mauve_alignment_results_xmfa=input_file_path_mauve_alignment_results_xmfa,
        output_file_path_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle=(
            output_file_path_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle),
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )

def get_region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info(
        other_nuccore_fasta_file_path,
        merged_cds_pair_region_with_margins_fasta_file_path,
        region_in_other_nuccore,
        is_region_in_other_nuccore_on_forward_strand,
        min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion,
        merged_cds_pair_region_with_margins,
        region_in_other_nuccore_output_dir_path,
):
    potential_breakpoint_containing_intervals_info_pickle_file_path = None
    mauve_alignment_results_xmfa_file_path = None
    mauve_alignment_results_backbone_csv_file_path = None
    # mauve_alignment_backbone_lcbs_info_pickle_file_path = None
    region_in_other_nuccore_potential_breakpoint_df_csv_file_path = None
    interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path = None
    mauve_alignment_covered_bases_in_region_in_other_nuccore_proportion = None
    mauve_total_match_proportion = None
    mauve_total_num_of_matches = None
    is_collinear_alignment = False
    region_in_other_nuccore_match_proportion = None
    min_interval_in_region_in_other_nuccore_match_proportion = None
    min_sub_alignment_min_match_proportion = None
    satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds = None

    if is_region_in_other_nuccore_on_forward_strand:
        region_in_other_nuccore_fasta_file_path = os.path.join(region_in_other_nuccore_output_dir_path, 'region_in_other_nuccore.fasta')
    else:
        region_in_other_nuccore_fasta_file_path = os.path.join(region_in_other_nuccore_output_dir_path, 'region_in_other_nuccore_reverse_complement.fasta')

    bio_utils.write_region_to_fasta_file(
        input_file_path_fasta=other_nuccore_fasta_file_path,
        region=region_in_other_nuccore,
        output_file_path_region_fasta=region_in_other_nuccore_fasta_file_path,
        write_reverse_complement_of_region=(not is_region_in_other_nuccore_on_forward_strand),
    )

    is_identical_to_merged_cds_pair_region_with_margins_file_path = os.path.join(region_in_other_nuccore_output_dir_path, 'is_identical_to_merged_cds_pair_region_with_margins.txt')
    bio_utils.write_whether_fasta_seqs_identical(
        input_file_path_fasta1=region_in_other_nuccore_fasta_file_path,
        input_file_path_fasta2=merged_cds_pair_region_with_margins_fasta_file_path,
        output_file_path_are_identical_txt=is_identical_to_merged_cds_pair_region_with_margins_file_path,
    )
    is_identical_to_merged_cds_pair_region_with_margins_str = generic_utils.read_text_file(is_identical_to_merged_cds_pair_region_with_margins_file_path)
    assert is_identical_to_merged_cds_pair_region_with_margins_str in ('0', '1')
    is_identical_to_merged_cds_pair_region_with_margins = is_identical_to_merged_cds_pair_region_with_margins_str == '1'

    if not is_identical_to_merged_cds_pair_region_with_margins:
        mauve_alignment_results_xmfa_file_path = os.path.join(
            region_in_other_nuccore_output_dir_path, 'mauve_region_in_other_nuccore_and_merged_cds_pair_region_with_margins_results.xmfa')
        mauve_alignment_results_backbone_csv_file_path = os.path.join(
            region_in_other_nuccore_output_dir_path, 'mauve_region_in_other_nuccore_and_merged_cds_pair_region_with_margins_results_backbone.txt')
        # try:
        mauve_interface_and_utils.progressive_mauve(
            input_file_path_seq0_fasta=region_in_other_nuccore_fasta_file_path,
            input_file_path_seq1_fasta=merged_cds_pair_region_with_margins_fasta_file_path,
            assume_input_sequences_are_collinear=False,
            output_file_path_alignment_xmfa=mauve_alignment_results_xmfa_file_path,
            output_file_path_backbone_csv=mauve_alignment_results_backbone_csv_file_path,
        )
        # except subprocess.SubprocessError:
        #     generic_utils.print_and_write_to_log(f'seems like mauve crashed.')
        #     return {
        #         'reason_region_in_other_nuccore_region_was_skipped': 'mauve crashed',
        #     }

        interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path = os.path.join(
            region_in_other_nuccore_output_dir_path, 'interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info.pickle')
        write_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info(
            input_file_path_mauve_alignment_results_xmfa=mauve_alignment_results_xmfa_file_path,
            output_file_path_interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle=(
                interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path),
        )
        with open(interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path, 'rb') as f:
            interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info = pickle.load(f)

        mauve_total_num_of_matches = sum(x['num_of_matches'] for x in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info.values())
        region_in_other_nuccore_len = region_in_other_nuccore[1] - region_in_other_nuccore[0] + 1

        aligned_part_of_merged_cds_pair_region_with_margins_len = (
            max(x[1] for x in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info) -
            min(x[0] for x in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info) + 1
        )
        longer_aligned_sequence_len = max(region_in_other_nuccore_len, aligned_part_of_merged_cds_pair_region_with_margins_len)
        mauve_total_match_proportion = mauve_total_num_of_matches / longer_aligned_sequence_len
        min_sub_alignment_min_match_proportion = min(
            x['min_match_proportion'] for x in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info.values())

        # print("[x['min_match_proportion'] for x in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info.values()]")
        # print([x['min_match_proportion'] for x in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info.values()])
        # print('min_sub_alignment_min_match_proportion')
        # print(min_sub_alignment_min_match_proportion)
        # print('min_min_sub_alignment_min_match_proportion')
        # print(min_min_sub_alignment_min_match_proportion)

        # print('mauve_total_match_proportion')
        # print(mauve_total_match_proportion)

        satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds = (
            (mauve_total_match_proportion >= min_mauve_total_match_proportion) and
            (min_sub_alignment_min_match_proportion >= min_min_sub_alignment_min_match_proportion)
        )

        if satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds:
            min_interval_in_region_in_other_nuccore_match_proportion = min(
                x['match_prop_rel_to_seq0'] for x in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info.values())
            region_in_other_nuccore_match_proportion = mauve_total_num_of_matches / region_in_other_nuccore_len
            merged_cds_pair_region_with_margins_start_index = merged_cds_pair_region_with_margins[0] - 1
            interval_in_nuccore_to_mauve_alignment_info = {
                (interval[0] + merged_cds_pair_region_with_margins_start_index, interval[1] + merged_cds_pair_region_with_margins_start_index): mauve_alignment_info
                for interval, mauve_alignment_info
                in interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info.items()
            }
            interval_in_nuccore_to_relevant_mauve_alignment_info = {
                interval_in_nuccore: {
                    'interval_in_region_in_other_nuccore': mauve_alignment_info['region_in_seq0'],
                    'is_other_nuccore_interval_on_forward_strand': mauve_alignment_info['is_seq0_on_forward_strand'],
                    'min_match_proportion': mauve_alignment_info['min_match_proportion'],
                    # 'interval_in_region_in_other_nuccore_match_proportion': mauve_alignment_info['match_prop_rel_to_seq0'],
                }
                for interval_in_nuccore, mauve_alignment_info
                in interval_in_nuccore_to_mauve_alignment_info.items()
            }


            region_in_other_nuccore_potential_breakpoint_df_csv_file_path = os.path.join(region_in_other_nuccore_output_dir_path,
                                                                                         'region_in_other_nuccore_potential_breakpoint_df.csv')
            write_region_in_other_nuccore_potential_breakpoint_df(
                interval_in_nuccore_to_relevant_mauve_alignment_info=interval_in_nuccore_to_relevant_mauve_alignment_info,
                output_file_path_region_in_other_nuccore_potential_breakpoint_df_csv=region_in_other_nuccore_potential_breakpoint_df_csv_file_path,
            )
            if generic_utils.is_file_empty(region_in_other_nuccore_potential_breakpoint_df_csv_file_path):
                region_in_other_nuccore_potential_breakpoint_df_csv_file_path = None

            all_other_nuccore_intervals_are_on_forward_strand = all(x['is_other_nuccore_interval_on_forward_strand']
                                                                    for x in interval_in_nuccore_to_relevant_mauve_alignment_info.values())
            if all_other_nuccore_intervals_are_on_forward_strand:
                sorted_intervals_in_region_in_other_nuccore = [alignment_info['interval_in_region_in_other_nuccore']
                                                               for _, alignment_info in sorted(interval_in_nuccore_to_relevant_mauve_alignment_info.items())]

                # lens_of_spacers_between_intervals_in_region_in_other_nuccore = [
                #     next_interval[0] - curr_interval[1] - 1
                #     for curr_interval, next_interval in zip(sorted_intervals_in_region_in_other_nuccore[:-1], sorted_intervals_in_region_in_other_nuccore[1:])
                # ]
                # total_lens_of_spacers_between_intervals_in_region_in_other_nuccore = sum(lens_of_spacers_between_intervals_in_region_in_other_nuccore)

                # collinear means that there aren't rearrangements.
                is_collinear_alignment = all(
                    next_interval[0] > curr_interval[1]
                    for curr_interval, next_interval in zip(sorted_intervals_in_region_in_other_nuccore[:-1], sorted_intervals_in_region_in_other_nuccore[1:])
                )




    region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info = {
        'region_in_other_nuccore_fasta_file_path': region_in_other_nuccore_fasta_file_path,
        'is_identical_to_merged_cds_pair_region_with_margins': is_identical_to_merged_cds_pair_region_with_margins,

        'mauve_alignment_results_xmfa_file_path': mauve_alignment_results_xmfa_file_path,
        'mauve_alignment_results_backbone_csv_file_path': mauve_alignment_results_backbone_csv_file_path,
        # 'mauve_alignment_backbone_lcbs_info_pickle_file_path': mauve_alignment_backbone_lcbs_info_pickle_file_path,
        'interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path': (
            interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path),
        'potential_breakpoint_containing_intervals_info_pickle_file_path': potential_breakpoint_containing_intervals_info_pickle_file_path,

        'region_in_other_nuccore_potential_breakpoint_df_csv_file_path': region_in_other_nuccore_potential_breakpoint_df_csv_file_path,
        'mauve_alignment_covered_bases_in_region_in_other_nuccore_proportion': mauve_alignment_covered_bases_in_region_in_other_nuccore_proportion,
        'mauve_total_num_of_matches': mauve_total_num_of_matches,
        'mauve_total_match_proportion': mauve_total_match_proportion,
        'is_collinear_alignment': is_collinear_alignment,
        'region_in_other_nuccore_match_proportion': region_in_other_nuccore_match_proportion,
        'min_interval_in_region_in_other_nuccore_match_proportion': min_interval_in_region_in_other_nuccore_match_proportion,
        'min_sub_alignment_min_match_proportion': min_sub_alignment_min_match_proportion,
        'satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds': (
            satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds),
    }

    return region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_regions_in_other_nuccores_preliminary_info(
        input_file_path_blast_left_and_right_margins_results_csv,
        left_margin_region,
        right_margin_region,
        max_dist_between_lens_of_spanning_regions_ratio_and_1,
        num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate,
        output_file_path_regions_in_other_nuccores_preliminary_info_pickle,
        blast_margins_output_dir_path,
        nuccore_accession,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    alignments_df = blast_interface_and_utils.read_blast_results_df(input_file_path_blast_left_and_right_margins_results_csv)
    alignments_df = alignments_df[alignments_df['sseqid'] != nuccore_accession]
    alignments_df['margin_in_region_in_other_nuccore_is_on_forward_strand'] = alignments_df['sstart'] < alignments_df['send']

    left_margin_alignments_df = alignments_df[alignments_df['qseqid'] == 'left_margin'].copy()
    right_margin_alignments_df = alignments_df[alignments_df['qseqid'] == 'right_margin'].copy()

    left_margin_alignments_df.loc[:, ['qstart', 'qend']] += left_margin_region[0] - 1
    right_margin_alignments_df.loc[:, ['qstart', 'qend']] += right_margin_region[0] - 1

    other_nuccore_accession_to_num_of_left_margin_alignments = left_margin_alignments_df['sseqid'].value_counts().to_dict()
    other_nuccore_accession_to_num_of_right_margin_alignments = right_margin_alignments_df['sseqid'].value_counts().to_dict()


    left_margin_alignments_df = left_margin_alignments_df.sort_values('evalue', ascending=True).groupby('sseqid', sort=False).head(
        num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate)
    right_margin_alignments_df = right_margin_alignments_df.sort_values('evalue', ascending=True).groupby('sseqid', sort=False).head(
        num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate)
    # print('\n\n\nleft_margin_alignments_df')
    # print(left_margin_alignments_df)

    margins_alignments_pairs_df = left_margin_alignments_df.merge(right_margin_alignments_df,
                                                                  on=['sseqid', 'margin_in_region_in_other_nuccore_is_on_forward_strand'],
                                                                suffixes=('_left', '_right'))
    num_of_margins_alignments_pairs_before_filtering = len(margins_alignments_pairs_df)


    margins_alignments_pairs_df.drop(
        margins_alignments_pairs_df[
            (
                    margins_alignments_pairs_df['margin_in_region_in_other_nuccore_is_on_forward_strand'] &
                    (margins_alignments_pairs_df['sstart_left'] >= margins_alignments_pairs_df['sstart_right'])
            )
            |
            (
                    (~margins_alignments_pairs_df['margin_in_region_in_other_nuccore_is_on_forward_strand']) &
                    (margins_alignments_pairs_df['sstart_left'] <= margins_alignments_pairs_df['sstart_right'])
            )
            ].index,
        inplace=True,
    )
    num_of_margins_alignments_pairs_such_that_both_margins_align_to_same_strand_and_in_the_right_order = len(margins_alignments_pairs_df)

    # print('margins_alignments_pairs_df')
    # print(margins_alignments_pairs_df)

    margins_alignments_pairs_df.loc[:, 'max_evalue'] = margins_alignments_pairs_df[['evalue_left', 'evalue_right']].max(axis=1)
    margins_alignments_pairs_df.loc[:, 'smax'] = margins_alignments_pairs_df[['sstart_left', 'send_left', 'sstart_right', 'send_right']].max(axis=1)
    margins_alignments_pairs_df.loc[:, 'smin'] = margins_alignments_pairs_df[['sstart_left', 'send_left', 'sstart_right', 'send_right']].min(axis=1)
    margins_alignments_pairs_df.loc[:, 'qmax'] = margins_alignments_pairs_df[['qstart_left', 'qend_left', 'qstart_right', 'qend_right']].max(axis=1)
    margins_alignments_pairs_df.loc[:, 'qmin'] = margins_alignments_pairs_df[['qstart_left', 'qend_left', 'qstart_right', 'qend_right']].min(axis=1)

    margins_alignments_pairs_df.loc[:, 'len_of_spanning_region_in_potential_mis_region_with_margins'] = (
            margins_alignments_pairs_df['qmax'] - margins_alignments_pairs_df['qmin'] + 1)
    margins_alignments_pairs_df.loc[:, 'len_of_spanning_region_in_other_nuccore_entry'] = (
            margins_alignments_pairs_df['smax'] - margins_alignments_pairs_df['smin'] + 1)

    # in the paper we say we calculated abs(x - y) / x, but that's equivalent to what we did, because:
    # abs(1 - y/x) == abs(x - y) / x
    margins_alignments_pairs_df.loc[:, 'lens_of_spanning_regions_ratio'] = (
            margins_alignments_pairs_df.loc[:, 'len_of_spanning_region_in_other_nuccore_entry'] /
            margins_alignments_pairs_df.loc[:, 'len_of_spanning_region_in_potential_mis_region_with_margins']
    )
    margins_alignments_pairs_df.loc[:, 'dist_between_lens_of_spanning_regions_ratio_and_1'] = (margins_alignments_pairs_df.loc[:, 'lens_of_spanning_regions_ratio'] - 1).abs()

    margins_alignments_pairs_df.loc[:, 'diff_between_lens_of_spanning_regions'] = (
            margins_alignments_pairs_df.loc[:, 'len_of_spanning_region_in_other_nuccore_entry'] -
            margins_alignments_pairs_df.loc[:, 'len_of_spanning_region_in_potential_mis_region_with_margins']
    )
    # margins_alignments_pairs_df.loc[:, 'abs_diff_between_lens_of_spanning_regions'] = margins_alignments_pairs_df.loc[:, 'diff_between_lens_of_spanning_regions'].abs()

    lens_of_spanning_regions_ratios = [float(x) for x in margins_alignments_pairs_df['lens_of_spanning_regions_ratio']]
    max_evalues = [float(x) for x in margins_alignments_pairs_df['max_evalue']]

    # print('margins_alignments_pairs_df')
    # print(margins_alignments_pairs_df)

    margins_alignments_pairs_df.drop(
        margins_alignments_pairs_df[margins_alignments_pairs_df['dist_between_lens_of_spanning_regions_ratio_and_1'] >
                                    max_dist_between_lens_of_spanning_regions_ratio_and_1].index,
        inplace=True,
    )
    # margins_alignments_pairs_df.drop(
    #     margins_alignments_pairs_df[margins_alignments_pairs_df['abs_diff_between_lens_of_spanning_regions'] > max_abs_diff_between_lens_of_spanning_regions].index,
    #     inplace=True,
    # )

    num_of_margins_alignments_pairs_such_that_both_margins_align_to_same_strand_and_in_the_right_order_and_with_wanted_spanning_region_length = len(
        margins_alignments_pairs_df)

    # print('margins_alignments_pairs_df')
    # print(margins_alignments_pairs_df)

    other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info = {}
    for other_nuccore_accession, curr_margins_alignments_pairs_df in margins_alignments_pairs_df.groupby('sseqid', sort=False):
        potential_regions_in_other_nuccore = curr_margins_alignments_pairs_df[['smin', 'smax']].to_records(index=False).tolist()
        potential_regions_in_other_nuccore_merged = generic_utils.get_merged_intervals(potential_regions_in_other_nuccore)

        region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info = {}
        for merged_region_according_to_which_potential_region_in_other_nuccore_was_chosen in potential_regions_in_other_nuccore_merged:
            merged_region_start, merged_region_end = merged_region_according_to_which_potential_region_in_other_nuccore_was_chosen
            margins_alignments_pairs_that_merged_region_contains_df = curr_margins_alignments_pairs_df[(curr_margins_alignments_pairs_df['smin'] >= merged_region_start) &
                                                                                                       (curr_margins_alignments_pairs_df['smax'] <= merged_region_end)]

            min_dist_between_lens_of_spanning_regions_ratio_and_1 = margins_alignments_pairs_that_merged_region_contains_df[
                'dist_between_lens_of_spanning_regions_ratio_and_1'].min()
            margins_alignments_pairs_that_merged_region_contains_with_min_abs_diff_df = margins_alignments_pairs_that_merged_region_contains_df[
                margins_alignments_pairs_that_merged_region_contains_df['dist_between_lens_of_spanning_regions_ratio_and_1'] ==
                min_dist_between_lens_of_spanning_regions_ratio_and_1
            ]
            best_margins_alignments_pair = margins_alignments_pairs_that_merged_region_contains_with_min_abs_diff_df.loc[
                margins_alignments_pairs_that_merged_region_contains_with_min_abs_diff_df['len_of_spanning_region_in_potential_mis_region_with_margins'].idxmax(), :
            ].to_dict()

            assert best_margins_alignments_pair['dist_between_lens_of_spanning_regions_ratio_and_1'] == min_dist_between_lens_of_spanning_regions_ratio_and_1

            region_in_other_nuccore = (int(best_margins_alignments_pair['smin']), int(best_margins_alignments_pair['smax']))
            is_region_in_other_nuccore_on_forward_strand = best_margins_alignments_pair['margin_in_region_in_other_nuccore_is_on_forward_strand']
            region_in_other_nuccore_preliminary_info = {
                'best_margins_alignments_pair': best_margins_alignments_pair,
                'merged_region_according_to_which_potential_region_in_other_nuccore_was_chosen': (
                    merged_region_according_to_which_potential_region_in_other_nuccore_was_chosen),
                'is_region_in_other_nuccore_on_forward_strand': is_region_in_other_nuccore_on_forward_strand,
            }
            region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info[region_in_other_nuccore] = region_in_other_nuccore_preliminary_info

        other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info[other_nuccore_accession] = (
            region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info)



    regions_in_other_nuccores_preliminary_info = {
        'blast_left_and_right_margins_results_csv_file_path': input_file_path_blast_left_and_right_margins_results_csv,
        'other_nuccore_accession_to_num_of_left_margin_alignments': other_nuccore_accession_to_num_of_left_margin_alignments,
        'other_nuccore_accession_to_num_of_right_margin_alignments': other_nuccore_accession_to_num_of_right_margin_alignments,
        'num_of_margins_alignments_pairs_before_filtering': num_of_margins_alignments_pairs_before_filtering,
        'num_of_margins_alignments_pairs_such_that_both_margins_align_to_same_strand_and_in_the_right_order': (
            num_of_margins_alignments_pairs_such_that_both_margins_align_to_same_strand_and_in_the_right_order),
        'num_of_margins_alignments_pairs_such_that_both_margins_align_to_same_strand_and_in_the_right_order_and_with_wanted_spanning_region_length': (
            num_of_margins_alignments_pairs_such_that_both_margins_align_to_same_strand_and_in_the_right_order_and_with_wanted_spanning_region_length),
        'other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info': (
            other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info),
        'lens_of_spanning_regions_ratios': lens_of_spanning_regions_ratios,
        'max_evalues': max_evalues,
    }
    with open(output_file_path_regions_in_other_nuccores_preliminary_info_pickle, 'wb') as f:
        pickle.dump(regions_in_other_nuccores_preliminary_info, f, protocol=4)


def write_regions_in_other_nuccores_preliminary_info(
        input_file_path_blast_left_and_right_margins_results_csv,
        left_margin_region,
        right_margin_region,
        max_dist_between_lens_of_spanning_regions_ratio_and_1,
        num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate,
        output_file_path_regions_in_other_nuccores_preliminary_info_pickle,
        blast_margins_output_dir_path,
        nuccore_accession,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_regions_in_other_nuccores_preliminary_info(
        input_file_path_blast_left_and_right_margins_results_csv=input_file_path_blast_left_and_right_margins_results_csv,
        left_margin_region=left_margin_region,
        right_margin_region=right_margin_region,
        max_dist_between_lens_of_spanning_regions_ratio_and_1=max_dist_between_lens_of_spanning_regions_ratio_and_1,
        num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate=num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate,
        output_file_path_regions_in_other_nuccores_preliminary_info_pickle=output_file_path_regions_in_other_nuccores_preliminary_info_pickle,
        blast_margins_output_dir_path=blast_margins_output_dir_path,
        nuccore_accession=nuccore_accession,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=11,
    )


def blast_margins_and_get_regions_in_other_nuccores_preliminary_info(
        blast_margins_output_dir_path,
        margins_fasta_file_path,
        blast_db_path,
        left_margin_region,
        right_margin_region,
        blast_margins_and_identify_regions_in_other_nuccores_args,
        blast_target_taxon_uids,
        nuccore_accession,
        blast_db_edit_time_repr_for_caching=None,
):
    if blast_target_taxon_uids:
        blast_left_and_right_margins_results_csv_file_path = os.path.join(blast_margins_output_dir_path,
                                                                          f'blast_left_and_right_margins_results_for_specified_taxa.csv')
    else:
        blast_left_and_right_margins_results_csv_file_path = os.path.join(blast_margins_output_dir_path,
                                                                          f'blast_left_and_right_margins_results.csv')

    with generic_utils.timing_context_manager('blast left_and_right_margins'):
        blast_interface_and_utils.blast_nucleotide(
            query_fasta_file_path=margins_fasta_file_path,
            blast_db_path=blast_db_path,
            target_species_taxon_uid=blast_target_taxon_uids,
            blast_results_file_path=blast_left_and_right_margins_results_csv_file_path,
            perform_gapped_alignment=False,
            query_strand_to_search='both',
            max_evalue=blast_margins_and_identify_regions_in_other_nuccores_args['max_evalue'],
            seed_len=blast_margins_and_identify_regions_in_other_nuccores_args['seed_len'],
            blast_db_edit_time_repr_for_caching=blast_db_edit_time_repr_for_caching,
            # verbose=True,
        )

    regions_in_other_nuccores_preliminary_info_pickle_file_path = os.path.join(blast_margins_output_dir_path, 'regions_in_other_nuccores_preliminary_info.pickle')
    write_regions_in_other_nuccores_preliminary_info(
        input_file_path_blast_left_and_right_margins_results_csv=blast_left_and_right_margins_results_csv_file_path,
        left_margin_region=left_margin_region,
        right_margin_region=right_margin_region,
        max_dist_between_lens_of_spanning_regions_ratio_and_1=blast_margins_and_identify_regions_in_other_nuccores_args[
            'max_dist_between_lens_of_spanning_regions_ratio_and_1'],
        num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate=blast_margins_and_identify_regions_in_other_nuccores_args[
            'num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate'],
        output_file_path_regions_in_other_nuccores_preliminary_info_pickle=regions_in_other_nuccores_preliminary_info_pickle_file_path,
        blast_margins_output_dir_path=blast_margins_output_dir_path,
        nuccore_accession=nuccore_accession,
    )

    with open(regions_in_other_nuccores_preliminary_info_pickle_file_path, 'rb') as f:
        regions_in_other_nuccores_preliminary_info = pickle.load(f)

    return regions_in_other_nuccores_preliminary_info


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_margins_fasta_file(
        nuccore_entry_fasta_file_path,
        left_margin_region,
        right_margin_region,
        output_file_path_margins_fasta,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    nuccore_entry_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(nuccore_entry_fasta_file_path)
    left_margin_seq = bio_utils.get_region_in_chrom_seq(
        chrom_seq=nuccore_entry_seq,
        start_position=left_margin_region[0],
        end_position=left_margin_region[1],
        region_name='left_margin',
    )
    if bio_utils.does_seq_record_contain_non_ACGT_bases(left_margin_seq, also_allow_acgt=True):
        generic_utils.write_empty_file(output_file_path_margins_fasta)
        return

    right_margin_seq = bio_utils.get_region_in_chrom_seq(
        chrom_seq=nuccore_entry_seq,
        start_position=right_margin_region[0],
        end_position=right_margin_region[1],
        region_name='right_margin',
    )
    if bio_utils.does_seq_record_contain_non_ACGT_bases(right_margin_seq, also_allow_acgt=True):
        generic_utils.write_empty_file(output_file_path_margins_fasta)
        return

    bio_utils.write_records_to_fasta_or_gb_file([left_margin_seq, right_margin_seq], output_file_path_margins_fasta)

def write_margins_fasta_file(
        nuccore_entry_fasta_file_path,
        left_margin_region,
        right_margin_region,
        output_file_path_margins_fasta,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_margins_fasta_file(
        nuccore_entry_fasta_file_path=nuccore_entry_fasta_file_path,
        left_margin_region=left_margin_region,
        right_margin_region=right_margin_region,
        output_file_path_margins_fasta=output_file_path_margins_fasta,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

def get_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore_sorted_by_dist_between_lens_of_spanning_regions_ratio_and_1(
        blast_target_to_regions_in_other_nuccores_preliminary_info,
):
    dist_between_lens_of_spanning_regions_ratio_and_1_and_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore = []
    for blast_target, regions_in_other_nuccores_preliminary_info in blast_target_to_regions_in_other_nuccores_preliminary_info.items():
        other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info = regions_in_other_nuccores_preliminary_info[
            'other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info']
        if other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info:
            for other_nuccore_accession, region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info in (
                    other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info.items()):
                for region_in_other_nuccore, region_in_other_nuccore_preliminary_info in region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info.items():
                    best_margins_alignments_pair = region_in_other_nuccore_preliminary_info['best_margins_alignments_pair']
                    dist_between_lens_of_spanning_regions_ratio_and_1 = best_margins_alignments_pair['dist_between_lens_of_spanning_regions_ratio_and_1']
                    dist_between_lens_of_spanning_regions_ratio_and_1_and_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore.append(
                        (dist_between_lens_of_spanning_regions_ratio_and_1, (blast_target, other_nuccore_accession, region_in_other_nuccore)))

    return [x for _, x in sorted(dist_between_lens_of_spanning_regions_ratio_and_1_and_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore)]

@generic_utils.execute_if_output_doesnt_exist_already
def cached_find_other_nuccore_entries_evidence_for_pi_for_merged_cds_pair_region(
        taxon_uid,
        nuccore_accession,
        merged_cds_pair_region,
        input_file_path_nuccore_entry_fasta,
        merged_cds_pair_region_margin_size,
        input_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle,
        more_taxon_uids_of_the_same_species_for_blasting_local_nt,
        input_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle,
        output_file_path_merged_cds_pair_region_evidence_for_pi_info_pickle,
        output_file_path_merged_cds_pair_region_region_in_other_nuccore_df_csv,
        output_file_path_merged_cds_pair_region_potential_breakpoint_df_csv,
        merged_cds_pair_region_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        local_blast_nt_database_path,
        local_blast_nt_database_update_log_for_caching_only,
        min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion,
        blast_margins_and_identify_regions_in_other_nuccores_args,
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
        input_file_path_debug_local_blast_database_fasta,
        debug_other_nuccore_accession_to_fasta_file_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with open(input_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle, 'rb') as f:
        taxon_local_blast_nt_database_nuccore_entries_info = pickle.load(f)
    with open(input_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle, 'rb') as f:
        downloaded_taxon_wgs_nuccore_entries_info = pickle.load(f)
    nuccore_accession_to_nt_nuccore_entry_len = (taxon_local_blast_nt_database_nuccore_entries_info['nuccore_accession_to_nt_nuccore_entry_len']
                                                 if taxon_local_blast_nt_database_nuccore_entries_info is not None
                                                 else None)
    merged_cds_pair_region_start = merged_cds_pair_region[0]
    merged_cds_pair_region_end = merged_cds_pair_region[1]
    # merged_cds_pair_region_as_str = f'{merged_cds_pair_region_start}_{merged_cds_pair_region_end}'

    regions_in_other_nuccores_dir_path = os.path.join(merged_cds_pair_region_output_dir_path, 'regions_in_other_nuccores')
    pathlib.Path(regions_in_other_nuccores_dir_path).mkdir(parents=True, exist_ok=True)

    merged_cds_pair_region_left_margin = (merged_cds_pair_region_start - merged_cds_pair_region_margin_size, merged_cds_pair_region_start - 1)
    merged_cds_pair_region_right_margin = (merged_cds_pair_region_end + 1, merged_cds_pair_region_end + merged_cds_pair_region_margin_size)

    assert merged_cds_pair_region_left_margin[1] - merged_cds_pair_region_left_margin[0] + 1 == merged_cds_pair_region_margin_size
    assert merged_cds_pair_region_right_margin[1] - merged_cds_pair_region_right_margin[0] + 1 == merged_cds_pair_region_margin_size

    margins_fasta_file_path = os.path.join(merged_cds_pair_region_output_dir_path, 'margins.fasta')
    write_margins_fasta_file(
        nuccore_entry_fasta_file_path=input_file_path_nuccore_entry_fasta,
        left_margin_region=merged_cds_pair_region_left_margin,
        right_margin_region=merged_cds_pair_region_right_margin,
        output_file_path_margins_fasta=margins_fasta_file_path,
    )
    merged_cds_pair_region_with_margins = (merged_cds_pair_region_left_margin[0], merged_cds_pair_region_right_margin[1])

    blast_target_to_regions_in_other_nuccores_preliminary_info = None
    other_nuccore_accession_and_region_in_other_nuccore_to_evidence_for_pi_info = None
    merged_cds_pair_region_with_margins_fasta_file_path = None
    other_nuccore_accession_and_region_in_other_nuccore_of_seqs_identical_to_merged_cds_pair_region_with_margins = None

    region_in_other_nuccore_potential_breakpoint_dfs = []
    region_in_other_nuccore_flat_dicts = []
    num_of_identical_regions_in_other_nuccores = 0
    num_of_non_identical_regions_in_other_nuccores = 0

    if generic_utils.is_file_empty(margins_fasta_file_path):
        merged_cds_pair_region_skipped_because_margins_contained_non_ACGT_bases = True
    else:
        merged_cds_pair_region_skipped_because_margins_contained_non_ACGT_bases = False

        blast_target_to_regions_in_other_nuccores_preliminary_info = {}
        blast_targets = ('local_debug',) if debug_other_nuccore_accession_to_fasta_file_path else ('local_wgs', 'local_nt')
        for blast_target in blast_targets:
            if blast_target == 'local_nt':
                if not nuccore_accession_to_nt_nuccore_entry_len:
                    continue
                blast_margins_output_dir_path = os.path.join(merged_cds_pair_region_output_dir_path, f'blast_margins_to_nt')
                blast_db_path = local_blast_nt_database_path
                # print(f'taxon_uid: {taxon_uid}')
                # print(f'nuccore_accession_to_nt_nuccore_entry_len: {nuccore_accession_to_nt_nuccore_entry_len}')
                blast_target_taxon_uids = [taxon_uid] + more_taxon_uids_of_the_same_species_for_blasting_local_nt
                blast_db_edit_time_repr_for_caching = local_blast_nt_database_update_log_for_caching_only
            elif blast_target == 'local_wgs':
                blast_db_path = downloaded_taxon_wgs_nuccore_entries_info['taxon_wgs_blast_db_path']
                if blast_db_path is None:
                    continue
                blast_margins_output_dir_path = os.path.join(merged_cds_pair_region_output_dir_path, f'blast_margins_to_wgs')
                blast_target_taxon_uids = None
                blast_db_edit_time_repr_for_caching = None
            else:
                assert blast_target == 'local_debug'
                blast_margins_output_dir_path = os.path.join(merged_cds_pair_region_output_dir_path, f'blast_margins_to_debug')
                blast_db_path = input_file_path_debug_local_blast_database_fasta
                blast_target_taxon_uids = None
                blast_db_edit_time_repr_for_caching = None

            pathlib.Path(blast_margins_output_dir_path).mkdir(parents=True, exist_ok=True)

            regions_in_other_nuccores_preliminary_info = blast_margins_and_get_regions_in_other_nuccores_preliminary_info(
                blast_margins_output_dir_path=blast_margins_output_dir_path,
                margins_fasta_file_path=margins_fasta_file_path,
                blast_db_path=blast_db_path,
                left_margin_region=merged_cds_pair_region_left_margin,
                right_margin_region=merged_cds_pair_region_right_margin,
                blast_margins_and_identify_regions_in_other_nuccores_args=blast_margins_and_identify_regions_in_other_nuccores_args,
                blast_target_taxon_uids=blast_target_taxon_uids,
                nuccore_accession=nuccore_accession,
                blast_db_edit_time_repr_for_caching=blast_db_edit_time_repr_for_caching,
            )
            blast_target_to_regions_in_other_nuccores_preliminary_info[blast_target] = regions_in_other_nuccores_preliminary_info

        sorted_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore = (
            get_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore_sorted_by_dist_between_lens_of_spanning_regions_ratio_and_1(
                blast_target_to_regions_in_other_nuccores_preliminary_info))
        if sorted_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore:
            merged_cds_pair_region_with_margins_fasta_file_path = os.path.join(merged_cds_pair_region_output_dir_path, 'merged_cds_pair_region_with_margins.fasta')
            bio_utils.write_region_to_fasta_file(
                input_file_path_fasta=input_file_path_nuccore_entry_fasta,
                region=merged_cds_pair_region_with_margins,
                output_file_path_region_fasta=merged_cds_pair_region_with_margins_fasta_file_path,
            )

        other_nuccore_accession_and_region_in_other_nuccore_to_evidence_for_pi_info = {}
        other_nuccore_accession_and_region_in_other_nuccore_of_seqs_identical_to_merged_cds_pair_region_with_margins = set()
        for blast_target, other_nuccore_accession, region_in_other_nuccore in sorted_blast_target_and_other_nuccore_accession_and_region_in_other_nuccore:
            assert num_of_non_identical_regions_in_other_nuccores <= max_num_of_non_identical_regions_in_other_nuccores_to_analyze
            if num_of_non_identical_regions_in_other_nuccores == max_num_of_non_identical_regions_in_other_nuccores_to_analyze:
                break

            regions_in_other_nuccores_preliminary_info = blast_target_to_regions_in_other_nuccores_preliminary_info[blast_target]

            if blast_target == 'local_nt':
                extracted_fasta_dir_path = os.path.join(other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path, other_nuccore_accession)
                extracted_fasta_file_path = os.path.join(extracted_fasta_dir_path, f'{other_nuccore_accession}.fasta')
                if not os.path.isfile(extracted_fasta_file_path):
                    pathlib.Path(extracted_fasta_dir_path).mkdir(parents=True, exist_ok=True)
                    blast_interface_and_utils.extract_specific_nuccore_fasta_file_from_blast_db(
                        blast_db_path=local_blast_nt_database_path,
                        sequence_identifier=other_nuccore_accession,
                        output_file_path_nuccore_fasta=extracted_fasta_file_path,
                        blast_db_edit_time_repr_for_caching=local_blast_nt_database_update_log_for_caching_only,
                    )
                other_nuccore_len = nuccore_accession_to_nt_nuccore_entry_len[other_nuccore_accession]
                assert (bio_utils.get_chrom_len_from_single_chrom_fasta_file(extracted_fasta_file_path) == other_nuccore_len)
                other_nuccore_fasta_file_path = extracted_fasta_file_path
            elif blast_target == 'local_wgs':
                other_nuccore_fasta_file_path = downloaded_taxon_wgs_nuccore_entries_info['wgs_nuccore_accession_to_wgs_nuccore_entry_fasta_file_path'][
                    other_nuccore_accession]
            else:
                assert blast_target == 'local_debug'
                other_nuccore_fasta_file_path = debug_other_nuccore_accession_to_fasta_file_path[other_nuccore_accession]

            region_in_other_nuccore_preliminary_info = regions_in_other_nuccores_preliminary_info[
                'other_nuccore_accession_to_region_in_other_nuccore_to_region_in_other_nuccore_preliminary_info'][other_nuccore_accession][region_in_other_nuccore]
            best_margins_alignments_pair = region_in_other_nuccore_preliminary_info['best_margins_alignments_pair']
            dist_between_lens_of_spanning_regions_ratio_and_1 = best_margins_alignments_pair['dist_between_lens_of_spanning_regions_ratio_and_1']
            len_of_matching_region_in_potential_mis_region_with_margins = int(best_margins_alignments_pair['len_of_spanning_region_in_potential_mis_region_with_margins'])
            region_in_other_nuccore_start = region_in_other_nuccore[0]
            region_in_other_nuccore_end = region_in_other_nuccore[1]
            region_in_other_nuccore_as_str = f'{region_in_other_nuccore_start}_{region_in_other_nuccore_end}'
            region_in_other_nuccore_output_dir_path = os.path.join(regions_in_other_nuccores_dir_path,
                                                                   f'{other_nuccore_accession}_{region_in_other_nuccore_as_str}')
            pathlib.Path(region_in_other_nuccore_output_dir_path).mkdir(parents=True, exist_ok=True)

            region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info = get_region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info(
                other_nuccore_fasta_file_path=other_nuccore_fasta_file_path,
                merged_cds_pair_region_with_margins_fasta_file_path=merged_cds_pair_region_with_margins_fasta_file_path,
                region_in_other_nuccore=region_in_other_nuccore,
                is_region_in_other_nuccore_on_forward_strand=region_in_other_nuccore_preliminary_info['is_region_in_other_nuccore_on_forward_strand'],
                min_mauve_total_match_proportion=min_mauve_total_match_proportion,
                min_min_sub_alignment_min_match_proportion=min_min_sub_alignment_min_match_proportion,
                merged_cds_pair_region_with_margins=merged_cds_pair_region_with_margins,
                region_in_other_nuccore_output_dir_path=region_in_other_nuccore_output_dir_path,
            )
            mauve_alignment_covered_bases_in_region_in_other_nuccore_proportion = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info[
                'mauve_alignment_covered_bases_in_region_in_other_nuccore_proportion']
            is_identical_to_merged_cds_pair_region_with_margins = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info[
                'is_identical_to_merged_cds_pair_region_with_margins']
            mauve_total_match_proportion = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info['mauve_total_match_proportion']
            min_interval_in_region_in_other_nuccore_match_proportion = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info[
                'min_interval_in_region_in_other_nuccore_match_proportion']
            min_sub_alignment_min_match_proportion = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info['min_sub_alignment_min_match_proportion']
            satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds = (
                region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info[
                    'satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds'])
            region_in_other_nuccore_match_proportion = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info['region_in_other_nuccore_match_proportion']
            mauve_total_num_of_matches = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info['mauve_total_num_of_matches']
            is_mauve_alignment_collinear = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info['is_collinear_alignment']
            assert type(is_mauve_alignment_collinear) == bool
            mauve_alignment_results_xmfa_file_path = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info[
                'mauve_alignment_results_xmfa_file_path']

            region_in_other_nuccore_potential_breakpoint_df_csv_file_path = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info[
                'region_in_other_nuccore_potential_breakpoint_df_csv_file_path']
            interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path = region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info[
                'interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path']
            if region_in_other_nuccore_potential_breakpoint_df_csv_file_path:
                region_in_other_nuccore_potential_breakpoint_df = pd.read_csv(region_in_other_nuccore_potential_breakpoint_df_csv_file_path, sep='\t',
                                                                              low_memory=False)
                region_in_other_nuccore_potential_breakpoint_df['other_nuccore_accession'] = other_nuccore_accession
                region_in_other_nuccore_potential_breakpoint_df['region_in_other_nuccore_start'] = region_in_other_nuccore_start
                region_in_other_nuccore_potential_breakpoint_df['region_in_other_nuccore_end'] = region_in_other_nuccore_end
                region_in_other_nuccore_potential_breakpoint_dfs.append(region_in_other_nuccore_potential_breakpoint_df)

            evidence_for_pi_info = {
                'region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info': (
                    region_in_other_nuccore_and_cds_pair_region_with_margins_alignment_info),
            }
            other_nuccore_accession_and_region_in_other_nuccore = (other_nuccore_accession, region_in_other_nuccore)
            if is_identical_to_merged_cds_pair_region_with_margins:
                num_of_identical_regions_in_other_nuccores += 1
                other_nuccore_accession_and_region_in_other_nuccore_of_seqs_identical_to_merged_cds_pair_region_with_margins.add(
                    other_nuccore_accession_and_region_in_other_nuccore)
            else:
                other_nuccore_accession_and_region_in_other_nuccore_to_evidence_for_pi_info[
                    other_nuccore_accession_and_region_in_other_nuccore] = evidence_for_pi_info

                region_in_other_nuccore_flat_dict = {
                    # 'taxon_uid': taxon_uid,
                    # 'nuccore_accession': nuccore_accession,
                    # 'merged_cds_pair_region_start': merged_cds_pair_region_start,
                    # 'merged_cds_pair_region_end': merged_cds_pair_region_end,
                    'other_nuccore_accession': other_nuccore_accession,
                    'other_nuccore_location': blast_target,
                    'region_in_other_nuccore_start': region_in_other_nuccore_start,
                    'region_in_other_nuccore_end': region_in_other_nuccore_end,
                    'is_region_in_other_nuccore_on_forward_strand': region_in_other_nuccore_preliminary_info['is_region_in_other_nuccore_on_forward_strand'],

                    'left_margin_alignment_start': min(best_margins_alignments_pair['sstart_left'], best_margins_alignments_pair['send_left']),
                    # 'left_margin_alignment_end' can be inferred from start and len.
                    'left_margin_alignment_len': best_margins_alignments_pair['length_left'],
                    'left_margin_alignment_num_of_mismatches': best_margins_alignments_pair['mismatch_left'],
                    'left_margin_alignment_evalue': best_margins_alignments_pair['evalue_left'],


                    # 'right_margin_alignment_start' can be inferred from end and len.
                    'right_margin_alignment_end': max(best_margins_alignments_pair['sstart_right'], best_margins_alignments_pair['send_right']),
                    'right_margin_alignment_len': best_margins_alignments_pair['length_right'],
                    'right_margin_alignment_num_of_mismatches': best_margins_alignments_pair['mismatch_right'],
                    'right_margin_alignment_evalue': best_margins_alignments_pair['evalue_right'],

                    'dist_between_lens_of_spanning_regions_ratio_and_1': dist_between_lens_of_spanning_regions_ratio_and_1,
                    'mauve_alignment_covered_bases_in_region_in_other_nuccore_proportion': mauve_alignment_covered_bases_in_region_in_other_nuccore_proportion,
                    'mauve_total_match_proportion': mauve_total_match_proportion,
                    'region_in_other_nuccore_match_proportion': region_in_other_nuccore_match_proportion,
                    'min_sub_alignment_min_match_proportion': min_sub_alignment_min_match_proportion,
                    'satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds': (
                        satisfied_mauve_total_match_proportion_and_min_sub_alignment_min_match_proportion_thresholds),
                    'mauve_total_num_of_matches': mauve_total_num_of_matches,
                    'min_interval_in_region_in_other_nuccore_match_proportion': min_interval_in_region_in_other_nuccore_match_proportion,
                    'is_mauve_alignment_collinear': is_mauve_alignment_collinear,
                    'mauve_alignment_results_xmfa_file_path': mauve_alignment_results_xmfa_file_path,
                    'interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path': (
                        interval_in_merged_cds_pair_region_with_margins_to_mauve_alignment_info_pickle_file_path),
                }
                region_in_other_nuccore_flat_dicts.append(region_in_other_nuccore_flat_dict)

                num_of_non_identical_regions_in_other_nuccores += 1
                assert num_of_non_identical_regions_in_other_nuccores <= max_num_of_non_identical_regions_in_other_nuccores_to_analyze
                if num_of_non_identical_regions_in_other_nuccores == max_num_of_non_identical_regions_in_other_nuccores_to_analyze:
                    break

    if region_in_other_nuccore_flat_dicts:
        merged_cds_pair_region_region_in_other_nuccore_df = pd.DataFrame(region_in_other_nuccore_flat_dicts)
        merged_cds_pair_region_region_in_other_nuccore_df.to_csv(output_file_path_merged_cds_pair_region_region_in_other_nuccore_df_csv, sep='\t', index=False)
    else:
        generic_utils.write_empty_file(output_file_path_merged_cds_pair_region_region_in_other_nuccore_df_csv)

    if region_in_other_nuccore_potential_breakpoint_dfs:
        potential_breakpoint_containing_interval_df = pd.concat(region_in_other_nuccore_potential_breakpoint_dfs, ignore_index=True)
        potential_breakpoint_containing_interval_df.to_csv(output_file_path_merged_cds_pair_region_potential_breakpoint_df_csv, sep='\t', index=False)
    else:
        generic_utils.write_empty_file(output_file_path_merged_cds_pair_region_potential_breakpoint_df_csv)

    merged_cds_pair_region_evidence_for_pi_info = {
        'merged_cds_pair_region_skipped_because_margins_contained_non_ACGT_bases': merged_cds_pair_region_skipped_because_margins_contained_non_ACGT_bases,
        'blast_target_to_regions_in_other_nuccores_preliminary_info': blast_target_to_regions_in_other_nuccores_preliminary_info,
        'other_nuccore_accession_and_region_in_other_nuccore_to_evidence_for_pi_info': (
            other_nuccore_accession_and_region_in_other_nuccore_to_evidence_for_pi_info),
        'margins_fasta_file_path': margins_fasta_file_path,
        'merged_cds_pair_region_with_margins': merged_cds_pair_region_with_margins,
        'merged_cds_pair_region_with_margins_fasta_file_path': merged_cds_pair_region_with_margins_fasta_file_path,
        'other_nuccore_accession_and_region_in_other_nuccore_of_seqs_identical_to_merged_cds_pair_region_with_margins': (
            other_nuccore_accession_and_region_in_other_nuccore_of_seqs_identical_to_merged_cds_pair_region_with_margins),

        'num_of_identical_regions_in_other_nuccores': num_of_identical_regions_in_other_nuccores,
        'num_of_non_identical_regions_in_other_nuccores': num_of_non_identical_regions_in_other_nuccores,
    }
    with open(output_file_path_merged_cds_pair_region_evidence_for_pi_info_pickle, 'wb') as f:
        pickle.dump(merged_cds_pair_region_evidence_for_pi_info, f, protocol=4)


def find_other_nuccore_entries_evidence_for_pi_for_merged_cds_pair_region(
        taxon_uid,
        nuccore_accession,
        merged_cds_pair_region,
        input_file_path_nuccore_entry_fasta,
        merged_cds_pair_region_margin_size,
        input_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle,
        more_taxon_uids_of_the_same_species_for_blasting_local_nt,
        input_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle,
        output_file_path_merged_cds_pair_region_evidence_for_pi_info_pickle,
        output_file_path_merged_cds_pair_region_region_in_other_nuccore_df_csv,
        output_file_path_merged_cds_pair_region_potential_breakpoint_df_csv,
        merged_cds_pair_region_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        local_blast_nt_database_path,
        local_blast_nt_database_update_log_for_caching_only,
        min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion,
        blast_margins_and_identify_regions_in_other_nuccores_args,
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
        input_file_path_debug_local_blast_database_fasta,
        debug_other_nuccore_accession_to_fasta_file_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_find_other_nuccore_entries_evidence_for_pi_for_merged_cds_pair_region(
        taxon_uid=taxon_uid,
        nuccore_accession=nuccore_accession,
        merged_cds_pair_region=merged_cds_pair_region,
        input_file_path_nuccore_entry_fasta=input_file_path_nuccore_entry_fasta,
        merged_cds_pair_region_margin_size=merged_cds_pair_region_margin_size,
        input_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle=input_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle,
        more_taxon_uids_of_the_same_species_for_blasting_local_nt=more_taxon_uids_of_the_same_species_for_blasting_local_nt,
        input_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle=input_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle,
        output_file_path_merged_cds_pair_region_evidence_for_pi_info_pickle=output_file_path_merged_cds_pair_region_evidence_for_pi_info_pickle,
        output_file_path_merged_cds_pair_region_region_in_other_nuccore_df_csv=output_file_path_merged_cds_pair_region_region_in_other_nuccore_df_csv,
        output_file_path_merged_cds_pair_region_potential_breakpoint_df_csv=output_file_path_merged_cds_pair_region_potential_breakpoint_df_csv,
        merged_cds_pair_region_output_dir_path=merged_cds_pair_region_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path=other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        local_blast_nt_database_path=local_blast_nt_database_path,
        local_blast_nt_database_update_log_for_caching_only=local_blast_nt_database_update_log_for_caching_only,
        min_mauve_total_match_proportion=min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion=min_min_sub_alignment_min_match_proportion,
        blast_margins_and_identify_regions_in_other_nuccores_args=blast_margins_and_identify_regions_in_other_nuccores_args,
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze=max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
        input_file_path_debug_local_blast_database_fasta=input_file_path_debug_local_blast_database_fasta,
        debug_other_nuccore_accession_to_fasta_file_path=debug_other_nuccore_accession_to_fasta_file_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=15,
    )

def find_other_nuccore_entries_evidence_for_pis_in_nuccore_entry(
        taxon_uid,
        nuccore_accession,
        nuccore_pairs_df,
        minimal_nuccore_entry_info,
        taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path,
        more_taxon_uids_of_the_same_species_for_blasting_local_nt,
        downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path,
        nuccore_entries_output_dir_path,
        other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
        local_blast_nt_database_update_log_for_caching_only,
        min_mauve_total_match_proportion,
        min_min_sub_alignment_min_match_proportion,
        merged_cds_pair_region_margin_size,
        blast_margins_and_identify_regions_in_other_nuccores_args,
        local_blast_nt_database_path,
        max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
        input_file_path_debug_local_blast_database_fasta,
        debug_other_nuccore_accession_to_fasta_file_path,
):
    nuccore_entry_output_dir_path = os.path.join(nuccore_entries_output_dir_path, nuccore_accession)
    nuccore_entry_fasta_file_path = minimal_nuccore_entry_info['fasta_file_path']
    nuccore_entry_len = minimal_nuccore_entry_info['chrom_len']
    nuccore_pairs_df = nuccore_pairs_df.copy()
    # ir_pairs = list(nuccore_pairs_df[['left1', 'right1', 'left2', 'right2']].drop_duplicates().itertuples(index=False, name=None))
    nuccore_pairs_df['cds_pair_region_start'] = nuccore_pairs_df[['left1', 'repeat1_cds_start_pos']].min(axis=1)
    nuccore_pairs_df['cds_pair_region_end'] = nuccore_pairs_df[['right2', 'repeat2_cds_end_pos']].max(axis=1)
    cds_pair_regions_df = nuccore_pairs_df[['cds_pair_region_start', 'cds_pair_region_end']].drop_duplicates()
    cds_pair_regions = list(cds_pair_regions_df.itertuples(index=False, name=None))
    merged_cds_pair_region_to_cds_pair_regions = generic_utils.naive_get_merged_interval_to_intervals(cds_pair_regions)

    merged_cds_pair_regions_output_dir_path  = os.path.join(nuccore_entry_output_dir_path, 'merged_cds_pair_regions')
    merged_cds_pair_region_to_result_file_paths = {}
    merged_cds_pair_region_flat_dicts = []
    merged_cds_pair_region_region_in_other_nuccore_dfs = []
    merged_cds_pair_region_potential_breakpoint_dfs = []
    num_of_merged_cds_pair_region_to_cds_pair_regions = len(merged_cds_pair_region_to_cds_pair_regions)
    for i, merged_cds_pair_region in enumerate(merged_cds_pair_region_to_cds_pair_regions):
        generic_utils.print_and_write_to_log(f'starting work on merged_cds_pair_region {i + 1}/{num_of_merged_cds_pair_region_to_cds_pair_regions} ({merged_cds_pair_region}).')
        merged_cds_pair_region_start = merged_cds_pair_region[0]
        merged_cds_pair_region_end = merged_cds_pair_region[1]

        if (merged_cds_pair_region_start <= merged_cds_pair_region_margin_size) or (merged_cds_pair_region_end > (nuccore_entry_len - merged_cds_pair_region_margin_size)):
            merged_cds_pair_region_flat_dicts.append({
                'taxon_uid': taxon_uid,
                'nuccore_accession': nuccore_accession,
                'merged_cds_pair_region_start': merged_cds_pair_region_start,
                'merged_cds_pair_region_end': merged_cds_pair_region_end,
                'merged_cds_pair_region_skipped_because_margins_exceed_nuccore': True,
            })
            continue

        merged_cds_pair_region_as_str = f'{merged_cds_pair_region_start}_{merged_cds_pair_region_end}'
        merged_cds_pair_region_output_dir_path = os.path.join(merged_cds_pair_regions_output_dir_path, merged_cds_pair_region_as_str)
        pathlib.Path(merged_cds_pair_region_output_dir_path).mkdir(parents=True, exist_ok=True)
        merged_cds_pair_region_evidence_for_pi_info_pickle_file_path = os.path.join(merged_cds_pair_region_output_dir_path,
                                                                                   'merged_cds_pair_region_evidence_for_pi_info.pickle')
        merged_cds_pair_region_region_in_other_nuccore_df_csv_file_path = os.path.join(merged_cds_pair_region_output_dir_path,
                                                                                   'merged_cds_pair_region_region_in_other_nuccore_df.csv')
        merged_cds_pair_region_potential_breakpoint_df_csv_file_path = os.path.join(merged_cds_pair_region_output_dir_path,
                                                                                   'merged_cds_pair_region_potential_breakpoint_df.csv')

        find_other_nuccore_entries_evidence_for_pi_for_merged_cds_pair_region(
            taxon_uid=taxon_uid,
            nuccore_accession=nuccore_accession,
            merged_cds_pair_region=merged_cds_pair_region,
            input_file_path_nuccore_entry_fasta=nuccore_entry_fasta_file_path,
            merged_cds_pair_region_margin_size=merged_cds_pair_region_margin_size,
            input_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle=taxon_local_blast_nt_database_nuccore_entries_info_pickle_file_path,
            more_taxon_uids_of_the_same_species_for_blasting_local_nt=more_taxon_uids_of_the_same_species_for_blasting_local_nt,
            input_file_path_downloaded_taxon_wgs_nuccore_entries_info_pickle=downloaded_taxon_wgs_nuccore_entries_info_pickle_file_path,
            output_file_path_merged_cds_pair_region_evidence_for_pi_info_pickle=merged_cds_pair_region_evidence_for_pi_info_pickle_file_path,
            output_file_path_merged_cds_pair_region_region_in_other_nuccore_df_csv=merged_cds_pair_region_region_in_other_nuccore_df_csv_file_path,
            output_file_path_merged_cds_pair_region_potential_breakpoint_df_csv=merged_cds_pair_region_potential_breakpoint_df_csv_file_path,
            merged_cds_pair_region_output_dir_path=merged_cds_pair_region_output_dir_path,
            other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path=other_nuccore_entries_extracted_from_local_nt_blast_db_dir_path,
            local_blast_nt_database_path=local_blast_nt_database_path,
            local_blast_nt_database_update_log_for_caching_only=local_blast_nt_database_update_log_for_caching_only,
            min_mauve_total_match_proportion=min_mauve_total_match_proportion,
            min_min_sub_alignment_min_match_proportion=min_min_sub_alignment_min_match_proportion,
            blast_margins_and_identify_regions_in_other_nuccores_args=blast_margins_and_identify_regions_in_other_nuccores_args,
            max_num_of_non_identical_regions_in_other_nuccores_to_analyze=max_num_of_non_identical_regions_in_other_nuccores_to_analyze,
            input_file_path_debug_local_blast_database_fasta=input_file_path_debug_local_blast_database_fasta,
            debug_other_nuccore_accession_to_fasta_file_path=debug_other_nuccore_accession_to_fasta_file_path,
        )
        with open(merged_cds_pair_region_evidence_for_pi_info_pickle_file_path, 'rb') as f:
            merged_cds_pair_region_evidence_for_pi_info = pickle.load(f)

        merged_cds_pair_region_flat_dicts.append({
            'taxon_uid': taxon_uid,
            'nuccore_accession': nuccore_accession,
            'merged_cds_pair_region_start': merged_cds_pair_region_start,
            'merged_cds_pair_region_end': merged_cds_pair_region_end,
            'merged_cds_pair_region_skipped_because_margins_exceed_nuccore': False,
            'merged_cds_pair_region_skipped_because_margins_contained_non_ACGT_bases': merged_cds_pair_region_evidence_for_pi_info[
                'merged_cds_pair_region_skipped_because_margins_contained_non_ACGT_bases'],
            'num_of_identical_regions_in_other_nuccores': merged_cds_pair_region_evidence_for_pi_info['num_of_identical_regions_in_other_nuccores'],
            'num_of_non_identical_regions_in_other_nuccores': merged_cds_pair_region_evidence_for_pi_info['num_of_non_identical_regions_in_other_nuccores'],
        })

        merged_cds_pair_region_to_result_file_paths[merged_cds_pair_region] = {
            'merged_cds_pair_region_evidence_for_pi_info_pickle_file_path': merged_cds_pair_region_evidence_for_pi_info_pickle_file_path,
            'merged_cds_pair_region_region_in_other_nuccore_df_csv_file_path': merged_cds_pair_region_region_in_other_nuccore_df_csv_file_path,
            'merged_cds_pair_region_potential_breakpoint_df_csv_file_path': merged_cds_pair_region_potential_breakpoint_df_csv_file_path,
        }

        if not generic_utils.is_file_empty(merged_cds_pair_region_region_in_other_nuccore_df_csv_file_path):
            merged_cds_pair_region_region_in_other_nuccore_df = pd.read_csv(merged_cds_pair_region_region_in_other_nuccore_df_csv_file_path, sep='\t', low_memory=False)
            merged_cds_pair_region_region_in_other_nuccore_df['taxon_uid'] = taxon_uid
            merged_cds_pair_region_region_in_other_nuccore_df['nuccore_accession'] = nuccore_accession
            merged_cds_pair_region_region_in_other_nuccore_df['merged_cds_pair_region_start'] = merged_cds_pair_region_start
            merged_cds_pair_region_region_in_other_nuccore_df['merged_cds_pair_region_end'] = merged_cds_pair_region_end
            merged_cds_pair_region_region_in_other_nuccore_dfs.append(merged_cds_pair_region_region_in_other_nuccore_df)

        if not generic_utils.is_file_empty(merged_cds_pair_region_potential_breakpoint_df_csv_file_path):
            merged_cds_pair_region_potential_breakpoint_df = pd.read_csv(merged_cds_pair_region_potential_breakpoint_df_csv_file_path, sep='\t', low_memory=False)
            merged_cds_pair_region_potential_breakpoint_df['taxon_uid'] = taxon_uid
            merged_cds_pair_region_potential_breakpoint_df['nuccore_accession'] = nuccore_accession
            merged_cds_pair_region_potential_breakpoint_df['merged_cds_pair_region_start'] = merged_cds_pair_region_start
            merged_cds_pair_region_potential_breakpoint_df['merged_cds_pair_region_end'] = merged_cds_pair_region_end
            merged_cds_pair_region_potential_breakpoint_dfs.append(merged_cds_pair_region_potential_breakpoint_df)

    if merged_cds_pair_region_flat_dicts:
        merged_cds_pair_region_df = pd.DataFrame(merged_cds_pair_region_flat_dicts)
    else:
        merged_cds_pair_region_df = None

    if merged_cds_pair_region_region_in_other_nuccore_dfs:
        nuccore_region_in_other_nuccore_df = pd.concat(merged_cds_pair_region_region_in_other_nuccore_dfs, ignore_index=True)
    else:
        nuccore_region_in_other_nuccore_df = None

    if merged_cds_pair_region_potential_breakpoint_dfs:
        nuccore_potential_breakpoint_df = pd.concat(merged_cds_pair_region_potential_breakpoint_dfs, ignore_index=True)
    else:
        nuccore_potential_breakpoint_df = None


    result_df_dict = {
        'merged_cds_pair_region_df': merged_cds_pair_region_df,
        'nuccore_region_in_other_nuccore_df': nuccore_region_in_other_nuccore_df,
        'nuccore_potential_breakpoint_df': nuccore_potential_breakpoint_df,
    }

    other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info = {
        'merged_cds_pair_region_to_result_file_paths': merged_cds_pair_region_to_result_file_paths,
        'merged_cds_pair_region_to_cds_pair_regions': merged_cds_pair_region_to_cds_pair_regions,
    }

    return other_nuccore_entries_evidence_for_pis_in_nuccore_entry_info, result_df_dict