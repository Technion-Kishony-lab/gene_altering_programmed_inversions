import pickle
import pickle

import numpy as np
import pandas as pd
from Bio import AlignIO

from generic import bio_utils
from generic import generic_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


_, mauve_version_stderr = generic_utils.run_cmd_and_get_stdout_and_stderr(['progressiveMauve', '--version'])

# print(f'mauve_version_stderr: {mauve_version_stderr}')
# progressiveMauve  build date Feb 13 2015 at 05:57:13
if 'progressiveMauve  build date Feb 13 2015' not in mauve_version_stderr:
    raise NotImplementedError()


@generic_utils.execute_if_output_doesnt_exist_already
def cached_progressive_mauve(
        input_file_path_seq0_fasta,
        input_file_path_seq1_fasta,
        assume_input_sequences_are_collinear,
        output_file_path_alignment_xmfa,
        output_file_path_backbone_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    # ugh. got unexpected results if when this file existed from previous runs. it was used as kind of a cache
    # (a bug or a feature??) by mauve internally , i guess...
    generic_utils.remove_file_silently(f'{input_file_path_seq0_fasta}.sslist')

    # http://darlinglab.org/mauve/user-guide/progressivemauve.html
    # http://darlinglab.org/mauve/user-guide/aligning.html
    # https://sourceforge.net/p/mauve/code/HEAD/tree/
    cmd_line_words = [
        'progressiveMauve',
        f'--output={output_file_path_alignment_xmfa}',
        f'--backbone-output={output_file_path_backbone_csv}',
        # '--seed-family',
        # '--solid-seeds',
        # '--skip-gapped-alignment',
        # '--no-recursion',
        # '--seed-weight=21',
        # '--seed-weight=7',
        # '--weight=100',
        # '--weight=0',
        # '--no-weight-scaling',
        # '--repeat-penalty=zero',
        # '--scoring-scheme=ancestral',
        # '--scoring-scheme=sp_ancestral',
        # '--collinear',
        # '--disable-cache',
        # '--debug',
        # '--disable-backbone',
        # '--mem-clean',
    ]
    if assume_input_sequences_are_collinear:
        cmd_line_words.append('--collinear')

    cmd_line_words += [input_file_path_seq0_fasta, input_file_path_seq1_fasta]

    generic_utils.run_cmd_and_check_ret_code_and_return_stdout(cmd_line_words, raise_exception_if_stderr_isnt_empty=True, print_stdout_if_verbose=False, verbose=True)

def progressive_mauve(
        input_file_path_seq0_fasta,
        input_file_path_seq1_fasta,
        assume_input_sequences_are_collinear,
        output_file_path_alignment_xmfa,
        output_file_path_backbone_csv,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_progressive_mauve(
        input_file_path_seq0_fasta=input_file_path_seq0_fasta,
        input_file_path_seq1_fasta=input_file_path_seq1_fasta,
        assume_input_sequences_are_collinear=assume_input_sequences_are_collinear,
        output_file_path_alignment_xmfa=output_file_path_alignment_xmfa,
        output_file_path_backbone_csv=output_file_path_backbone_csv,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

def drop_non_alignment_rows_in_backbone_df_in_place(backbone_df):
    backbone_df.drop(
        backbone_df[(backbone_df == 0).any(axis=1)].index,
        inplace=True,
    )
    assert ~(backbone_df == 0).all().all()

def plot_backbone_on_ax(
        ax,
        backbone_df,
        seq0_on_x_axis,
        x_axis_offset=0,
        y_axis_offset=0,
):
    backbone_df = backbone_df.copy()
    drop_non_alignment_rows_in_backbone_df_in_place(backbone_df)

    for _, lcb_row in backbone_df.iterrows():
        lcb_region_in_seq0 = np.array(lcb_row[['seq0_leftend', 'seq0_rightend']])
        lcb_region_in_seq1 = np.array(lcb_row[['seq1_leftend', 'seq1_rightend']])

        if (lcb_region_in_seq0 < [0,0]).all():
            lcb_region_in_seq0 = -lcb_region_in_seq0[::-1]

        assert (lcb_region_in_seq0 > [0,0]).all()
        assert (lcb_region_in_seq1 > [0,0]).all()

        x_axis_lcb = lcb_region_in_seq0
        y_axis_lcb = lcb_region_in_seq1
        if not seq0_on_x_axis:
            x_axis_lcb, y_axis_lcb = y_axis_lcb, x_axis_lcb

        ax.plot(
            x_axis_offset + x_axis_lcb,
            y_axis_offset + y_axis_lcb,
        )




def get_list_of_matching_positions_df_for_each_pairwise_alignment_in_xmfa(xmfa_file_path, str_that_seq0_id_should_contain=None, str_that_seq1_id_should_contain=None,
                                                                          return_only_region_in_seq1_to_min_match_proportion=False):
    # print(f'xmfa_file_path: {xmfa_file_path}')
    alignments = list(AlignIO.parse(xmfa_file_path, 'mauve'))
    list_of_matching_positions_df_per_pairwise_alignment = []
    region_in_seq1_to_min_match_proportion = {}
    for alignment in alignments:
        # print(dir(alignment))
        # exit()
        num_of_seqs = len(alignment)
        if num_of_seqs != 2:
            # raise RuntimeError()
            continue

        seq0, seq1 = alignment

        # seq0_len = len(seq0)
        # seq1_len = len(seq1)
        # print(f'seq0_len: {seq0_len}')
        # print(f'seq1_len: {seq1_len}')

        # print(f"str(seq0).count('-'): {str(seq0).count('-')}")
        # print(f"str(seq1).count('-'): {str(seq1).count('-')}")

        if str_that_seq0_id_should_contain:
            if str_that_seq0_id_should_contain not in seq0.id:
                print(f'seq0.id: {seq0.id}')
                assert False
        if str_that_seq1_id_should_contain:
            if str_that_seq1_id_should_contain not in seq1.id:
                print(f'seq1.id: {seq1.id}')
                assert False

        seq0_as_str = bio_utils.seq_record_to_str(seq0)
        seq1_as_str = bio_utils.seq_record_to_str(seq1)

        seq0_step = seq0.annotations['strand']
        seq1_step = seq1.annotations['strand']
        assert seq0_step in (1, -1)
        assert seq1_step in (1, -1)

        is_seq0_on_forward_strand = seq0_step == 1
        is_seq1_on_forward_strand = seq1_step == 1

        # print(f'\nseq0.annotations: {seq0.annotations}')
        # print(f'seq1.annotations: {seq1.annotations}\n')

        start_pos_in_seq0 = seq0.annotations['start'] + 1
        end_pos_in_seq0 = seq0.annotations['end']
        if not is_seq0_on_forward_strand:
            start_pos_in_seq0, end_pos_in_seq0 = end_pos_in_seq0, start_pos_in_seq0

        start_pos_in_seq1 = seq1.annotations['start'] + 1
        end_pos_in_seq1 = seq1.annotations['end']
        if not is_seq1_on_forward_strand:
            start_pos_in_seq1, end_pos_in_seq1 = end_pos_in_seq1, start_pos_in_seq1

        # print(f'(start_pos_in_seq1, end_pos_in_seq1): {(start_pos_in_seq1, end_pos_in_seq1)}')

        # print('annoying.')
        # print('start_pos_in_seq0, end_pos_in_seq0')
        # print(start_pos_in_seq0, end_pos_in_seq0)
        # print('start_pos_in_seq1, end_pos_in_seq1')
        # print(start_pos_in_seq1, end_pos_in_seq1)
        # continue
        # exit()

        matching_positions_df_rows = []
        curr_pos_in_seq0 = start_pos_in_seq0
        curr_pos_in_seq1 = start_pos_in_seq1
        assert len(seq0_as_str) == len(seq1_as_str)
        for seq0_base, seq1_base in zip(seq0_as_str, seq1_as_str):
            if seq0_base == seq1_base:
                assert seq0_base != '-'
                matching_positions_df_rows.append({'seq0_position': curr_pos_in_seq0,
                                                   'seq1_position': curr_pos_in_seq1})

            if seq0_base != '-':
                curr_pos_in_seq0 += seq0_step
            if seq1_base != '-':
                curr_pos_in_seq1 += seq1_step

        # print(matching_positions_df_rows)
        # print('curr_pos_in_seq0, end_pos_in_seq0, seq0_base')
        # print(curr_pos_in_seq0, end_pos_in_seq0, seq0_base)
        assert curr_pos_in_seq0 - seq0_step == end_pos_in_seq0

        # print('curr_pos_in_seq1, end_pos_in_seq1, seq1_base')
        # print(curr_pos_in_seq1, end_pos_in_seq1, seq1_base)
        assert curr_pos_in_seq1 - seq1_step == end_pos_in_seq1

        match_prop_rel_to_seq0 = len(matching_positions_df_rows) / len(seq0_as_str)
        match_prop_rel_to_seq1 = len(matching_positions_df_rows) / len(seq1_as_str)

        min_match_proportion = min(match_prop_rel_to_seq0, match_prop_rel_to_seq1)
        region_in_seq1_to_min_match_proportion[(start_pos_in_seq1, end_pos_in_seq1)] = min_match_proportion

        # print(f'(start_pos_in_seq1, end_pos_in_seq1, match_prop_rel_to_seq1): {(start_pos_in_seq1, end_pos_in_seq1, match_prop_rel_to_seq1)}')
        # print(f'(start_pos_in_seq1, end_pos_in_seq1, match_prop_rel_to_seq1): {(start_pos_in_seq1, end_pos_in_seq1, match_prop_rel_to_seq1)}')

        if match_prop_rel_to_seq0 < 0.25:
            print(f'\n\nmaybe something is wrong. seems like there are too few matching positions: {match_prop_rel_to_seq0}\n\n')
            # raise RuntimeError('something is wrong. there are too few matching positions.')

        list_of_matching_positions_df_per_pairwise_alignment.append(pd.DataFrame(matching_positions_df_rows))


    if return_only_region_in_seq1_to_min_match_proportion:
        return region_in_seq1_to_min_match_proportion
    return list_of_matching_positions_df_per_pairwise_alignment


def get_region_in_seq1_to_mauve_alignment_info(xmfa_file_path):
    # print(f'xmfa_file_path: {xmfa_file_path}')
    alignments = list(AlignIO.parse(xmfa_file_path, 'mauve'))
    region_in_seq1_to_mauve_alignment_info = {}
    for alignment in alignments:
        num_of_seqs = len(alignment)

        if num_of_seqs not in (1, 2):
            print(f'num_of_seqs: {num_of_seqs}')
            print(f'alignment: {alignment}')
        assert num_of_seqs in (1, 2)

        if num_of_seqs == 1:
            continue

        seq0, seq1 = alignment

        seq0_as_str = bio_utils.seq_record_to_str(seq0)
        seq1_as_str = bio_utils.seq_record_to_str(seq1)

        seq0_step = seq0.annotations['strand']
        seq1_step = seq1.annotations['strand']
        assert seq0_step in (1, -1)
        # assert seq1_step in (1, -1)
        assert seq1_step == 1 # that's just the way it is in mauve (at least the current version i use).

        is_seq0_on_forward_strand = seq0_step == 1
        # is_seq1_on_forward_strand = seq1_step == 1


        start_pos_in_seq0 = seq0.annotations['start'] + 1
        end_pos_in_seq0 = seq0.annotations['end']
        region_in_seq0 = (start_pos_in_seq0, end_pos_in_seq0)
        assert generic_utils.is_interval(region_in_seq0)
        if not is_seq0_on_forward_strand:
            start_pos_in_seq0, end_pos_in_seq0 = end_pos_in_seq0, start_pos_in_seq0

        start_pos_in_seq1 = seq1.annotations['start'] + 1
        end_pos_in_seq1 = seq1.annotations['end']
        region_in_seq1 = (start_pos_in_seq1, end_pos_in_seq1)
        if not generic_utils.is_interval(region_in_seq1):
            assert region_in_seq1 == (1, 0)
            # print(region_in_seq1, region_in_seq0)
            # print(alignment)
            # print(f'seq0.annotations: {seq0.annotations}')
            # print(f'seq1.annotations: {seq1.annotations}')
            num_of_matches = 0
            match_prop_rel_to_seq0 = 0
            match_prop_rel_to_seq1 = 0
            min_match_proportion = 0
        else:
            assert generic_utils.is_interval(region_in_seq1)
            # if not is_seq1_on_forward_strand:
            #     start_pos_in_seq1, end_pos_in_seq1 = end_pos_in_seq1, start_pos_in_seq1

            num_of_matches = 0
            curr_pos_in_seq0 = start_pos_in_seq0
            curr_pos_in_seq1 = start_pos_in_seq1
            assert len(seq0_as_str) == len(seq1_as_str)
            for seq0_base, seq1_base in zip(seq0_as_str, seq1_as_str):
                if seq0_base == seq1_base:
                    assert seq0_base != '-'
                    num_of_matches += 1

                if seq0_base != '-':
                    curr_pos_in_seq0 += seq0_step
                if seq1_base != '-':
                    curr_pos_in_seq1 += seq1_step

            assert curr_pos_in_seq0 - seq0_step == end_pos_in_seq0
            assert curr_pos_in_seq1 - seq1_step == end_pos_in_seq1

            match_prop_rel_to_seq0 = num_of_matches / len(seq0_as_str)
            match_prop_rel_to_seq1 = num_of_matches / len(seq1_as_str)

            min_match_proportion = min(match_prop_rel_to_seq0, match_prop_rel_to_seq1)

        mauve_alignment_info = {
            'region_in_seq0': region_in_seq0,
            'is_seq0_on_forward_strand': is_seq0_on_forward_strand,
            'num_of_matches': num_of_matches,
            'match_prop_rel_to_seq0': match_prop_rel_to_seq0,
            'match_prop_rel_to_seq1': match_prop_rel_to_seq1,
            'min_match_proportion': min_match_proportion,
        }
        region_in_seq1_to_mauve_alignment_info[region_in_seq1] = mauve_alignment_info

    return region_in_seq1_to_mauve_alignment_info


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_total_mauve_alignment_info(
        input_file_path_xmfa,
        output_file_path_total_mauve_alignment_info_pickle,
):
    region_in_seq1_to_mauve_alignment_info = get_region_in_seq1_to_mauve_alignment_info(input_file_path_xmfa)

    num_of_sub_alignments = len(region_in_seq1_to_mauve_alignment_info)
    total_len_of_aligned_regions_in_seq0 = sum(x['region_in_seq0'][1] - x['region_in_seq0'][0] + 1 for x in region_in_seq1_to_mauve_alignment_info.values())
    total_len_of_aligned_regions_in_seq1 = sum(x[1] - x[0] + 1 for x in region_in_seq1_to_mauve_alignment_info)
    total_num_of_matches = sum(x['num_of_matches'] for x in region_in_seq1_to_mauve_alignment_info.values())
    all_sub_alignments_are_to_seq0_forward_strand = all(x['is_seq0_on_forward_strand'] for x in region_in_seq1_to_mauve_alignment_info.values())

    total_mauve_alignment_info = {
        'num_of_sub_alignments': num_of_sub_alignments,
        'total_len_of_aligned_regions_in_seq0': total_len_of_aligned_regions_in_seq0,
        'total_len_of_aligned_regions_in_seq1': total_len_of_aligned_regions_in_seq1,
        'total_num_of_matches': total_num_of_matches,
        'all_sub_alignments_are_to_seq0_forward_strand': all_sub_alignments_are_to_seq0_forward_strand,
    }

    with open(output_file_path_total_mauve_alignment_info_pickle, 'wb') as f:
        pickle.dump(total_mauve_alignment_info, f, protocol=4)

def write_total_mauve_alignment_info(
        input_file_path_xmfa,
        output_file_path_total_mauve_alignment_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_total_mauve_alignment_info(
        input_file_path_xmfa=input_file_path_xmfa,
        output_file_path_total_mauve_alignment_info_pickle=output_file_path_total_mauve_alignment_info_pickle,
    )


# def write_seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand_pickle(
#         input_file_path_backbone_csv,
#         output_file_path_seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand_pickle,
# ):
#     backbone_df = pd.read_csv(input_file_path_backbone_csv, sep='\t')
#
#     drop_non_alignment_rows_in_backbone_df_in_place(backbone_df)
#
#     with open(output_file_path_seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand_pickle, 'wb') as f:
#         pickle.dump(file_path_seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand, f, protocol=4)

def get_backbone_lcbs_info(backbone_csv_file_path):
    if generic_utils.get_num_of_lines_in_text_file(backbone_csv_file_path) <= 1:
        backbone_lcbs_info = {
            'total_num_of_bases_in_backbone_lcbs_of_seq0': 0,
            'total_num_of_bases_in_backbone_lcbs_of_seq1': 0,
            'seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand': {},
            'seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand': {},
        }
    else:
        backbone_df = pd.read_csv(backbone_csv_file_path, sep='\t')

        drop_non_alignment_rows_in_backbone_df_in_place(backbone_df)

        total_num_of_bases_in_backbone_lcbs_of_seq0 = ((backbone_df['seq0_leftend'] - backbone_df['seq0_rightend']).abs() + 1).sum()
        total_num_of_bases_in_backbone_lcbs_of_seq1 = ((backbone_df['seq1_leftend'] - backbone_df['seq1_rightend']).abs() + 1).sum()

        seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand = {}
        for _, lcb_row in backbone_df.iterrows():
            lcb_region_in_seq0 = (int(lcb_row['seq0_leftend']), int(lcb_row['seq0_rightend']))
            lcb_region_in_seq1 = (int(lcb_row['seq1_leftend']), int(lcb_row['seq1_rightend']))

            seq0_region_signs = set(np.sign(lcb_region_in_seq0))
            seq1_region_signs = set(np.sign(lcb_region_in_seq1))
            assert len(seq0_region_signs) == 1
            assert len(seq1_region_signs) == 1
            assert -1 not in seq1_region_signs # that's just the way it is in mauve (at least the current version i use).

            is_seq0_on_forward_strand = None if (seq0_region_signs == {0}) else (seq0_region_signs == {1})
            if not is_seq0_on_forward_strand:
                lcb_region_in_seq0 = (-lcb_region_in_seq0[0], -lcb_region_in_seq0[1])

            assert generic_utils.is_interval(lcb_region_in_seq0)
            assert generic_utils.is_interval(lcb_region_in_seq1)

            assert lcb_region_in_seq0 not in seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand
            seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand[lcb_region_in_seq0] = (lcb_region_in_seq1, is_seq0_on_forward_strand)

        seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand = {
            lcb_region_in_seq1: (lcb_region_in_seq0, is_seq0_on_forward_strand)
            for lcb_region_in_seq0, (lcb_region_in_seq1, is_seq0_on_forward_strand)
            in seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand.items()
        }
        assert len(seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand) == len(seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand)

        # seq0_aligned_region_len = (
        #     max(x[1] for x in seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand) -
        #     min(x[0] for x in seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand) + 1
        # )
        # seq1_aligned_region_len = (
        #     max(x[1] for x in seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand) -
        #     min(x[0] for x in seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand) + 1
        # )


        backbone_lcbs_info = {
            'total_num_of_bases_in_backbone_lcbs_of_seq0': total_num_of_bases_in_backbone_lcbs_of_seq0,
            'total_num_of_bases_in_backbone_lcbs_of_seq1': total_num_of_bases_in_backbone_lcbs_of_seq1,
            'seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand': seq0_backbone_lcb_to_seq1_backbone_lcb_and_is_seq0_on_forward_strand,
            'seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand': seq1_backbone_lcb_to_seq0_backbone_lcb_and_is_seq0_on_forward_strand,
        }

    return backbone_lcbs_info

def get_base_depth_df_per_alignment(
        xmfa_file_path,
        min_alignment_len=1,
):
    alignments = list(AlignIO.parse(xmfa_file_path, 'mauve'))
    # list_of_matching_positions_df_per_pairwise_alignment = []
    base_depth_df_per_alignment = []
    for alignment in alignments:
        # print(alignment.annotations)
        # print(alignment.column_annotations)
        # print(alignment)
        # print(alignment.format())
        # print(dir(alignment))
        # exit()
        alignment_len = alignment.get_alignment_length()
        # print(f'len(alignment): {len(alignment)}')
        num_of_aligned_seqs = len(alignment)
        assert num_of_aligned_seqs >= 1
        if (alignment_len < min_alignment_len) or (num_of_aligned_seqs == 1):
            continue

        base_depth_df_rows = [{base: 0 for base in bio_utils.DNA_BASES} for _ in range(alignment_len)]
        for seq in alignment:
            # print(f'seq.description: {seq.description}')
            # print(f'seq.name: {seq.name}')
            # print(f'seq.id: {seq.id}')
            # print(seq)
            # print(dir(seq))
            # print(seq.annotations['strand'])
            for i, base in enumerate(bio_utils.seq_record_to_str(seq)):
                if base == '-':
                    continue
                assert base in bio_utils.DNA_BASES
                # is_dna_base = base in bio_utils.DNA_BASES
                base_depth_df_rows[i][base] += 1
            # print(bio_utils.seq_record_to_str(seq)[:50])
            # print(bio_utils.seq_record_to_str(seq))
        # DNA_BASE_TO_BASE_INDEX
        base_depth_df = pd.DataFrame(base_depth_df_rows)
        base_depth_df_per_alignment.append(base_depth_df)

    return base_depth_df_per_alignment

def get_base_depth_and_freq_df_per_alignment(
        xmfa_file_path,
        min_alignment_len=1,
):
    base_depth_df_per_alignment = get_base_depth_df_per_alignment(
        xmfa_file_path=xmfa_file_path,
        min_alignment_len=min_alignment_len,
    )
    base_depth_and_freq_df_per_alignment = []
    for base_depth_df in base_depth_df_per_alignment:
        base_depth_and_freq_df = base_depth_df.copy()
        base_depth_and_freq_df.loc[:, 'TOTAL_COVERAGE'] = base_depth_and_freq_df.sum(axis=1)
        assert (base_depth_and_freq_df['TOTAL_COVERAGE'] >= 1).all()
        for base in bio_utils.DNA_BASES:
            base_depth_and_freq_df.loc[:, f'{base}_FREQ'] = base_depth_and_freq_df[base] / base_depth_and_freq_df['TOTAL_COVERAGE']

        assert np.isclose(base_depth_and_freq_df.loc[:, [f'{base}_FREQ' for base in bio_utils.DNA_BASES]].sum(axis=1), 1).all()
        base_depth_and_freq_df_per_alignment.append(base_depth_and_freq_df)

    return base_depth_and_freq_df_per_alignment

def get_minor_allele_freq_series_per_alignment(
        xmfa_file_path,
        min_alignment_len=1,
        min_total_coverage=0,
):
    base_depth_and_freq_df_per_alignment = get_base_depth_and_freq_df_per_alignment(
        xmfa_file_path=xmfa_file_path,
        min_alignment_len=min_alignment_len,
    )
    minor_allele_freq_series_per_alignment = []
    for base_depth_and_freq_df in base_depth_and_freq_df_per_alignment:
        filtered_base_depth_and_freq_df = base_depth_and_freq_df.copy()
        filtered_base_depth_and_freq_df.drop(
            filtered_base_depth_and_freq_df[filtered_base_depth_and_freq_df['TOTAL_COVERAGE'] < min_total_coverage].index,
            inplace=True,
        )
        base_freqs_mat = filtered_base_depth_and_freq_df[[f'{base}_FREQ' for base in bio_utils.DNA_BASES]].to_numpy()
        base_freqs_mat.sort(axis=1)
        minor_allele_freq_series = pd.Series(base_freqs_mat[:, -2], index=filtered_base_depth_and_freq_df.index)
        minor_allele_freq_series_per_alignment.append(minor_allele_freq_series)

    return minor_allele_freq_series_per_alignment

def plot_xmfa_on_ax(
        ax,
        mauve_alignment_results_xmfa_file_path,
        str_that_seq0_id_should_contain,
        str_that_seq1_id_should_contain,
        x_axis_offset=0,
        y_axis_offset=0,
):
    list_of_matching_positions_df_per_pairwise_alignment = get_list_of_matching_positions_df_for_each_pairwise_alignment_in_xmfa(
        mauve_alignment_results_xmfa_file_path,
        str_that_seq0_id_should_contain=str_that_seq0_id_should_contain,
        str_that_seq1_id_should_contain=str_that_seq1_id_should_contain,
    )

    print(f'len(list_of_matching_positions_df_per_pairwise_alignment): {len(list_of_matching_positions_df_per_pairwise_alignment)}')

    # there isn't a real need for the color arg. it would have been a different color anyway due to a different call to ax.scatter.
    # for matching_positions_df, color in zip(list_of_matching_positions_df_per_pairwise_alignment, itertools.cycle(plot_utils.DISTINGUISHABLE_VISIBLE_MATPLOTLIB_COLORS[1:])):

    # print(list_of_matching_positions_df_per_pairwise_alignment)

    for matching_positions_df in list_of_matching_positions_df_per_pairwise_alignment:
        # print('matching_positions_df')
        # print(matching_positions_df)
        ax.scatter(
            matching_positions_df['seq1_position'] + x_axis_offset,
            matching_positions_df['seq0_position'] + y_axis_offset,
            s=4,
            # color=color,
        )

