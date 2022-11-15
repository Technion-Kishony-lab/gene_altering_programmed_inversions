#!/usr/bin/env python
import io
import warnings

import numpy as np
import pandas as pd

from generic import bio_utils
from generic import blast_interface_and_utils
from generic import generic_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

INVERTED_OR_DIRECT_OR_BOTH_TO_BLAST_STRAND_ARG = {
    'inverted': 'minus',
    'direct': 'plus',
    'both': 'both',
}

MY_BLAST_LIKE_FORMAT_REPEATS_PAIRS_COLUMNS = [
    'chrom',
    'matches_percents',
    'mismatch',
    'gapopen',
    'evalue',
    'is_inverted_repeats_pair',
    'left1',
    'right1',
    'left2',
    'right2',
    'spacer_len',
]

def write_repeats_pairs_df_to_csv_in_my_blast_like_format(pairs_df, csv_file_path):
    if pairs_df.empty:
        pairs_df = pd.DataFrame(columns=MY_BLAST_LIKE_FORMAT_REPEATS_PAIRS_COLUMNS)
    else:
        assert set(pairs_df) >= set(MY_BLAST_LIKE_FORMAT_REPEATS_PAIRS_COLUMNS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        pairs_df.to_csv(csv_file_path, sep='\t', index=False)

def read_repeats_pairs_df_from_csv_in_my_blast_like_format(csv_file_path):
    pairs_df = pd.read_csv(csv_file_path, sep='\t')
    assert set(pairs_df) >= set(MY_BLAST_LIKE_FORMAT_REPEATS_PAIRS_COLUMNS)
    return pairs_df

def filter_imperfect_repeats_pairs_df(
        pairs_df,
        min_repeat_len,
        max_spacer_len,
        min_spacer_len,
):
    verify_arguments_for_filtering_imperfect_repeats_pairs(
        min_repeat_len=min_repeat_len,
        max_spacer_len=max_spacer_len,
        min_spacer_len=min_spacer_len,
    )
    # filter according to min_spacer_len threshold
    pairs_df = pairs_df[(pairs_df['spacer_len'] >= min_spacer_len)]

    # filter according to max_spacer_len and min_repeat_len_to_ignore_max_spacer_len thresholds
    pairs_df = pairs_df[(pairs_df['spacer_len'] <= max_spacer_len)]
    # print('after filter according to max_spacer_len and min_repeat_len_to_ignore_max_spacer_len thresholds')
    # print(pairs_df)

    # filter according to min_repeat_len threshold
    pairs_df = pairs_df[pairs_df['repeat_len'] >= min_repeat_len]
    # print('after filter according to min_repeat_len threshold')
    # print(pairs_df)

    return pairs_df

def verify_arguments_for_filtering_imperfect_repeats_pairs(
        min_repeat_len=4,
        max_spacer_len=1,
        min_repeat_len_to_ignore_max_spacer_len=4,
        min_spacer_len=0,
):
    if min_repeat_len < 4:
        raise RuntimeError('min_repeat_len must be at least 4.')

    if min_repeat_len_to_ignore_max_spacer_len < 4:
        raise RuntimeError('min_repeat_len_to_ignore_max_spacer_len must be at least 4.')

    if max_spacer_len < 1:
        raise RuntimeError('max_spacer_len must be at least 1.')

    # if min_spacer_len < 0:
    #     raise RuntimeError('min_spacer_len must be at least 0.')


def verify_seed_len(seed_len):
    if seed_len < 4:
        raise RuntimeError('seed_len must be at least 4.')


def do_inplace_init_processing_of_blast_results_df(pairs_df):
    assert (pairs_df['qseqid'] == pairs_df['sseqid']).all()
    pairs_df.drop('sseqid', axis=1, inplace=True)
    pairs_df.rename(columns={
        'qseqid': 'chrom',
        'length': 'repeat_len',
        'pident': 'matches_percents',
    }, inplace=True)

    assert (pairs_df['qstart'] < pairs_df['qend']).all()
    pairs_df.loc[:, 'is_inverted_repeats_pair'] = pairs_df['sstart'] > pairs_df['send']
    pairs_df.loc[pairs_df['is_inverted_repeats_pair'], ['sstart', 'send']] = pairs_df.loc[pairs_df['is_inverted_repeats_pair'], ['send', 'sstart']].rename(
        columns={'send': 'sstart', 'sstart': 'send'})  # Turns out that loc[] cares about names.
    assert (pairs_df['sstart'] < pairs_df['send']).all()
    pairs_df.rename(columns={
        'qstart': 'q_left',
        'qend': 'q_right',
        'sstart': 's_left',
        'send': 's_right',
    }, inplace=True)

    pairs_df.loc[:, 's_is_repeat1'] = pairs_df['s_left'] < pairs_df['q_left']
    pairs_df.loc[pairs_df['s_is_repeat1'], ['q_left', 'q_right', 's_left', 's_right']] = pairs_df.loc[pairs_df['s_is_repeat1'], ['q_left', 'q_right', 's_left', 's_right']].rename(
        columns={'q_left': 's_left', 'q_right': 's_right', 's_left': 'q_left', 's_right': 'q_right'})

    pairs_df.rename(columns={
        'q_left': 'left1',
        'q_right': 'right1',
        's_left': 'left2',
        's_right': 'right2',
    }, inplace=True)

    pairs_df.drop_duplicates(subset=['left1', 'right1', 'left2', 'right2', 'is_inverted_repeats_pair'], inplace=True)
    # print('after drop_duplicates')
    # print(pairs_df)

    pairs_df.drop([
        'bitscore',
        's_is_repeat1',
    ], axis=1, inplace=True)

    pairs_df.loc[:, 'spacer_len'] = pairs_df['left2'] - pairs_df['right1'] - 1


@generic_utils.execute_if_output_doesnt_exist_already
def cached_find_imperfect_repeats_pairs(
        input_file_path_fasta,
        min_repeat_len,
        seed_len,
        max_spacer_len,
        max_evalue,
        output_file_path_imperfect_repeats_pairs_csv,
        inverted_or_direct_or_both,
        min_spacer_len,
        match_and_mismatch_scores,
):
    verify_arguments_for_filtering_imperfect_repeats_pairs(
        min_repeat_len=min_repeat_len,
        max_spacer_len=max_spacer_len,
        min_spacer_len=min_spacer_len,
    )

    verify_seed_len(seed_len)

    if inverted_or_direct_or_both not in INVERTED_OR_DIRECT_OR_BOTH_TO_BLAST_STRAND_ARG:
        raise RuntimeError(f'inverted_or_direct_or_both must be one of these options: {tuple(sorted(INVERTED_OR_DIRECT_OR_BOTH_TO_BLAST_STRAND_ARG))}.')

    chrom_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(input_file_path_fasta)

    chrom_name = chrom_seq.id
    chrom_len = len(chrom_seq)
    max_spacer_len = min(max_spacer_len, chrom_len - 3)

    # num_of_bases_to_concatenate_to_simulate_circularity = 0
    chrom_seq_to_blast = chrom_seq
    chrom_seq_to_blast.id = f'{chrom_name}'

    chrom_seq_to_blast_fasta_file_path = input_file_path_fasta

    if 0:
        # ATTENTION: beware of changing this calculation of max_evalue_arg, as it seems that the evalue threshold also affects BLAST's choice of x_dropoff. beware.
        # IMPORTANT: why do we need to specify max_evalue_arg?
        #    because the evalue of a 20 bp IR pair for a nuccore of length 10e6 (for example) is >10, and so by default BLAST would not report such IR pair.
        chrom_seq_to_blast_len = len(chrom_seq_to_blast)
        approx_num_of_possible_repeats_pairs = chrom_seq_to_blast_len ** 2
        # I would have multiplied by 2 for direct/inverted (while ignoring palindromes), but BLAST ignores (when it calculates the evalue of a hit) whether we asked to search
        # 1 or 2 strands of the query for alignments to the database. anyway, my calculation is not the one that BLAST uses. only a simplified approximation.
        # for exactly how to calculate BLAST's evalue, see https://www.ncbi.nlm.nih.gov/BLAST/tutorial/Altschul-1.html.
        approx_probability_of_perfect_repeats_pair_of_length_seed_len_to_be_found = 1 / 4 ** seed_len
        approx_evalue_of_perfect_repeats_pair_of_length_seed_len = (approx_num_of_possible_repeats_pairs *
                                                                    approx_probability_of_perfect_repeats_pair_of_length_seed_len_to_be_found)
        # * 5 just to be safe.
        max_evalue_arg = approx_evalue_of_perfect_repeats_pair_of_length_seed_len * 5
        if max_evalue_arg <= 10:
            max_evalue_arg = None # the default is 10, so no need to specify a lower one. (i am also worried that for a very low evalue BLAST would do weird stuff to x_dropoff...)

    # print(f'max_evalue_arg: {max_evalue_arg}')
    blast_strand_arg = INVERTED_OR_DIRECT_OR_BOTH_TO_BLAST_STRAND_ARG[inverted_or_direct_or_both]

    blast_results_stringio = io.StringIO(blast_interface_and_utils.blast_nucleotide(
        query_fasta_file_path=chrom_seq_to_blast_fasta_file_path,
        blast_db_path=chrom_seq_to_blast_fasta_file_path,
        perform_gapped_alignment=False,
        query_strand_to_search=blast_strand_arg,
        seed_len=seed_len,
        # max_evalue=MAX_EVALUE_BIG_ENOUGH_TO_EFFECTIVELY_DISABLE_THRESHOLD, # this seems to be a bad idea, because it causes BLAST to choose a smaller x_dropoff
        # oh my. what a bad call. indeed it wasn't ok to comment this out. the default evalue is 10, and so for a nuccore of length 16Mbps, the evalue of
        # a prefcet IR pair of length 20 is around 232. ugh.
        max_evalue=max_evalue,
        match_and_mismatch_scores=match_and_mismatch_scores,
        # verbose=False,
    ))

    # blast_results_str = blast_results_stringio.getvalue()

    pairs_df = blast_interface_and_utils.read_blast_results_df(blast_results_stringio)
    del blast_results_stringio
    # print(pairs_df)

    do_inplace_init_processing_of_blast_results_df(pairs_df)
    # print(pairs_df)

    assert (pairs_df['repeat_len'] == (pairs_df['right1'] - pairs_df['left1'] + 1)).all()
    assert (pairs_df['repeat_len'] == (pairs_df['right2'] - pairs_df['left2'] + 1)).all()

    # 220311: ugh. this assert sometimes fail. seems like an actual bug in BLAST. e.g., for NZ_JAIWNA010000001.1 blast reported an alignment of length 19, even though the seed
    # length was 20. ugh.
    # assert (pairs_df['repeat_len'] >= seed_len).all()
    pairs_df = pairs_df[pairs_df['repeat_len'] >= seed_len]

    assert (pairs_df['left1'] >= 1).all()
    assert (pairs_df['right2'] <= chrom_len).all()

    pairs_df = filter_imperfect_repeats_pairs_df(
        pairs_df=pairs_df,
        min_repeat_len=min_repeat_len,
        max_spacer_len=max_spacer_len,
        min_spacer_len=min_spacer_len,
    )

    # print(pairs_df)

    write_repeats_pairs_df_to_csv_in_my_blast_like_format(pairs_df, output_file_path_imperfect_repeats_pairs_csv)

def find_imperfect_repeats_pairs(
        input_file_path_fasta,
        min_repeat_len,
        seed_len,
        max_spacer_len,
        max_evalue,
        output_file_path_imperfect_repeats_pairs_csv,
        inverted_or_direct_or_both='both',
        min_spacer_len=1,
        match_and_mismatch_scores=None,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_find_imperfect_repeats_pairs(
        input_file_path_fasta=input_file_path_fasta,
        min_repeat_len=min_repeat_len,
        seed_len=seed_len,
        max_spacer_len=max_spacer_len,
        max_evalue=max_evalue,
        output_file_path_imperfect_repeats_pairs_csv=output_file_path_imperfect_repeats_pairs_csv,
        inverted_or_direct_or_both=inverted_or_direct_or_both,
        min_spacer_len=min_spacer_len,
        match_and_mismatch_scores=match_and_mismatch_scores,
    )

if __name__ == '__main__':
    raise RuntimeError('This is a library. No main() here.')
