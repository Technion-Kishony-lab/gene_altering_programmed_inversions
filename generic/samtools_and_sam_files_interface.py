import os
import os.path
import warnings
import contextlib
import io
import re
import csv
import collections

import numpy as np
import pandas as pd

from generic import generic_utils
from generic import bio_utils
import pickle

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

SAM_FILE_MANDATORY_COLUMN_NAMES = [
    'QNAME',
    'FLAG',
    'RNAME',
    'POS',
    'MAPQ',
    'CIGAR',
    'RNEXT',
    'PNEXT',
    'TLEN',
    'SEQ',
    'QUAL',
]
NUM_OF_MANDATORY_COLUMNS_IN_SAME_FILE = len(SAM_FILE_MANDATORY_COLUMN_NAMES)

CIGAR_OPERATIONS_THAT_CONSUME_QUERY = set('MIS=X')
CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE = set('MDN=X')
CIGAR_OPERATIONS_THAT_CONSUME_QUERY_BUT_NOT_REFERENCE = CIGAR_OPERATIONS_THAT_CONSUME_QUERY - CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE
CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE_BUT_NOT_QUERY = CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE - CIGAR_OPERATIONS_THAT_CONSUME_QUERY
ALL_CIGAR_OPERATIONS = set('MIDNSHP=X')

READ_REVERSE_STRAND_BITMASK = 0x10


def check_samtools_version():
    samtools_version_output, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['samtools', '--version'])
    # print(samtools_version_output)
    if ('samtools 1.9\nUsing htslib 1.9' not in samtools_version_output) and ('samtools 1.7\nUsing htslib 1.7' not in samtools_version_output):
        raise NotImplementedError(f'sorry. a wrapper for this version hasnt been implemented yet.')


@generic_utils.execute_if_output_doesnt_exist_already
def cached_discard_unmapped_reads(
        input_file_path_sam,
        output_file_path_sam_without_unmapped_reads,
        keep_header,
):
    check_samtools_version()

    cmd_line_words = [
        'samtools', 'view',
        '-F', '4',
        '-o', output_file_path_sam_without_unmapped_reads,
    ]
    if keep_header:
        cmd_line_words.append('-h')

    cmd_line_words.append(input_file_path_sam)

    generic_utils.run_cmd_and_check_ret_code_and_return_stdout(
        cmd_line_words,
        raise_exception_if_stderr_isnt_empty=True,
        # verbose=True,
        verbose=False,
    )


def discard_unmapped_reads(
        input_file_path_sam,
        output_file_path_sam_without_unmapped_reads,
        keep_header=True,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_discard_unmapped_reads(
        input_file_path_sam=input_file_path_sam,
        output_file_path_sam_without_unmapped_reads=output_file_path_sam_without_unmapped_reads,
        keep_header=keep_header,
    )


def get_alignment_df(sam_file_path):
    sam_file_contents_without_header, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(
        ['samtools', 'view', sam_file_path]
    )
    aligned_reads_df = pd.read_csv(
        io.StringIO(sam_file_contents_without_header),
        sep='\t',
        names=SAM_FILE_MANDATORY_COLUMN_NAMES,
        usecols=range(NUM_OF_MANDATORY_COLUMNS_IN_SAME_FILE),
        # added quoting according to https://stackoverflow.com/questions/18016037/pandas-parsererror-eof-character-when-reading-multiple-csv-files-to-hdf5/29857126#29857126
        # (which discussed the error i got before I added it).
        quoting=csv.QUOTE_NONE,
    )
    return aligned_reads_df

@contextlib.contextmanager
def horrifying_hack_for_samtools_1_7_libcrypto_error_context_manager():
    open_ssl_path = generic_utils.run_cmd_and_get_stdout_and_stderr(['which', 'openssl'], verbose=False)[0].strip()
    libcrypto_so_1_1_path = os.path.join(os.path.dirname(os.path.dirname(open_ssl_path)), 'lib', 'libcrypto.so.1.1')
    assert os.path.isfile(libcrypto_so_1_1_path)
    horrifying_hack_symlink_path = os.path.join(os.path.dirname(libcrypto_so_1_1_path), 'libcrypto.so.1.0.0')
    os.symlink(libcrypto_so_1_1_path, horrifying_hack_symlink_path)
    try:
        yield
    finally:
        os.unlink(horrifying_hack_symlink_path)

def get_alignments_df_with_int_tags_columns(sam_file_path, int_tags_column_names):
    cmd_line_words = ['samtools', 'view']
    for tag_to_discard in {
                              'AS',
                              'NM',
                              'MD',
                              'MC',
                              'XS',
                              'XN',
                              'XM',
                              'XO',
                              'XG',
                              'YS',
                              'YT',
                              'SA',
                              'XA',
                          } - set(int_tags_column_names):
        cmd_line_words += ['-x', tag_to_discard]

    cmd_line_words.append(sam_file_path)

    alignments_sam_file_contents, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(
        cmd_line_words,
        # verbose=True,
        verbose=False,
    )
    alignments_df = pd.read_csv(
        io.StringIO(alignments_sam_file_contents),
        sep='\t',
        names=(SAM_FILE_MANDATORY_COLUMN_NAMES + int_tags_column_names),
        # added quoting according to https://stackoverflow.com/questions/18016037/pandas-parsererror-eof-character-when-reading-multiple-csv-files-to-hdf5/29857126#29857126
        # (which discussed the error i got before I added it).
        quoting=csv.QUOTE_NONE,
    )

    for int_tag_column_name in int_tags_column_names:
        assert alignments_df[int_tag_column_name].str.startswith(f'{int_tag_column_name}:i:').all()
        alignments_df.loc[:, int_tag_column_name] = alignments_df.loc[:, int_tag_column_name].str.slice(start=len(f'{int_tag_column_name}:i:')).astype(int)

    return alignments_df


def get_primary_alignments_df(alignments_df):
    return alignments_df[(alignments_df['FLAG'] & 0x900) == 0]

def get_primary_alignments_df_from_sam_file(sam_file_path):
    return get_primary_alignments_df(get_alignment_df(sam_file_path))

def get_primary_alignments_df_with_int_tags_columns(sam_file_path, int_tags_column_names, columns_to_replace_with_star=None):
    alignments_df_with_non_primary = get_alignments_df_with_int_tags_columns(sam_file_path, int_tags_column_names)
    primary_alignments_df = get_primary_alignments_df(alignments_df_with_non_primary).copy()

    if columns_to_replace_with_star:
        # primary_alignments_df.drop(columns_to_replace_with_star, axis=1, inplace=True)
        primary_alignments_df.loc[:, columns_to_replace_with_star] = '*'

    return primary_alignments_df


def get_cigar_as_list(cigar_str):
    # https://samtools.github.io/hts-specs/SAMv1.pdf makes stuff very clear.
    cigar_as_list = []
    for match_obj in re.finditer('\\d+[MIDNSHP=X]', cigar_str):
        match = match_obj.group()
        operation = match[-1]
        operation_len = int(match[:-1])
        cigar_as_list.append((operation_len, operation))
    return cigar_as_list

def get_total_length_of_query_according_to_cigar(cigar_as_list):
    return sum(operation_len for operation_len, operation in cigar_as_list if operation in CIGAR_OPERATIONS_THAT_CONSUME_QUERY)

def get_total_length_of_reference_according_to_cigar(cigar_as_list):
    return sum(operation_len for operation_len, operation in cigar_as_list if operation in CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE)

def get_alignment_start_and_end_pos_in_query_and_aligned_region_in_ref_len_and_total_query_len(cigar_str):
    # https://samtools.github.io/hts-specs/SAMv1.pdf makes stuff very clear.
    cigar_as_list = get_cigar_as_list(cigar_str)
    total_query_len = get_total_length_of_query_according_to_cigar(cigar_as_list)
    # total_length_of_reference_according_to_cigar = get_total_length_of_reference_according_to_cigar(cigar_as_list)

    alignment_start_pos_in_query = 1
    for operation_len, operation in cigar_as_list:
        if operation in CIGAR_OPERATIONS_THAT_CONSUME_QUERY_BUT_NOT_REFERENCE:
            alignment_start_pos_in_query += operation_len
        else:
            break

    alignment_end_pos_in_query = total_query_len
    for operation_len, operation in cigar_as_list[::-1]:
        if operation in CIGAR_OPERATIONS_THAT_CONSUME_QUERY_BUT_NOT_REFERENCE:
            alignment_end_pos_in_query -= operation_len
        else:
            break

    # IIUC, these asserts must be true only if the clipping penalty is zero.
    # assert cigar_as_list[0][1] in CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE
    # assert cigar_as_list[-1][1] in CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE
    aligned_region_in_ref_len = sum(operation_len for operation_len, operation in cigar_as_list if operation in CIGAR_OPERATIONS_THAT_CONSUME_REFERENCE)

    return (alignment_start_pos_in_query, alignment_end_pos_in_query, aligned_region_in_ref_len, total_query_len)


def add_and_get_alignment_start_and_stop_positions_in_query_and_aligned_region_in_query_len_and_alignment_stop_position_in_reference_and_total_query_len(alignment_df_row, aligner):
    alignment_start_pos_in_query, alignment_end_pos_in_query, aligned_region_in_ref_len, total_query_len = (
        get_alignment_start_and_end_pos_in_query_and_aligned_region_in_ref_len_and_total_query_len(alignment_df_row['CIGAR']))

    aligned_region_in_query_len = alignment_end_pos_in_query - alignment_start_pos_in_query + 1
    if aligner in {'bwa'}:
        if alignment_df_row['FLAG'] & READ_REVERSE_STRAND_BITMASK:
            # print(total_query_len, alignment_start_pos_in_query, alignment_end_pos_in_query)
            new_alignment_start_pos_in_query = total_query_len - alignment_end_pos_in_query + 1
            new_alignment_end_pos_in_query = total_query_len - alignment_start_pos_in_query + 1
            # print(total_query_len, alignment_start_pos_in_query, alignment_end_pos_in_query)
            assert new_alignment_end_pos_in_query - new_alignment_start_pos_in_query + 1 == aligned_region_in_query_len
            alignment_start_pos_in_query, alignment_end_pos_in_query = new_alignment_start_pos_in_query, new_alignment_end_pos_in_query
    else:
        # https://www.biostars.org/p/289583/#289592 - i don't know whether it would be the same for other aligners. just write tests i guess.
        raise NotImplementedError()

    new_alignment_df_row = alignment_df_row.copy()
    new_alignment_df_row['alignment_start_pos_in_query'] = alignment_start_pos_in_query
    new_alignment_df_row['alignment_end_pos_in_query'] = alignment_end_pos_in_query
    new_alignment_df_row['aligned_region_in_query_len'] = aligned_region_in_query_len
    new_alignment_df_row['alignment_end_pos_in_ref'] = new_alignment_df_row['POS'] + aligned_region_in_ref_len - 1
    new_alignment_df_row['total_query_len'] = total_query_len
    return new_alignment_df_row


def add_columns_for_start_and_end_positions_of_alignment_in_query_and_aligned_region_in_query_len_and_end_position_of_alignment_in_reference_and_total_query_len(alignments_df, aligner):
    if alignments_df.empty:
        alignments_df_with_new_columns = alignments_df.reindex(columns=alignments_df.columns.values.tolist() + [
            'alignment_start_pos_in_query',
            'alignment_end_pos_in_query',
            'aligned_region_in_query_len',
            'alignment_end_pos_in_ref',
            'total_query_len',
        ])
    else:
        alignments_df_with_new_columns = alignments_df.apply(
            lambda row: add_and_get_alignment_start_and_stop_positions_in_query_and_aligned_region_in_query_len_and_alignment_stop_position_in_reference_and_total_query_len(row, aligner),
            axis=1,
        )

    return alignments_df_with_new_columns

def set_aligned_bases_indices_according_to_cigar(cigar_str, base_index_to_is_aligned):
    # https://samtools.github.io/hts-specs/SAMv1.pdf makes it very clear what we care about - CIGAR operations described by "consumes query" - [MIS=X]. neat.
    curr_index = 0
    for match_obj in re.finditer('\\d+[MIS=X]', cigar_str):
        match = match_obj.group()
        operation = match[-1]
        operation_len = int(match[:-1])
        next_index = curr_index + operation_len
        if operation not in 'SI':
            base_index_to_is_aligned[curr_index:next_index] = 1
        curr_index = next_index

def get_num_of_aligned_bases_of_read(alignments_df, read_len):
    # read_len can be bigger than the read's actual length.
    base_index_to_is_aligned = np.zeros(read_len)
    alignments_df.apply(
        lambda row: set_aligned_bases_indices_according_to_cigar(row['CIGAR'], base_index_to_is_aligned),
        axis=1,
    )
    return np.count_nonzero(base_index_to_is_aligned)


def get_read_name_and_num_of_aligned_bases_df_without_distinguishing_between_1st_and_2nd_in_pair(aligned_reads_df, max_read_len):
    if aligned_reads_df.empty:
        return pd.DataFrame(columns=['QNAME', 'num_of_aligned_bases'])
    read_name_and_num_of_aligned_bases_df = aligned_reads_df.groupby('QNAME', sort=False).apply(
        lambda group_df: pd.Series({'num_of_aligned_bases': get_num_of_aligned_bases_of_read(group_df, max_read_len)}),
    ).reset_index()
    return read_name_and_num_of_aligned_bases_df


def get_num_of_aligned_bases_according_to_cigar_str(cigar_str):
    # https://samtools.github.io/hts-specs/SAMv1.pdf
    # unfortunately, M is often used, so we can't know only from the CIGAR how many matching bases we have.
    num_of_aligned_bases = 0
    for match_obj in re.finditer('\\d+[M=X]', cigar_str):
        match = match_obj.group()
        # operation = match[-1]
        operation_len = int(match[:-1])
        num_of_aligned_bases += operation_len
    return num_of_aligned_bases

def get_first_mates_alignments_df(alignments_df):
    return alignments_df[(alignments_df['FLAG'] & 0x40) != 0]

def write_only_alignments_with_seq_field_longer_than_threshold(
        input_file_path_alignment_sam,
        min_seq_field_len,
        output_file_path_filtered_alignment_sam,
):
    alignment_df = get_alignment_df(input_file_path_alignment_sam)
    filtered_alignment_df = alignment_df[alignment_df['SEQ'].str.len() >= min_seq_field_len]
    filtered_alignment_df.to_csv(output_file_path_filtered_alignment_sam, sep='\t', index=False)

def get_clipping_positions(
        input_file_path_alignment_sam,
):
    alignment_df = get_alignment_df(input_file_path_alignment_sam)

def write_aligned_reads_fasta(
        input_file_path_alignment_sam,
        output_file_path_aligned_reads_fasta,
):
    alignment_df = get_alignment_df(input_file_path_alignment_sam)
    read_name_to_seqs = collections.defaultdict(set)
    for _, row in alignment_df.iterrows():
        read_name = row['QNAME']
        read_seq = row['SEQ']
        read_name_to_seqs[read_name].add(read_seq)

    aligned_read_seqs = []
    for read_name, read_seqs in read_name_to_seqs.items():
        num_of_read_seqs = len(read_seqs)
        if num_of_read_seqs > 1:
            for i, read_seq in enumerate(read_seqs):
                curr_seq = bio_utils.str_to_seq_record(read_seq)
                curr_seq.name = curr_seq.description = curr_seq.id = f'{read_name}_suffix_added_by_oren_{i}'
                aligned_read_seqs.append(curr_seq)
        else:
            curr_seq = bio_utils.str_to_seq_record(next(iter(read_seqs)))
            curr_seq.name = curr_seq.description = curr_seq.id = read_name
            aligned_read_seqs.append(curr_seq)

    bio_utils.write_records_to_fasta_or_gb_file(aligned_read_seqs, output_file_path_aligned_reads_fasta, 'fasta')




@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_paired_read_name_to_pair_max_score_assuming_primary_paired_alignment_has_max_score(
        input_file_path_alignments_sam,
        output_file_path_paired_read_name_to_pair_max_score_pickle,
):
    primary_alignments_df = get_primary_alignments_df_with_int_tags_columns(input_file_path_alignments_sam, ['AS', 'YS'])
    first_mate_primary_alignments_df = get_first_mates_alignments_df(primary_alignments_df)
    assert first_mate_primary_alignments_df['QNAME'].is_unique

    paired_read_name_to_pair_max_score = {
        row['QNAME']: row['AS'] + row['YS']
        for _, row in first_mate_primary_alignments_df.iterrows()
    }

    with open(output_file_path_paired_read_name_to_pair_max_score_pickle, 'wb') as f:
        pickle.dump(paired_read_name_to_pair_max_score, f, protocol=4)


def write_paired_read_name_to_pair_max_score_assuming_primary_paired_alignment_has_max_score(
        input_file_path_alignments_sam,
        output_file_path_paired_read_name_to_pair_max_score_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_paired_read_name_to_pair_max_score_assuming_primary_paired_alignment_has_max_score(
        input_file_path_alignments_sam=input_file_path_alignments_sam,
        output_file_path_paired_read_name_to_pair_max_score_pickle=output_file_path_paired_read_name_to_pair_max_score_pickle,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_single_read_name_to_max_score_assuming_primary_alignment_has_max_score(
        input_file_path_alignments_sam,
        output_file_path_single_read_name_to_max_score_pickle,
):
    primary_alignments_df = get_primary_alignments_df_with_int_tags_columns(input_file_path_alignments_sam, ['AS'])
    assert primary_alignments_df['QNAME'].is_unique

    single_read_name_to_max_score = {
        row['QNAME']: row['AS']
        for _, row in primary_alignments_df.iterrows()
    }

    with open(output_file_path_single_read_name_to_max_score_pickle, 'wb') as f:
        pickle.dump(single_read_name_to_max_score, f, protocol=4)


def write_single_read_name_to_max_score_assuming_primary_alignment_has_max_score(
        input_file_path_alignments_sam,
        output_file_path_single_read_name_to_max_score_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_single_read_name_to_max_score_assuming_primary_alignment_has_max_score(
        input_file_path_alignments_sam=input_file_path_alignments_sam,
        output_file_path_single_read_name_to_max_score_pickle=output_file_path_single_read_name_to_max_score_pickle,
    )


def get_paired_read_name_to_num_of_max_score_paired_alignments(
        paired_alignments_of_relevant_reads_to_variant_sam_file_path,
        paired_read_name_to_pair_max_score_in_alignment_to_mis_regions_pickle_file_path,
):
    with open(paired_read_name_to_pair_max_score_in_alignment_to_mis_regions_pickle_file_path, 'rb') as f:
        paired_read_name_to_pair_max_score_in_alignment_to_mis_regions = pickle.load(f)

    paired_read_name_and_pair_max_score_in_alignment_to_mis_regions_df = pd.DataFrame(
        [(paired_read_name, pair_max_score_in_alignment_to_mis_regions)
         for paired_read_name, pair_max_score_in_alignment_to_mis_regions
         in paired_read_name_to_pair_max_score_in_alignment_to_mis_regions.items()],
        columns=['QNAME', 'pair_max_score'],
    )

    alignments_df = get_alignments_df_with_int_tags_columns(
        paired_alignments_of_relevant_reads_to_variant_sam_file_path, ['AS', 'YS']
    )
    first_mate_alignments_df = get_first_mates_alignments_df(alignments_df)

    assert set(alignments_df['QNAME']) <= set(paired_read_name_and_pair_max_score_in_alignment_to_mis_regions_df['QNAME'])
    first_mate_alignments_and_max_scores_df = first_mate_alignments_df.merge(
        paired_read_name_and_pair_max_score_in_alignment_to_mis_regions_df, on='QNAME', how='inner',
    )
    max_score_first_mate_alignments_df = first_mate_alignments_and_max_scores_df[
        (first_mate_alignments_and_max_scores_df['AS'] + first_mate_alignments_and_max_scores_df['YS']) ==
        first_mate_alignments_and_max_scores_df['pair_max_score']
        ]

    grouped_max_score_first_mates_alignments = max_score_first_mate_alignments_df.groupby('QNAME', sort=False)
    # In rare cases (I guess they are rare), the same first mate can be part of more than one valid paired alignments.
    # This isn't fair, as only one of them could be true (the others have an insert-size that doesn't fit that specific
    # sequencing, i guess). Thus, we count the number of unique positions of first mates.
    paired_read_name_and_num_of_max_score_paired_alignments_df = grouped_max_score_first_mates_alignments.apply(
        lambda group_df: pd.Series({'num_of_max_score_paired_alignments': group_df['POS'].nunique()})
    ).reset_index()

    # print(paired_read_name_and_num_of_max_score_paired_alignments_df)
    # print(list(paired_read_name_and_num_of_max_score_paired_alignments_df))
    return {
        row['QNAME']: row['num_of_max_score_paired_alignments']
        for _, row in paired_read_name_and_num_of_max_score_paired_alignments_df.iterrows()
    }

@generic_utils.execute_if_output_doesnt_exist_already
def cached_create_sorted_and_indexed_bam_files(
        input_file_path_aligned_reads_sam,
        output_file_path_aligned_reads_bam,
        output_file_path_sorted_aligned_reads_bam,
        output_file_path_sorted_aligned_reads_bam_bai,
):
    assert output_file_path_sorted_aligned_reads_bam_bai == f'{output_file_path_sorted_aligned_reads_bam}.bai'
    # samtools view -b -S -o aligned.bam aligned.sam
    cmd_line_words = [
        'samtools', 'view',
        '-b',  # output BAM
        '-S',  # auto detect input format.
        '-o', output_file_path_aligned_reads_bam,
        input_file_path_aligned_reads_sam]
    generic_utils.run_cmd_and_assert_stdout_and_stderr_are_empty(cmd_line_words)

    # samtools sort aligned.bam -o aligned-v1.3.sorted.bam
    cmd_line_words = [
        'samtools', 'sort',
        output_file_path_aligned_reads_bam,
        '-o', output_file_path_sorted_aligned_reads_bam]
    generic_utils.run_cmd_and_assert_stdout_and_stderr_are_empty(cmd_line_words)

    # samtools index aligned-v1.3.sorted.bam
    cmd_line_words = [
        'samtools', 'index',
        output_file_path_sorted_aligned_reads_bam]
    generic_utils.run_cmd_and_assert_stdout_and_stderr_are_empty(cmd_line_words)

def create_sorted_and_indexed_bam_files(
        input_file_path_aligned_reads_sam,
        output_file_path_aligned_reads_bam,
        output_file_path_sorted_aligned_reads_bam,
        output_file_path_sorted_aligned_reads_bam_bai,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_create_sorted_and_indexed_bam_files(
        input_file_path_aligned_reads_sam=input_file_path_aligned_reads_sam,
        output_file_path_aligned_reads_bam=output_file_path_aligned_reads_bam,
        output_file_path_sorted_aligned_reads_bam=output_file_path_sorted_aligned_reads_bam,
        output_file_path_sorted_aligned_reads_bam_bai=output_file_path_sorted_aligned_reads_bam_bai,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_create_fasta_index(
        input_file_path_fasta,
        output_file_path_fai,
):
    assert output_file_path_fai == f'{input_file_path_fasta}.fai'
    cmd_line_words = ['samtools', 'faidx', input_file_path_fasta]
    generic_utils.run_cmd_and_assert_stdout_and_stderr_are_empty(cmd_line_words)

def create_fasta_index(
        input_file_path_fasta,
        output_file_path_fai,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_create_fasta_index(
        input_file_path_fasta=input_file_path_fasta,
        output_file_path_fai=output_file_path_fai,
    )

