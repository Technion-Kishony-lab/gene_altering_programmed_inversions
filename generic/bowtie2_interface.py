import os
import os.path
import warnings

import numpy as np
import pandas as pd

from generic import generic_utils
# from generic import samtools_and_sam_files_interface

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

BOWTIE2_NUM_OF_THREADS = 20

def check_bowtie2_version():
    # bowtie2_version_output, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['bowtie2', '--version'], raise_exception_if_subproc_returned_non_zero=False)
    bowtie2_version_output, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['bowtie2', '--version'])
    # if 'version 2.4' not in bowtie2_version_output:
    #     raise NotImplementedError(f'sorry. a wrapper for this version hasnt been implemented yet.')
    if 'version 2.2.5' not in bowtie2_version_output:
        raise NotImplementedError(f'sorry. a wrapper for this version hasnt been implemented yet.')

@generic_utils.execute_if_output_doesnt_exist_already
def cached_internal_bowtie2_build(
        input_file_path_fasta,
        output_file_path_1_bt2,
        output_file_path_2_bt2,
        output_file_path_3_bt2,
        output_file_path_4_bt2,
        output_file_path_rev_1_bt2,
        output_file_path_rev_2_bt2,
        check_version,
):
    if check_version:
        check_bowtie2_version()

    cmd_line_words = [
        'bowtie2-build',
        input_file_path_fasta,
        input_file_path_fasta,
    ]
    generic_utils.run_cmd_and_check_ret_code_and_return_stdout(
        cmd_line_words,
        # verbose=True,
        verbose=False,
    )

def internal_bowtie2_build(
        input_file_path_fasta,
        output_file_path_1_bt2,
        output_file_path_2_bt2,
        output_file_path_3_bt2,
        output_file_path_4_bt2,
        output_file_path_rev_1_bt2,
        output_file_path_rev_2_bt2,
        check_version,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_internal_bowtie2_build(
        input_file_path_fasta=input_file_path_fasta,
        output_file_path_1_bt2=output_file_path_1_bt2,
        output_file_path_2_bt2=output_file_path_2_bt2,
        output_file_path_3_bt2=output_file_path_3_bt2,
        output_file_path_4_bt2=output_file_path_4_bt2,
        output_file_path_rev_1_bt2=output_file_path_rev_1_bt2,
        output_file_path_rev_2_bt2=output_file_path_rev_2_bt2,
        check_version=check_version,
    )

def bowtie2_build_index(fasta_file_path, check_version=True):
    internal_bowtie2_build(
        input_file_path_fasta=fasta_file_path,
        output_file_path_1_bt2=f'{fasta_file_path}.1.bt2',
        output_file_path_2_bt2=f'{fasta_file_path}.2.bt2',
        output_file_path_3_bt2=f'{fasta_file_path}.3.bt2',
        output_file_path_4_bt2=f'{fasta_file_path}.4.bt2',
        output_file_path_rev_1_bt2=f'{fasta_file_path}.rev.1.bt2',
        output_file_path_rev_2_bt2=f'{fasta_file_path}.rev.2.bt2',
        check_version=check_version,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_bowtie2_align_paired_reads(
        input_file_path_reference_fasta,
        input_file_path_raw_reads_file1,
        input_file_path_raw_reads_file2,
        output_file_path_sam,
        output_file_path_stderr,
        end_to_end_mode,
        seed_len,
        interval_between_seeds_func_str_repr,
        min_score_func_str_repr,
        require_both_reads_to_be_aligned,
        require_paired_alignments,
        min_insert_size,
        max_insert_size,
        report_all_alignments,
):
    check_bowtie2_version()
    # samtools_and_sam_files_interface.check_samtools_version() # keep it here to not waste time.

    cmd_line_words = [
        'bowtie2',
        '--end-to-end' if end_to_end_mode else '--local',
        '--no-unal',
     ]
    if min_score_func_str_repr:
        cmd_line_words += ['--score-min', min_score_func_str_repr]
    if report_all_alignments:
        cmd_line_words.append('-a')
    if require_both_reads_to_be_aligned:
        cmd_line_words.append('--no-mixed')
    if require_paired_alignments:
        cmd_line_words.append('--no-discordant')
    if seed_len:
        cmd_line_words += ['-L', str(seed_len)]
    if interval_between_seeds_func_str_repr:
        cmd_line_words += ['-i', interval_between_seeds_func_str_repr]
    if BOWTIE2_NUM_OF_THREADS:
        cmd_line_words += ['-p', str(BOWTIE2_NUM_OF_THREADS)]

    if min_insert_size:
        cmd_line_words += ['--minins', str(min_insert_size)]
    if max_insert_size:
        cmd_line_words += ['--maxins', str(max_insert_size)]



    cmd_line_words += ['-x', input_file_path_reference_fasta]
    if input_file_path_raw_reads_file1 == input_file_path_raw_reads_file2:
        cmd_line_words += ['-U', input_file_path_raw_reads_file1]
        raise NotImplementedError('sorry. only implemented paired-end alignment.')
    else:
        cmd_line_words += ['-1', input_file_path_raw_reads_file1, '-2', input_file_path_raw_reads_file2]

    cmd_line_words += ['-S', output_file_path_sam]

    generic_utils.run_cmd_and_write_stdout_and_stderr_to_files(
        cmd_line_words,
        stderr_file_path=output_file_path_stderr,
        verbose=True,
        # verbose=False,
    )

def bowtie2_align_paired_reads(
        input_file_path_reference_fasta,
        input_file_path_raw_reads_file1,
        input_file_path_raw_reads_file2,
        output_file_path_sam,
        output_file_path_stderr,
        end_to_end_mode=True,
        seed_len=None,
        # interval_between_seeds_func_str_repr='C,1',
        interval_between_seeds_func_str_repr=None,
        min_score_func_str_repr=None,
        min_insert_size=None,
        max_insert_size=None,
        require_both_reads_to_be_aligned=False,
        require_paired_alignments=False,
        report_all_alignments=True,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_bowtie2_align_paired_reads(
        input_file_path_reference_fasta=input_file_path_reference_fasta,
        input_file_path_raw_reads_file1=input_file_path_raw_reads_file1,
        input_file_path_raw_reads_file2=input_file_path_raw_reads_file2,
        output_file_path_sam=output_file_path_sam,
        output_file_path_stderr=output_file_path_stderr,
        end_to_end_mode=end_to_end_mode,
        seed_len=seed_len,
        interval_between_seeds_func_str_repr=interval_between_seeds_func_str_repr,
        min_score_func_str_repr=min_score_func_str_repr,
        min_insert_size=min_insert_size,
        max_insert_size=max_insert_size,
        require_both_reads_to_be_aligned=require_both_reads_to_be_aligned,
        require_paired_alignments=require_paired_alignments,
        report_all_alignments=report_all_alignments,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_bowtie2_align_single_reads(
        input_file_path_reference_fasta,
        input_file_path_raw_reads_file,
        output_file_path_sam,
        output_file_path_stderr,
        end_to_end_mode,
        seed_len,
        interval_between_seeds_func_str_repr,
        min_score_func_str_repr,
        report_all_alignments,
        num_of_threads,
):
    check_bowtie2_version()
    # samtools_and_sam_files_interface.check_samtools_version() # keep it here to not waste time.

    cmd_line_words = [
        'bowtie2',
        '--end-to-end' if end_to_end_mode else '--local',
        '-L', str(seed_len),
        '-i', interval_between_seeds_func_str_repr,
        '--no-unal',
        '--threads', str(num_of_threads),
     ]
    if min_score_func_str_repr:
        cmd_line_words += ['--score-min', min_score_func_str_repr]
    if report_all_alignments:
        cmd_line_words.append('-a')
    if input_file_path_raw_reads_file.endswith('.fasta'):
        cmd_line_words.append('-f')

    # cmd_line_words += ['--threads', '4']

    cmd_line_words += ['-x', input_file_path_reference_fasta]
    cmd_line_words += ['-U', input_file_path_raw_reads_file]
    cmd_line_words += ['-S', output_file_path_sam]


    generic_utils.run_cmd_and_write_stdout_and_stderr_to_files(
        cmd_line_words,
        stderr_file_path=output_file_path_stderr,
        verbose=True,
        # verbose=False,
    )

def bowtie2_align_single_reads(
        input_file_path_reference_fasta,
        input_file_path_raw_reads_file,
        output_file_path_sam,
        output_file_path_stderr,
        end_to_end_mode=True,
        seed_len=20,
        interval_between_seeds_func_str_repr='C,1',
        min_score_func_str_repr=None,
        report_all_alignments=True,
        num_of_threads=1,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_bowtie2_align_single_reads(
        input_file_path_reference_fasta=input_file_path_reference_fasta,
        input_file_path_raw_reads_file=input_file_path_raw_reads_file,
        output_file_path_sam=output_file_path_sam,
        output_file_path_stderr=output_file_path_stderr,
        end_to_end_mode=end_to_end_mode,
        seed_len=seed_len,
        interval_between_seeds_func_str_repr=interval_between_seeds_func_str_repr,
        min_score_func_str_repr=min_score_func_str_repr,
        report_all_alignments=report_all_alignments,
        num_of_threads=num_of_threads,
    )
