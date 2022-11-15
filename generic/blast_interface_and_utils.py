import itertools
# from Bio.Blast import NCBIWWW
import os.path
import pickle
import subprocess
import tempfile

import pandas as pd

from generic import bio_utils
from generic import generic_utils

NUM_OF_SECONDS_TO_WAIT_SO_THAT_NCBI_BLAST_DOESNT_BAN_US_OR_SOMETHING = 60


QUERY_STRAND_TO_SEARCH_TO_OLD_BLASTALL_S_ARG = {'plus': '1',
                                                'minus': '2',
                                                'both': '3'}

BLAST_TABULAR_OUTPUT_COLUMN_NAMES = ('qseqid',
                                     'sseqid',
                                     'pident',
                                     'length',
                                     'mismatch',
                                     'gapopen',
                                     # quite sure it always holds that qstart < qend.
                                     'qstart',
                                     'qend',
                                     'sstart',
                                     'send',
                                     'evalue',
                                     'bitscore')


# a good intro to BLAST (and the different programs it includes), i think: https://github.com/elucify/blast-docs/wiki/The-Developer%27s-Guide-to-BLAST#-blast-programs

# blastn_version_stdout, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['blastn', '-version'])
# blast_db_cmd_version_stdout, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['makeblastdb', '-version'])
# make_blast_db_version_stdout, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['makeblastdb', '-version'])

def get_file_used_for_blast_db_caching_path(blast_db_path):
    return f'{blast_db_path}.file_used_for_blast_db_caching.txt'

@generic_utils.execute_if_output_doesnt_exist_already
def cached_extract_specific_nuccore_fasta_file_from_blast_db_internal(
        blast_db_path,
        argument_that_is_used_only_to_determine_whether_to_use_cached_output,
        sequence_identifier,
        output_file_path_nuccore_fasta,
):
    cmd_line_words = [
        'blastdbcmd',
        '-db', blast_db_path,
        '-entry', sequence_identifier,
        '-out', output_file_path_nuccore_fasta,
    ]
    generic_utils.run_cmd_and_check_ret_code_and_return_stdout(cmd_line_words)


def extract_specific_nuccore_fasta_file_from_blast_db_internal(
        blast_db_path,
        argument_that_is_used_only_to_determine_whether_to_use_cached_output,
        sequence_identifier,
        output_file_path_nuccore_fasta,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_extract_specific_nuccore_fasta_file_from_blast_db_internal(
        blast_db_path=blast_db_path,
        argument_that_is_used_only_to_determine_whether_to_use_cached_output=argument_that_is_used_only_to_determine_whether_to_use_cached_output,
        sequence_identifier=sequence_identifier,
        output_file_path_nuccore_fasta=output_file_path_nuccore_fasta,
    )

def extract_specific_nuccore_fasta_file_from_blast_db(blast_db_path, sequence_identifier, output_file_path_nuccore_fasta, blast_db_edit_time_repr_for_caching=None):
    extract_specific_nuccore_fasta_file_from_blast_db_internal(
        blast_db_path=blast_db_path,
        # a bit ugly. oh well.
        argument_that_is_used_only_to_determine_whether_to_use_cached_output=(
            blast_db_edit_time_repr_for_caching if blast_db_edit_time_repr_for_caching else
            generic_utils.read_text_file(get_file_used_for_blast_db_caching_path(blast_db_path))),
        sequence_identifier=sequence_identifier,
        output_file_path_nuccore_fasta=output_file_path_nuccore_fasta,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_make_blast_nucleotide_db_internal(
        input_file_path_fasta,
        output_file_path_for_caching_only, # this is generic_utils.OUTPUT_FILE_PATH_FOR_CACHING_ONLY_ARG. the caching system would write the current time to this file.
        verbose,
):
    cmd_line_words = [
            'makeblastdb',
            '-in', input_file_path_fasta,
            '-dbtype', 'nucl',  # nucleotide
            '-parse_seqids', # https://ncbi.github.io/magicblast/cook/blastdb.html: "The -parse_seqids option is required to keep the original sequence identifiers."
        ]
    generic_utils.run_cmd_and_check_ret_code_and_return_stdout(cmd_line_words, verbose=verbose)

def make_blast_nucleotide_db_internal(
        input_file_path_fasta,
        output_file_path_for_caching_only,
        verbose,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_make_blast_nucleotide_db_internal(
        input_file_path_fasta=input_file_path_fasta,
        output_file_path_for_caching_only=output_file_path_for_caching_only,
        verbose=verbose,
    )

def make_blast_nucleotide_db(fasta_file_path, verbose=True):
    make_blast_nucleotide_db_internal(
        input_file_path_fasta=fasta_file_path,
        output_file_path_for_caching_only=get_file_used_for_blast_db_caching_path(fasta_file_path),
        verbose=verbose,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_make_blast_nucleotide_db_for_multiple_fasta_files(
        multiple_input_file_paths_fasta_files,
        output_file_path_blast_db_with_dummy_suffix,
):
    dummy_suffix = '.dummy_suffix'
    assert output_file_path_blast_db_with_dummy_suffix.endswith(dummy_suffix)
    blast_db_path = output_file_path_blast_db_with_dummy_suffix[:(-len(dummy_suffix))]
    concatenated_fasta_file_contents = ''.join(generic_utils.read_text_file(fasta_file_path) for fasta_file_path in multiple_input_file_paths_fasta_files)
    generic_utils.write_text_file(blast_db_path, concatenated_fasta_file_contents)
    make_blast_nucleotide_db(blast_db_path)
    os.remove(blast_db_path)
    generic_utils.write_text_file(output_file_path_blast_db_with_dummy_suffix, 'success')

def make_blast_nucleotide_db_for_multiple_fasta_files(
        multiple_input_file_paths_fasta_files,
        blast_db_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_make_blast_nucleotide_db_for_multiple_fasta_files(
        multiple_input_file_paths_fasta_files=multiple_input_file_paths_fasta_files,
        output_file_path_blast_db_with_dummy_suffix=(f'{blast_db_path}.dummy_suffix'),
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_accessions_in_blast_db_to_text_file(
        blast_db_path,
        blast_db_edit_time_repr_for_caching,
        output_file_path_nuccore_accessions_in_blast_db,
):
    # blastdbcmd -help
    with open(output_file_path_nuccore_accessions_in_blast_db, 'w') as f:
        subprocess.run(['blastdbcmd', '-db', blast_db_path, '-entry', 'all', '-outfmt', '%a'], check=True, stdout=f)


def write_nuccore_accessions_in_blast_db_to_text_file(
        blast_db_path,
        blast_db_edit_time_repr_for_caching,
        output_file_path_nuccore_accessions_in_blast_db,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_accessions_in_blast_db_to_text_file(
        blast_db_path=blast_db_path,
        blast_db_edit_time_repr_for_caching=blast_db_edit_time_repr_for_caching,
        output_file_path_nuccore_accessions_in_blast_db=output_file_path_nuccore_accessions_in_blast_db,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_get_is_nuccore_accession_in_local_blast_db(
        nuccore_accession,
        input_file_path_nuccore_accessions_in_blast_db_txt,
        output_file_path_is_nuccore_accession_in_local_blast_db_txt,
):
    # https://stackoverflow.com/questions/4749330/how-to-test-if-string-exists-in-file-with-bash/4749368#4749368
    #
    grep_stdout, grep_stderr, grep_return_code = generic_utils.run_cmd_and_get_stdout_and_stderr(
        ['grep', '-Fx', '-m', '1', nuccore_accession, input_file_path_nuccore_accessions_in_blast_db_txt],
        raise_exception_if_subproc_returned_non_zero=False,
        also_return_return_code=True,
        verbose=True,
    )
    if grep_return_code not in {0, 1}:
        print(f'grep_stdout: {grep_stdout}')
        print(f'grep_stderr: {grep_stderr}')
        print(f'grep_return_code: {grep_return_code}')
        raise subprocess.SubprocessError(f'grep failed with return code {grep_return_code}')
    is_nuccore_accession_in_local_blast_db = grep_return_code == 0

    generic_utils.write_text_file(output_file_path_is_nuccore_accession_in_local_blast_db_txt, str(is_nuccore_accession_in_local_blast_db))

def get_is_nuccore_accession_in_local_blast_db(
        nuccore_accession,
        input_file_path_nuccore_accessions_in_blast_db_txt,
        output_file_path_is_nuccore_accession_in_local_blast_db_txt,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_get_is_nuccore_accession_in_local_blast_db(
        nuccore_accession=nuccore_accession,
        input_file_path_nuccore_accessions_in_blast_db_txt=input_file_path_nuccore_accessions_in_blast_db_txt,
        output_file_path_is_nuccore_accession_in_local_blast_db_txt=output_file_path_is_nuccore_accession_in_local_blast_db_txt,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_execute_blast_nucleotide(
        input_file_path_query_fasta,
        blast_db_edit_time_repr_for_caching,
        cmd_line_words,
        output_file_path_blast_results,
        verbose,
):
    generic_utils.run_cmd_and_check_ret_code_and_return_stdout(cmd_line_words, raise_exception_if_stderr_isnt_empty=True, print_stdout_if_verbose=True, verbose=verbose)

def execute_blast_nucleotide(
        input_file_path_query_fasta,
        blast_db_edit_time_repr_for_caching,
        cmd_line_words,
        output_file_path_blast_results,
        verbose,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_execute_blast_nucleotide(
        input_file_path_query_fasta=input_file_path_query_fasta,
        blast_db_edit_time_repr_for_caching=blast_db_edit_time_repr_for_caching,
        cmd_line_words=cmd_line_words,
        output_file_path_blast_results=output_file_path_blast_results,
        verbose=verbose,
    )



def blast_nucleotide(
        query_fasta_file_path,
        blast_db_path=None,
        target_species_taxon_uid=None, # a better name would be target_taxon_uids, but the caching mechanism would ignore existing caches if i change the name now.
        blast_results_file_path=None,
        perform_gapped_alignment=True,
        query_strand_to_search='both', # other options: 'plus' and 'minus'
        seed_len=None,
        min_matches_percentage=None,
        max_evalue=None,
        xdropoff_for_ungapped_alignment_in_bits_iiuc=None,
        max_num_of_overlaps_to_report=None,
        one_hit_algorithm=True,
        disable_dust_low_complexity_query_filter=True,
        match_and_mismatch_scores=None,
        region_in_query_sequence=None,
        blast_db_edit_time_repr_for_caching=None,
        verbose=True,
):
    assert not (perform_gapped_alignment and xdropoff_for_ungapped_alignment_in_bits_iiuc)

    # echo ">aoeu" > tmp_for_blast.fasta
    # echo CGAACTTTGAGAAGATACCTGGAAGTCCGATAGCTTATTGGGCGAGTAAACCCCTTATTTCCGATTTTGAAATTGGCATACCACTCAAAGACTTAGTTGATCCTAAAGTGGGTCTGCAAACTGGTGATAACAGTCGCTTTCTACGTCAATGGTTTGAAGTAAATGTGCATAATATTAGTTTTAACACAAAGAGCACCGCAGAATCGCTTAGGTCGATCAAGAAATGGTTTCCTTACAACAAGGGTGGATCATATCGCAAATGGTATGGCAATTTTGACTACATTGTAAATTGGCAACATGATGG >> tmp_for_blast.fasta
    # /usr/bin/blastn -outfmt 6 -db nt -remote -strand both -query tmp_for_blast.fasta

    cmd_line_words = [
        'blastn',
        '-query', query_fasta_file_path,
        # '-task', 'blastn',
        '-outfmt', '6',  # tabular,
        '-strand', query_strand_to_search,
    ]

    if blast_db_path:
        cmd_line_words += ['-db', blast_db_path]
    if target_species_taxon_uid:
        cmd_line_words += ['-taxids', ','.join(str(x) for x in target_species_taxon_uid)]
    if blast_results_file_path:
        cmd_line_words += ['-out', blast_results_file_path]
    if max_num_of_overlaps_to_report:
        cmd_line_words += ['-max_target_seqs', str(max_num_of_overlaps_to_report)]
    if not perform_gapped_alignment:
        cmd_line_words += ['-ungapped']
    if seed_len:
        cmd_line_words += ['-word_size', str(seed_len)]
    if min_matches_percentage:
        raise RuntimeError('you probably dont want this. you might miss a very important overlap just because BLAST kept extending to also include a "not so good stretch" '
                           '(i.e., many matches, but also not too few mismatches).\n'
                           'IIUC, to get what you want, choose a larger penalty for mismatches. The defaults can be found in BLAST\'s web interface.')
        # cmd_line_words += ['-perc_identity', str(min_matches_percentage)]
    if max_evalue:
        cmd_line_words += ['-evalue', str(max_evalue)]
    if xdropoff_for_ungapped_alignment_in_bits_iiuc:
        cmd_line_words += ['-xdrop_ungap', str(xdropoff_for_ungapped_alignment_in_bits_iiuc)]
    if one_hit_algorithm:
        # https://blast.advbiocomp.com/doc/parameters.html: "The 1-hit BLAST algorithm will always be more sensitive than the 2-hit algorithm, with all else equal"
        cmd_line_words += ['-window_size', '0']
    if disable_dust_low_complexity_query_filter:
        cmd_line_words += ['-dust', 'no']
    if match_and_mismatch_scores:
        match_score, mismatch_score = match_and_mismatch_scores
        cmd_line_words += ['-penalty', str(mismatch_score)]
        cmd_line_words += ['-reward', str(match_score)]
    if region_in_query_sequence:
        assert (type(region_in_query_sequence) == tuple) and (len(region_in_query_sequence) == 2)
        cmd_line_words += ['-query_loc', f'{region_in_query_sequence[0]}-{region_in_query_sequence[1]}']

    if blast_results_file_path:
        if not blast_db_edit_time_repr_for_caching:
            blast_db_edit_time_repr_for_caching = generic_utils.read_text_file(get_file_used_for_blast_db_caching_path(blast_db_path))
            # print(f'blast_db_edit_time_repr_for_caching: {blast_db_edit_time_repr_for_caching}')
        execute_blast_nucleotide(
            input_file_path_query_fasta=query_fasta_file_path,
            blast_db_edit_time_repr_for_caching=blast_db_edit_time_repr_for_caching,
            cmd_line_words=cmd_line_words,
            output_file_path_blast_results=blast_results_file_path,
            verbose=verbose,
        )
    else:
        return generic_utils.run_cmd_and_check_ret_code_and_return_stdout(cmd_line_words, raise_exception_if_stderr_isnt_empty=True,
                                                                          print_stdout_if_verbose=False, verbose=verbose)



def read_blast_results_df(blast_results_file_path_or_stringio_obj, names=BLAST_TABULAR_OUTPUT_COLUMN_NAMES):
    return pd.read_csv(blast_results_file_path_or_stringio_obj, sep='\t', names=names)

def cloud_blast(
):
    raise NotImplementedError('''
        201101: hmmm. This sounds problematic:
        https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=DeveloperInfo says:
        The NCBI servers are a shared resource and not intended for projects that involve a large number of BLAST searches. We provide Stand-alone BLAST and the RESTful API at a cloud provider for such projects.
        I guess that using blastalign is not really a problem, as most people don't use it. With regard to cloud blast, i guess if i do only a single search for a species, this is also ok… (means i must do a single search for left and right margins).
        201111: also, we downloaded the nt blast database to the server. we should probably never use cloud blast. good.
                              ''')


def get_matching_aligned_subject_subregion_of_aligned_query_subregion(
        aligned_query_subregion,
        alignment_row,
):
    # print(f'aligned_query_subregion: {aligned_query_subregion}')
    # print(f'alignment_row: {alignment_row}')

    qstart = alignment_row['qstart']
    qend = alignment_row['qend']
    aligned_query_subregion_start, aligned_query_subregion_end = aligned_query_subregion

    # print(f'{qstart} <= {aligned_query_subregion_start} <= {aligned_query_subregion_end} <= {qend}')
    assert qstart <= aligned_query_subregion_start <= aligned_query_subregion_end <= qend

    sstart = alignment_row['sstart']
    send = alignment_row['send']
    assert abs(send - sstart) == qend - qstart

    aligned_query_subregion_start_index_in_query_aligned_region = aligned_query_subregion_start - qstart
    aligned_query_subregion_end_index_in_query_aligned_region = aligned_query_subregion_end - qstart

    if sstart <= send:
        # aligned_subject_subregion_is_on_reverse_strand = False
        aligned_subject_subregion_start = sstart + aligned_query_subregion_start_index_in_query_aligned_region
        aligned_subject_subregion_end = sstart + aligned_query_subregion_end_index_in_query_aligned_region
        assert aligned_subject_subregion_end - aligned_subject_subregion_start == aligned_query_subregion_end - aligned_query_subregion_start
    else:
        # aligned_subject_subregion_is_on_reverse_strand = True
        aligned_subject_subregion_start = sstart - aligned_query_subregion_start_index_in_query_aligned_region
        aligned_subject_subregion_end = sstart - aligned_query_subregion_end_index_in_query_aligned_region
        assert aligned_subject_subregion_start - aligned_subject_subregion_end == aligned_query_subregion_end - aligned_query_subregion_start

    return (aligned_subject_subregion_start, aligned_subject_subregion_end)


def get_interval_in_subject_to_1_to_1_matching_interval_in_query(alignments_df):
    if alignments_df.empty:
        return {}

    if not (alignments_df['qstart'] <= alignments_df['qend']).all():
        print('alignments_df')
        print(alignments_df)
    assert (alignments_df['qstart'] <= alignments_df['qend']).all()

    alignments_df.loc[:, 'smin'] = alignments_df[['sstart', 'send']].min(axis=1)
    alignments_df.loc[:, 'smax'] = alignments_df[['sstart', 'send']].max(axis=1)

    interval_in_query_to_num_of_containing_alignments = (
        generic_utils.find_partition_to_subintervals_and_return_subinterval_to_num_of_containing_given_intervals(
            given_intervals=alignments_df[['qstart', 'qend']].to_records(index=False).tolist(),
        )
    )
    interval_in_subject_to_num_of_containing_alignments = (
        generic_utils.find_partition_to_subintervals_and_return_subinterval_to_num_of_containing_given_intervals(
            given_intervals=alignments_df[['smin', 'smax']].to_records(index=False).tolist(),
        )
    )
    intervals_in_query_sorted = sorted(interval_in_query_to_num_of_containing_alignments)
    intervals_in_subject_sorted = sorted(interval_in_subject_to_num_of_containing_alignments)

    # print('interval_in_query_to_num_of_containing_alignments')
    # print(interval_in_query_to_num_of_containing_alignments)
    #
    # print('interval_in_subject_to_num_of_containing_alignments')
    # print(interval_in_subject_to_num_of_containing_alignments)

    interval_in_subject_to_1_to_1_matching_interval_in_query = {}
    for _, row in alignments_df.iterrows():
        assert row['qstart'] <= row['qend']
        is_alignment_to_forward_strand_of_subject = row['sstart'] <= row['send']

        relevant_intervals_in_query = itertools.dropwhile(lambda interval: interval[0] < row['qstart'], intervals_in_query_sorted)
        relevant_intervals_in_query = itertools.takewhile(lambda interval: interval[1] <= row['qend'], relevant_intervals_in_query)
        relevant_intervals_in_query = list(relevant_intervals_in_query)
        assert relevant_intervals_in_query[0][0] == row['qstart']
        assert relevant_intervals_in_query[-1][1] == row['qend']

        relevant_intervals_in_query = [interval for interval in relevant_intervals_in_query
                                               if interval_in_query_to_num_of_containing_alignments[interval] == 1]
        if not relevant_intervals_in_query:
            continue

        relevant_intervals_in_subject = itertools.dropwhile(lambda interval: interval[0] < row['smin'], intervals_in_subject_sorted)
        relevant_intervals_in_subject = itertools.takewhile(lambda interval: interval[1] <= row['smax'], relevant_intervals_in_subject)
        relevant_intervals_in_subject = list(relevant_intervals_in_subject)
        assert relevant_intervals_in_subject[0][0] == row['smin']
        assert relevant_intervals_in_subject[-1][1] == row['smax']

        relevant_intervals_in_subject = [interval for interval in relevant_intervals_in_subject
                                                                   if interval_in_subject_to_num_of_containing_alignments[interval] == 1]
        if not relevant_intervals_in_subject:
            continue

        # print('relevant_intervals_in_query')
        # print(relevant_intervals_in_query)
        # print('relevant_intervals_in_subject')
        # print(relevant_intervals_in_subject)

        relevant_intervals_in_alignment_to_query = [
            (
                interval[0] - row['qstart'],
                interval[1] - row['qstart'],
            )
            for interval in relevant_intervals_in_query
        ]

        if is_alignment_to_forward_strand_of_subject:
            assert row['smin'] == row['sstart']
            relevant_intervals_in_alignment_to_subject = [
                (
                    interval[0] - row['smin'],
                    interval[1] - row['smin'],
                )
                for interval in relevant_intervals_in_subject
            ]
        else:
            assert row['smax'] == row['sstart']
            relevant_intervals_in_alignment_to_subject = [
                (
                    row['smax'] - interval[1],
                    row['smax'] - interval[0],
                )
                for interval in relevant_intervals_in_subject
            ]

        # print('relevant_intervals_in_alignment_to_query')
        # print(relevant_intervals_in_alignment_to_query)
        # print('relevant_intervals_in_alignment_to_subject')
        # print(relevant_intervals_in_alignment_to_subject)

        final_relevant_intervals_in_alignment = [
            subinterval
            for subinterval, num_of_containing_given_intervals
            in generic_utils.find_partition_to_subintervals_and_return_subinterval_to_num_of_containing_given_intervals(
                given_intervals=(relevant_intervals_in_alignment_to_query + relevant_intervals_in_alignment_to_subject),
            ).items()
            if num_of_containing_given_intervals == 2
        ]

        for interval in final_relevant_intervals_in_alignment:
            interval_in_query = (
                row['qstart'] + interval[0],
                row['qstart'] + interval[1],
            )
            if is_alignment_to_forward_strand_of_subject:
                assert row['smin'] == row['sstart']
                interval_in_subject = (
                    row['smin'] + interval[0],
                    row['smin'] + interval[1],
                )
            else:
                assert row['smax'] == row['sstart']
                interval_in_subject = (
                    row['smax'] - interval[1],
                    row['smax'] - interval[0],
                )

            interval_in_subject_to_1_to_1_matching_interval_in_query[interval_in_subject] = (
                interval_in_query,
                is_alignment_to_forward_strand_of_subject,
            )

    return interval_in_subject_to_1_to_1_matching_interval_in_query

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_interval_in_subject_to_1_to_1_matching_interval_in_query(
        input_file_path_alignments_csv,
        output_file_path_interval_in_subject_to_1_to_1_matching_interval_in_query_pickle,
):
    alignments_df = pd.read_csv(input_file_path_alignments_csv, sep='\t')

    interval_in_subject_to_1_to_1_matching_interval_in_query = get_interval_in_subject_to_1_to_1_matching_interval_in_query(alignments_df)

    with open(output_file_path_interval_in_subject_to_1_to_1_matching_interval_in_query_pickle, 'wb') as f:
        pickle.dump(interval_in_subject_to_1_to_1_matching_interval_in_query, f, protocol=4)

def write_interval_in_subject_to_1_to_1_matching_interval_in_query(
        input_file_path_alignments_csv,
        output_file_path_interval_in_subject_to_1_to_1_matching_interval_in_query_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_interval_in_subject_to_1_to_1_matching_interval_in_query(
        input_file_path_alignments_csv=input_file_path_alignments_csv,
        output_file_path_interval_in_subject_to_1_to_1_matching_interval_in_query_pickle=output_file_path_interval_in_subject_to_1_to_1_matching_interval_in_query_pickle,
    )

def get_max_subject_position(alignments_df):
    return alignments_df[['sstart', 'send']].max().max()

def get_min_subject_position(alignments_df):
    return alignments_df[['sstart', 'send']].min().min()


def blast_two_dna_seqs_as_strs(str1, str2, **kwargs):
    with tempfile.TemporaryDirectory(dir='.') as temp_dir_path:
        str1_fasta_file_path = os.path.join(temp_dir_path, 'str1.fasta')
        str2_fasta_file_path = os.path.join(temp_dir_path, 'str2.fasta')
        blast_results_csv_file_path = os.path.join(temp_dir_path, 'blast_results.csv')

        str1_seq = bio_utils.str_to_seq_record(str1)
        str2_seq = bio_utils.str_to_seq_record(str2)
        str1_seq.name = str1_seq.description = str1_seq.id = 'str1'
        str2_seq.name = str2_seq.description = str2_seq.id = 'str2'

        bio_utils.write_records_to_fasta_or_gb_file([str1_seq], str1_fasta_file_path)
        bio_utils.write_records_to_fasta_or_gb_file([str2_seq], str2_fasta_file_path)
        make_blast_nucleotide_db(str2_fasta_file_path, verbose=False)
        blast_nucleotide(
            query_fasta_file_path=str1_fasta_file_path,
            blast_db_path=str2_fasta_file_path,
            blast_results_file_path=blast_results_csv_file_path,
            **kwargs,
        )
        alignments_df = read_blast_results_df(blast_results_csv_file_path)
        return alignments_df


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_taxon_uids_of_nuccore_in_blast_db(
        local_blast_db_path,
        local_blast_nt_database_update_log_for_caching_only,
        nuccore_uid_or_accession,
        output_file_path_taxon_uids_of_nuccore_in_blast_db,
):
    # blastdbcmd -entry FM179322.1 -db ../DBs/blast_nt_database/nt -outfmt "%T"
    blastdbcmd_out_as_str, blastdbcmd_err_as_str = generic_utils.run_cmd_and_get_stdout_and_stderr([
        'blastdbcmd',
        '-entry', nuccore_uid_or_accession,
        '-db', local_blast_db_path,
        '-outfmt', '%T',
    ], verbose=True)
    assert not blastdbcmd_err_as_str
    generic_utils.write_text_file(output_file_path_taxon_uids_of_nuccore_in_blast_db, blastdbcmd_out_as_str)

def write_taxon_uids_of_nuccore_in_blast_db(
        local_blast_db_path,
        local_blast_nt_database_update_log_for_caching_only,
        nuccore_uid_or_accession,
        output_file_path_taxon_uids_of_nuccore_in_blast_db,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_taxon_uids_of_nuccore_in_blast_db(
        local_blast_db_path=local_blast_db_path,
        local_blast_nt_database_update_log_for_caching_only=local_blast_nt_database_update_log_for_caching_only,
        nuccore_uid_or_accession=nuccore_uid_or_accession,
        output_file_path_taxon_uids_of_nuccore_in_blast_db=output_file_path_taxon_uids_of_nuccore_in_blast_db,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_build_taxon_local_blast_nt_database_nuccore_entries_info(
        local_blast_db_path,
        local_blast_nt_database_update_log_for_caching_only,
        taxon_uids,
        taxon_primary_assembly_nuccore_total_len,
        output_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    nuccore_accession_to_nt_nuccore_entry_len = {}

    # blastdbcmd -db /zdata/user-data/DBs/blast_nt_database/nt -taxids 2907820 -outfmt "%a _ %l"
    blastdbcmd_out_as_str, blastdbcmd_err_as_str, blastdbcmd_err_code = generic_utils.run_cmd_and_get_stdout_and_stderr(
        ['blastdbcmd', '-db', local_blast_db_path, '-taxids', ','.join(str(x) for x in taxon_uids), '-outfmt', '%a _ %l'],
        raise_exception_if_subproc_returned_non_zero=False, also_return_return_code=True, verbose=True)
    if blastdbcmd_err_code != 0:
        # print('blastdbcmd_out_as_str, blastdbcmd_err_as_str, blastdbcmd_err_code')
        # print(blastdbcmd_out_as_str, blastdbcmd_err_as_str, blastdbcmd_err_code)
        # 220117: ugh. got this error also for cases in which the taxon_uid was of a species, e.g., 32056, for which entries did exist in the local nt database
        #   (e.g., KM982554.1). when I used blastdbcmd again, this time specifying the strain taxon uid (1610718), the entry (KM982554.1) was found.
        #   that's why i added also trying for strain taxon uids of the same species.
        assert '[blastdbcmd] Taxonomy ID(s) not found. This could be because the ID(s) provided are not at or below the species level.' in blastdbcmd_err_as_str
        taxon_local_blast_nt_database_nuccore_entries_info = None
    else:
        for line in blastdbcmd_out_as_str.splitlines():
            nuccore_accession, nuccore_entry_len = line.split(' _ ')
            nuccore_accession_to_nt_nuccore_entry_len[nuccore_accession] = int(nuccore_entry_len)

        total_len_of_nt_nuccore_entries = sum(nuccore_accession_to_nt_nuccore_entry_len.values())
        estimated_num_of_genomes_in_nt = total_len_of_nt_nuccore_entries / taxon_primary_assembly_nuccore_total_len

        taxon_local_blast_nt_database_nuccore_entries_info = {
            'nuccore_accession_to_nt_nuccore_entry_len': nuccore_accession_to_nt_nuccore_entry_len,
            'total_len_of_nt_nuccore_entries': total_len_of_nt_nuccore_entries,
            'estimated_num_of_genomes_in_nt': estimated_num_of_genomes_in_nt,
        }

    with open(output_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle, 'wb') as f:
        pickle.dump(taxon_local_blast_nt_database_nuccore_entries_info, f, protocol=4)

def build_taxon_local_blast_nt_database_nuccore_entries_info(
        local_blast_db_path,
        local_blast_nt_database_update_log_for_caching_only,
        taxon_uids,
        taxon_primary_assembly_nuccore_total_len,
        output_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_build_taxon_local_blast_nt_database_nuccore_entries_info(
        local_blast_db_path=local_blast_db_path,
        local_blast_nt_database_update_log_for_caching_only=local_blast_nt_database_update_log_for_caching_only,
        taxon_uids=taxon_uids,
        taxon_primary_assembly_nuccore_total_len=taxon_primary_assembly_nuccore_total_len,
        output_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle=output_file_path_taxon_local_blast_nt_database_nuccore_entries_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_uids_of_contained_taxa_according_to_blast_get_species_taxids(
        species_taxon_uid,
        output_file_path_uids_of_contained_taxa_pickle,
):
    # get_species_taxids.sh -t 2907820
    cmd_out_as_str, cmd_err_as_str, cmd_err_code = generic_utils.run_cmd_and_get_stdout_and_stderr(
        ['get_species_taxids.sh', '-t', str(species_taxon_uid)],
        raise_exception_if_subproc_returned_non_zero=False,
        also_return_return_code=True,
        verbose=True,
    )
    if cmd_err_code == 1:
        assert 'Taxonomy ID not found' in cmd_err_as_str
        uids_of_contained_taxa = None
    else:
        assert cmd_err_code == 0
        uids_of_contained_taxa = [int(x) for x in cmd_out_as_str.strip().split()]

    with open(output_file_path_uids_of_contained_taxa_pickle, 'wb') as f:
        pickle.dump(uids_of_contained_taxa, f, protocol=4)

def write_uids_of_contained_taxa_according_to_blast_get_species_taxids(
        species_taxon_uid,
        output_file_path_uids_of_contained_taxa_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_uids_of_contained_taxa_according_to_blast_get_species_taxids(
        species_taxon_uid=species_taxon_uid,
        output_file_path_uids_of_contained_taxa_pickle=output_file_path_uids_of_contained_taxa_pickle,
    )


