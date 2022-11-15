import glob
import time
import warnings
import tempfile
import pickle
import io
import pandas as pd
import itertools
import os.path
import subprocess
from generic import generic_utils
from generic import bio_utils

PAIRWISE_IDENTITY_DEFINITION_TYPE_TO_ARG = {
    'matching_columns_divided_by_shortest_sequence_length': 0,
    'matching_columns_divided_by_alignment_length_such_that_terminal_gaps_are_penalized': 1,
    'matching_columns_divided_by_alignment_length_ignoring_terminal_gaps': 2, # i think this is the default.
}

UC_FORMAT_COLUMN_NAMES = (
    'record_type',
    'cluster_index',
    'centroid_or_query_length_or_cluster_size',
    'percent_similarity_to_centroid',
    'match_orientation',
    'not_used1',
    'not_used2',
    'alignment_cigar',
    'query_or_centroid_label',
    'centroid_label',
)

def read_clusters_uc_file(clusters_uc_file_path):
    return pd.read_csv(clusters_uc_file_path, sep='\t', low_memory=False, names=UC_FORMAT_COLUMN_NAMES)

@generic_utils.execute_if_output_doesnt_exist_already
def cached_vsearch_cluster(
        input_file_path_seqs_fasta,
        min_pairwise_identity_with_cluster_centroid,
        pairwise_identity_definition_type,
        also_use_reverse_complement_when_comparing_to_centroid,
        min_seq_len,
        max_seq_len,
        num_of_seqs,
        verify_all_sequences_were_clustered,
        output_file_path_centroids_fasta,
        output_file_path_clusters_uc,
        output_file_path_vsearch_stderr,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    # vsearch --cluster_fast queries.fas --id 0.97 --centroids centroids.fas --uc clusters.uc
    cmd_line_words = [
        'vsearch',
        '--cluster_fast',
        input_file_path_seqs_fasta,
        '--id', str(min_pairwise_identity_with_cluster_centroid),
        '--iddef', str(PAIRWISE_IDENTITY_DEFINITION_TYPE_TO_ARG[pairwise_identity_definition_type]),
        '--strand', 'both' if also_use_reverse_complement_when_comparing_to_centroid else 'plus',
        '--centroids', output_file_path_centroids_fasta,
        '--uc', output_file_path_clusters_uc,
        '--minseqlength', str(min_seq_len),
        '--maxseqlength', str(max_seq_len),
        '--qmask', 'none',
    ]

    stdout, stderr = generic_utils.run_cmd_and_get_stdout_and_stderr(cmd_line_words, verbose=True)
    assert stdout == ''
    print(f'stderr: {stderr}')

    # from the manual:
    # Output clustering results in filename using a tab-separated uclust-like format with 10
    # columns and 3 different type of entries (S, H or C). Each fasta sequence in the input file
    # can be either a cluster centroid (S) or a hit (H) assigned to a cluster. Cluster records (C)
    # summarize information (size, centroid label) for each cluster. In the context of clustering, the option --uc_allhits has no effect on the --uc output. Column content varies with
    # the type of entry (S, H or C)

    if verify_all_sequences_were_clustered:
        clusters_uc_df = read_clusters_uc_file(output_file_path_clusters_uc)
        record_type_count_df = clusters_uc_df['record_type'].value_counts()
        assert record_type_count_df['S'] == record_type_count_df['C']
        assert record_type_count_df['S'] + (record_type_count_df['H'] if ('H' in record_type_count_df) else 0) == num_of_seqs

    generic_utils.write_text_file(output_file_path_vsearch_stderr, stderr)

def vsearch_cluster(
        input_file_path_seqs_fasta,
        min_pairwise_identity_with_cluster_centroid,
        pairwise_identity_definition_type,
        also_use_reverse_complement_when_comparing_to_centroid,
        min_seq_len,
        max_seq_len,
        num_of_seqs,
        verify_all_sequences_were_clustered,
        output_file_path_centroids_fasta,
        output_file_path_clusters_uc,
        output_file_path_vsearch_stderr,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_vsearch_cluster(
        input_file_path_seqs_fasta=input_file_path_seqs_fasta,
        min_pairwise_identity_with_cluster_centroid=min_pairwise_identity_with_cluster_centroid,
        pairwise_identity_definition_type=pairwise_identity_definition_type,
        also_use_reverse_complement_when_comparing_to_centroid=also_use_reverse_complement_when_comparing_to_centroid,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        num_of_seqs=num_of_seqs,
        verify_all_sequences_were_clustered=verify_all_sequences_were_clustered,
        output_file_path_centroids_fasta=output_file_path_centroids_fasta,
        output_file_path_clusters_uc=output_file_path_clusters_uc,
        output_file_path_vsearch_stderr=output_file_path_vsearch_stderr,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=5,
    )