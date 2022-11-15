import os
import pickle

import numpy as np
import pandas as pd

from generic import generic_utils
from generic import vsearch_interface_and_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_pairs_df_with_cluster_indices(
        input_file_path_cds_pairs_df_csv,
        input_file_path_repeat_cds_seqs_fasta,
        input_file_path_repeat_seq_name_df_csv,
        min_pairwise_identity_with_cluster_centroid,
        pairwise_identity_definition_type,
        output_file_path_pairs_with_cluster_indices_df_csv,
        output_file_path_pair_clustering_info_pickle,
        clustering_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_pairs_df = pd.read_csv(input_file_path_cds_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_cds_pairs = len(cds_pairs_df)

    num_of_repeat_cdss_to_cluster = len(pd.concat(
        [
            cds_pairs_df[['nuccore_accession', f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file']].rename(
                columns={f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file': 'index_in_nuccore_cds_features_gb_file'})
            for repeat_num in (1, 2)
        ],
        ignore_index=True,
    ).drop_duplicates())
    # min_repeat_cds_len = ir_pairs_linked_df_extension.add_repeat_cds_len_columns(cds_pairs_df)['min_repeat_cds_len'].min()
    min_repeat_cds_len = cds_pairs_df['min_repeat_cds_len'].min()
    assert min_repeat_cds_len == int(min_repeat_cds_len)
    min_repeat_cds_len = int(min_repeat_cds_len)
    # max_repeat_cds_len = ir_pairs_linked_df_extension.add_repeat_cds_len_columns(cds_pairs_df)['max_repeat_cds_len'].max()
    max_repeat_cds_len = cds_pairs_df['max_repeat_cds_len'].max()
    assert max_repeat_cds_len == int(max_repeat_cds_len)
    max_repeat_cds_len = int(max_repeat_cds_len)

    cluster_centroids_fasta_file_path = os.path.join(clustering_out_dir_path, 'cluster_centroids.fasta')
    clusters_uc_file_path = os.path.join(clustering_out_dir_path, 'clusters.uc')
    vsearch_stderr_txt_file_path = os.path.join(clustering_out_dir_path, 'vsearch_stderr.txt')
    vsearch_interface_and_utils.vsearch_cluster(
        input_file_path_seqs_fasta=input_file_path_repeat_cds_seqs_fasta,
        min_pairwise_identity_with_cluster_centroid=min_pairwise_identity_with_cluster_centroid,
        pairwise_identity_definition_type=pairwise_identity_definition_type,
        also_use_reverse_complement_when_comparing_to_centroid=False, # we always wrote CDS to fasta 5' to 3'.
        min_seq_len=min_repeat_cds_len,
        max_seq_len=max_repeat_cds_len,
        num_of_seqs=num_of_repeat_cdss_to_cluster,
        verify_all_sequences_were_clustered=True,
        output_file_path_centroids_fasta=cluster_centroids_fasta_file_path,
        output_file_path_clusters_uc=clusters_uc_file_path,
        output_file_path_vsearch_stderr=vsearch_stderr_txt_file_path,
    )

    clusters_uc_df = vsearch_interface_and_utils.read_clusters_uc_file(clusters_uc_file_path)
    clusters_s_and_h_records_df = clusters_uc_df[(clusters_uc_df['record_type'] == 'S') | (clusters_uc_df['record_type'] == 'H')]

    minimal_cdss_with_cluster_indices_df = clusters_s_and_h_records_df[['cluster_index', 'query_or_centroid_label']].rename(
        columns={'query_or_centroid_label': 'cds_seq_name'}).merge(
            pd.read_csv(input_file_path_repeat_seq_name_df_csv, sep='\t', low_memory=False)
        ).drop('cds_seq_name', axis=1)
    # print('minimal_cdss_with_cluster_indices_df')
    # print(minimal_cdss_with_cluster_indices_df)
    # print(f'num_of_repeat_cdss_to_cluster: {num_of_repeat_cdss_to_cluster}')
    assert len(minimal_cdss_with_cluster_indices_df) == num_of_repeat_cdss_to_cluster

    pairs_with_cluster_indices_df = cds_pairs_df.copy()
    for repeat_num in (1, 2):
        pairs_with_cluster_indices_df = pairs_with_cluster_indices_df.merge(
            minimal_cdss_with_cluster_indices_df.rename(columns={
                'cluster_index': f'repeat{repeat_num}_cds_cluster_index',
                'index_in_nuccore_cds_features_gb_file': f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'
            })
        )
        assert len(pairs_with_cluster_indices_df) == orig_num_of_cds_pairs

    assert not pairs_with_cluster_indices_df[['repeat1_cds_cluster_index', 'repeat2_cds_cluster_index']].isna().any(axis=None)
    pairs_with_cluster_indices_df['min_cluster_index'] = pairs_with_cluster_indices_df[['repeat1_cds_cluster_index', 'repeat2_cds_cluster_index']].min(axis=1)
    pairs_with_cluster_indices_df['max_cluster_index'] = pairs_with_cluster_indices_df[['repeat1_cds_cluster_index', 'repeat2_cds_cluster_index']].max(axis=1)
    final_cluster_indices_df = pairs_with_cluster_indices_df[['min_cluster_index', 'max_cluster_index']].drop_duplicates()
    final_cluster_indices_df['cds_pair_cluster_index'] = np.arange(len(final_cluster_indices_df))
    num_of_final_clusters = len(final_cluster_indices_df)

    pairs_with_cluster_indices_df = pairs_with_cluster_indices_df.merge(final_cluster_indices_df).drop(['min_cluster_index', 'max_cluster_index'], axis=1)
    assert pairs_with_cluster_indices_df['cds_pair_cluster_index'].nunique() == num_of_final_clusters
    assert len(pairs_with_cluster_indices_df) == orig_num_of_cds_pairs
    pairs_with_cluster_indices_df.to_csv(output_file_path_pairs_with_cluster_indices_df_csv, sep='\t', index=False)

    vsearch_stderr = generic_utils.read_text_file(vsearch_stderr_txt_file_path)
    pair_clustering_info = {
        'cluster_centroids_fasta_file_path': cluster_centroids_fasta_file_path,
        'clusters_uc_file_path': clusters_uc_file_path,
        'vsearch_stderr': vsearch_stderr,
    }
    with open(output_file_path_pair_clustering_info_pickle, 'wb') as f:
        pickle.dump(pair_clustering_info, f, protocol=4)


def write_pairs_df_with_cluster_indices(
        input_file_path_cds_pairs_df_csv,
        input_file_path_repeat_cds_seqs_fasta,
        input_file_path_repeat_seq_name_df_csv,
        min_pairwise_identity_with_cluster_centroid,
        pairwise_identity_definition_type,
        output_file_path_pairs_with_cluster_indices_df_csv,
        output_file_path_pair_clustering_info_pickle,
        clustering_out_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_pairs_df_with_cluster_indices(
        input_file_path_cds_pairs_df_csv=input_file_path_cds_pairs_df_csv,
        input_file_path_repeat_cds_seqs_fasta=input_file_path_repeat_cds_seqs_fasta,
        input_file_path_repeat_seq_name_df_csv=input_file_path_repeat_seq_name_df_csv,
        min_pairwise_identity_with_cluster_centroid=min_pairwise_identity_with_cluster_centroid,
        pairwise_identity_definition_type=pairwise_identity_definition_type,
        output_file_path_pairs_with_cluster_indices_df_csv=output_file_path_pairs_with_cluster_indices_df_csv,
        output_file_path_pair_clustering_info_pickle=output_file_path_pair_clustering_info_pickle,
        clustering_out_dir_path=clustering_out_dir_path,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=3,
    )