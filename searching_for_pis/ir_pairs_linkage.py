import pickle
import random

import numpy as np
import pandas as pd

from generic import generic_utils
from searching_for_pis import index_column_names

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

APPROX_OPTIMAL_CROSS_PRODUCT_SIZE = int(2e5)

def get_chunks_df(df1, df2, df1_num_column_name, df2_num_column_name):
    cross_product_df = df1['nuccore_accession'].value_counts().reset_index(name=df1_num_column_name).merge(
        df2['nuccore_accession'].value_counts().reset_index(name=df2_num_column_name)).rename(columns={'index': 'nuccore_accession'})
    cross_product_df['cross_product_size'] = cross_product_df[df1_num_column_name] * cross_product_df[df2_num_column_name]
    cross_product_df.sort_values(by='cross_product_size', inplace=True, ascending=False)
    # print(cross_product_df)

    curr_chunk_index = 0
    curr_chunk_cross_product_total_size = 0
    chunk_indices = []
    for curr_cross_product_size in cross_product_df['cross_product_size']:
        chunk_indices.append(curr_chunk_index)
        curr_chunk_cross_product_total_size += curr_cross_product_size
        if curr_chunk_cross_product_total_size >= APPROX_OPTIMAL_CROSS_PRODUCT_SIZE:
            curr_chunk_index += 1
            curr_chunk_cross_product_total_size = 0

    return pd.concat([cross_product_df['nuccore_accession'].reset_index(drop=True), pd.Series(chunk_indices, name='chunk_index')], axis=1)


def get_all_ir_pairs_linked_df(
        all_cds_df,
        all_ir_pairs_df,
        debug___num_of_nuccore_entries_to_go_over=None,
):
    nuccore_accessions = pd.Series(list(set(all_ir_pairs_df['nuccore_accession'].drop_duplicates()) & set(all_cds_df['nuccore_accession'].drop_duplicates())),
                                   name='nuccore_accession')
    num_of_nuccore_entries = len(nuccore_accessions)

    if (debug___num_of_nuccore_entries_to_go_over is not None) and (num_of_nuccore_entries > debug___num_of_nuccore_entries_to_go_over):
        nuccore_accessions = sorted(nuccore_accessions)
        print(nuccore_accessions[:20])
        random.seed(0)
        random.shuffle(nuccore_accessions)
        print(nuccore_accessions[:20])
        nuccore_accessions = nuccore_accessions[:debug___num_of_nuccore_entries_to_go_over]
        num_of_nuccore_entries = debug___num_of_nuccore_entries_to_go_over
        nuccore_accessions = pd.Series(nuccore_accessions, name='nuccore_accession')

    all_cds_df = all_cds_df.merge(nuccore_accessions, how='inner')
    all_ir_pairs_df = all_ir_pairs_df.merge(nuccore_accessions, how='inner')

    chunks_df = get_chunks_df(
        df1=all_cds_df,
        df2=all_ir_pairs_df,
        df1_num_column_name='num_of_cds',
        df2_num_column_name='num_of_ir_pairs',
    )


    # print(chunks_df)
    chunk_index_to_chunk_size = chunks_df['chunk_index'].value_counts().to_dict()
    num_of_chunks = chunks_df['chunk_index'].nunique()

    minimal_cds_df_column_names = ['nuccore_accession', 'start_pos', 'end_pos', 'index_in_nuccore_cds_features_gb_file', 'strand']
    minimal_all_cds_df = all_cds_df[minimal_cds_df_column_names].merge(chunks_df)

    minimal_linked_repeat_column_names = index_column_names.REPEAT_INDEX_COLUMN_NAMES + ['index_in_nuccore_cds_features_gb_file']

    minimal_all_ir_pairs_df = all_ir_pairs_df[['nuccore_accession', 'left1', 'right1', 'left2', 'right2', 'index_in_nuccore_ir_pairs_df_csv_file']].merge(chunks_df)

    minimal_all_ir_pairs_df_grouped = minimal_all_ir_pairs_df.groupby('chunk_index', sort=False)
    minimal_all_cds_df_grouped = minimal_all_cds_df.groupby('chunk_index', sort=False)

    minimal_strictly_contained_repeats_linked_dfs = []


    for i, (chunk_index, curr_ir_pairs_df) in enumerate(minimal_all_ir_pairs_df_grouped):
        chunk_size = chunk_index_to_chunk_size[chunk_index]
        generic_utils.print_and_write_to_log(f'(get_all_ir_pairs_linked_df 1) '
                                             f'starting work on chunk {i + 1}/{num_of_chunks} (chunk_index: {chunk_index}, chunk_size: {chunk_size})')
        curr_chunk_repeats_linked_df = curr_ir_pairs_df.drop('chunk_index', axis=1).merge(
            minimal_all_cds_df_grouped.get_group(chunk_index).drop('chunk_index', axis=1), on='nuccore_accession', how='inner')
        generic_utils.print_and_write_to_log(f'num of rows in the cross product df: {len(curr_chunk_repeats_linked_df)}')

        for repeat_num in (1, 2):
            repeat_curr_chunk_repeats_linked_df = curr_chunk_repeats_linked_df[
                (
                    (curr_chunk_repeats_linked_df['start_pos'] <= curr_chunk_repeats_linked_df[f'left{repeat_num}']) &
                    (curr_chunk_repeats_linked_df[f'right{repeat_num}'] <= curr_chunk_repeats_linked_df['end_pos'])
                ) &
                (
                    (curr_chunk_repeats_linked_df['start_pos'] < curr_chunk_repeats_linked_df[f'left{repeat_num}']) |
                    (curr_chunk_repeats_linked_df[f'right{repeat_num}'] < curr_chunk_repeats_linked_df['end_pos'])
                )
            ].copy()
            repeat_curr_chunk_repeats_linked_df['repeat_dist_from_cds_end'] = (repeat_curr_chunk_repeats_linked_df['end_pos'] -
                                                                                repeat_curr_chunk_repeats_linked_df[f'right{repeat_num}'])
            repeat_curr_chunk_repeats_linked_df['repeat_dist_from_cds_start'] = (repeat_curr_chunk_repeats_linked_df[f'left{repeat_num}'] -
                                                                                  repeat_curr_chunk_repeats_linked_df['start_pos'])
            repeat_curr_chunk_repeats_linked_df['min_repeat_dist_from_cds_edge'] = repeat_curr_chunk_repeats_linked_df[[
                'repeat_dist_from_cds_end',
                'repeat_dist_from_cds_start',
            ]].min(axis=1)

            repeat_curr_chunk_repeats_linked_df = repeat_curr_chunk_repeats_linked_df.sort_values(
                by='min_repeat_dist_from_cds_edge', ascending=False,
            ).drop_duplicates(index_column_names.IR_PAIR_INDEX_COLUMN_NAMES, keep='first').drop(['start_pos', 'end_pos',
                                                                              'repeat_dist_from_cds_end', 'repeat_dist_from_cds_start',
                                                                              'min_repeat_dist_from_cds_edge'], axis=1)
            repeat_curr_chunk_repeats_linked_df['linked_repeat_num'] = repeat_num
            minimal_strictly_contained_repeats_linked_dfs.append(repeat_curr_chunk_repeats_linked_df)

    minimal_strictly_contained_repeats_linked_df = pd.concat(minimal_strictly_contained_repeats_linked_dfs, ignore_index=True)
    ir_pairs_with_sum_of_linked_repeat_nums_df = minimal_strictly_contained_repeats_linked_df.groupby(index_column_names.IR_PAIR_INDEX_COLUMN_NAMES)[
        'linked_repeat_num'].sum().reset_index(name='sum_of_linked_repeat_nums')

    ir_pairs_with_two_strictly_contained_repeats_df = ir_pairs_with_sum_of_linked_repeat_nums_df[
        ir_pairs_with_sum_of_linked_repeat_nums_df['sum_of_linked_repeat_nums'] == 3][index_column_names.IR_PAIR_INDEX_COLUMN_NAMES]
    final_repeats_of_ir_pairs_with_two_strictly_contained_repeats_df = minimal_strictly_contained_repeats_linked_df.merge(
        ir_pairs_with_two_strictly_contained_repeats_df)[minimal_linked_repeat_column_names]
    final_repeats_of_ir_pairs_with_two_strictly_contained_repeats_df['repeat_strictly_contained_in_its_cds'] = True

    ir_pairs_with_one_strictly_contained_repeat_df = ir_pairs_with_sum_of_linked_repeat_nums_df[
        (ir_pairs_with_sum_of_linked_repeat_nums_df['sum_of_linked_repeat_nums'] == 1) |
        (ir_pairs_with_sum_of_linked_repeat_nums_df['sum_of_linked_repeat_nums'] == 2)
    ][index_column_names.IR_PAIR_INDEX_COLUMN_NAMES]

    if ir_pairs_with_one_strictly_contained_repeat_df.empty:
        minimal_all_repeats_linked_df = final_repeats_of_ir_pairs_with_two_strictly_contained_repeats_df
    else:
        minimal_strictly_contained_repeats_left_to_link_df = minimal_strictly_contained_repeats_linked_df.merge(ir_pairs_with_one_strictly_contained_repeat_df)
        minimal_strictly_contained_repeats_left_to_link_df['wanted_cds_strand'] = -minimal_strictly_contained_repeats_left_to_link_df['strand']
        minimal_strictly_contained_repeats_left_to_link_df['linked_repeat_num'] = 3 - minimal_strictly_contained_repeats_left_to_link_df['linked_repeat_num']
        minimal_strictly_contained_repeats_left_to_link_df.drop(['strand', 'index_in_nuccore_cds_features_gb_file'], axis=1, inplace=True)

        repeats_left_to_link_df = pd.concat(
            [
                minimal_strictly_contained_repeats_left_to_link_df[minimal_strictly_contained_repeats_left_to_link_df['linked_repeat_num'] == repeat_num].drop(
                    [f'left{3 - repeat_num}', f'right{3 - repeat_num}'], axis=1).rename(columns={f'left{repeat_num}': f'left', f'right{repeat_num}': 'right'})
                for repeat_num in (1, 2)
            ],
            ignore_index=True,
        )
        assert len(repeats_left_to_link_df) == len(repeats_left_to_link_df[index_column_names.IR_PAIR_INDEX_COLUMN_NAMES].drop_duplicates())
        # print('repeats_left_to_link_df')
        # print(repeats_left_to_link_df)

        cds_relevant_for_repeats_left_to_link_df = all_cds_df.merge(repeats_left_to_link_df['nuccore_accession'].drop_duplicates())
        chunks_df = get_chunks_df(
            df1=cds_relevant_for_repeats_left_to_link_df,
            df2=repeats_left_to_link_df,
            df1_num_column_name='num_of_cds',
            df2_num_column_name='num_of_ir_pairs',
        )
        num_of_chunks = chunks_df['chunk_index'].nunique()

        repeats_left_to_link_df = repeats_left_to_link_df.merge(chunks_df)
        minimal_cds_relevant_for_repeats_left_to_link_df = cds_relevant_for_repeats_left_to_link_df[minimal_cds_df_column_names].merge(chunks_df)

        assert (repeats_left_to_link_df['wanted_cds_strand'].abs() == 1).all()
        repeats_left_to_link_df_grouped = repeats_left_to_link_df.groupby(['chunk_index', 'wanted_cds_strand'], sort=False)
        minimal_cds_relevant_for_repeats_left_to_link_df_grouped = minimal_cds_relevant_for_repeats_left_to_link_df.groupby('chunk_index', sort=False)



        minimal_non_strictly_contained_repeats_linked_dfs = []
        for i, ((chunk_index, wanted_cds_strand), curr_repeats_df) in enumerate(repeats_left_to_link_df_grouped):
            generic_utils.print_and_write_to_log(f'(get_all_ir_pairs_linked_df 2) '
                                                 f'starting work on chunk {i + 1}/{num_of_chunks}(*2 because we work on each strand separately) '
                                                 f'(chunk_index: {chunk_index}, wanted_cds_strand: {wanted_cds_strand})')
            curr_chunk_repeats_linked_df = curr_repeats_df.drop('chunk_index', axis=1).merge(
                minimal_cds_relevant_for_repeats_left_to_link_df_grouped.get_group(chunk_index).drop('chunk_index', axis=1), on='nuccore_accession', how='inner')
            generic_utils.print_and_write_to_log(f'num of rows in the cross product df: {len(curr_chunk_repeats_linked_df)}')

            # discard any repeat that contains the CDS
            curr_chunk_minimal_cds_containing_repeats_df = curr_chunk_repeats_linked_df[
                (curr_chunk_repeats_linked_df['left'] <= curr_chunk_repeats_linked_df['start_pos']) &
                (curr_chunk_repeats_linked_df['end_pos'] <= curr_chunk_repeats_linked_df['right'])
            ][index_column_names.IR_PAIR_INDEX_COLUMN_NAMES].drop_duplicates()
            curr_chunk_repeats_linked_df = curr_chunk_repeats_linked_df.merge(curr_chunk_minimal_cds_containing_repeats_df, how='left', indicator=True)
            curr_chunk_repeats_linked_df = curr_chunk_repeats_linked_df[curr_chunk_repeats_linked_df['_merge'] == 'left_only'].drop('_merge', axis=1)

            if wanted_cds_strand == 1:
                curr_chunk_repeats_linked_df = curr_chunk_repeats_linked_df[curr_chunk_repeats_linked_df['left'] <= curr_chunk_repeats_linked_df['start_pos']]
                assert (curr_chunk_repeats_linked_df['left'] < curr_chunk_repeats_linked_df['start_pos']).all()
                curr_chunk_repeats_linked_df['repeat_dist_from_cds'] = curr_chunk_repeats_linked_df['start_pos'] - curr_chunk_repeats_linked_df['left']
            else:
                curr_chunk_repeats_linked_df = curr_chunk_repeats_linked_df[curr_chunk_repeats_linked_df['end_pos'] <= curr_chunk_repeats_linked_df['right']]
                assert (curr_chunk_repeats_linked_df['end_pos'] < curr_chunk_repeats_linked_df['right']).all()
                curr_chunk_repeats_linked_df['repeat_dist_from_cds'] = curr_chunk_repeats_linked_df['right'] - curr_chunk_repeats_linked_df['end_pos']

            curr_chunk_repeats_linked_df = curr_chunk_repeats_linked_df.sort_values('repeat_dist_from_cds', ascending=True).drop_duplicates(
                index_column_names.IR_PAIR_INDEX_COLUMN_NAMES, keep='first')
            curr_chunk_repeats_linked_df = curr_chunk_repeats_linked_df[curr_chunk_repeats_linked_df['strand'] == wanted_cds_strand][minimal_linked_repeat_column_names]
            minimal_non_strictly_contained_repeats_linked_dfs.append(curr_chunk_repeats_linked_df)

        final_non_contained_repeats_df = pd.concat(minimal_non_strictly_contained_repeats_linked_dfs, ignore_index=True)
        final_non_contained_repeats_df['repeat_strictly_contained_in_its_cds'] = False

        final_contained_repeats_of_ir_pairs_with_one_strictly_contained_repeat_df = minimal_strictly_contained_repeats_linked_df.merge(
            ir_pairs_with_one_strictly_contained_repeat_df)[minimal_linked_repeat_column_names]
        final_contained_repeats_of_ir_pairs_with_one_strictly_contained_repeat_df['repeat_strictly_contained_in_its_cds'] = True

        minimal_all_repeats_linked_df = pd.concat([final_non_contained_repeats_df,
                                                   final_repeats_of_ir_pairs_with_two_strictly_contained_repeats_df,
                                                   final_contained_repeats_of_ir_pairs_with_one_strictly_contained_repeat_df], ignore_index=True)
    # print('final_non_contained_repeats_df')
    # print(final_non_contained_repeats_df)
    # print('final_repeats_of_ir_pairs_with_two_strictly_contained_repeats_df')
    # print(final_repeats_of_ir_pairs_with_two_strictly_contained_repeats_df)
    # print('final_contained_repeats_of_ir_pairs_with_one_strictly_contained_repeat_df')
    # print(final_contained_repeats_of_ir_pairs_with_one_strictly_contained_repeat_df)
    assert len(minimal_all_repeats_linked_df) == len(minimal_all_repeats_linked_df[index_column_names.REPEAT_INDEX_COLUMN_NAMES].drop_duplicates())

    # This isn't guaranteed, because we might fail to link some of the repeats.
    # assert len(minimal_all_repeats_linked_df) == len(minimal_all_repeats_linked_df[index_column_names.IR_PAIR_INDEX_COLUMN_NAMES].drop_duplicates()) * 2

    minimal_all_repeats_linked_df = minimal_all_repeats_linked_df.merge(
        all_cds_df,
        on=['nuccore_accession', 'index_in_nuccore_cds_features_gb_file'], how='inner',
    )
    assert len(minimal_all_repeats_linked_df) == len(minimal_all_repeats_linked_df[index_column_names.REPEAT_INDEX_COLUMN_NAMES].drop_duplicates())

    cds_column_names_to_add_prefix = (set(all_cds_df) - {'nuccore_accession'})
    get_columns_dict = lambda repeat_num: {
        **{name: f'repeat{repeat_num}_cds_{name}' for name in cds_column_names_to_add_prefix},
        'repeat_strictly_contained_in_its_cds': f'repeat{repeat_num}_strictly_contained_in_its_cds',
    }

    repeat_num_to_minimal_all_repeats_linked_df = {
        repeat_num: minimal_all_repeats_linked_df[
            minimal_all_repeats_linked_df['linked_repeat_num'] == repeat_num
        ].rename(columns=get_columns_dict(repeat_num)).drop('linked_repeat_num', axis=1)
        for repeat_num in (1, 2)
    }
    # inner join here means that we only keep pairs with both repeats linked.
    minimal_all_ir_pairs_linked_df = repeat_num_to_minimal_all_repeats_linked_df[1].merge(repeat_num_to_minimal_all_repeats_linked_df[2],
                                                                                          how='inner', on=index_column_names.IR_PAIR_INDEX_COLUMN_NAMES)

    assert not minimal_all_ir_pairs_linked_df[['repeat1_strictly_contained_in_its_cds', 'repeat2_strictly_contained_in_its_cds']].isna().any().any()
    minimal_all_ir_pairs_linked_df['each_repeat_is_strictly_contained_in_its_cds'] = minimal_all_ir_pairs_linked_df[['repeat1_strictly_contained_in_its_cds',
                                                                                                              'repeat2_strictly_contained_in_its_cds']].all(axis=1)

    assert (minimal_all_ir_pairs_linked_df[['repeat1_cds_strand', 'repeat2_cds_strand']].abs() == 1).all(axis=None)

    assert minimal_all_ir_pairs_linked_df[
        minimal_all_ir_pairs_linked_df['repeat1_cds_strand'] == minimal_all_ir_pairs_linked_df['repeat2_cds_strand']
    ]['each_repeat_is_strictly_contained_in_its_cds'].all()

    minimal_all_ir_pairs_linked_df = minimal_all_ir_pairs_linked_df[
        minimal_all_ir_pairs_linked_df['repeat1_cds_strand'] != minimal_all_ir_pairs_linked_df['repeat2_cds_strand']
    ]

    final_all_ir_pairs_linked_df = minimal_all_ir_pairs_linked_df.merge(all_ir_pairs_df, on=['nuccore_accession', 'index_in_nuccore_ir_pairs_df_csv_file'])

    assert len(final_all_ir_pairs_linked_df) == len(final_all_ir_pairs_linked_df[index_column_names.IR_PAIR_INDEX_COLUMN_NAMES].drop_duplicates())
    return final_all_ir_pairs_linked_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_ir_pairs_linked_df_csv(
        input_file_path_all_cds_df_csv,
        input_file_path_all_ir_pairs_df_csv,
        output_file_path_ir_pairs_linked_df_csv,
        output_file_path_ir_pair_linkage_info_pickle,
        debug___num_of_nuccore_entries_to_go_over,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    # in case of a nuccore without any cds, any IR pair in it won't appear in all_ir_pairs_linked_df.

    all_cds_df = pd.read_csv(input_file_path_all_cds_df_csv, sep='\t', low_memory=False)
    all_ir_pairs_df = pd.read_csv(input_file_path_all_ir_pairs_df_csv, sep='\t', low_memory=False)

    orig_num_of_ir_pairs = len(all_ir_pairs_df)

    all_ir_pairs_linked_df = get_all_ir_pairs_linked_df(
        all_cds_df=all_cds_df,
        all_ir_pairs_df=all_ir_pairs_df,
        debug___num_of_nuccore_entries_to_go_over=debug___num_of_nuccore_entries_to_go_over,
    )

    num_of_linked_ir_pairs = len(all_ir_pairs_linked_df)
    num_of_not_linked_ir_pairs = orig_num_of_ir_pairs - num_of_linked_ir_pairs
    not_linked_ir_pair_proportion = num_of_not_linked_ir_pairs / orig_num_of_ir_pairs

    all_ir_pairs_linked_df.to_csv(output_file_path_ir_pairs_linked_df_csv, sep='\t', index=False)
    ir_pair_linkage_info = {
        'orig_num_of_ir_pairs': orig_num_of_ir_pairs,
        'num_of_linked_ir_pairs': num_of_linked_ir_pairs,

        'num_of_not_linked_ir_pairs': num_of_not_linked_ir_pairs,
        'not_linked_ir_pair_proportion': not_linked_ir_pair_proportion,
    }

    with open(output_file_path_ir_pair_linkage_info_pickle, 'wb') as f:
        pickle.dump(ir_pair_linkage_info, f, protocol=4)

def write_ir_pairs_linked_df_csv(
        input_file_path_all_cds_df_csv,
        input_file_path_all_ir_pairs_df_csv,
        output_file_path_ir_pairs_linked_df_csv,
        output_file_path_ir_pair_linkage_info_pickle,
        debug___num_of_nuccore_entries_to_go_over,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_ir_pairs_linked_df_csv(
        input_file_path_all_cds_df_csv=input_file_path_all_cds_df_csv,
        input_file_path_all_ir_pairs_df_csv=input_file_path_all_ir_pairs_df_csv,
        output_file_path_ir_pairs_linked_df_csv=output_file_path_ir_pairs_linked_df_csv,
        output_file_path_ir_pair_linkage_info_pickle=output_file_path_ir_pair_linkage_info_pickle,
        debug___num_of_nuccore_entries_to_go_over=debug___num_of_nuccore_entries_to_go_over,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=27,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_discard_ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds(
        input_file_path_pairs_df_csv,
        output_file_path_filtered_pairs_df_csv,
        output_file_path_discard_pairs_without_any_repeat_strictly_contained_in_its_cds_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    pairs_df = pd.read_csv(input_file_path_pairs_df_csv, sep='\t', low_memory=False)
    orig_num_of_ir_pairs = len(pairs_df)

    for repeat_num in (1, 2):
        assert not pairs_df[f'repeat{repeat_num}_cds_index_in_nuccore_cds_features_gb_file'].isna().any()

        assert (pairs_df[f'left{repeat_num}'] <= pairs_df[f'repeat{repeat_num}_cds_end_pos']).all()
        assert (pairs_df[f'repeat{repeat_num}_cds_start_pos'] <= pairs_df[f'right{repeat_num}']).all()

        pairs_df[f'repeat{repeat_num}_strictly_contained_in_its_cds'] = (
            (pairs_df[f'repeat{repeat_num}_cds_start_pos'] < pairs_df[f'left{repeat_num}']) &
            (pairs_df[f'right{repeat_num}'] < pairs_df[f'repeat{repeat_num}_cds_end_pos'])
        )

    pairs_df['any_repeat_strictly_contained_in_its_cds'] = pairs_df[['repeat1_strictly_contained_in_its_cds', 'repeat2_strictly_contained_in_its_cds']].any(axis=1)

    cds_pairs_with_any_repeat_strictly_contained_in_its_cds_df = pairs_df.groupby(index_column_names.CDS_PAIR_INDEX_COLUMN_NAMES)['any_repeat_strictly_contained_in_its_cds'].any().reset_index(
        name='any_ir_pair_repeat_strictly_contained_in_its_cds')
    cds_pairs_with_any_repeat_strictly_contained_in_its_cds_df = cds_pairs_with_any_repeat_strictly_contained_in_its_cds_df[
        cds_pairs_with_any_repeat_strictly_contained_in_its_cds_df['any_ir_pair_repeat_strictly_contained_in_its_cds']].drop('any_ir_pair_repeat_strictly_contained_in_its_cds', axis=1)

    filtered_pairs_df = pairs_df.merge(cds_pairs_with_any_repeat_strictly_contained_in_its_cds_df)

    num_of_ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds = orig_num_of_ir_pairs - len(filtered_pairs_df)
    ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds_proportion = (num_of_ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds /
                                                                                      orig_num_of_ir_pairs)

    filtered_pairs_df.to_csv(output_file_path_filtered_pairs_df_csv, sep='\t', index=False)
    discard_pairs_without_any_repeat_strictly_contained_in_its_cds_info = {
        'num_of_ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds': num_of_ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds,
        'ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds_proportion': ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds_proportion,
    }
    with open(output_file_path_discard_pairs_without_any_repeat_strictly_contained_in_its_cds_info_pickle, 'wb') as f:
        pickle.dump(discard_pairs_without_any_repeat_strictly_contained_in_its_cds_info, f, protocol=4)

def discard_ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds(
        input_file_path_pairs_df_csv,
        output_file_path_filtered_pairs_df_csv,
        output_file_path_discard_pairs_without_any_repeat_strictly_contained_in_its_cds_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_discard_ir_pairs_of_cds_pairs_without_any_repeat_strictly_contained_in_its_cds(
        input_file_path_pairs_df_csv=input_file_path_pairs_df_csv,
        output_file_path_filtered_pairs_df_csv=output_file_path_filtered_pairs_df_csv,
        output_file_path_discard_pairs_without_any_repeat_strictly_contained_in_its_cds_info_pickle=output_file_path_discard_pairs_without_any_repeat_strictly_contained_in_its_cds_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=4,
    )


