IR_PAIR_INDEX_COLUMN_NAMES = ['nuccore_accession', 'index_in_nuccore_ir_pairs_df_csv_file']

REPEAT_INDEX_COLUMN_NAMES = IR_PAIR_INDEX_COLUMN_NAMES + ['linked_repeat_num']

CDS_INDEX_COLUMN_NAMES = ['nuccore_accession', 'index_in_nuccore_cds_features_gb_file']

OPERON_INDEX_COLUMN_NAMES = ['nuccore_accession', 'operon_index']

CDS_PAIR_INDEX_COLUMN_NAMES = [
    'nuccore_accession',
    'repeat1_cds_index_in_nuccore_cds_features_gb_file',
    'repeat2_cds_index_in_nuccore_cds_features_gb_file',
]

LONGER_REPEAT_CDS_INDEX_COLUMN_NAMES = [
    'nuccore_accession',
    'longer_repeat_cds_index_in_nuccore_cds_features_gb_file',
]



MERGED_CDS_PAIR_REGION_INDEX_COLUMN_NAMES = ['nuccore_accession', 'merged_cds_pair_region_start', 'merged_cds_pair_region_end']

REGION_IN_OTHER_NUCCORE_INDEX_COLUMN_NAMES = [
    'nuccore_accession',
    'merged_cds_pair_region_start',
    'merged_cds_pair_region_end',
    'other_nuccore_accession',
    'region_in_other_nuccore_start',
    'region_in_other_nuccore_end',
]


MERGED_PADDED_CDS_INDEX_INTERVAL_INDEX_COLUMN_NAMES = ['nuccore_accession', 'merged_padded_cds_index_interval_start', 'merged_padded_cds_index_interval_end']