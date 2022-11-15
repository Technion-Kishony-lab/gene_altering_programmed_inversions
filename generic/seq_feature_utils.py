import collections
import math
import pickle
import random
import re
import warnings

import Bio
import numpy as np
import pandas as pd

from generic import bio_utils
from generic import blast_interface_and_utils
from generic import generic_utils

TRANSPOSASE_EC_NUMBER_WHICH_IS_LEVEL_3 = '2.7.7'  # https://en.wikipedia.org/wiki/Transposase

POTENTIALLY_INTERESTING_SOMETIME_MULTIPLE_VALUE_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT = {
    'gene_synonym',
    'rpt_unit_range',
}

POTENTIALLY_INTERESTING_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT = {
    # according to http://www.insdc.org/documents/feature_table.html#7.3.1
    'codon_start',
    # 'country', # NAH.
    # 'experiment' # NAH. not structured enough for us to perform relatively-fast data-mining, i think.
    'gene',
    'locus_tag',
    # 'comment', # NAH. not structured at all.
    'phenotype',  # probably won't help much, but whatever.
    'product',
    'protein_id',
    'rpt_unit_seq',
    'standard_name',
}

POTENTIALLY_INTERESTING_NON_VALUELESS_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT = (
        POTENTIALLY_INTERESTING_SOMETIME_MULTIPLE_VALUE_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT |
        POTENTIALLY_INTERESTING_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT
)

POTENTIALLY_INTERESTING_VALUELESS_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT = {
    'proviral',
    'pseudo',
    'transgenic',
}

SEQ_FEATURE_BASIC_COLUMNS = [
    'start_position_ignoring_fuzziness',
    'end_position_ignoring_fuzziness',
    'strand',
    'is_part_of_compound_location',
    'type',
    'id',
    'has_only_a_single_qualifier',
]

POTENTIALLY_INTERESTING_AND_REQUIRING_SPECIAL_TREATMENT_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT = {
    'bound_moiety',
    'EC_number',
    'function',
    'inference',  # turns out this one is structured!
    'mobile_element_type',
    'ncRNA_class',  # probably won't help much, but whatever.
    'pseudogene',
    'recombination_class',
    'regulatory_class',
    'rpt_type',
    'satellite',
}

POTENTIALLY_INTERESTING_AND_REQUIRING_SPECIAL_TREATMENT_QUALIFIERS_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT_COLUMN_NAMES = [
    'bound_moiety',
    'ec_number_level1_class',
    'ec_number_level2_class',
    'ec_number_level3_class',
    'ec_number_level4_class',
    'function',
    'inference_type',
    'inference_type_internal_id',
    'is_inference_from_same_species',
    'mobile_element_type',
    'mobile_element_type_type_internal_id',
    'non_coding_rna_class',
    'non_coding_rna_class_internal_id',
    'pseudogene_type',
    'pseudogene_type_internal_id',
    'recombination_class',
    'recombination_class_internal_id',
    'regulatory_class',
    'regulatory_class_internal_id',
    'rpt_type',
    'rpt_type_internal_id',
    'satellite',
    'satellite_type',
    'satellite_type_internal_id',
]

ALL_POTENTIALLY_INTERESTING_SEQ_FEATURE_COLUMNS = (
        SEQ_FEATURE_BASIC_COLUMNS +
        POTENTIALLY_INTERESTING_AND_REQUIRING_SPECIAL_TREATMENT_QUALIFIERS_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT_COLUMN_NAMES +
        sorted(POTENTIALLY_INTERESTING_NON_VALUELESS_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT |
               POTENTIALLY_INTERESTING_VALUELESS_QUALIFIERS_NAMES_AT_LEAST_FOR_MULTIPLE_VARIANT_PHASE_VARIATION_PROJECT)
)

SEQ_FEATURE_COLUMN_NAME_TO_DTYPE = {
    # Why I bother? https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options/27232309#27232309
    # dtypes according to:
    #   https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes
    #   http://www.insdc.org/documents/feature_table.html#7.3.1
    #   my code.
    'start_position_ignoring_fuzziness': int,  # can't be nan
    'end_position_ignoring_fuzziness': int,  # can't be nan
    'strand': np.float64,
    'is_part_of_compound_location': bool,  # can't be nan
    'type': 'category',
    'id': 'string',
    'has_only_a_single_qualifier': bool,  # can't be nan

    'bound_moiety': 'string',
    'ec_number_level1_class': 'string',
    'ec_number_level2_class': 'string',
    'ec_number_level3_class': 'string',
    'ec_number_level4_class': 'string',
    'function': 'string',
    'inference_type': 'string',
    'inference_type_internal_id': np.float64,
    'is_inference_from_same_species': 'boolean',
    'mobile_element_type': 'string',
    'mobile_element_type_type_internal_id': np.float64,
    'non_coding_rna_class': 'string',
    'non_coding_rna_class_internal_id': np.float64,
    'pseudogene_type': 'string',
    'pseudogene_type_internal_id': np.float64,
    'recombination_class': 'string',
    'recombination_class_internal_id': np.float64,
    'regulatory_class': 'string',
    'regulatory_class_internal_id': np.float64,
    'rpt_type': 'string',
    'rpt_type_internal_id': np.float64,
    'satellite': 'string',
    'satellite_type': 'string',
    'satellite_type_internal_id': np.float64,

    'codon_start': np.float64,
    'gene': 'string',
    'gene_synonym': 'string',
    'locus_tag': 'string',
    'phenotype': 'string',
    'product': 'string',
    'protein_id': 'string',
    'proviral': 'boolean',
    'pseudo': 'boolean',
    'rpt_unit_range': 'string',
    'rpt_unit_seq': 'string',
    'standard_name': 'string',
    'transgenic': 'boolean',
}

assert set(SEQ_FEATURE_COLUMN_NAME_TO_DTYPE) == set(ALL_POTENTIALLY_INTERESTING_SEQ_FEATURE_COLUMNS)

QUALIFIER_VALUES_SEPARATOR = ';;;'

QUALIFIER_OPTIONAL_CATEGORY_REGEX_UNGROUPED = '(?:(?:COORDINATES|DESCRIPTION|EXISTENCE):)?'
QUALIFIER_INFERENCE_TYPES = [
    # the order matters!! if you add something, it must be last so that the previous IDs aren't messed up.
    'non-experimental evidence, no additional details recorded',
    'similar to RNA sequence, other RNA',
    'similar to RNA sequence, mRNA',
    'similar to RNA sequence, EST',
    'similar to RNA sequence',
    'similar to DNA sequence',
    'similar to AA sequence',
    'ab initio prediction',
    'similar to sequence',
    'nucleotide motif',
    'protein motif',
    'alignment',
    'profile',
    'PHASTER annotation server',  # only because of https://www.ncbi.nlm.nih.gov/nuccore/1476651252. i don't think this is an official option. whatever.
]
QUALIFIER_INFERENCE_TYPE_INTERNAL_ID_TO_TYPE = dict(enumerate(QUALIFIER_INFERENCE_TYPES))
QUALIFIER_INFERENCE_TYPE_TO_TYPE_INTERNAL_ID = {v: k for k, v in QUALIFIER_INFERENCE_TYPE_INTERNAL_ID_TO_TYPE.items()}
QUALIFIER_INFERENCE_TYPE_REGEX = '|'.join(sorted(QUALIFIER_INFERENCE_TYPES, key=len, reverse=True))  # A somewhat ugly hack to make the regex greedy...

# why I added ' ?' after QUALIFIER_OPTIONAL_CATEGORY_REGEX_UNGROUPED? because I encountered (in some gb file) a value with a space there...
QUALIFIER_INFERENCE_REGEX_WITH_TYPE_AND_SAME_SPECIES_GROUPED___WITHOUT_VERIFYING_EVIDENCE_BASIS_STRUCTURE = (
    f'{QUALIFIER_OPTIONAL_CATEGORY_REGEX_UNGROUPED} ?({QUALIFIER_INFERENCE_TYPE_REGEX})( \\(same species\\))?(?::.+)?')

QUALIFIER_MOBILE_ELEMENT_TYPE_TYPES = [
    "transposon",
    "retrotransposon",
    "integron",
    "insertion sequence",
    "non-LTR retrotransposon",
    "SINE",
    "MITE",
    "LINE",
    "other",
]
QUALIFIER_MOBILE_ELEMENT_TYPE_TYPE_INTERNAL_ID_TO_TYPE = dict(enumerate(QUALIFIER_MOBILE_ELEMENT_TYPE_TYPES))
QUALIFIER_MOBILE_ELEMENT_TYPE_TYPE_TO_TYPE_INTERNAL_ID = {v: k for k, v in QUALIFIER_MOBILE_ELEMENT_TYPE_TYPE_INTERNAL_ID_TO_TYPE.items()}
QUALIFIER_MOBILE_ELEMENT_TYPE_TYPE_REGEX = '|'.join(sorted(QUALIFIER_MOBILE_ELEMENT_TYPE_TYPES, key=len, reverse=True))  # A somewhat ugly hack to make the regex greedy...
QUALIFIER_MOBILE_ELEMENT_TYPE_REGEX_WITH_TYPE_GROUPED = f'({QUALIFIER_MOBILE_ELEMENT_TYPE_TYPE_REGEX})(?::.+)?'

QUALIFIER_NON_CODING_RNA_CLASSES = [
    'antisense_RNA',
    'autocatalytically_spliced_intron',
    'ribozyme',
    'hammerhead_ribozyme',
    'lncRNA',
    'RNase_P_RNA',
    'RNase_MRP_RNA',
    'telomerase_RNA',
    'guide_RNA',
    'sgRNA',
    'rasiRNA',
    'scRNA',
    'scaRNA',
    'siRNA',
    'pre_miRNA',
    'miRNA',
    'piRNA',
    'snoRNA',
    'snRNA',
    'SRP_RNA',
    'vault_RNA',
    'Y_RNA',
    'other',
]
QUALIFIER_NON_CODING_RNA_CLASS_INTERNAL_ID_TO_CLASS = dict(enumerate(QUALIFIER_NON_CODING_RNA_CLASSES))
QUALIFIER_NON_CODING_RNA_CLASS_TO_CLASS_INTERNAL_ID = {v: k for k, v in QUALIFIER_NON_CODING_RNA_CLASS_INTERNAL_ID_TO_CLASS.items()}

QUALIFIER_PSEUDOGENE_TYPES = [
    'processed',
    'unprocessed',
    'unitary',
    'allelic',
    'unknown',
]
QUALIFIER_PSEUDOGENE_TYPE_INTERNAL_ID_TO_TYPE = dict(enumerate(QUALIFIER_PSEUDOGENE_TYPES))
QUALIFIER_PSEUDOGENE_TYPE_TO_TYPE_INTERNAL_ID = {v: k for k, v in QUALIFIER_PSEUDOGENE_TYPE_INTERNAL_ID_TO_TYPE.items()}

QUALIFIER_RECOMBINATION_CLASSES = [
    'meiotic',
    'mitotic',
    'non_allelic_homologous',
    'chromosome_breakpoint',
    'other',
]
QUALIFIER_RECOMBINATION_CLASS_INTERNAL_ID_TO_CLASS = dict(enumerate(QUALIFIER_RECOMBINATION_CLASSES))
QUALIFIER_RECOMBINATION_CLASS_TO_CLASS_INTERNAL_ID = {v: k for k, v in QUALIFIER_RECOMBINATION_CLASS_INTERNAL_ID_TO_CLASS.items()}

QUALIFIER_REGULATORY_CLASSES = [
    'attenuator',
    'CAAT_signal',
    'enhancer',
    'enhancer_blocking_element',
    'GC_signal',
    'imprinting_control_region',
    'insulator',
    'locus_control_region',
    'minus_35_signal',
    'minus_10_signal',
    'polyA_signal_sequence',
    'promoter',
    'response_element',
    'ribosome_binding_site',
    'riboswitch',
    'silencer',
    'TATA_box',
    'terminator',
    'other',
]
QUALIFIER_REGULATORY_CLASS_INTERNAL_ID_TO_CLASS = dict(enumerate(QUALIFIER_REGULATORY_CLASSES))
QUALIFIER_REGULATORY_CLASS_TO_CLASS_INTERNAL_ID = {v: k for k, v in QUALIFIER_REGULATORY_CLASS_INTERNAL_ID_TO_CLASS.items()}

QUALIFIER_RPT_TYPES_LOWERCASE = [
    'tandem',
    'direct',
    'inverted',
    'flanking',
    'nested',
    'dispersed',
    'terminal',
    'long_terminal_repeat',
    'non_ltr_retrotransposon_polymeric_tract',
    'centromeric_repeat',
    'telomeric_repeat',
    'x_element_combinatorial_repeat',
    'y_prime_element',
    'other',
]
QUALIFIER_RPT_TYPE_INTERNAL_ID_TO_TYPE_LOWERCASE = dict(enumerate(QUALIFIER_RPT_TYPES_LOWERCASE))
QUALIFIER_RPT_TYPE_LOWERCASE_TO_TYPE_INTERNAL_ID = {v: k for k, v in QUALIFIER_RPT_TYPE_INTERNAL_ID_TO_TYPE_LOWERCASE.items()}

QUALIFIER_SATELLITE_TYPES = ["satellite", "microsatellite", "minisatellite"]
QUALIFIER_SATELLITE_TYPE_INTERNAL_ID_TO_TYPE = dict(enumerate(QUALIFIER_SATELLITE_TYPES))
QUALIFIER_SATELLITE_TYPE_TO_TYPE_INTERNAL_ID = {v: k for k, v in QUALIFIER_SATELLITE_TYPE_INTERNAL_ID_TO_TYPE.items()}
QUALIFIER_SATELLITE_TYPE_REGEX = '|'.join(sorted(QUALIFIER_SATELLITE_TYPES, key=len, reverse=True))  # A somewhat ugly hack to make the regex greedy...
QUALIFIER_SATELLITE_TYPE_REGEX_WITH_TYPE_GROUPED = f'({QUALIFIER_SATELLITE_TYPE_REGEX})'

def get_feature_start_pos(feature):
    # they follow the python convention, so they use the start and stop indices, but use "end" instead of "stop". ugh.
    return int(feature.location.start) + 1

def get_feature_end_pos(feature):
    return int(feature.location.end)

def get_feature_middle_pos(feature):
    return (get_feature_start_pos(feature) + get_feature_end_pos(feature)) / 2

def get_feature_dist_from_position(feature, position):
    start_pos = get_feature_start_pos(feature)
    end_pos = get_feature_end_pos(feature)
    if start_pos <= position <= end_pos:
        return 0
    return min(abs(position - start_pos),
               abs(position - end_pos))

def do_features_have_same_edges(feature1, feature2):
    return (get_feature_start_pos(feature1) == get_feature_start_pos(feature2)) and (
            get_feature_end_pos(feature1) == get_feature_end_pos(feature2))

def get_index_of_leftmost_feature(features, predicate=lambda x: True):
    predicate_results = [predicate(x) for x in features]
    if not any(predicate_results):
        return None
    return min(range(len(features)), key=lambda i: get_feature_start_pos(features[i]) if predicate_results[i] else np.inf)

def get_index_of_rightmost_feature(features, predicate=lambda x: True):
    predicate_results = [predicate(x) for x in features]
    if not any(predicate_results):
        return None
    return max(range(len(features)), key=lambda i: get_feature_end_pos(features[i]) if predicate_results[i] else -np.inf)

def get_most_upstream_position_in_cds(cds_feature):
    assert cds_feature.type == 'CDS'
    assert abs(cds_feature.location.strand) == 1
    if cds_feature.location.strand == 1:
        return get_feature_start_pos(cds_feature)
    else:
        return get_feature_end_pos(cds_feature)

def get_most_downstream_position_in_cds(cds_feature):
    assert cds_feature.type == 'CDS'
    assert abs(cds_feature.location.strand) == 1
    if cds_feature.location.strand == 1:
        return get_feature_end_pos(cds_feature)
    else:
        return get_feature_start_pos(cds_feature)

def get_cds_first_codon_first_position(cds_feature):
    assert cds_feature.type == 'CDS'
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    assert len(cds_feature.qualifiers['codon_start']) == 1
    codon_start_qualifier = int(cds_feature.qualifiers['codon_start'][0])
    if strand == 1:
        first_codon_first_pos = get_feature_start_pos(cds_feature) + 1 - codon_start_qualifier
    else:
        first_codon_first_pos = get_feature_end_pos(cds_feature) - 1 + codon_start_qualifier

    return first_codon_first_pos

def get_cds_first_codon_third_position(cds_feature):
    assert cds_feature.type == 'CDS'
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    first_codon_first_pos = get_cds_first_codon_first_position(cds_feature)
    if strand == 1:
        first_codon_third_pos = first_codon_first_pos + 2
    else:
        first_codon_third_pos = first_codon_first_pos - 2

    return first_codon_third_pos

def get_cds_second_codon_start_position(cds_feature):
    assert cds_feature.type == 'CDS'
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    first_codon_first_pos = get_cds_first_codon_first_position(cds_feature)
    if strand == 1:
        second_codon_start_pos = first_codon_first_pos + 3
    else:
        second_codon_start_pos = first_codon_first_pos - 3

    return second_codon_start_pos

def get_cds_last_codon_first_position(cds_feature):
    assert cds_feature.type == 'CDS'
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    first_codon_start_position = get_cds_first_codon_first_position(cds_feature)
    # cds_len = get_feature_len(cds_feature)
    # num_of_codons = math.ceil(cds_len / 3)
    if strand == 1:
        end_pos = get_feature_end_pos(cds_feature)
        num_of_codons = math.ceil((end_pos - first_codon_start_position + 1) / 3)
        last_codon_start_position = first_codon_start_position + (num_of_codons - 1) * 3
    else:
        start_pos = get_feature_start_pos(cds_feature)
        num_of_codons = math.ceil((first_codon_start_position - start_pos + 1) / 3)
        last_codon_start_position = first_codon_start_position - (num_of_codons - 1) * 3

    return last_codon_start_position


def get_position_frame_for_cds_feature(cds_feature, position):
    assert cds_feature.type == 'CDS'
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    first_codon_first_pos = get_cds_first_codon_first_position(cds_feature)
    if strand == 1:
        position_frame = (position - first_codon_first_pos) % 3 + 1
    else:
        position_frame = (first_codon_first_pos - position) % 3 + 1

    return position_frame


def find_closest_upstream_in_frame_stop_codon(cds_feature, nuccore_seq, pos_in_codon_to_return=1):
    assert pos_in_codon_to_return in {1, 2, 3}
    assert cds_feature.type == 'CDS'
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    first_codon_first_pos = get_cds_first_codon_first_position(cds_feature)
    if strand == 1:
        # we don't start 3 bases to the left because first_codon_first_pos could actually be outside the CDS!
        # (the CDS annotation could start from a base which isn't the first in the codon.)
        for curr_pos in range(first_codon_first_pos, 0, -3):
            if bio_utils.seq_record_to_str(nuccore_seq[curr_pos - 1:curr_pos + 2]) in bio_utils.BACTERIA_STOP_CODONS:
                return curr_pos + pos_in_codon_to_return - 1
        return None # in case we didnt find any stop codon
    else:
        assert strand == -1
        # we don't start 3 bases to the right because first_codon_first_pos could actually be outside the CDS!
        # (the CDS annotation could start from a base which isn't the first in the codon.)
        for curr_pos in range(first_codon_first_pos, len(nuccore_seq) + 1, 3):
            if bio_utils.seq_record_to_str(nuccore_seq[curr_pos - 3:curr_pos]) in bio_utils.BACTERIA_STOP_CODONS_INVERTED:
                return curr_pos - pos_in_codon_to_return + 1
        return None # in case we didnt find any stop codon

def find_closest_downstream_in_frame_stop_codon(cds_feature, nuccore_seq, pos_in_codon_to_return=1):
    assert pos_in_codon_to_return in {1, 2, 3}
    assert cds_feature.type == 'CDS'
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    last_codon_start_pos = get_cds_last_codon_first_position(cds_feature)
    if strand == 1:
        # we don't start 3 bases to the right because last_codon_start_pos+2 could actually be outside the CDS, IIUC.
        # (the CDS annotation could end in a base which isn't the last in the codon, IIUC.)
        # More importantly, usually the last codon is a stop codon.
        for curr_pos in range(last_codon_start_pos, len(nuccore_seq) - 1, 3):
            if bio_utils.seq_record_to_str(nuccore_seq[curr_pos - 1:curr_pos + 2]) in bio_utils.BACTERIA_STOP_CODONS:
                return curr_pos + pos_in_codon_to_return - 1
        return None # in case we didnt find any stop codon
    else:
        assert strand == -1
        # we don't start 3 bases to the left because first_codon_first_pos-2 could actually be outside the CDS, IIUC.
        # (the CDS annotation could end in a base which isn't the last in the codon, IIUC.)
        # More importantly, usually the last codon is a stop codon.
        for curr_pos in range(last_codon_start_pos, 2, -3):
            if bio_utils.seq_record_to_str(nuccore_seq[curr_pos - 3:curr_pos]) in bio_utils.BACTERIA_STOP_CODONS_INVERTED:
                return curr_pos - pos_in_codon_to_return + 1
        return None # in case we didnt find any stop codon

def find_closest_in_frame_stop_codon_left_to_cds(cds_feature, nuccore_seq, pos_in_codon_to_return=1):
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    if strand == 1:
        return find_closest_upstream_in_frame_stop_codon(
            cds_feature=cds_feature,
            nuccore_seq=nuccore_seq,
            pos_in_codon_to_return=pos_in_codon_to_return,
        )
    else:
        return find_closest_downstream_in_frame_stop_codon(
            cds_feature=cds_feature,
            nuccore_seq=nuccore_seq,
            pos_in_codon_to_return=pos_in_codon_to_return,
        )

def find_closest_in_frame_stop_codon_right_to_cds(cds_feature, nuccore_seq, pos_in_codon_to_return=1):
    strand = cds_feature.location.strand
    assert abs(strand) == 1
    if strand == 1:
        return find_closest_downstream_in_frame_stop_codon(
            cds_feature=cds_feature,
            nuccore_seq=nuccore_seq,
            pos_in_codon_to_return=pos_in_codon_to_return,
        )
    else:
        return find_closest_upstream_in_frame_stop_codon(
            cds_feature=cds_feature,
            nuccore_seq=nuccore_seq,
            pos_in_codon_to_return=pos_in_codon_to_return,
        )

def is_joined_feature(feature):
    return feature.location_operator == 'join'

def get_joined_feature_part_regions(feature):
    assert feature.location_operator == 'join'
    return sorted([(int(x.start), int(x.end)) for x in feature.location.parts])

def get_total_dist_between_joined_feature_parts(feature):
    joined_feature_part_regions = get_joined_feature_part_regions(feature)
    total_dist_between_joined_feature_parts = 0
    for prev_part_region, part_region in zip(joined_feature_part_regions[:-1], joined_feature_part_regions[1:]):
        dist_between_prev_and_curr_regions = max(0, part_region[0] - prev_part_region[1] - 1)
        total_dist_between_joined_feature_parts += dist_between_prev_and_curr_regions
    return total_dist_between_joined_feature_parts

def discard_joined_features_with_large_total_dist_between_joined_parts_internal(features, max_total_dist_between_joined_parts):
    return [x for x in features
            if (not is_joined_feature(x)) or (get_total_dist_between_joined_feature_parts(x) <= max_total_dist_between_joined_parts)]

@generic_utils.execute_if_output_doesnt_exist_already
def cached_discard_joined_features_with_large_total_dist_between_joined_parts(
        input_file_path_gb,
        max_total_dist_between_joined_parts,
        discard_non_cds,
        output_file_path_filtered_gb,
        output_file_path_num_of_filtered_features,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    gb_record = bio_utils.get_gb_record(input_file_path_gb)
    if discard_non_cds:
        features = [x for x in gb_record.features if x.type == 'CDS']
    else:
        features = gb_record.features
    filtered_features = discard_joined_features_with_large_total_dist_between_joined_parts_internal(features, max_total_dist_between_joined_parts)
    gb_record.features = filtered_features
    bio_utils.write_records_to_fasta_or_gb_file(gb_record, output_file_path_filtered_gb, 'gb')

    num_of_filtered_features = len(filtered_features)
    generic_utils.write_text_file(output_file_path_num_of_filtered_features, str(num_of_filtered_features))

def discard_joined_features_with_large_total_dist_between_joined_parts(
        input_file_path_gb,
        max_total_dist_between_joined_parts,
        discard_non_cds,
        output_file_path_filtered_gb,
        output_file_path_num_of_filtered_features,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_discard_joined_features_with_large_total_dist_between_joined_parts(
        input_file_path_gb=input_file_path_gb,
        max_total_dist_between_joined_parts=max_total_dist_between_joined_parts,
        discard_non_cds=discard_non_cds,
        output_file_path_filtered_gb=output_file_path_filtered_gb,
        output_file_path_num_of_filtered_features=output_file_path_num_of_filtered_features,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )

def get_feature_len(feature):
    return get_feature_end_pos(feature) - get_feature_start_pos(feature) + 1

def get_seq_features(gb_file_path, return_source=False):
    gb_record = bio_utils.get_gb_record(gb_file_path)
    seq_features = gb_record.features

    if (not return_source) and (len(seq_features) > 0):
        # seems to me like every feature type except for 'source' might be interesting.
        if seq_features[0].type == 'source':
            seq_features = seq_features[1:]

    return seq_features


def get_cds_seq_features(gb_file_path):
    return [x for x in get_seq_features(gb_file_path) if x.type == 'CDS']

def get_product_qualifier(feature):
    qualifiers = feature.qualifiers
    if 'product' in qualifiers:
        products = qualifiers['product']
        assert len(products) == 1
        product = products[0]
    else:
        product = None
    return product

def any_stop_codon_in_seq(seq, already_a_multiple_of_3=True):
    if not already_a_multiple_of_3:
        region_len_before_truncating_to_multiple_of_3 = len(seq)
        seq = seq[:(region_len_before_truncating_to_multiple_of_3 - (region_len_before_truncating_to_multiple_of_3 % 3))]

    seq_len = len(seq)
    assert (seq_len % 3) == 0
    orig_num_of_codons = seq_len // 3

    seq_with_added_start_and_stop_codons = bio_utils.str_to_seq_record('ATG') + seq + bio_utils.str_to_seq_record('TGA')
    try:
        peptide_with_added_methionine_seq = seq_with_added_start_and_stop_codons.translate(table=bio_utils.BACTERIA_CODON_TABLE_ID, cds=True)
    except Bio.Data.CodonTable.TranslationError:
        return True

    assert len(peptide_with_added_methionine_seq) == orig_num_of_codons + 1
    return False
    # num_of_codons_until_stop_codon = len(peptide_with_added_methionine_seq)
    # assert num_of_codons_until_stop_codon <= orig_num_of_codons + 1 # because we added a methionine
    # return num_of_codons_until_stop_codon != orig_num_of_codons + 1

def any_stop_codon_in_region(nuccore_seq, region, strand, already_a_multiple_of_3=True):
    # print(f'any_stop_codon_in_region, region: {region}')
    # print(f'nuccore_seq.name: {nuccore_seq.name}')
    if region[1] - region[0] + 1 == 0:
        return False
    assert region[1] - region[0] + 1 > 0

    region_seq = bio_utils.get_region_in_chrom_seq(nuccore_seq, region[0], region[1])
    assert abs(strand) == 1
    if strand == -1:
        region_seq = region_seq.reverse_complement()

    return any_stop_codon_in_seq(region_seq, already_a_multiple_of_3)



def find_features_for_which_position_is_downstream(features, position, max_downstream_dist_from_feature):
    feature_infos = []
    for i, feature in enumerate(features):
        if feature.strand == 1:
            feature_end = get_feature_end_pos(feature)
            dist = position - feature_end
            if 1 <= dist <= max_downstream_dist_from_feature:
                feature_infos.append({
                    'feature_index': i,
                    'dist_from_feature': dist
                })
        else:
            assert feature.strand == -1
            feature_start = get_feature_start_pos(feature)
            dist = feature_start - position
            if 1 <= dist <= max_downstream_dist_from_feature:
                feature_infos.append({
                    'feature_index': i,
                    'dist_from_feature': dist
                })
    return feature_infos

def find_features_for_which_position_is_upstream(features, position, max_upstream_dist_from_feature):
    feature_infos = []
    for i, feature in enumerate(features):
        if feature.strand == 1:
            feature_start = get_feature_start_pos(feature)
            dist = feature_start - position
            if 1 <= dist <= max_upstream_dist_from_feature:
                feature_infos.append({
                    'feature_index': i,
                    'dist_from_feature': dist
                })
        else:
            assert feature.strand == -1
            feature_end = get_feature_end_pos(feature)
            dist = position - feature_end
            if 1 <= dist <= max_upstream_dist_from_feature:
                feature_infos.append({
                    'feature_index': i,
                    'dist_from_feature': dist
                })
    return feature_infos

def find_features_that_contain_position(features, position):
    containing_feature_infos = []
    for i, feature in enumerate(features):
        start_pos = get_feature_start_pos(feature)
        end_pos = get_feature_end_pos(feature)
        if start_pos <= position <= end_pos:
            normalized_position_in_feature = get_normalized_position_in_feature(feature, position)
            containing_feature_infos.append({
                'feature_index': i,
                'normalized_position_in_feature': normalized_position_in_feature,
            })

    return containing_feature_infos

def find_potential_operons_internal(cds_features, indices_of_recombinase_cds, max_dist_between_cds_in_operon):
    # max_dist_between_cds_in_operon == 0 means the cds must overlap (i.e., at least one bp is shared).
    assert all(x.type == 'CDS' for x in cds_features)
    potential_operon_infos = []
    curr_operon_indices = [0]
    for prev_feature_i, feature in enumerate(cds_features[1:]):
        prev_feature = cds_features[prev_feature_i]
        feature_start = get_feature_start_pos(feature)
        # feature_end = get_feature_end_pos(feature)
        prev_feature_start = get_feature_start_pos(prev_feature)
        prev_feature_end = get_feature_end_pos(prev_feature)
        if not feature_start > prev_feature_start:
            print(f'(feature_start, prev_feature_start): {(feature_start, prev_feature_start)}')
        assert feature_start > prev_feature_start

        feature_i = prev_feature_i + 1
        if (prev_feature.strand == feature.strand) and (prev_feature_end >= feature_start - max_dist_between_cds_in_operon):
            curr_operon_indices.append(feature_i)
        else:
            if len(curr_operon_indices) > 1:
                potential_operon_infos.append({
                    'feature_indices': curr_operon_indices,
                    'recombinase_feature_indices': sorted(set(curr_operon_indices) & indices_of_recombinase_cds),
                    'strand': prev_feature.strand,
                })
            curr_operon_indices = [feature_i]

    if len(curr_operon_indices) > 1:
        potential_operon_infos.append({
            'feature_indices': curr_operon_indices,
            'recombinase_feature_indices': sorted(set(curr_operon_indices) & indices_of_recombinase_cds),
            'strand': prev_feature.strand,
        })

    all_potential_operon_cds_feature_indices = set()
    for potential_operon_info in potential_operon_infos:
        all_potential_operon_cds_feature_indices.update(potential_operon_info['feature_indices'])

    potential_operons_info = {
        'potential_operon_infos': potential_operon_infos,
        'all_potential_operon_cds_feature_indices': all_potential_operon_cds_feature_indices,
    }
    return potential_operons_info

@generic_utils.execute_if_output_doesnt_exist_already
def cached_find_potential_operons(
        input_file_path_gb,
        indices_of_recombinase_cds,
        max_dist_between_cds_in_operon,
        output_file_path_potential_operons_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    cds_features = get_cds_seq_features(input_file_path_gb)
    potential_operons_info = find_potential_operons_internal(
        cds_features=cds_features,
        indices_of_recombinase_cds=indices_of_recombinase_cds,
        max_dist_between_cds_in_operon=max_dist_between_cds_in_operon,
    )

    with open(output_file_path_potential_operons_info_pickle, 'wb') as f:
        pickle.dump(potential_operons_info, f, protocol=4)

def find_potential_operons(
        input_file_path_gb,
        indices_of_recombinase_cds,
        max_dist_between_cds_in_operon,
        output_file_path_potential_operons_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_find_potential_operons(
        input_file_path_gb=input_file_path_gb,
        indices_of_recombinase_cds=indices_of_recombinase_cds,
        max_dist_between_cds_in_operon=max_dist_between_cds_in_operon,
        output_file_path_potential_operons_info_pickle=output_file_path_potential_operons_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )


def find_other_cds_in_operon(feature_index, potential_operon_infos):
    feature_infos = []
    for potential_operon_info in potential_operon_infos:
        if feature_index in potential_operon_info['feature_indices']:
            strand = potential_operon_info['strand']
            assert strand in (1, -1)
            is_plus_strand = strand == 1
            for other_feature_index in potential_operon_info['feature_indices']:
                if other_feature_index != feature_index:
                    if is_plus_strand ^ (other_feature_index > feature_index):
                        relative_position_in_operon = 'upstream'
                    else:
                        relative_position_in_operon = 'downstream'
                    feature_infos.append({
                        'feature_index': other_feature_index,
                        'relative_position_in_operon': relative_position_in_operon,
                    })
            return feature_infos
    assert False


def find_affected_cds_features_and_group_by_type(
        cds_features, effect_position,
        max_downstream_dist_from_feature, max_upstream_dist_from_feature, max_dist_between_cds_in_operon,
):
    assert all(x.type == 'CDS' for x in cds_features)
    features_for_which_position_is_upstream_infos = find_features_for_which_position_is_upstream(
        cds_features, effect_position, max_upstream_dist_from_feature)
    features_for_which_position_is_downstream_infos = find_features_for_which_position_is_downstream(
        cds_features, effect_position, max_downstream_dist_from_feature)
    containing_feature_infos = find_features_that_contain_position(cds_features, effect_position)
    potential_operon_infos = find_potential_operons(cds_features, max_dist_between_cds_in_operon)

    all_potential_operon_cds_feature_indices = set()
    for potential_operon_info in potential_operon_infos:
        all_potential_operon_cds_feature_indices.update(potential_operon_info['feature_indices'])

    features_for_which_position_is_upstream_to_other_feature_in_operon_infos = []
    for feature_info in features_for_which_position_is_upstream_infos:
        feature_index = feature_info['feature_index']
        if feature_index in all_potential_operon_cds_feature_indices:
            features_for_which_position_is_upstream_to_other_feature_in_operon_infos.extend(
                find_other_cds_in_operon(feature_index, potential_operon_infos))

    features_for_which_position_is_downstream_to_other_feature_in_operon_infos = []
    for feature_info in features_for_which_position_is_downstream_infos:
        feature_index = feature_info['feature_index']
        if feature_index in all_potential_operon_cds_feature_indices:
            features_for_which_position_is_downstream_to_other_feature_in_operon_infos.extend(
                find_other_cds_in_operon(feature_index, potential_operon_infos))

    features_for_which_position_is_in_other_feature_in_operon_infos = []
    for feature_info in containing_feature_infos:
        feature_index = feature_info['feature_index']
        if feature_index in all_potential_operon_cds_feature_indices:
            features_for_which_position_is_in_other_feature_in_operon_infos.extend(
                find_other_cds_in_operon(feature_index, potential_operon_infos))

    effect_type_to_affected_cds_feature_infos = {
        'upstream_to_CDS': features_for_which_position_is_upstream_infos,
        'downstream_to_CDS': features_for_which_position_is_downstream_infos,
        'in_CDS': containing_feature_infos,
        'upstream_to_other_CDS_in_operon': features_for_which_position_is_upstream_to_other_feature_in_operon_infos,
        'downstream_to_other_CDS_in_operon': features_for_which_position_is_downstream_to_other_feature_in_operon_infos,
        'in_other_CDS_in_operon': features_for_which_position_is_in_other_feature_in_operon_infos,
    }
    all_affected_cds_feature_indices = set()
    for feature_infos in effect_type_to_affected_cds_feature_infos.values():
        all_affected_cds_feature_indices.update(x['feature_index'] for x in feature_infos)
    # all_affected_cds_features = [cds_features[i] for i in sorted(all_affected_cds_feature_indices)]

    return effect_type_to_affected_cds_feature_infos, all_affected_cds_feature_indices

def get_fraction_of_region_that_features_cover(features, region):
    feature_intervals = {(get_feature_start_pos(x), get_feature_end_pos(x)) for x in features}
    merged_feature_intervals = generic_utils.get_merged_intervals(feature_intervals)
    intersections_of_features_with_region = generic_utils.get_intersections_of_intervals_with_interval(
        merged_feature_intervals, region)
    fraction = sum(x[1] - x[0] + 1 for x in intersections_of_features_with_region) / (region[1] - region[0] + 1)
    assert 0 <= fraction <= 1
    return fraction

def get_num_of_features_overlapping_region(features, region):
    feature_intervals = {(get_feature_start_pos(x), get_feature_end_pos(x)) for x in features}
    intersections_of_features_with_region = generic_utils.get_intersections_of_intervals_with_interval(feature_intervals, region)
    return len(intersections_of_features_with_region)

def get_indices_of_features_overlapping_region(features, region):
    assert generic_utils.is_interval(region)
    region_start, region_end = region
    indices_of_features_overlapping_region = []
    for i, feature in enumerate(features):
        feature_start = get_feature_start_pos(feature)
        feature_end = get_feature_end_pos(feature)
        if (region_start <= feature_start <= region_end) or (region_start <= feature_end <= region_end) or (feature_start <= region_start <= feature_end):
            indices_of_features_overlapping_region.append(i)
    return indices_of_features_overlapping_region

def get_position_as_num_of_bases_downstream_to_start_codon_start(cds_feature, position):
    assert cds_feature.type == 'CDS'
    start_codon_start_pos = get_most_upstream_position_in_cds(cds_feature)
    assert abs(cds_feature.location.strand) == 1
    if cds_feature.location.strand == 1:
        return position - start_codon_start_pos
    else:
        return start_codon_start_pos - position

def get_position_as_num_of_bases_upstream_to_stop_codon_end(cds_feature, position):
    assert cds_feature.type == 'CDS'
    stop_codon_end_pos = get_most_downstream_position_in_cds(cds_feature)
    assert abs(cds_feature.location.strand) == 1
    if cds_feature.location.strand == 1:
        return stop_codon_end_pos - position
    else:
        return position - stop_codon_end_pos


def get_closest_feature(features, position):
    return min(features, key=lambda x: get_feature_dist_from_position(x, position))

def get_normalized_position_in_feature(feature, position):
    # in curr implementation, normalized position is in [0,1)
    # i.e., lowest value is 0, and highest value is (feature_len - 1) / feature_len
    start_pos = get_feature_start_pos(feature)
    end_pos = get_feature_end_pos(feature)
    feature_len = get_feature_len(feature)
    strand = feature.strand
    if strand == 1:
            dist_from_feature_start = position - start_pos
    else:
        assert strand == -1
        dist_from_feature_start = end_pos - position
    return dist_from_feature_start / feature_len

def get_normalized_position_in_feature_or_diff_from_closest_edge(feature, position):
    start_pos = get_feature_start_pos(feature)
    end_pos = get_feature_end_pos(feature)
    strand = feature.strand
    if start_pos <= position <= end_pos:
        return get_normalized_position_in_feature(feature, position)
    else:
        if strand == 1:
            if position < start_pos:
                return position - start_pos
            else:
                assert position > end_pos
                return position - end_pos
        else:
            assert strand == -1
            if position > end_pos:
                return end_pos - position
            else:
                assert position < start_pos
                return start_pos - position
    assert False

def get_length_of_overlap_between_feature_and_region(feature, region):
    feature_start_pos = get_feature_start_pos(feature)
    feature_end_pos = get_feature_end_pos(feature)
    feature_interval = (feature_start_pos, feature_end_pos)
    overlap_interval = generic_utils.get_intersection_of_2_intervals(feature_interval, region)
    if overlap_interval is None:
        return 0
    return overlap_interval[1] - overlap_interval[0] + 1

def get_index_of_feature_with_longest_overlap(features, region, random_seed=0):
    feature_index_to_overlap_len = {i: get_length_of_overlap_between_feature_and_region(features[i], region) for i in range(len(features))}
    max_overlap_len = max(feature_index_to_overlap_len.values())
    max_overlap_feature_indices = [i for i, overlap_len in feature_index_to_overlap_len.items() if overlap_len == max_overlap_len]
    random.seed(random_seed)
    feature_with_longest_overlap_i = random.choice(max_overlap_feature_indices)
    # feature_with_longest_overlap_i = next(iter(max_overlap_feature_indices))
    return feature_with_longest_overlap_i, feature_index_to_overlap_len[feature_with_longest_overlap_i], len(max_overlap_feature_indices)

def find_evidence_for_indel_splitting_orig_cds(
        nuccore_fasta_file_path,
        region_potentially_on_other_side_of_indel1,
        region_potentially_on_other_side_of_indel2,
        evidence_alignment_for_indel_splitting_orig_cds_seed_len,
        blast_evidence_for_indel_splitting_orig_cds_info_pickle_file_path,
):
    blast_for_evidence_for_indel_splitting_orig_cds(
        input_file_path_nuccore_fasta=nuccore_fasta_file_path,
        region_potentially_on_other_side_of_indel1=region_potentially_on_other_side_of_indel1,
        region_potentially_on_other_side_of_indel2=region_potentially_on_other_side_of_indel2,
        evidence_alignment_for_indel_splitting_orig_cds_seed_len=evidence_alignment_for_indel_splitting_orig_cds_seed_len,
        output_file_path_blast_evidence_for_indel_splitting_orig_cds_info_pickle=blast_evidence_for_indel_splitting_orig_cds_info_pickle_file_path,
    )
    with open(blast_evidence_for_indel_splitting_orig_cds_info_pickle_file_path, 'rb') as f:
        blast_evidence_for_indel_splitting_orig_cds_info = pickle.load(f)

    if blast_evidence_for_indel_splitting_orig_cds_info['did_not_blast_due_to_bases_other_than_ACGT']:
        assert blast_evidence_for_indel_splitting_orig_cds_info['min_evalue_alignment_info'] is None
        blast_evidence_for_indel_splitting_orig_cds_info['min_evalue_alignment_info'] = {'evalue': -np.inf}
    elif blast_evidence_for_indel_splitting_orig_cds_info['blast_found_no_alignments']:
        assert blast_evidence_for_indel_splitting_orig_cds_info['min_evalue_alignment_info'] is None
        blast_evidence_for_indel_splitting_orig_cds_info['min_evalue_alignment_info'] = {'evalue': np.inf}

    evidence_for_indel_splitting_orig_cds_info = {
        'blast_evidence_for_indel_splitting_orig_cds_info': blast_evidence_for_indel_splitting_orig_cds_info,
        'blast_evidence_for_indel_splitting_orig_cds_info_pickle_file_path': blast_evidence_for_indel_splitting_orig_cds_info_pickle_file_path,
    }
    return evidence_for_indel_splitting_orig_cds_info

@generic_utils.execute_if_output_doesnt_exist_already
def cached_blast_for_evidence_for_indel_splitting_orig_cds(
        input_file_path_nuccore_fasta,
        region_potentially_on_other_side_of_indel1,
        region_potentially_on_other_side_of_indel2,
        evidence_alignment_for_indel_splitting_orig_cds_seed_len,
        output_file_path_blast_evidence_for_indel_splitting_orig_cds_info_pickle,
):
    nuccore_seq = bio_utils.get_chrom_seq_from_single_chrom_fasta_file(input_file_path_nuccore_fasta)
    region1_str = bio_utils.seq_record_to_str(bio_utils.get_region_in_chrom_seq(nuccore_seq, *region_potentially_on_other_side_of_indel1))
    region2_str = bio_utils.seq_record_to_str(bio_utils.get_region_in_chrom_seq(nuccore_seq, *region_potentially_on_other_side_of_indel2))
    blast_found_no_alignments = None
    # print(f'region1_str: {region1_str}')
    # print(f'region2_str: {region2_str}')

    did_not_blast_due_to_bases_other_than_ACGT = not ((set(region1_str) <= bio_utils.DNA_BASES_SET) and (set(region2_str) <= bio_utils.DNA_BASES_SET))

    if did_not_blast_due_to_bases_other_than_ACGT:
        min_evalue_alignment_info = None
    else:
        alignments_df = blast_interface_and_utils.blast_two_dna_seqs_as_strs(
            region1_str,
            region2_str,
            perform_gapped_alignment=True,
            seed_len=evidence_alignment_for_indel_splitting_orig_cds_seed_len,
            query_strand_to_search='minus',
            verbose=False,
        )
        if alignments_df.empty:
            min_evalue_alignment_info = None
            blast_found_no_alignments = True
        else:
            min_evalue_alignment_info = alignments_df.loc[alignments_df['evalue'].idxmin(), :].to_dict()
            blast_found_no_alignments = False

    blast_evidence_for_indel_splitting_orig_cds_info = {
        'min_evalue_alignment_info': min_evalue_alignment_info,
        'did_not_blast_due_to_bases_other_than_ACGT': did_not_blast_due_to_bases_other_than_ACGT,
        'blast_found_no_alignments': blast_found_no_alignments,
    }

    with open(output_file_path_blast_evidence_for_indel_splitting_orig_cds_info_pickle, 'wb') as f:
        pickle.dump(blast_evidence_for_indel_splitting_orig_cds_info, f, protocol=4)

def blast_for_evidence_for_indel_splitting_orig_cds(
        input_file_path_nuccore_fasta,
        region_potentially_on_other_side_of_indel1,
        region_potentially_on_other_side_of_indel2,
        evidence_alignment_for_indel_splitting_orig_cds_seed_len,
        output_file_path_blast_evidence_for_indel_splitting_orig_cds_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_blast_for_evidence_for_indel_splitting_orig_cds(
        input_file_path_nuccore_fasta=input_file_path_nuccore_fasta,
        region_potentially_on_other_side_of_indel1=region_potentially_on_other_side_of_indel1,
        region_potentially_on_other_side_of_indel2=region_potentially_on_other_side_of_indel2,
        evidence_alignment_for_indel_splitting_orig_cds_seed_len=evidence_alignment_for_indel_splitting_orig_cds_seed_len,
        output_file_path_blast_evidence_for_indel_splitting_orig_cds_info_pickle=output_file_path_blast_evidence_for_indel_splitting_orig_cds_info_pickle,
    )

def get_features_that_contain_the_location(position_on_chromosome, seq_features):
    # could be much faster if i used binary search. but i didn't have performance issues yet, and also didn't find a respectable source claiming that the features are guaranteed to be sorted according to location.
    features_found = []
    for feature in seq_features:
        zero_based_position_on_chromosome = position_on_chromosome - 1
        raise NotImplementedError() # seems to me like this is wrong. should instead be feature.location.start <= zero_based_position_on_chromosome < get_feature_end_pos(feature)

        if feature.location.start <= zero_based_position_on_chromosome <= get_feature_end_pos(feature):
            features_found.append(feature)
    return features_found


def write_potentially_interesting_seq_features_to_csv(features_df, csv_file_path):
    if features_df.empty:
        features_df = pd.DataFrame([], columns=ALL_POTENTIALLY_INTERESTING_SEQ_FEATURE_COLUMNS)
    assert set(features_df) == set(ALL_POTENTIALLY_INTERESTING_SEQ_FEATURE_COLUMNS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        features_df.to_csv(csv_file_path, sep='\t', index=False)


def read_potentially_interesting_seq_features_from_csv(csv_file_path):
    features_df = pd.read_csv(csv_file_path, sep='\t', dtype=SEQ_FEATURE_COLUMN_NAME_TO_DTYPE)
    assert set(features_df) == set(ALL_POTENTIALLY_INTERESTING_SEQ_FEATURE_COLUMNS)
    return features_df



def is_final_codon_is_not_a_stop_codon_err_str(err_str):
    return 'Final codon ' in err_str and ' is not a stop codon' in err_str


def is_first_codon_is_not_a_start_codon_err_str(err_str):
    return 'First codon ' in err_str and ' is not a start codon' in err_str


def is_extra_in_frame_stop_codon_found_err_str(err_str):
    return 'Extra in frame stop codon found.' in err_str


def get_mutation_info_using_annotations(position_on_chromosome, seq_features, ref_genome_chrom_seq, base1, base2):
    relevant_features = get_features_that_contain_the_location(position_on_chromosome, seq_features)
    if not relevant_features:
        return None, None, None, None, None
    features_types = {f.type for f in relevant_features}
    if features_types == {'gene', 'CDS'}:
        gene_features = [f for f in relevant_features if f.type == 'gene']
        cds_features = [f for f in relevant_features if f.type == 'CDS']
        assert len(gene_features) == len(cds_features)
        annotation_types = []
        annotation_locus_tags = []
        annotation_names = []
        annotation_additional_infos = []
        annotation_strands = []
        annotation_start_positions = []
        annotation_end_positions = []
        mutation_position_in_annotation_in_forward_strand_list = []
        mutation_types = []
        for gene_feature, cds_feature in zip(gene_features, cds_features):
            assert gene_feature.location == cds_feature.location
            annotation_types.append('gene')
            annotation_names.append(gene_feature.qualifiers['gene'][0] if 'gene' in gene_feature.qualifiers else None)
            # annotation_additional_infos.append(cds_feature.qualifiers['product'][0] if 'product' in cds_feature.qualifiers else None)
            annotation_locus_tags.append(cds_feature.qualifiers['locus_tag'][0])
            annotation_additional_infos.append(cds_feature.qualifiers['product'][0])
            strand = cds_feature.location.strand
            annotation_strands.append(strand)

            raise NotImplementedError() # seems to me like this is wrong, because of the comment above.
            annotation_start_pos_on_chrom = cds_feature.location.start
            annotation_start_positions.append(annotation_start_pos_on_chrom)
            annotation_end_positions.append(get_feature_end_pos(cds_feature))
            mutation_position_in_annotation_in_forward_strand_list.append(position_on_chromosome - annotation_start_pos_on_chrom + 1)

            if strand in (-1, 1):
                # https://biopython.org/DIST/docs/api/Bio.Seq-module.html
                translated_gene_in_reference = cds_feature.qualifiers['translation'][0]
                translated_gene_with_bases = []
                is_first_codon_not_start_codon_with_bases = [False, False]
                cds_translation_err_strs = [None, None]
                for i, base in enumerate((base1, base2)):
                    # strand_1_of_coding_seq_with_base = (ref_genome_chrom_seq[annotation_start_pos_on_chrom - 1:position_on_chromosome - 1] +
                    #                                     SeqIO.SeqRecord(base) +
                    #                                     ref_genome_chrom_seq[position_on_chromosome:get_feature_end_pos(cds_feature)])
                    chrom_seq_with_base = ref_genome_chrom_seq[:position_on_chromosome - 1] + base + ref_genome_chrom_seq[position_on_chromosome:]
                    try:
                        translated_gene_with_base = cds_feature.translate(chrom_seq_with_base).seq
                    except Bio.Data.CodonTable.TranslationError as err:
                        err_str = str(err)
                        cds_translation_err_strs[i] = err_str
                        if not (is_final_codon_is_not_a_stop_codon_err_str(err_str) or
                                is_first_codon_is_not_a_start_codon_err_str(err_str) or
                                is_extra_in_frame_stop_codon_found_err_str(err_str)):
                            print(f'position_on_chromosome: {position_on_chromosome}')
                            print(f'cds_feature.location: {cds_feature.location}')
                            raise

                        if is_first_codon_is_not_a_start_codon_err_str(err_str):
                            is_first_codon_not_start_codon_with_bases[i] = True
                            translation_table_id = int(cds_feature.qualifiers['transl_table'][0])
                            all_start_codons_list = Bio.Data.CodonTable.generic_by_id[translation_table_id].start_codons
                            all_stop_codons_list = Bio.Data.CodonTable.generic_by_id[translation_table_id].stop_codons

                            coding_seq_if_on_strand1 = chrom_seq_with_base[annotation_start_pos_on_chrom - 1:get_feature_end_pos(cds_feature)]
                            coding_seq = coding_seq_if_on_strand1 if strand == 1 else coding_seq_if_on_strand1.reverse_complement()
                            coding_seq_as_str = str(coding_seq.seq)
                            # this assertion is for simplicity. it means that if we don't find a start codon with both bases, then the mutation must be synonymous, as in both cases
                            # the mutation doesn't affect the product protein, which is probably completely lost, i guess, as the start codon is too far from the promoter. i think.
                            assert len(coding_seq_as_str) >= 5
                            for start_codon_i in range(len(coding_seq_as_str) - 2):
                                if coding_seq_as_str[start_codon_i:start_codon_i + CODON_LEN] in all_start_codons_list:
                                    for stop_codon_i in range(start_codon_i + CODON_LEN, len(coding_seq_as_str), CODON_LEN):
                                        if coding_seq_as_str[stop_codon_i:stop_codon_i + CODON_LEN] in all_stop_codons_list:
                                            translated_gene_with_base = coding_seq[start_codon_i:stop_codon_i + CODON_LEN].translate(table=translation_table_id, cds=True).seq
                                            break
                                    else:
                                        # stop_codon_i = None
                                        # ugly hack in order to use cds=True and thus not explicitly replace the first amino acid with M.
                                        coding_seq_from_start_codon = coding_seq[start_codon_i:]
                                        coding_seq_from_start_codon_len_mod_3 = len(coding_seq_from_start_codon) % 3
                                        if coding_seq_from_start_codon_len_mod_3 == 0:
                                            coding_seq_from_start_codon_trimmed_to_multiple_of_three = coding_seq_from_start_codon
                                        else:
                                            coding_seq_from_start_codon_trimmed_to_multiple_of_three = coding_seq_from_start_codon[:-coding_seq_from_start_codon_len_mod_3]
                                        translated_gene_with_base = (coding_seq_from_start_codon_trimmed_to_multiple_of_three + all_stop_codons_list[0]).translate(
                                            table=translation_table_id, cds=True).seq + '...'
                                    break
                            else:
                                translated_gene_with_base = '...'
                                start_codon_i = np.inf
                            translated_gene_with_base = (start_codon_i, translated_gene_with_base)


                        else:
                            translated_seq_as_str = str(cds_feature.translate(chrom_seq_with_base, cds=False).seq)
                            if is_final_codon_is_not_a_stop_codon_err_str(err_str):
                                assert not '*' in translated_seq_as_str
                                translated_gene_with_base = translated_seq_as_str + '...'
                            else:
                                assert is_extra_in_frame_stop_codon_found_err_str(err_str)
                                assert '*' in translated_seq_as_str
                                translated_gene_with_base = translated_seq_as_str.partition('*')[0]

                    translated_gene_with_bases.append(translated_gene_with_base)
                # print(translated_gene_with_bases)
                # print(translated_gene_in_reference)

                if not translated_gene_in_reference in translated_gene_with_bases:
                    print('Warning: Both the major allele and the second major allele are not like the reference, and substituting each of them results in a '
                          'nonsynonymous substitution (relative to the reference).')
                    print(f'position_on_chromosome: {position_on_chromosome}')
                    print(f'cds_feature.location: {cds_feature.location}')
                    print(f'translated gene for major allele: {translated_gene_with_bases[0]}')
                    print(f'translated gene for second major allele: {translated_gene_with_bases[1]}')
                    ### DEBUG ### todo: remove this exit()? or maybe it is better to keep it? not sure.
                    exit()

                if any(is_first_codon_not_start_codon_with_bases):
                    if all(is_first_codon_not_start_codon_with_bases):
                        if translated_gene_with_bases[0] == translated_gene_with_bases[1]:
                            mutation_type = 'synonymous'
                        else:
                            major_allele_start_codon_i = translated_gene_with_bases[0][0]
                            second_major_allele_start_codon_i = translated_gene_with_bases[1][0]
                            if major_allele_start_codon_i != second_major_allele_start_codon_i:
                                mutation_type = 'different_start_codon'
                            else:
                                assert major_allele_start_codon_i == second_major_allele_start_codon_i
                                translated_gene_with_major_allele = translated_gene_with_bases[0][1]
                                translated_gene_with_second_major_allele = translated_gene_with_bases[1][1]
                                assert translated_gene_with_major_allele != translated_gene_with_second_major_allele
                                print('Warning: This is quite suspicious. It is a case in which both the major and the second major allele are such that the start codon in the '
                                      'reference genome is lost, the new start codon is at the same position in both cases, but still the translation is not the same, i.e., the'
                                      'start codon is translated to different amino acid in each case (which is impossible with the current bacteria codon table (11).')
                                mutation_type = 'missense'
                    elif is_first_codon_not_start_codon_with_bases[0]:
                        mutation_type = 'start_gain'
                    else:
                        assert is_first_codon_not_start_codon_with_bases[1]
                        mutation_type = 'start_loss'
                else:
                    translated_gene_with_bases_lens = [len(x) for x in translated_gene_with_bases]
                    if translated_gene_with_bases_lens[0] == translated_gene_with_bases_lens[1]:
                        # note that it is possible to get here when both major and second major are different than the reference. They could even both be nonstop, but it would still be
                        # fine, because then they either differ in the amino acid that replaced the stop codon, or they don't. all of the following codons are the same, so we are good.
                        if translated_gene_with_bases[0] == translated_gene_with_bases[1]:
                            mutation_type = 'synonymous'
                        else:
                            hamming_dist = generic_utils.get_hamming_dist_between_same_len_strs(translated_gene_with_bases[0], translated_gene_with_bases[1])
                            assert hamming_dist == 1
                            mutation_type = 'missense'
                    elif translated_gene_with_bases_lens[0] > translated_gene_with_bases_lens[1]:
                        if cds_translation_err_strs[0] is not None and is_final_codon_is_not_a_stop_codon_err_str(cds_translation_err_strs[0]):
                            mutation_type = 'non_stop'
                        else:
                            assert is_extra_in_frame_stop_codon_found_err_str(cds_translation_err_strs[1])
                            mutation_type = 'stop_gain'
                    else:
                        assert translated_gene_with_bases_lens[0] < translated_gene_with_bases_lens[1]
                        if cds_translation_err_strs[1] is not None and is_final_codon_is_not_a_stop_codon_err_str(cds_translation_err_strs[1]):
                            mutation_type = 'non_stop'
                        else:
                            assert is_extra_in_frame_stop_codon_found_err_str(cds_translation_err_strs[0])
                            mutation_type = 'stop_gain'
                mutation_types.append(mutation_type)
                # if mutation_type not in ['missense', 'synonymous', 'stop_gain']:
                #     print()
                #     print(f'mutation_type: {mutation_type}')
                #     print(f'position_on_chromosome: {position_on_chromosome}')
                #     print(f'cds_feature.location: {cds_feature.location}')
                #     print(f'translated gene for major allele: {translated_gene_with_bases[0]}')
                #     print(f'translated gene for second major allele: {translated_gene_with_bases[1]}')
            else:
                mutation_types.append(None)
        annotation_type_value_for_df = ','.join(str(x) for x in annotation_types)
        annotation_locus_tag_value_for_df = ','.join(str(x) for x in annotation_locus_tags)
        annotation_name_value_for_df = ','.join(str(x) for x in annotation_names)
        annotation_additional_info_value_for_df = ','.join(str(x) for x in annotation_additional_infos)
        annotation_strand_value_for_df = ','.join(str(x) for x in annotation_strands)
        annotation_start_position_value_for_df = ','.join(str(x) for x in annotation_start_positions)
        annotation_end_position_value_for_df = ','.join(str(x) for x in annotation_end_positions)
        mutation_position_in_annotation_in_forward_strand_value_for_df = ','.join(str(x) for x in mutation_position_in_annotation_in_forward_strand_list)
        mutation_type_value_for_df = ','.join(str(x) for x in mutation_types)

    else:
        print(f'position_on_chromosome: {position_on_chromosome}')
        print('features:')
        for f in relevant_features:
            print(f)

        raise NotImplementedError
    return (
        annotation_type_value_for_df,
        annotation_locus_tag_value_for_df,
        annotation_name_value_for_df,
        annotation_additional_info_value_for_df,
        annotation_strand_value_for_df,
        annotation_start_position_value_for_df,
        annotation_end_position_value_for_df,
        mutation_position_in_annotation_in_forward_strand_value_for_df,
        mutation_type_value_for_df
    )

def get_features_in_and_near_region(features_df, start_position, end_position, chrom_len, near_distance, is_chrom_circular):
    start_position_minus_margins = start_position - near_distance
    end_position_plus_margins = end_position + near_distance
    if is_chrom_circular:
        if end_position_plus_margins > chrom_len:
            features_after_ori_df = features_df.copy()
            features_after_ori_df.loc[:, 'start_position_ignoring_fuzziness'] = features_after_ori_df['start_position_ignoring_fuzziness'] + chrom_len
            features_after_ori_df.loc[:, 'end_position_ignoring_fuzziness'] = features_after_ori_df['end_position_ignoring_fuzziness'] + chrom_len
            features_df = pd.concat([features_df, features_after_ori_df])
        if start_position_minus_margins < 1:
            features_before_ori_df = features_df.copy()
            features_before_ori_df.loc[:, 'start_position_ignoring_fuzziness'] = features_before_ori_df['start_position_ignoring_fuzziness'] - chrom_len
            features_before_ori_df.loc[:, 'end_position_ignoring_fuzziness'] = features_before_ori_df['end_position_ignoring_fuzziness'] - chrom_len
            features_df = pd.concat([features_df, features_before_ori_df])
    features_df = features_df[(
            ((features_df['start_position_ignoring_fuzziness'] >= start_position_minus_margins) & (features_df['start_position_ignoring_fuzziness'] <= end_position_plus_margins)) |
            ((features_df['end_position_ignoring_fuzziness'] >= start_position_minus_margins) & (features_df['end_position_ignoring_fuzziness'] <= end_position_plus_margins)) |
            ((features_df['start_position_ignoring_fuzziness'] <= start_position_minus_margins) & (features_df['end_position_ignoring_fuzziness'] >= end_position_plus_margins))
    )]
    # print(f'start_position_minus_margins: {start_position_minus_margins}')
    # print(f'end_position_plus_margins: {end_position_plus_margins}')
    # print(f'chrom_len: {chrom_len}')
    return features_df

def get_seq_features_in_or_near_regions(gb_record, regions, near_distance):
    features_in_or_near_regions = []
    for region in sorted(regions):
        region_start, region_end = region
        region_with_margins_start = region_start - near_distance
        region_with_margins_end = region_end + near_distance
        for feature in gb_record.features:
            feature_start_pos = get_feature_start_pos(feature)
            feature_end_pos = get_feature_end_pos(feature)
            if (
                ((feature_start_pos >= region_with_margins_start) & (feature_start_pos <= region_with_margins_end)) |
                ((feature_end_pos >= region_with_margins_start) & (feature_end_pos <= region_with_margins_end)) |
                ((feature_start_pos <= region_with_margins_start) & (feature_end_pos >= region_with_margins_end))
            ):
                features_in_or_near_regions.append(feature)

    return features_in_or_near_regions


def get_seq_features_without_redundant_gene_features(features):
    non_gene_feature_start_and_stop_positions_and_strands_and_locus_tags_and_gene_names = {
        (
            get_feature_start_pos(feature),
            get_feature_end_pos(feature),
            feature.location.strand,
            feature.qualifiers['locus_tag'][0] if 'locus_tag' in feature.qualifiers else None,
            feature.qualifiers['gene'][0] if 'gene' in feature.qualifiers else None,
        )
        for feature in features
        if feature.type != 'gene'
    }
    # print(f'non_gene_feature_start_and_stop_positions_and_strands_and_locus_tags_and_gene_names:\n{non_gene_feature_start_and_stop_positions_and_strands_and_locus_tags_and_gene_names}')
    filtered_features = [
        feature
        for feature in features
        if (
                (feature.type != 'gene') or
                (
                    get_feature_start_pos(feature),
                    get_feature_end_pos(feature),
                    feature.location.strand,
                    feature.qualifiers['locus_tag'][0] if 'locus_tag' in feature.qualifiers else None,
                    feature.qualifiers['gene'][0] if 'gene' in feature.qualifiers else None,
                ) not in non_gene_feature_start_and_stop_positions_and_strands_and_locus_tags_and_gene_names or
                not (set(feature.qualifiers) <= {'gene', 'locus_tag', 'old_locus_tag','db_xref', 'pseudo'})
        )
    ]
    # print(f'filtered_features:\n{filtered_features}')
    return filtered_features


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_column_word_counts_and_freqs_csv(
        input_file_path_seq_features_csv,
        column_name,
        output_file_path_column_word_counts_and_freqs_csv,
        output_file_path_total_word_count,
):
    features_df = read_potentially_interesting_seq_features_from_csv(input_file_path_seq_features_csv)
    distinct_column_values = features_df[['start_position_ignoring_fuzziness', 'end_position_ignoring_fuzziness', column_name]].dropna().drop_duplicates()[column_name]
    generic_utils.write_word_counts_and_freqs_to_csv_and_total_word_count_to_text_file(
        text_series=distinct_column_values,
        word_counts_and_freqs_csv_file_path=output_file_path_column_word_counts_and_freqs_csv,
        total_word_count_txt_file_path=output_file_path_total_word_count,
    )


def write_column_word_counts_and_freqs_csv(
        input_file_path_seq_features_csv,
        column_name,
        output_file_path_column_word_counts_and_freqs_csv,
        output_file_path_total_word_count,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_column_word_counts_and_freqs_csv(
        input_file_path_seq_features_csv=input_file_path_seq_features_csv,
        column_name=column_name,
        output_file_path_column_word_counts_and_freqs_csv=output_file_path_column_word_counts_and_freqs_csv,
        output_file_path_total_word_count=output_file_path_total_word_count,
)


TRANSPOSON_DESCRIPTION_REGEX = 'transposon|transposable|transposase'
def any_transposon_annotation_in_features(features_df):
    # Removed that because it turns out that at least one type of recombinase is of the same EC class: https://en.wikipedia.org/wiki/Cre_recombinase
    # if (features_df['ec_number_level3_class'] == TRANSPOSASE_EC_NUMBER_WHICH_IS_LEVEL_3).any():
    #     return True
    if features_df['mobile_element_type_type_internal_id'].notna().any():
        return True
    if features_df['product'].str.contains(TRANSPOSON_DESCRIPTION_REGEX, case=False, regex=True).any():
        return True
    if features_df['function'].str.contains(TRANSPOSON_DESCRIPTION_REGEX, case=False, regex=True).any():
        return True

    return False




# # https://en.wikipedia.org/wiki/Hin_recombinase - invertase. but https://en.wikipedia.org/wiki/Invertase is something else. problematic? or maybe this one doesn't exist in bacteria?
# # https://proteopedia.org/wiki/index.php/Resolvase
# # also had 'holliday' here previously, but it found (unsurprisingly, really) RuvA and RuvB in rhamnosus, so i removed it.
# # also had 'recombination' here previously, but it found various recombination-related stuff, so i removed it.
# INDICATIVE_RECOMBINASE_STRS = {'recombinase','integrase','invertase','resolvase','inversion'}
# INDICATIVE_TRANSPOSON_STRS = {'transpos'}
# # https://en.wikipedia.org/wiki/Lambda_phage#Protein_function_overview
# # https://proteopedia.org/wiki/index.php/Resolvase
# # https://www.uniprot.org/docs/similar.txt (this is big. download it by going to https://www.uniprot.org/docs/similar and using save link as on the "download" link.
# # there would be a lot of false positives, e.g., a recombinase gene in human whose name is identical to an unrelated gene in bacteria.
# RECOMBINASE_GENE_NAMES = {'Xer','XerC','XerD','XerCD','XerH','int','hin','HRec','gin','Cre','RecA','lamint','phiC31'}

DEFAULT_RECOMBINASE_PRODUCT_REGEX = r'recombinase|integrase|invertase|resolvase|inversion'
# DEFAULT_TRANSPOSON_PRODUCT_REGEX = r'insertion sequence|insertion element|transpos|TnsA|TnsB|TnpV|TniQ|TniA|TniB|IstB'
DEFAULT_TRANSPOSON_PRODUCT_REGEX = r'insertion sequence|insertion element|transpos'

DEFAULT_MEMBRANE_TRANSPORT_PRODUCT_REGEX = r'transporter|symporter|antiporter|porin|permease'

DEFAULT_HYPOTHETICAL_AND_UNKNOWN_FUNCTION_PRODUCT_REGEX = r'hypothetical protein|DUF\d+ domain-containing protein'

# DEFAULT_DNA_SPECIFICITY_PRODUCT_REGEX = r'spec'


def is_recombinase_feature(feature, recombinase_product_regex=DEFAULT_RECOMBINASE_PRODUCT_REGEX):
    product = get_product_qualifier(feature)
    return (
        (product is not None) and
        bool(re.search(recombinase_product_regex, product))
    )
    # return bool(
    #     (('product' in qualifiers) and generic_utils.does_any_str_in_strs1_contain_any_str_in_strs2(qualifiers['product'], INDICATIVE_RECOMBINASE_STRS))
    #     or
    #     (('function' in qualifiers) and generic_utils.does_any_str_in_strs1_contain_any_str_in_strs2(qualifiers['function'], INDICATIVE_RECOMBINASE_STRS))
    #     or
    #     (('gene' in qualifiers) and (set(qualifiers['gene']) & RECOMBINASE_GENE_NAMES))
    #     or
    #     (('gene_synonym' in qualifiers) and (set(qualifiers['gene_synonym']) & RECOMBINASE_GENE_NAMES))
    # )

def is_transposon_feature(feature, transposon_product_regex=DEFAULT_TRANSPOSON_PRODUCT_REGEX):
    product = get_product_qualifier(feature)
    return (
        (product is not None) and
        bool(re.search(transposon_product_regex, product))
    )
    # return (
    #     (('product' in qualifiers) and generic_utils.does_any_str_in_strs1_contain_any_str_in_strs2(qualifiers['product'], INDICATIVE_TRANSPOSON_STRS))
    #     or
    #     (('function' in qualifiers) and generic_utils.does_any_str_in_strs1_contain_any_str_in_strs2(qualifiers['function'], INDICATIVE_TRANSPOSON_STRS))
    # )

# def is_recombinase_and_not_transposon_feature(feature):
#     # why would you ever want that? to avoid false positives, i guess...
#     return is_recombinase_feature(feature) and (not is_transposon_feature(feature))

def product_series_to_is_transposon_series(product_series):
    # return (~(product_series.isna())) & product_series.str.contains(DEFAULT_TRANSPOSON_PRODUCT_REGEX, regex=True).astype(bool)
    return (~(product_series.isna())) & (product_series.str.contains(DEFAULT_TRANSPOSON_PRODUCT_REGEX, regex=True) == True)

def product_series_to_is_recombinase_series(product_series):
    # return (~(product_series.isna())) & product_series.str.contains(DEFAULT_RECOMBINASE_PRODUCT_REGEX, regex=True).astype(bool)
    return (~(product_series.isna())) & (product_series.str.contains(DEFAULT_RECOMBINASE_PRODUCT_REGEX, regex=True) == True)

def product_series_to_is_recombinase_and_not_transposon_series(product_series):
    return product_series_to_is_recombinase_series(product_series) & (~product_series_to_is_transposon_series(product_series))

def product_series_to_is_membrane_transport_product_series(product_series):
    # return (~(product_series.isna())) & product_series.str.contains(DEFAULT_TRANSPOSON_PRODUCT_REGEX, regex=True).astype(bool)
    return (~(product_series.isna())) & (product_series.str.contains(DEFAULT_MEMBRANE_TRANSPORT_PRODUCT_REGEX, regex=True) == True)

def product_series_to_is_hypothetical_or_unknown_function_product_series(product_series):
    # return (~(product_series.isna())) & product_series.str.contains(DEFAULT_TRANSPOSON_PRODUCT_REGEX, regex=True).astype(bool)
    return (~(product_series.isna())) & (product_series.str.contains(DEFAULT_HYPOTHETICAL_AND_UNKNOWN_FUNCTION_PRODUCT_REGEX, regex=True) == True)

def is_TonB_related_feature(feature):
    return 'TonB' in get_product_qualifier(feature)

@generic_utils.execute_if_output_doesnt_exist_already
def cached_find_indices_of_recombinase_cds(
        input_file_path_gb,
        recombinase_product_regex,
        output_file_path_indices_of_recombinase_cds_pickle,
):
    cds_features = get_cds_seq_features(input_file_path_gb)
    indices_of_recombinase_cds = {i for i,cds in enumerate(cds_features)
                                  if is_recombinase_feature(cds, recombinase_product_regex=recombinase_product_regex)}
    with open(output_file_path_indices_of_recombinase_cds_pickle, 'wb') as f:
        pickle.dump(indices_of_recombinase_cds, f, protocol=4)

def find_indices_of_recombinase_cds(
        input_file_path_gb,
        recombinase_product_regex,
        output_file_path_indices_of_recombinase_cds_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_find_indices_of_recombinase_cds(
        input_file_path_gb=input_file_path_gb,
        recombinase_product_regex=recombinase_product_regex,
        output_file_path_indices_of_recombinase_cds_pickle=output_file_path_indices_of_recombinase_cds_pickle,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_recombinase_cds_seq_features(
        input_file_path_gb,
        recombinase_product_regex,
        ignore_transposon_recombinases,
        margin_size_for_merging_recombinase_cds_seq_feature_regions,
        output_file_path_recombinase_seq_features_info_pickle,
        output_file_path_recombinase_seq_features_gb,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    gb_record = bio_utils.get_gb_record(input_file_path_gb)

    recombinase_features = []
    recombinase_cds_feature_regions_and_strands = []
    num_of_cds_seq_features = 0
    num_of_recombinase_cds_skipped_because_they_contain_ori = 0
    num_of_ignored_recombinase_transposon_cds_features = 0
    for feature in gb_record.features:
        if feature.type == 'CDS':
            num_of_cds_seq_features += 1
            if not is_recombinase_feature(feature, recombinase_product_regex=recombinase_product_regex):
                continue
            if ignore_transposon_recombinases:
                if is_transposon_feature(feature):
                    num_of_ignored_recombinase_transposon_cds_features += 1
                    continue

            cds_start_pos = get_feature_start_pos(feature)
            cds_end_pos = get_feature_end_pos(feature)
            strand = feature.location.strand
            if cds_start_pos > cds_end_pos:
                num_of_recombinase_cds_skipped_because_they_contain_ori += 1
            else:
                recombinase_features.append(feature)
                recombinase_cds_feature_regions_and_strands.append((cds_start_pos, cds_end_pos, strand))


    num_of_not_skipped_recombinase_features = len(recombinase_features)
    if num_of_cds_seq_features:
        not_skipped_recombinase_cds_seq_feature_proportion = num_of_not_skipped_recombinase_features / num_of_cds_seq_features
        assert 0 <= not_skipped_recombinase_cds_seq_feature_proportion <= 1
    else:
        not_skipped_recombinase_cds_seq_feature_proportion = np.nan

    recombinase_cds_feature_regions_with_margins = {
        (x[0] - margin_size_for_merging_recombinase_cds_seq_feature_regions,
         x[1] + margin_size_for_merging_recombinase_cds_seq_feature_regions)
        for x in recombinase_cds_feature_regions_and_strands
    }
    merged_recombinase_cds_seq_feature_regions_with_margins = generic_utils.get_merged_intervals(recombinase_cds_feature_regions_with_margins)
    merged_recombinase_cds_seq_feature_regions = {
        (x[0] + margin_size_for_merging_recombinase_cds_seq_feature_regions,
         x[1] - margin_size_for_merging_recombinase_cds_seq_feature_regions)
        for x in merged_recombinase_cds_seq_feature_regions_with_margins
    }

    recombinase_seq_features_gb_record = gb_record
    recombinase_seq_features_gb_record.seq = recombinase_seq_features_gb_record.seq[:0]
    recombinase_seq_features_gb_record.features = recombinase_features
    bio_utils.write_records_to_fasta_or_gb_file(recombinase_seq_features_gb_record, output_file_path_recombinase_seq_features_gb, 'gb')

    recombinase_feature_index_in_recombinase_gb_to_merged_recombinase_region = []
    merged_recombinase_region_to_recombinase_feature_indices = collections.defaultdict(set)

    for i, feature_region_and_strand in enumerate(recombinase_cds_feature_regions_and_strands):
        # print(f'\nfeature_region_and_strand: {feature_region_and_strand}')
        curr_merged_recombinase_region = None
        for merged_recombinase_region in merged_recombinase_cds_seq_feature_regions:
            # print(f'merged_recombinase_region: {merged_recombinase_region}')
            if feature_region_and_strand[0] >= merged_recombinase_region[0] and feature_region_and_strand[1] <= merged_recombinase_region[1]:
                recombinase_feature_index_in_recombinase_gb_to_merged_recombinase_region.append(merged_recombinase_region)
                merged_recombinase_region_to_recombinase_feature_indices[merged_recombinase_region].add(i)
                curr_merged_recombinase_region = merged_recombinase_region
                break
        assert curr_merged_recombinase_region

    merged_recombinase_region_to_recombinase_feature_indices = dict(merged_recombinase_region_to_recombinase_feature_indices)  # I don't want a defaultdict moving around.

    recombinase_seq_features_info = {
        'num_of_cds_seq_features': num_of_cds_seq_features,
        'num_of_recombinases_that_contain_ori': num_of_recombinase_cds_skipped_because_they_contain_ori,
        'num_of_not_skipped_recombinase_features': num_of_not_skipped_recombinase_features,
        'num_of_ignored_recombinase_transposon_cds_features': num_of_ignored_recombinase_transposon_cds_features,
        'recombinase_cds_feature_regions_and_strands': recombinase_cds_feature_regions_and_strands,
        'not_skipped_recombinase_cds_seq_feature_proportion': not_skipped_recombinase_cds_seq_feature_proportion,
        'margin_size_for_merging_recombinase_cds_seq_feature_regions': margin_size_for_merging_recombinase_cds_seq_feature_regions,
        'merged_recombinase_cds_seq_feature_regions': merged_recombinase_cds_seq_feature_regions,
        'recombinase_feature_index_in_recombinase_gb_to_merged_recombinase_region': recombinase_feature_index_in_recombinase_gb_to_merged_recombinase_region,
        'merged_recombinase_region_to_recombinase_feature_indices': merged_recombinase_region_to_recombinase_feature_indices,
        'recombinase_cds_seq_features_gb_file_path': output_file_path_recombinase_seq_features_gb,
    }

    with open(output_file_path_recombinase_seq_features_info_pickle, 'wb') as f:
        pickle.dump(recombinase_seq_features_info, f, protocol=4)

def write_recombinase_cds_seq_features(
        input_file_path_gb,
        recombinase_product_regex,
        ignore_transposon_recombinases,
        margin_size_for_merging_recombinase_cds_seq_feature_regions,
        output_file_path_recombinase_seq_features_info_pickle,
        output_file_path_recombinase_seq_features_gb,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_recombinase_cds_seq_features(
        input_file_path_gb=input_file_path_gb,
        recombinase_product_regex=recombinase_product_regex,
        ignore_transposon_recombinases=ignore_transposon_recombinases,
        margin_size_for_merging_recombinase_cds_seq_feature_regions=margin_size_for_merging_recombinase_cds_seq_feature_regions,
        output_file_path_recombinase_seq_features_info_pickle=output_file_path_recombinase_seq_features_info_pickle,
        output_file_path_recombinase_seq_features_gb=output_file_path_recombinase_seq_features_gb,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=11,
    )

def is_hypothetical_protein_cds(cds_product):
    return cds_product == 'hypothetical protein'

STR_BEFORE_PROTEIN_ID_USED_IN_INFERENCE = 'similar to AA sequence:RefSeq:'
STR_BEFORE_PROTEIN_ID_USED_IN_INFERENCE_LEN = len(STR_BEFORE_PROTEIN_ID_USED_IN_INFERENCE)
def extract_any_homolog_protein_id_from_inferences(cds_inferences):
    for inference in cds_inferences:
        start_i_of_str_before_protein_id_used_in_inference = inference.find(STR_BEFORE_PROTEIN_ID_USED_IN_INFERENCE)
        if start_i_of_str_before_protein_id_used_in_inference != -1:
            homolog_protein_id = inference[start_i_of_str_before_protein_id_used_in_inference +
                                           STR_BEFORE_PROTEIN_ID_USED_IN_INFERENCE_LEN:]
            # in rare cases, we need extra processing. e.g., in https://www.ncbi.nlm.nih.gov/nuccore/NZ_BAFQ01000167.1
            # we now have homolog_protein_id == 'WP_005439383.1,RefSeq:WP_010595403.1, RefSeq:WP_018836696.1'
            homolog_protein_id = homolog_protein_id.partition(',')[0]
            # print(f'homolog_protein_id: {homolog_protein_id}')
            assert ' ' not in homolog_protein_id
            return homolog_protein_id
    return None

@generic_utils.execute_if_output_doesnt_exist_already
def cached_extract_cds_products_from_gb(
        input_file_path_gb,
        output_file_path_products,
        output_file_path_products_info_pickle,
        ignore_recombinases,
        ignore_transposon_recombinases_when_ignoring_recombinases,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    gb_record = bio_utils.get_gb_record(input_file_path_gb)

    products_info = extract_cds_products(
        features=gb_record.features,
        products_file_path=output_file_path_products,
        ignore_recombinases=ignore_recombinases,
        ignore_transposon_recombinases_when_ignoring_recombinases=ignore_transposon_recombinases_when_ignoring_recombinases,
    )

    with open(output_file_path_products_info_pickle, 'wb') as f:
        pickle.dump(products_info, f, protocol=4)

def extract_cds_products_from_gb(
        input_file_path_gb,
        output_file_path_products,
        output_file_path_products_info_pickle,
        ignore_recombinases,
        ignore_transposon_recombinases_when_ignoring_recombinases,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_extract_cds_products_from_gb(
        input_file_path_gb=input_file_path_gb,
        output_file_path_products=output_file_path_products,
        output_file_path_products_info_pickle=output_file_path_products_info_pickle,
        ignore_recombinases=ignore_recombinases,
        ignore_transposon_recombinases_when_ignoring_recombinases=ignore_transposon_recombinases_when_ignoring_recombinases,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )