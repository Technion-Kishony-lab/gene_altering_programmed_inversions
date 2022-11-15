import os.path
import pathlib

import numpy as np

from generic import generic_utils


def get_ir_pair_str_repr(ir_pair):
    return f'({ir_pair[0]}-{ir_pair[1]},{ir_pair[2]}-{ir_pair[3]})'

def get_variant_ir_pair_str_repr(variant_ir_pairs, replace_last_comma_with_and=False):
    ir_pair_str_reprs = [get_ir_pair_str_repr(x) for x in sorted(variant_ir_pairs)]
    if replace_last_comma_with_and:
        return ', '.join(ir_pair_str_reprs[:-1]) + ' and ' + ir_pair_str_reprs[-1]
    return ', '.join(ir_pair_str_reprs)

def get_mauve_failed_explanation(variant_ir_pairs):
    variant_ir_pair_str_repr = get_variant_ir_pair_str_repr(variant_ir_pairs, replace_last_comma_with_and=True)
    return (f'blastn alignments indicate the read matches a variant such that the inverted repeats {variant_ir_pair_str_repr} '
            f'(presumably) promote programmed inversions allowing switching from the reference variant to this variant. '
            f'However, progressiveMauve alignments seems to be inaccurate, maybe due to lower sensitivity of progressiveMauve '
            f'(relative to blastn)')

TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION = get_mauve_failed_explanation({(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)})

READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION = ('read seems to not match any variant that can be reached from '
                                                                                'the reference variant by programmed inversions')


POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS = {'TonB-dependent receptor', '___SusC/RagA'}
ALL_POSSIBLE_TONB_DEPENDENT_RECEPTOR_PRODUCTS = POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS | {'___SusD/RagB'}

POSSIBLE_DUF1016_PRODUCTS = {'___DUF1016', 'YhcG family protein', 'PDDEXK nuclease domain-containing protein'}


POSSIBLE_HSDS_PRODUCTS = {
    '___restriction endonuclease subunit S',

    # see https://www.ncbi.nlm.nih.gov/protein/OUP32334.1 and https://www.ncbi.nlm.nih.gov/protein/WP_087249171.1
    # (from nuccore NZ_NFKD01000019.1 / NFKD01000019.1)
    # 'type I restriction endonuclease', # even though I have an excuse (see comment above), this seems too wrong, so I comment it out.
}

POSSIBLE_TRUNCATED_HSDS_PRODUCTS = POSSIBLE_HSDS_PRODUCTS | {
    'hypothetical protein',
}

POSSIBLE_HSDR_PRODUCTS = {
    '___DEAD/DEAH box helicase',
    '___restriction endonuclease subunit R',

    # see https://www.ncbi.nlm.nih.gov/protein/WP_013483871.1 (from nuccore NC_014825.1)
    'type I restriction-modification system endonuclease',
}

POSSIBLE_MTASE_PRODUCTS = {
    'N-6 DNA methylase',
    'class I SAM-dependent DNA methyltransferase',
    'SAM-dependent DNA methyltransferase',
    'Eco57I restriction-modification methylase domain-containing protein',
    '___restriction-modification system subunit M',
    'DNA methylase',
    'DNA adenine methylase',
    'DNA modification methylase',
    'DNA (cytosine-5-)-methyltransferase',

    # blastp https://www.ncbi.nlm.nih.gov/protein/WP_128833139.1 to https://www.ncbi.nlm.nih.gov/protein/WP_155765198.1, and you will get 87% alignment coverage.
    'type II restriction endonuclease',

    # blastp https://www.ncbi.nlm.nih.gov/protein/WP_157851054.1 to https://www.ncbi.nlm.nih.gov/protein/WP_037182160.1, and you will get 99% alignment coverage.
    # 'restriction endonuclease', # even though I have an excuse (see comment above), this seems too wrong, so I comment it out.

    # this was the annotation of https://www.ncbi.nlm.nih.gov/protein/WP_217490363.1 sometime in the past (during which i downloaded the genbank file). you can see that by
    # blasting it to nr. anyway, blastp https://www.ncbi.nlm.nih.gov/protein/WP_217490363.1 to https://www.ncbi.nlm.nih.gov/protein/MBQ6704743.1,
    # and you get 99% alignment coverage.
    'type IIS restriction enzyme R and M protein',
}

POSSIBLE_PILUS_PRODUCTS = {
    'pilin',
    'shufflon system plasmid conjugative transfer pilus tip adhesin PilV',
    'prepilin-type N-terminal cleavage/methylation domain-containing protein',
    'PilT/PilU family type 4a pilus ATPase',
    'Flp pilus assembly complex ATPase component TadA',
    'type IV pilus twitching motility protein PilT',
    'type IV pili methyl-accepting chemotaxis transducer N-terminal domain-containing protein',
    'type IV-A pilus assembly ATPase PilB',

    'prepilin peptidase',
    'pilus assembly protein PilX',
    'Flp pilus assembly complex ATPase component TadA',
    'type IV pilus biogenesis protein PilP',
    'type 4b pilus protein PilO2',
    'PilN family type IVB pilus formation outer membrane protein',
    'type IV pilus biogenesis protein PilM',
    'TcpQ domain-containing protein', # https://www.uniprot.org/uniprot/P29490
}

POSSIBLE_TRUNCATED_MTASE_PRODUCTS = POSSIBLE_MTASE_PRODUCTS | {
    'hypothetical protein',
}
POSSIBLE_TRUNCATED_OR_SHORT_PGLX_PRODUCTS = POSSIBLE_TRUNCATED_MTASE_PRODUCTS | {'BREX-1 system adenine-specific DNA-methyltransferase PglX'}
POSSIBLE_TRUNCATED_PHAGE_TAIL_PRODUCTS = {'tail fiber domain-containing protein', '___phage tail or tail fiber protein', 'hypothetical protein'}
POSSIBLE_RHUM_PRODUCTS = {'virulence RhuM family protein', '___RhuM', 'RhuM protein'}

POSSIBLE_GENERIC_HELICASE_PRODUCTS = {'___DEAD/DEAH box helicase','helicase','DNA helicase','ATP-dependent helicase', 'helicase C-terminal domain-containing protein'}
POSSIBLE_DISARM_DRMA_PRODUCTS = POSSIBLE_GENERIC_HELICASE_PRODUCTS | {'DISARM system helicase DrmA'}
POSSIBLE_DISARM_DRMD_PRODUCTS = POSSIBLE_GENERIC_HELICASE_PRODUCTS | {'DISARM system SNF2-like helicase DrmD', '___SNF2/RAD54 family helicase'}
POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS = POSSIBLE_GENERIC_HELICASE_PRODUCTS | {'phospholipase D-like domain-containing protein', '___SNF2/RAD54 family helicase',
                                                                                'NgoFVII family restriction endonuclease'}

ALL_POSSIBLE_MTASE_PRODUCTS = POSSIBLE_MTASE_PRODUCTS | {'BREX-1 system adenine-specific DNA-methyltransferase PglX'}
ALL_POSSIBLE_HELICASE_PRODUCTS = (POSSIBLE_GENERIC_HELICASE_PRODUCTS | POSSIBLE_DISARM_DRMA_PRODUCTS |
                                  POSSIBLE_DISARM_DRMD_PRODUCTS | POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS | {'UvrD-helicase domain-containing protein'})
POSSIBLE_UVRD_HELICASE_IN_DISARM_LIKE_CONTEXT_PRODUCTS = {'UvrD-helicase domain-containing protein', 'AAA family ATPase'}

PHAGE_TAIL_RELATED_PRODUCTS = {'___phage tail or tail fiber protein', 'phage tail protein I', 'tail fiber assembly protein'}
POSSIBLE_PHAGE_PRODUCTS = {
    '___phage tail or tail fiber protein', 'phage tail protein I', 'tail fiber assembly protein',
    '___phage baseplate assembly protein',
    '___phage GP46 or GPW/gp25 family protein',
    'phage tail sheath protein',
    'phage virion morphogenesis protein',
    'phage late control D family protein',
    'phage major tail tube protein',
    'phage holin',
    'GpE family phage tail protein',
    'phage Gp37/Gp68 family protein',
    'phage tail sheath subtilisin-like domain-containing protein',
    'phage tail assembly protein',
    'phage protein D',
    'phage portal protein',
    'phage tail tape measure protein',
    'phage repressor protein',
    'phage antirepressor KilAC domain-containing protein',
    'phage holin family protein',
    'phage major capsid protein',
    'phage tail length tape measure family protein',
    '___baseplate protein',
    'tail fiber domain-containing protein',
    'tail assembly protein',
    'tail protein X',
    'Fels-2 prophage protein',
    'prophage tail fiber N-terminal domain-containing protein',
    'phage terminase large subunit',
    'phage tail fiber',
    'DUF2612 domain-containing protein', # according to https://pfam.xfam.org/family/PF11041

    # https://www.ncbi.nlm.nih.gov/Structure/cdd/pfam10076 says: "Uncharacterized protein conserved in bacteria (DUF2313)
    #    Members of this family of proteins comprise various hypothetical and putative bacteriophage tail proteins."
    'DUF2313 domain-containing protein',
}

POSSIBLE_TRANSPOSON_PRODUCTS = {
    'Tn3 family transposase',
    'IS66 family transposase',
    'IS3 family transposase',
    'IS6 family transposase',
    'transposase',
    'IS5 family transposase',
    'IS91 family transposase',
    'IS21 family transposase',
    'IS481 family transposase',
    'IS1 family transposase',
    'IS110 family transposase',
    'ISL3 family transposase',
    'IS630 family transposase',
    'IS256 family transposase',
    'IS30 family transposase',
    'IS1595 family transposase',
    'IS66-like element ISPpu19 family transposase',
    'IS701 family transposase',
    'ISKra4 family transposase',
    'IS1182 family transposase',
    'IS1380-like element ISBvu1 family transposase',
    'IS1380 family transposase',
    'IS1634 family transposase',
    'transposase family protein',
    'IS21-like element ISIde2 family transposase',
    'IS4 family transposase',
    'ISAs1 family transposase',
    'Rpn family recombination-promoting nuclease/putative transposase',
    'Tn3-like element TnXax1 family transposase',
    'conjugative transposon protein TraM',
    'IS5/IS1182 family transposase',
}

POSSIBLE_HELIX_TURN_HELIX_PRODUCTS = {
    'helix-turn-helix domain-containing protein',
    'helix-turn-helix transcriptional regulator',
    'winged helix-turn-helix transcriptional regulator',
    'HTH domain-containing protein',
    '___helix-turn-helix',
}

UPSTREAM_CDS_CONTEXT_IN_BREX_TYPE_1 = {
    3: ({'same_strand'}, {'DUF1819 family protein'}),
    2: ({'same_strand'}, {'DUF1788 domain-containing protein'}),
    1: ({'same_strand'}, {'BREX system P-loop protein BrxC'}),
}
SOLITARY_PGLX_UPSTREAM_CDS_CONTEXT = {
    2: ({'not'}, POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS),
    1: ({'not'},
        POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS | POSSIBLE_HSDR_PRODUCTS | POSSIBLE_DUF1016_PRODUCTS | POSSIBLE_DISARM_DRMD_PRODUCTS |
        POSSIBLE_MTASE_PRODUCTS),
}

DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE = {
    1: ({'other_strand'}, POSSIBLE_TRUNCATED_MTASE_PRODUCTS),
    2: (set(), {'___recombinase'}),
}
NON_DISARM_DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE = {
    **DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE,
    3: ({'not'}, POSSIBLE_DISARM_DRMA_PRODUCTS),
}
DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE_WITHOUT_DOWNSTREAM_PLD_AND_SNF2 = {
    **DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE,
    3: ({'not'}, POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS),
    4: ({'not'}, POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS),
}


TRUNCATED_HSDS_AND_RECOMBINASE_DOWNSTREAM_CDS_CONTEXTS = [
    {
        1: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
        2: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
        3: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
        4: (set(), {'___recombinase'}),
    },
    {
        1: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
        2: (set(), {'___recombinase'}),
    },
    {
        1: (set(), {'___recombinase'}),
        2: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
    },
    {
        1: (set(), {'___recombinase'}),
        2: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
        3: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
    }
]






SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT = {
    # enter the path to your local BLAST nt database here, e.g.: '/DBs/blast_nt_database/nt'.
    # If you don't have a local copy of the BLAST nt database, but you do have BLAST installed
    # (assuming you followed the installation instructions in README), you could download a copy of the BLAST nt database by
    # navigating to the desired path, and running there:
    # >>> update_blastdb.pl nt
    # If you have enough disk space, and there are no issues with slow connection to NCBI (we had this issue on campus, and so we downloaded the
    # database at home and then copied it to the lab's server), it should just work.
    'local_blast_nt_database_path': None,

    # enter here the path to a file that contains a timestamp indicative of when you local blast nt database was downloaded.
    # the exact content of the file is not important, but it should be changed if and only if you update your local blast nt database.
    # the reason is that our caching machinery would use this file as an indication whether it can used a cached result or not.
    'local_blast_nt_database_update_log_file_path': None,

    'max_total_dist_between_joined_parts_per_joined_feature': 100,
    'debug_local_blast_database_path': None,
    'debug_other_nuccore_accession_to_fasta_file_path': None,

    'stage1': {
        'output_dir_path': 's1_output_of_massive_screening',
        'results_pickle_file_name': 'stage1_results.pickle',

        # note that False also means the shuffling of the taxon uids would be different, so the first 5000 would be different ones.
        # also, the chosen reference genome for each species might be different. beware.
        'use_cached_refseq_bacteria_assembly_summary_file': True,
        # 'use_cached_refseq_bacteria_assembly_summary_file': False,

        # https://ftp.ncbi.nlm.nih.gov/genomes/README_assembly_summary.txt
        'refseq_bacteria_assembly_summary_file_url': 'ftp://ftp.ncbi.nih.gov/genomes/refseq/bacteria/assembly_summary.txt',

        'allowed_assembly_level_values_sorted_by_preference': [
            'Complete Genome',
            'Chromosome',
            'Scaffold',
            'Contig',
        ],
    },

    'stage2': {
        'output_dir_path': 's2_output_of_massive_screening',
        'results_pickle_file_name': 'stage2_results_info.pickle',
        'repeat_pairs': {
            'seed_len': 20,
            'min_repeat_len': 20,
            'max_spacer_len': int(15e3),
            'min_spacer_len': 1,
            # assuming that when BLASTing a 20Mbp chromosome to itself, the evalue of a perfect 20bp IR pair is around 20e6**2 * (1/4**20) == 363.8,
            # an evalue of 1000 should be high enough for all bacterial chromosomes (the longest one I encountered was around 16Mbp).
            'max_evalue': 1000,
        },
    },

    'stage3': {
        'output_dir_path': 's3_output_of_massive_screening',
        'results_pickle_file_name': 'stage3_results_info.pickle',

        # maybe I could have just specified 22 in stage2, but this way we can see the distribution of repeat lens starting at 20, and
        # honestly, i didn't want to run stage2 again, and this was much faster...
        'min_repeat_len': 22,

        'min_max_estimated_copy_num_to_classify_as_mobile_element': 3,
        'blast_repeat_to_its_taxon_genome_to_find_copy_num': {
            'min_dist_from_ir_pair_region_for_alignments': int(50e3),
            'max_evalue': 1e-4,
            'seed_len': 15,
        },
    },

    'stage4': {
        'output_dir_path': 's4_output_of_massive_screening',
        'results_pickle_file_name': 'stage4_results_info.pickle',

        'max_num_of_taxon_uids_to_search_local_nt_per_taxon': 100,
        'min_wgs_nuccore_entry_len': int(40e3),
        'max_num_of_wgs_nuccore_entries_per_taxon_for_entrez_query': 500,
    },

    'stage5': {
        'output_dir_path': 's5_output_of_massive_screening',
        'results_pickle_file_name': 'stage5_results_info.pickle',
        'other_nuccore_entries_extracted_from_local_nt_blast_db_dir_name': 'other_nuccore_entries_extracted_from_local_nt_blast_db',

        'merged_cds_pair_region_margin_size': 200,

        'blast_margins_and_identify_regions_in_other_nuccores': {
            'max_evalue': 1e-5, # this isn't really relevant, as we have num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate.
            'seed_len': 20, # maybe this is too strict?
            # 'min_alignment_len': 50, # don't add it back. i think a much better alternative is to decrease max_evalue.
            'num_of_best_alignments_of_each_margin_to_other_nuccore_to_investigate': 20,
            'max_dist_between_lens_of_spanning_regions_ratio_and_1': 0.05,
        },
        'max_num_of_non_identical_regions_in_other_nuccores_to_analyze_per_merged_cds_pair_region': 100,

        'min_mauve_total_match_proportion': 0.95,
        'min_min_sub_alignment_min_match_proportion': 0.95,
        # 'max_breakpoint_containing_interval_len': 10,
        'max_breakpoint_containing_interval_len': np.inf,
        'max_max_dist_between_potential_breakpoint_containing_interval_and_repeat': 10,
    },

    'enrichment_analysis': {
        'output_dir_path': 'enrichment_analysis_output_of_massive_screening',
        'results_pickle_file_name': 'enrichment_analysis_results_info.pickle',

        # 'DEBUG___RANDOM_ANY_HIGH_CONFIDENCE_IR_PAIR_LINKED_TO_CDS_PAIR': True,
        'DEBUG___RANDOM_ANY_HIGH_CONFIDENCE_IR_PAIR_LINKED_TO_CDS_PAIR': False,

        # 'DEBUG___SHUFFLED_PRODUCTS': True,
        'DEBUG___SHUFFLED_PRODUCTS': False,

        # 'DEBUG___SHUFFLE_OPERON_ASYMMETRY': True,
        'DEBUG___SHUFFLE_OPERON_ASYMMETRY': False,

        'clustering': {
            'min_pairwise_identity_with_cluster_centroid': 0.95,
            'pairwise_identity_definition_type': 'matching_columns_divided_by_alignment_length_such_that_terminal_gaps_are_penalized',
        },

        # 'test_column_name_for_each_product_comparing_merged_cds_pair_regions': 'any_high_confidence_ir_pair_linked_to_cds_pair',
        'test_column_name_for_each_product_comparing_merged_cds_pair_regions': 'predicted_any_high_confidence_ir_pair_linked_to_cds_pair',


        'min_num_of_cds_pairs_with_product_for_enrichment_test': 12,
        'max_corrected_pvalue_for_product_to_be_considered_significantly_enriched': 0.05,
        # 'num_of_cds_on_each_side_for_context_analysis': 10,
        'logistic_regression_dependent_var_column_name': 'any_high_confidence_ir_pair_linked_to_cds_pair',
        'names_of_columns_whose_binarized_versions_are_used_in_logistic_regression': [
            # 'cds_asymmetry',
            'operon_asymmetry',
            # 'cds_spacer_len',
            'operon_spacer_len',
            'max_repeat_len',
            # 'cds_closest_repeat_position_orientation_matching',
            'operon_closest_repeat_position_orientation_matching',
            # 'operon_furthest_repeat_position_orientation_matching',
            # 'cds_furthest_repeat_position_orientation_matching',
        ],
        'min_predicted_rearrangement_probability': 0.05,

        # Could look at the distribution of distances between consecutive CDSs on the same strand. hopefully, there would be two peaks:
        # one for consecutive CDSs not on the same operon, and one for consecutive CDSs on the same operon.
        'max_dist_between_cds_in_operon': 20,



        'list_of_product_and_product_family': [
            ('master DNA invertase Mpi family serine-type recombinase', '___recombinase'),
            ('recombinase family protein', '___recombinase'),
            ('tyrosine-type recombinase/integrase', '___recombinase'),
            ('site-specific integrase', '___recombinase'),
            ('DDE-type integrase/transposase/recombinase', '___recombinase'),
            ('Hin recombinase', '___recombinase'),
            ('recombinase', '___recombinase'),
            ('recombinase XerD', '___recombinase'),
            ('recombinase RecA', '___recombinase'),
            ('tyrosine recombinase', '___recombinase'),
            ('serine recombinase', '___recombinase'),
            ('integrase', '___recombinase'),
            ('phage integrase SAM-like domain-containing protein', '___recombinase'),
            ('integrase core domain-containing protein', '___recombinase'),
            ('integrase arm-type DNA-binding domain-containing protein', '___recombinase'),
            ('Holliday junction resolvase RuvX', '___recombinase'),
            ('tyrosine recombinase XerS', '___recombinase'),
            ('tyrosine recombinase XerC', '___recombinase'),
            ('phage integrase family protein', '___recombinase'),
            ('integrase domain-containing protein', '___recombinase'),
            ('tyrosine-type recombinase/integrase family protein', '___recombinase'),
            ('tyrosine-type DNA invertase PsrA', '___recombinase'),
            ('Tsr0667 family tyrosine-type DNA invertase', '___recombinase'),





            ('sulfotransferase', '___sulfotransferase'),
            ('sulfotransferase family protein', '___sulfotransferase'),
            ('tetratricopeptide repeat-containing sulfotransferase family protein', '___sulfotransferase'),

            ('lytic transglycosylase', '___lytic transglycosylase'),
            ('transglycosylase SLT domain-containing protein', '___lytic transglycosylase'),
            ('lytic transglycosylase domain-containing protein', '___lytic transglycosylase'),

            ('alpha/beta hydrolase', '___alpha/beta hydrolase'),
            ('alpha/beta fold hydrolase', '___alpha/beta hydrolase'),

            ('RagB/SusD family nutrient uptake outer membrane protein', '___SusD/RagB'),
            ('SusD/RagB family nutrient-binding outer membrane lipoprotein', '___SusD/RagB'),

            ('SusC/RagA family TonB-linked outer membrane protein', '___SusC/RagA'),
            # ('TonB-dependent receptor', '___SusC/RagA'), # not every TonB-dependent receptor is a SusC/RagA OM protein

            ('helix-turn-helix domain-containing protein', '___helix-turn-helix'),
            ('helix-turn-helix transcriptional regulator', '___helix-turn-helix'),

            ('HsdR family type I site-specific deoxyribonuclease', '___restriction endonuclease subunit R'),
            ('restriction endonuclease subunit R', '___restriction endonuclease subunit R'),
            ('type I restriction endonuclease subunit R', '___restriction endonuclease subunit R'),

            ('type I restriction endonuclease subunit S', '___restriction endonuclease subunit S'),
            ('restriction endonuclease subunit S', '___restriction endonuclease subunit S'),
            ('restriction endonuclease S subunit', '___restriction endonuclease subunit S'),
            ('type I restriction-modification enzyme, S subunit', '___restriction endonuclease subunit S'),
            ('type I site-specific deoxyribonuclease specificity subunit HsdS', '___restriction endonuclease subunit S'),
            ('type I R-M system S protein', '___restriction endonuclease subunit S'),

            ('type I restriction-modification system subunit M', '___restriction-modification system subunit M'),
            ('restriction endonuclease subunit M', '___restriction-modification system subunit M'),
            ('type II restriction endonuclease subunit M', '___restriction-modification system subunit M'),

            ('DEAD/DEAH box helicase', '___DEAD/DEAH box helicase'),
            ('DEAD/DEAH box helicase family protein', '___DEAD/DEAH box helicase'),

            ('SNF2/RAD54 family helicase', '___SNF2/RAD54 family helicase'),
            ('helicase SNF2/RAD54 family', '___SNF2/RAD54 family helicase'),

            ('baseplate protein', '___baseplate protein'),
            ('baseplate assembly protein', '___baseplate protein'),
            ('baseplate J/gp47 family protein', '___baseplate protein'),

            ('phage baseplate assembly protein', '___phage baseplate assembly protein'),
            ('phage baseplate assembly protein V', '___phage baseplate assembly protein'),

            ('phage GP46 family protein', '___phage GP46 or GPW/gp25 family protein'),
            ('GPW/gp25 family protein', '___phage GP46 or GPW/gp25 family protein'),

            ('phage tail protein', '___phage tail or tail fiber protein'),
            ('tail fiber protein', '___phage tail or tail fiber protein'),

            ('DUF1016 family protein', '___DUF1016'),
            ('DUF1016 domain-containing protein', '___DUF1016'),

            ('Fic family protein', '___Fic'),
            ('cell filamentation protein Fic', '___Fic'),
            ('toxin Fic', '___Fic'),

            ('type II toxin-antitoxin system death-on-curing family toxin', '___death-on-curing protein'),
            ('death-on-curing protein', '___death-on-curing protein'),

            ('virulence RhuM family protein', '___RhuM'),
            ('RhuM protein', '___RhuM'),
        ],

        'name_to_cds_context_info': {
            'MTase: BREX type 1': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': {'BREX-1 system adenine-specific DNA-methyltransferase PglX'},
                'cds_contexts': [
                    {
                        'upstream': [UPSTREAM_CDS_CONTEXT_IN_BREX_TYPE_1],
                        'downstream': [{
                            1: ({'other_strand'}, {'___recombinase'}),
                            2: ({'other_strand'}, POSSIBLE_TRUNCATED_OR_SHORT_PGLX_PRODUCTS),
                            # 3: ({'same_strand'}, {'BREX-1 system phosphatase PglZ type A'}), # adding this just discards a case in which we have a DUF262 before the PglZ, so better just not having it, I guess.
                        }],
                    },
                ],
                'locus_description': 'BREX type 1',
            },
            'MTase: BREX type 1, downstream extra short PglX': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': {'BREX-1 system adenine-specific DNA-methyltransferase PglX'},
                'cds_contexts': [
                    {
                        'upstream': [UPSTREAM_CDS_CONTEXT_IN_BREX_TYPE_1],
                        'downstream': [{
                            1: ({'other_strand'}, POSSIBLE_TRUNCATED_OR_SHORT_PGLX_PRODUCTS),
                            2: (set(), {'___recombinase'}),
                            3: ({'same_strand'}, POSSIBLE_TRUNCATED_OR_SHORT_PGLX_PRODUCTS),
                        }],
                    },
                ],
                'locus_description': 'BREX type 1',
            },
            'MTase: upstream PLD&SNF2 helicase': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': POSSIBLE_MTASE_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            2: ({'not'}, POSSIBLE_GENERIC_HELICASE_PRODUCTS),
                            1: ({'same_strand'}, POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS),
                        }],
                        'downstream': [NON_DISARM_DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE],
                    },
                ],
            },
            'MTase: Class 1 DISARM-like': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': POSSIBLE_MTASE_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            1: ({'same_strand'}, {'___DEAD/DEAH box helicase'}), # a SNF2 helicase
                        }],
                        'downstream': [{
                            1: ({'other_strand'}, POSSIBLE_TRUNCATED_MTASE_PRODUCTS | {'restriction endonuclease'}),
                            2: (set(), {'___recombinase'}),
                            3: ({'same_strand'}, {'___DEAD/DEAH box helicase'}), # a helicase containing DUF1998
                            4: ({'same_strand'}, POSSIBLE_UVRD_HELICASE_IN_DISARM_LIKE_CONTEXT_PRODUCTS), # a UvrD helicase
                        }],
                    },
                ],
                'general context description': 'Class 1 DISARM-like',
            },
            'MTase: Class 1 DISARM': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': POSSIBLE_MTASE_PRODUCTS | {'hypothetical protein'},
                'cds_contexts': [
                    {
                        'upstream': [{
                            1: ({'same_strand'}, POSSIBLE_DISARM_DRMD_PRODUCTS),
                        }],
                        'downstream': [{
                            **DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE,
                            3: ({'same_strand'}, POSSIBLE_DISARM_DRMA_PRODUCTS),
                            4: ({'same_strand'}, {'DUF1998 domain-containing protein'}),
                        }],
                    },
                ],
                'general context description': 'Class 1 DISARM',
            },
            'MTase: upstream drmD': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': {'hypothetical protein'},
                'cds_contexts': [
                    {
                        'upstream': [{
                            3: ({'same_strand'}, {'serine/threonine protein kinase'}),
                            2: ({'same_strand'}, {'serine/threonine protein kinase'}),
                            1: ({'same_strand'}, POSSIBLE_DISARM_DRMD_PRODUCTS),
                        }],
                        'downstream': [{
                            **DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE,
                            3: ({'not'}, POSSIBLE_DISARM_DRMA_PRODUCTS),
                        }],
                    },
                ],
                'general context description': 'Class 1 DISARM',
            },
            'MTase: solitary': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': POSSIBLE_MTASE_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [SOLITARY_PGLX_UPSTREAM_CDS_CONTEXT],
                        'downstream': [DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE_WITHOUT_DOWNSTREAM_PLD_AND_SNF2],
                    },
                ],
            },
            'MTase: upstream DUF1016': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': POSSIBLE_MTASE_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            2: ({'not'}, POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS),
                            1: ({'same_strand'}, POSSIBLE_DUF1016_PRODUCTS),
                        }],
                        'downstream': [DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE_WITHOUT_DOWNSTREAM_PLD_AND_SNF2],
                    },
                ],
            },
            'MTase: downstream PLD&SNF2 helicase': {
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': POSSIBLE_MTASE_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [SOLITARY_PGLX_UPSTREAM_CDS_CONTEXT],
                        'downstream': [
                            {
                                **DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE,
                                3: ({'same_strand'}, POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS),
                            },
                            {
                                **DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE,
                                3: ({'same_strand'}, {'putative DNA binding domain-containing protein'}),
                                4: ({'same_strand'}, POSSIBLE_PLD_AND_SNF2_HELICASE_PRODUCTS),
                            },
                        ],
                    },
                ],
            },
            'MTase: two upstream helicases': { # AKA 'YprA-YprA-DUF1998-HepA system'. formerly named 'N-6 DNA methylase in a context reminiscent of DISARM'.
                'linked_repeat_cds_product_class': 'DNA MTase',
                'longer_linked_repeat_cds_product_families': POSSIBLE_MTASE_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            2: ({'same_strand'}, POSSIBLE_GENERIC_HELICASE_PRODUCTS),
                            1: ({'same_strand'}, POSSIBLE_GENERIC_HELICASE_PRODUCTS),
                        }],
                        'downstream': [NON_DISARM_DOWNSTREAM_CDS_CONTEXT_TRUNCATED_MTASE_AND_RECOMBINASE],
                    },
                ],
            },
            'OM receptor: downstream SusD/RagB': {
                'linked_repeat_cds_product_class': 'OM receptor',
                'longer_linked_repeat_cds_product_families': POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            3: ({'other_strand'}, {'___SusD/RagB'}),
                            2: ({'other_strand'}, POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS),
                            1: (set(), {'___recombinase'}),
                        }],
                        'downstream': [{1: ({'same_strand'}, {'___SusD/RagB'})}],
                    },
                ],
            },
            # 'OM receptor: downstream SusD/RagB MIS': {
            #     'longer_linked_repeat_cds_product_families': POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS,
            #     'cds_contexts': [
            #         {
            #             'upstream': [
            #                 {
            #                     5: ({'other_strand'}, {'___SusD/RagB'}),
            #                     4: ({'other_strand'}, POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS),
            #                     3: ({'other_strand'}, {'___SusD/RagB'}),
            #                     2: ({'other_strand'}, POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS),
            #                     1: (set(), {'___recombinase'}),
            #                 },
            #                 {
            #                     6: ({'other_strand'}, {'___SusD/RagB'}),
            #                     5: ({'other_strand'}, POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS),
            #                     3: ({'other_strand'}, {'___SusD/RagB'}),
            #                     2: ({'other_strand'}, POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS),
            #                     1: (set(), {'___recombinase'}),
            #                 },
            #                 {
            #                     7: ({'other_strand'}, {'___SusD/RagB'}),
            #                     6: ({'other_strand'}, POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS),
            #                     3: ({'other_strand'}, {'___SusD/RagB'}),
            #                     2: ({'other_strand'}, POSSIBLE_SUSC_RAGA_TONB_DEPENDENT_RECEPTOR_PRODUCTS),
            #                     1: (set(), {'___recombinase'}),
            #                 },
            #             ],
            #             'downstream': [{
            #                 1: ({'same_strand'}, {'___SusD/RagB'}),
            #             }],
            #         },
            #     ],
            # },

            'phage tail: upstream phage tail protein I and downstream tail fiber assembly': {
                'linked_repeat_cds_product_class': 'phage tail',
                'longer_linked_repeat_cds_product_families': {'___phage tail or tail fiber protein'},
                'cds_contexts': [
                    {
                        'upstream': [{
                            4: ({'same_strand'}, {'___phage baseplate assembly protein'}),
                            3: ({'same_strand'}, {'___phage GP46 or GPW/gp25 family protein'}),
                            2: ({'same_strand'}, {'___baseplate protein'}),
                            1: ({'same_strand'}, {'phage tail protein I'}),
                        }],
                        'downstream': [{
                            1: ({'same_strand'}, {'tail fiber assembly protein'}),
                            2: ({'other_strand'}, {'tail fiber assembly protein'}),
                            3: ({'other_strand'}, POSSIBLE_TRUNCATED_PHAGE_TAIL_PRODUCTS),
                            4: (set(), {'___recombinase'}),
                        }],
                    },
                ],
                # actually, it is just a guess that this always appears inside a prophage, even though this seems like a very reasonable guess...
                # 'general context description': 'prophage',
            },
            'phage tail: upstream DUF2313 and downstream tail fiber assembly': {
                'linked_repeat_cds_product_class': 'phage tail',
                'longer_linked_repeat_cds_product_families': {'___phage tail or tail fiber protein'},
                'cds_contexts': [
                    {
                        'upstream': [{
                            4: ({'same_strand'}, {'___phage baseplate assembly protein'}),
                            3: ({'same_strand'}, {'___phage GP46 or GPW/gp25 family protein'}),
                            2: ({'same_strand'}, {'___baseplate protein'}),
                            1: ({'same_strand'}, {'DUF2313 domain-containing protein'}),
                        }],
                        'downstream': [{
                            1: ({'same_strand'}, {'tail fiber assembly protein'}),
                            2: ({'other_strand'}, {'tail fiber assembly protein'}),
                            3: ({'other_strand'}, POSSIBLE_TRUNCATED_PHAGE_TAIL_PRODUCTS),
                            4: (set(), {'___recombinase'}),
                        }],
                    },
                ],
            },
            'phage tail: downstream transporter and endonuclease': {
                # what is the purpose of this?
                # the paper 'Burkholderia cenocepacia Prophages—Prevalence, Chromosome Location and Major Genes Involved' says:
                # "MFS transporter genes were the most commonly observed in BCC genomes analyzed which is probably related to host drug resistance.
                # MFS have a very broad substrate spectrum, thus their influence on the host virulence could not be precisely indicated."
                'linked_repeat_cds_product_class': 'phage tail',
                'longer_linked_repeat_cds_product_families': {'___phage tail or tail fiber protein'},
                'cds_contexts': [
                    {
                        'upstream': [{
                            4: ({'same_strand'}, {'___phage baseplate assembly protein'}),
                            3: ({'same_strand'}, {'___phage GP46 or GPW/gp25 family protein'}),
                            2: ({'same_strand'}, {'___baseplate protein'}),
                            1: ({'same_strand'}, {'phage tail protein I'}),
                        }],
                        'downstream': [{
                            1: ({'same_strand'}, {'hypothetical protein', 'esterase-like activity of phytase family protein'}),
                            2: ({'same_strand'}, {'aromatic acid/H+ symport family MFS transporter'}),
                            3: ({'other_strand'}, {'DNA endonuclease SmrA'}),
                            4: ({'other_strand'}, POSSIBLE_TRUNCATED_PHAGE_TAIL_PRODUCTS),
                            5: (set(), {'___recombinase'}),
                            6: ({'same_strand'}, POSSIBLE_TRUNCATED_PHAGE_TAIL_PRODUCTS),
                            7: ({'same_strand'}, {'tail fiber assembly protein'}),
                        }],
                    },
                ],
            },
            'RM specificity: M-S-R': {
                'linked_repeat_cds_product_class': 'Type I restriction-modification HsdS',
                'longer_linked_repeat_cds_product_families': POSSIBLE_HSDS_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            2: ({'not'}, POSSIBLE_HSDR_PRODUCTS),
                            1: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                        }],
                        'downstream': [{
                            1: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            2: (set(), {'___recombinase'}),
                            3: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                        }],
                    },
                ],
            },
            'RM specificity: M-invertibleSs-R': {
                'linked_repeat_cds_product_class': 'Type I restriction-modification HsdS',
                'longer_linked_repeat_cds_product_families': POSSIBLE_HSDS_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            2: ({'not'}, POSSIBLE_HSDR_PRODUCTS),
                            1: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                        }],
                        'downstream': [
                            {
                                1: (set(), {'___recombinase'}),
                                2: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                                3: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            },
                            {
                                1: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                                2: (set(), {'___recombinase'}),
                                3: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            },
                            # decided to not the following because the ones above are much better. the first of the following has only 2 cds pairs, and the second
                            # has a hypoth without any homologies, so feels problematic...
                            # {
                            #     1: (set(), {'___recombinase'}),
                            #     2: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                            #     3: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                            #     4: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            # },
                            # {
                            #     1: (set(), {'___recombinase'}),
                            #     2: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                            #     3: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                            #     4: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                            #     5: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            # },
                        ],
                    },
                    {
                        # formerly called this one 'RM subunit S in M-S-invertibleSs-R', but after I found some direct repeats between the hsdS downstream
                        # to the hsdM, I realized it might actually become an inverted repeat after an inversion, and so it might not be constant as I previously
                        # thought. It also makes more sense that it isn't constant due to its shortness.
                        'upstream': [{
                            5: ({'not'}, POSSIBLE_HSDR_PRODUCTS), # is this necessary?
                            4: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                            3: ({'same_strand'}, POSSIBLE_HSDS_PRODUCTS),
                            2: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                            1: (set(), {'___recombinase'}),
                        }],
                        'downstream': [
                            {
                                1: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            },
                            {
                                1: ({'same_strand'}, {'hypothetical protein', 'FRG domain-containing protein'}),
                                2: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            },
                            {
                                1: ({'same_strand'}, {'hypothetical protein'}),
                                2: ({'same_strand'}, {'FRG domain-containing protein'}),
                                3: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            },
                        ],
                    },
                    # {
                    #     # Decided to comment this out, but I am not sure about it.
                    #     # This presumably identifies mistakes in CDS lengths, which wrongly choose the longer repeat CDS...
                    #     # unless there really are two promoters, but I guess this is not the case?
                    #     # there are only a few of these, so I feel better skipping them. I guess it makes sense to only consider cases in which the longer repeat CDS is
                    #     # part of an operon...
                    #     'upstream': [{
                    #         2: ({'other_strand'}, POSSIBLE_HSDR_PRODUCTS),
                    #         1: (set(), {'___recombinase'}),
                    #     }],
                    #     'downstream': [
                    #         {
                    #             1: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #             2: ({'other_strand'}, POSSIBLE_MTASE_PRODUCTS),
                    #         },
                    #     ],
                    # },
                    # decided against this one mainly because 3 HsdS CDSs suggests it might be 2 invertible HsdS CDSs and another constant HsdS, but there are too few cases here,
                    # so I prefer skipping it (see comments below also.
                    # {
                    #     'upstream': [{
                    #                     5: ({'not'}, POSSIBLE_HSDR_PRODUCTS),
                    #                     4: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                    #                     3: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #                     2: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #                     1: (set(), {'___recombinase'}),
                    #                 }],
                    #     'downstream': [
                    #         # only 4 cases, and at least in one of them one of the HsdS CDSs seems to not have IR pairs with others, so I guess I should better skip this.
                    #         {
                    #             1: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                    #         },
                    #         # decided to not have this one, because it is only 4 examples, and in one of them 1th downstream is a hypoth which is homologous to
                    #         # a few 'HEPN family nuclease' proteins.
                    #         # {
                    #         #     1: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #         #     2: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                    #         # },
                    #     ],
                    # },

                ],
            },
            'RM specificity: R-M-S': {
                'linked_repeat_cds_product_class': 'Type I restriction-modification HsdS',
                'longer_linked_repeat_cds_product_families': POSSIBLE_HSDS_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [
                            {
                                2: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                                1: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                            },
                            # {
                            #     # thought about adding this, but there were no common downstream CDS patterns for it (seemed pretty messy overall), so decided to not have it.
                            #     3: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            #     2: ({'same_strand'}, {'hypothetical protein', '___DEAD/DEAH box helicase', 'AAA family ATPase', 'putative DNA binding domain-containing protein'}),
                            #     1: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                            # }
                        ],
                        'downstream': TRUNCATED_HSDS_AND_RECOMBINASE_DOWNSTREAM_CDS_CONTEXTS,
                    },
                ],
            },
            'RM specificity: R-M-DUF1016-S': {
                'linked_repeat_cds_product_class': 'Type I restriction-modification HsdS',
                'longer_linked_repeat_cds_product_families': POSSIBLE_HSDS_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [{
                            3: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                            2: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                            1: ({'same_strand'}, POSSIBLE_DUF1016_PRODUCTS),
                        }],
                        'downstream': TRUNCATED_HSDS_AND_RECOMBINASE_DOWNSTREAM_CDS_CONTEXTS,
                    },
                ],
            },
            'RM specificity: R-M-RhuM-S': {
                'linked_repeat_cds_product_class': 'Type I restriction-modification HsdS',
                'longer_linked_repeat_cds_product_families': POSSIBLE_HSDS_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [
                            {
                                3: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                                2: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                                1: ({'same_strand'}, POSSIBLE_RHUM_PRODUCTS),
                            },
                            {
                                4: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                                3: ({'same_strand'}, {'hypothetical protein', 'DUF4062 domain-containing protein', 'GIY-YIG nuclease family protein',
                                                      'four helix bundle protein', 'DUF262 domain-containing protein'}),
                                2: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                                # NOTE: The ___Fic should be here. this is not a mistake.
                                # This doesn't seem like a old genbank file that I am using: At least for one of these Fic CDSs,
                                # it seems like the genbank I am using is up to date (December 2021). so i guess ncbi-genome-download
                                # downloaded the version with Fic instead of the one with RhuM, i.e., QTMU01000025.1 instead of NZ_QTMU01000025.1.
                                # Saw the same for QTNE01000004.1 and QTNB01000001.1 (instead of NZ_QTNE01000004.1 and NZ_QTNB01000001.1).
                                1: ({'same_strand'}, POSSIBLE_RHUM_PRODUCTS | {'___Fic'}),
                            },
                        ],
                        'downstream': TRUNCATED_HSDS_AND_RECOMBINASE_DOWNSTREAM_CDS_CONTEXTS,
                    },
                    # {
                    #     # Decided to comment this out, but I am not sure about it.
                    #     # I don't know about it. This presumably identifies mistakes in CDS lengths, which wrongly choose the longer repeat CDS...
                    #     # unless there really are two promoters, but I guess this is not the case?
                    #     # there are only a few of these, so I feel better skipping them. I guess it makes sense to only consider cases in which the longer repeat CDS is
                    #     # part of an operon...
                    #     'upstream': [
                    #         {
                    #             1: (set(), {'___recombinase'}),
                    #         },
                    #     ],
                    #     'downstream': [
                    #         {
                    #             1: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #             2: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #             3: ({'other_strand'}, POSSIBLE_HSDS_PRODUCTS),
                    #             4: ({'other_strand'}, POSSIBLE_RHUM_PRODUCTS | {'___Fic'}),
                    #             5: ({'other_strand'}, POSSIBLE_MTASE_PRODUCTS),
                    #             6: ({'other_strand'}, {'hypothetical protein', 'DUF4062 domain-containing protein', 'GIY-YIG nuclease family protein',
                    #                                    'four helix bundle protein', 'DUF262 domain-containing protein', 'DUF4268 domain-containing protein'}),
                    #             7: ({'other_strand'}, POSSIBLE_HSDR_PRODUCTS),
                    #         },
                    #         # Conceptually, this should also be here, as the mirror image of the first CDS context, but there are no cases matching that currently.
                    #         # {
                    #         #     1: ({'other_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #         #     2: ({'same_strand'}, POSSIBLE_TRUNCATED_HSDS_PRODUCTS),
                    #         #     3: ({'other_strand'}, POSSIBLE_HSDS_PRODUCTS),
                    #         #     4: ({'other_strand'}, POSSIBLE_RHUM_PRODUCTS),
                    #         #     5: ({'other_strand'}, POSSIBLE_MTASE_PRODUCTS),
                    #         #     6: ({'other_strand'}, POSSIBLE_HSDR_PRODUCTS),
                    #         # },
                    #     ],
                    # },
                ],
            },
            'RM specificity: R-M-Fic/DOC-S': {
                'linked_repeat_cds_product_class': 'Type I restriction-modification HsdS',
                'longer_linked_repeat_cds_product_families': POSSIBLE_HSDS_PRODUCTS,
                'cds_contexts': [
                    {
                        'upstream': [
                            {
                                4: ({'same_strand'}, POSSIBLE_HSDR_PRODUCTS),
                                3: ({'same_strand'}, POSSIBLE_MTASE_PRODUCTS),
                                2: ({'same_strand'}, {'hypothetical protein', 'DNA-binding protein'}),
                                1: ({'same_strand'}, {'___Fic', '___death-on-curing protein'}),
                            },
                        ],
                        'downstream': TRUNCATED_HSDS_AND_RECOMBINASE_DOWNSTREAM_CDS_CONTEXTS,
                    },
                ],
            }
        },
        'conflicting_and_final_cds_context_names': [
            ('OM receptor: downstream SusD/RagB', 'OM receptor: downstream SusD/RagB MIS', 'OM receptor: downstream SusD/RagB MIS'),
        ],
    },

    'stage6': {
        'output_dir_path': 's6_output_of_massive_screening',
        'results_pickle_file_name': 'stage6_results_info.pickle',

        # the code assumes SRA files were already downloaded and moved into the directory specified by 'sra_entries_dir_name'.
        # so in the default case (as you downloaded this code), the path of this directory would be
        # s6_output_of_massive_screening/sra_entries.
        'sra_entries_dir_name': 'sra_entries',

        'DEBUG___sra_entry_fasta_file_path': None,
        'DEBUG___nuccore_fasta_file_path': None,
        'DEBUG___nuccore_gb_file_path': None,
        'DEBUG___assembly_fasta_file_path': None,

        'blast_longer_linked_repeat_cds_to_each_taxon_genome': {
            'seed_len': 10,
            'max_evalue': 1e-5,
            'min_repeat_cds_covered_bases_proportion': 0.5,
            'min_alignment_bases_covered_by_cds_proportion': 0.9,
        },
        'blast_longer_linked_repeat_cds_homolog_to_its_margins': {
            'margin_size': int(10e3),
            'seed_len': 14,
            'max_evalue': 1e-4,
            'max_evalue_to_consider_as_potential_programmed_inversion': 1e-6, # what we actually show in Figure 4D
        },

        'blast_nuccore_to_find_ir_pairs': {
            'seed_len': 7,
            'min_repeat_len': 14,
            'max_evalue': 1000,
        },

        'blast_alignment_region_to_long_reads': {
            'min_ir_pair_region_margin_size_for_evidence_read': 500,
            'min_num_of_read_bases_covered_by_any_alignment': int(2e3),
            'min_alignment_region_margin_size': int(10e3),
            'seed_len': 8,
            'max_evalue': 1e-4,
        },
        'blast_alignment_assembly_to_relevant_long_reads': {
            'seed_len': 8,
            'max_evalue': 1e-4,
        },
        'rna_seq_analysis': {
            'inverted_repeat_max_evalue': 1e-3,
            'min_abs_score_diff': 100,
        },

        'cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args': {
            # IMPORTANT: 'Beyond Restriction Modification: Epigenomic Roles of DNA Methylation in Prokaryotes' (2021) says:
            #   "genes encoding Type II REases are often difficult to identify by sequence similarity"
            # BLASTP all validated MTase longer_linked_repeat_cds_region to the classic PglX (which is one of the validated MTases), classic Eco57I (E. coli),
            #   Cj0031 of C. jejuni, and DISARM DrmMI?
            #
            # WP_197984793.1 ('two upstream helicases') - best alignment to WP_045072894.1 (DISARM) - evalue=4e-126 (other alignments are not close to that)
            #
            # best alignment to WP_015765013.1 (classic truncated PglX) - evalue=5e-148.
            # second best alignment to WP_015765014.1 (classic PglX) - evalue=8e-146 (other alignments are not close to that)
            # WP_201670672.1 ('BREX type 1, downstream extra short PglX')
            #
            # best alignment to WP_045072894.1 (DISARM) - evalue=3e-13 (other alignments are not too far behind, but there
            # was also another not-bad alignment to DISARM, which makes DISARM seem much closer than others)
            # WP_191009655.1 ('Class 1 DISARM-like')
            #
            # best alignment to YP_002343503.1 (C. jejuni Cj0031) - evalue=5e-115 (from aa 319 to 1111 out of 1122 (the length of UBN58241.1)
            # also, blasted UBN58241.1 to QFG45918.1, and got a single alignment, evalue=1e-104, from aa 365 to 1043 out of 1062 (the length of QFG45918.1)
            # UBN58241.1 ('upstream PLD&SNF2 helicase')
            #
            # best alignment to YP_002343503.1 (C. jejuni Cj0031) - evalue=7e-74
            # also, blasted UBN58241.1 to QFG45918.1, and got a single alignment, evalue=1e-104, from aa 365 to 1043 out of 1062 (the length of QFG45918.1)
            # QFG45918.1 ('downstream PLD&SNF2 helicase')
            #
            # best alignment to YP_002343503.1 (C. jejuni Cj0031) - evalue=0 (alignment contains aa 8 to 1241 out of 1241 (the length of WP_223381307.1))
            # also, blasted WP_223381307.1 to AYY86337.1, and got a single alignment, evalue=0, from aa 1 to 1239 out of 1241 (the length of WP_223381307.1)
            # WP_223381307.1 ('solitary')
            #
            # best alignment to YP_002343503.1 (C. jejuni Cj0031) - evalue=0 (alignment contains aa 29 to 1243 out of 1245 (the length of AYY86337.1))
            # AYY86337.1 ('upstream DUF1016')
            #
            #
            #
            # WP_197984793.1 ('two upstream helicases')
            # WP_201670672.1 ('BREX type 1, downstream extra short PglX')
            # WP_191009655.1 ('Class 1 DISARM-like')
            # UBN58241.1 ('upstream PLD&SNF2 helicase')
            # QFG45918.1 ('downstream PLD&SNF2 helicase')
            # WP_223381307.1 ('solitary')
            # AYY86337.1 ('upstream DUF1016')
            # WP_015765014.1 - classic PglX
            # WP_015765013.1 - classic truncated PglX
            # YP_002343503.1 - Cj0031 of C. jejuni. Has Eco57I (pfam07669) and TaqI_C (pfam12950) domains. (https://academic.oup.com/nar/article/44/10/4581/2515808 - Phase variation of a Type IIG restriction-modification enzyme alters site-specific methylation patterns and gene expression in Campylobacter jejuni strain NCTC11168)
            # AAD08395.1 - M.HpyAIV of H. Pylori (mentioned in "Beyond Restriction Modification: Epigenomic Roles of DNA Methylation in Prokaryotes") - blasted it to all others, and got no alignments.
            # WP_045072894.1 - DISARM methylase from 220126_another_disarm_with_rearrangements.png
            # AAA23389.1 - Eco57I of E. coli (https://www.uniprot.org/uniprot/P25239) - has PglX, Eco57I and HsdM domains (https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=AAA23389.1&FULL)
            #
            #

            'MTase: two upstream helicases': {
                # WP_228426257.1 - the MTase with SNF2 (length: 2194 aa)
                # WP_115169831.1 - the MTase with 3 other adjacent MTases (length: 1154 aa)
                # WP_197984793.1 - the MTase we show in fig4 (length: 1243 aa)
                # blastp WP_115169831.1 to WP_197984793.1 results in a very good alignment (evalue=0) of aa 1 to 1130 out of 1154.
                # blastp WP_228426257.1 to WP_197984793.1 results in a very good alignment (evalue=0) of aa 1 to 1222 out of 1243.

                # shared domains upstream to the MTase, upstream to downstream:
                # YprA domain
                # ~600 aa without a shared domain
                # YprA domain
                # ~370 aa without a shared domain
                # DUF1998
                # exactly 215 aa without a shared domain (in both NZ_UGYW01000002.1 and NZ_UFVQ01000003.1)
                # HepA domain, which contains two SNF2 domains (in WP_228426257.1 the first SNF2 domain isn't identified, but i guess it is still pretty much there?)

                'NZ_CP068294.1': [ # Flavobacterium sp. CLA17 (Bacteroidetes)
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP068294.1?report=graph&v=3119745:3133985
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP068294.1?report=graph&v=3120509-3123718
                        'presumably_relevant_cds_region': (3119745, 3133985),
                        # homology claim is based on blasting WP_197984793.1 to WP_052482302.1
                        'target_gene_product_description': 'Type II restriction-modification enzyme, homologous to Class 1 DISARM DrmMI',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_171569986.1&FULL
                            'A protein containing two short YprA (COG1205, helicase) domains and a DUF1998 (pfam09369) domain; ' 
                            'SNF2 (COG0553, helicase) domain-containing protein' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_171569985.1&FULL
                        ),

                        'locus_description_for_table_3': 'Similar to Class 1 DISARM in terms of some encoded protein domains, but with a different gene order, '
                                                         'and with a single CDS encoding two short YprA (COG1205, helicase) domains and '
                                                         'a DUF1998 (pfam09369) domain',

                        'locus_definition': r'$\mathit{Flavobacterium}$ sp. CLA17',
                        'phylum': 'Bacteroidetes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((3122130, 3125861), 'WP_197984793.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_197984793.1
                        # WP_171569986.1, the upstream helicase further from the MTase, contains two YprA domains and a DUF1998 domain.
                        # WP_171569985.1, the upstream helicase closer to the MTase, contains a HepA domain (which contains two SNF2 domains).
                        # WP_197984793.1, the MTase, contains a PglX domain
                        # The downstream hypoth WP_171567931.1 has no identified conserved domains. It is homologous only to hypoth (first 100 hits)
                        'ir_pair_region_with_margins': (3120509, 3123718),
                        'alignment_region': (3100000, 3144000),
                        # (3121009, 3121053, 3123174, 3123218) # blue
                        # (3121370, 3121397, 3122788, 3122815) # green
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR11301546': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR11301546.1.1050215.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion
                                    'SRR11301546.1.1050216.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion
                                    'SRR11301546.1.1112598.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch?

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR11301546.1.1112599.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch?

                                    'SRR11301546.1.1112600.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch?
                                    'SRR11301546.1.218423.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion
                                    # 'SRR11301546.1.238992.1', # cassette switch. doesn't cover the whole region
                                    'SRR11301546.1.238993.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    'SRR11301546.1.238994.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    'SRR11301546.1.380567.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    # 'SRR11301546.1.380568.1', # cassette switch. doesn't cover the whole region
                                    'SRR11301546.1.454354.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion
                                    'SRR11301546.1.650314.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    'SRR11301546.1.685108.1': {(3121370, 3121397, 3122788, 3122815)}, # inversion
                                    'SRR11301546.1.685109.1': {(3121370, 3121397, 3122788, 3122815)}, # inversion
                                    'SRR11301546.1.725072.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    'SRR11301546.1.79703.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR11301546.1.917252.1': {(3121370, 3121397, 3122788, 3122815)}, # inversion

                                    'SRR11301546.1.971711.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    # 'SRR11301546.1.971712.1', # cassette switch. doesn't cover the whole region
                                    # 'SRR11301546.1.971713.1', # cassette switch. doesn't cover the whole region
                                    'SRR11301546.1.971714.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    'SRR11301546.1.971715.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    'SRR11301546.1.971716.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)}, # cassette switch
                                    # 'SRR11301546.1.978874.1', # inversion + inverted duplication. doesn't span the whole region

                                    # added late (on 220304)
                                    'SRR11301546.1.818374.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion
                                    'SRR11301546.1.818375.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion
                                    'SRR11301546.1.818376.1': {(3121009, 3121053, 3123174, 3123218)}, # inversion


                                    'SRR11301546.1.1124627.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.1124628.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.1124629.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.1124633.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.1124634.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.1177469.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177473.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177474.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177475.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177476.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177477.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177480.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177481.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177482.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.1177483.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135003.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135004.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135005.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135006.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135008.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135009.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135010.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135011.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135012.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135014.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.135015.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.145788.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.145790.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.145791.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.145792.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.145793.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.145794.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.145795.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.159829.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.159830.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.159831.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.160495.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160496.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160497.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160498.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160499.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160500.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160501.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160503.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.160504.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.218421.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.218422.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.238991.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.238995.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.252260.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.252261.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.276040.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.276041.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.276042.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.276043.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.289822.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.289824.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.454353.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543178.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543179.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543180.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543181.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543182.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543183.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543184.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543185.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.543187.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.5990.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.5992.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.5994.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.650313.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.650315.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.685105.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.685106.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.685107.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.686943.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.686944.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.686945.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.686946.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.686947.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.686948.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.686949.1': {(3121370, 3121397, 3122788, 3122815), (3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.780360.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.780361.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.780362.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.79704.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.818377.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.978875.1': {(3121009, 3121053, 3123174, 3123218)},
                                    'SRR11301546.1.978892.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978893.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978894.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978895.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978896.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978898.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978900.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978901.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978902.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978903.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978904.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978905.1': {(3121370, 3121397, 3122788, 3122815)},
                                    'SRR11301546.1.978906.1': {(3121370, 3121397, 3122788, 3122815)},
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            # blastp WP_197984792.1, and many of the top scoring homologs which aren't as short as WP_197984792.1 (e.g., WP_115169831.1)
                            # were annotated as DNA MTases.
                            (3121003, 3122133): 'DNA MTase'
                        },

                        'describe_in_the_paper': True,
                    },
                ],

                'NZ_UGYW01000002.1': [ # Sphingobacterium spiritivorum strain NCTC11388
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_UGYW01000002.1?report=graph&v=1267325:1281438
                        'presumably_relevant_cds_region': (1267325, 1281438),
                        # homology claim is based on blasting WP_115169831.1 to WP_052482302.1
                        'target_gene_product_description': 'Type II restriction-modification enzyme, homologous to Class 1 DISARM DrmMI',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'A protein containing two short YprA (COG1205, helicase) domains; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_115169834.1&FULL
                            'DUF1998 (pfam09369) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_115169833.1&FULL
                            'SNF2 (COG0553, helicase) domain-containing protein' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_115169832.1&FULL
                        ),

                        'locus_description_for_table_3': 'Similar to Class 1 DISARM in terms of some encoded protein domains, but with a different gene order',

                        # CDS from upstream to downstream:
                        # DEAD/DEAH box helicase (with 2 YprA domains (YprA is a helicase with a DUF1998 in its C-terminus), but doesn't have a DUF1998 domain)
                        #       (https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_115169834.1&FULL)
                        # DUF1998 domain-containing protein (https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_115169833.1&FULL)
                        # SNF2-related protein (looks like a helicase. two SNF2 domains. https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_115169832.1&FULL)
                        # four N-6 DNA methylase CDSs. the first (WP_115169831.1) is much longer.
                        # site-specific integrase
                        'locus_definition': r'$\mathit{Sphingobacterium}$ $\mathit{spiritivorum}$',
                        'phylum': 'Bacteroidetes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((1269931, 1273395), 'WP_115169831.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_115169831.1
                        'ir_pair_region_with_margins': (1268070, 1271260),
                        'alignment_region': (1248000, 1292000),
                        #     repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0           76         0  1268579  1268654  1270678  1270753  6.450000e-38        2023
                        # 1           25         0  1268941  1268965  1270367  1270391  2.130000e-08        1401
                        # 2           67        15  1269390  1269456  1269795  1269861  1.160000e-06         338
                        # 3           49        12  1268700  1268748  1270584  1270632  1.900000e-01        1835
                        # 4           18         2  1270800  1270817  1271065  1271082  7.100000e-01         247
                        # 5           17         2  1269421  1269437  1270408  1270424  2.700000e+00         970
                        # 6           16         2  1268137  1268152  1269278  1269293  1.000000e+01        1125
                        # 7           16         2  1268524  1268539  1269534  1269549  1.000000e+01         994
                        # 8           16         2  1268690  1268705  1270123  1270138  1.000000e+01        1417
                        # 9           15         2  1269616  1269630  1269687  1269701  3.900000e+01          56
                        # 10          15         2  1270152  1270166  1271206  1271220  3.900000e+01        1039
                        'sra_accession_to_variants_and_reads_info': {
                            'ERR768077': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # 'ERR768077.1.102310.1': {()}, # not perfect, but good enough, i guess?
                                    # 'ERR768077.1.109180.3': {()}, # inversion in the N-terminus of the big hypoth.
                                    # 'ERR768077.1.140547.1': {()}, # not perfect, but good enough, i guess?
                                    # 'ERR768077.1.157800.1': {()}, # inversion in the middle of the big hypoth.
                                    # 'ERR768077.1.17310.1': {()}, # inverted duplication nearby

                                    'ERR768077.1.146166.3': {(1268579, 1268654, 1270678, 1270753)}, # inversion
                                    'ERR768077.1.25308.1': {(1268941, 1268965, 1270367, 1270391)}, # inversion
                                    'ERR768077.1.54217.1': {(1268579, 1268654, 1270678, 1270753)}, # inversion
                                    'ERR768077.1.80667.1': {(1268579, 1268654, 1270678, 1270753)}, # inversion
                                    'ERR768077.1.9194.1': {(1268941, 1268965, 1270367, 1270391)}, # inversion

                                    # Ugh. mauve fails to identify the cassette switch here. I guess mauve is wrong, but for the sake of time, I grudgingly mark this read as a
                                    # not_non_ref_variant. ugh.
                                    # in blast 1268070 in ref aligns to ~3684 in the read, and 1271260 in ref aligns to ~795, which is 2890 bps.
                                    # 'ERR768077.1.21200.1': {(1268579, 1268654, 1270678, 1270753), (1268941, 1268965, 1270367, 1270391)}, # a beautiful cassette switch.



                                    'ERR768077.1.142130.1': {(1268579, 1268654, 1270678, 1270753)}, # inversion
                                    'ERR768077.1.35223.3': {(1268579, 1268654, 1270678, 1270753)}, # inversion
                                },
                                'inaccurate_or_not_beautiful_mauve_alignment': {
                                    'ERR768077.1.146166.3',
                                },
                                'not_non_ref_variant_read_name_to_anomaly_description': {
                                    # I actually think ERR768077.1.21200.1 is perfectly fine, but it is here because mauve failed for it. see above.
                                    'ERR768077.1.21200.1': get_mauve_failed_explanation(
                                        {(1268579, 1268654, 1270678, 1270753), (1268941, 1268965, 1270367, 1270391)}),
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            (1273399, 1276236): 'helicase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_115169832.1&FULL
                            # blastp WP_115169828.1, and many of the top scoring homologs which aren't as short as WP_115169828.1 (e.g., WP_115169831.1)
                            # were annotated as DNA MTases.
                            (1268580, 1269335): 'DNA MTase',
                            # blastp WP_147284221.1, and many of the top scoring homologs which aren't as short as WP_147284221.1 (e.g., WP_078796408.1)
                            # were annotated as DNA MTases.
                            (1269328, 1269624): 'DNA MTase',
                            # blastp WP_115169830.1, and many of the top scoring homologs which aren't as short as WP_115169830.1 (e.g., WP_170148734.1)
                            # were annotated as DNA MTases.
                            (1269621, 1269941): 'DNA MTase',
                        },

                        'describe_in_the_paper': True,
                    },
                ],
                'NZ_UFVQ01000003.1': [ # Chryseobacterium carnipullorum strain NCTC13533
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_UFVQ01000003.1?report=graph&v=2773030:2790172
                        # 'presumably_relevant_cds_region': (2774620, 2790172),
                        'presumably_relevant_cds_region': (2773030, 2790172),
                        # homology claim is based on blasting WP_228426257.1 to WP_052482302.1
                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_228426257.1&FULL
                        'target_gene_product_description': 'Type II restriction-modification enzyme with an SNF2 (COG0553, helicase) domain, '
                                                           'with its C-terminus part homologous to Class 1 DISARM DrmMI',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'A protein containing a short YprA (COG1205, helicase) domain; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_128124891.1&FULL
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_128124892.1&FULL
                            'A protein containing a short YprA (COG1205, helicase) domain and a DUF1998 (pfam09369) domain'
                        ),

                        'locus_description_for_table_3': 'Similar to Class 1 DISARM in terms of some encoded protein domains, but with a different gene order, '
                                                         'and with a single CDS encoding a short YprA (COG1205, helicase) domain and '
                                                         'a DUF1998 (pfam09369) domain',

                        # blasted this region to the region in NZ_UGYW01000002.1, and they seem pretty similar, except for the extra integrase CDSs here.
                        #
                        # CDS from upstream to downstream:
                        # DEAD/DEAH box helicase. has a big YprA domain (YprA is a helicase with a DUF1998 in its C-ternminus), but doesn't have a DUF1998.
                        #     (https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_128124891.1&FULL)
                        # A helicase that contains a DUF1998. has a YprA domain. (https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_128124892.1&FULL)
                        # three N-6 DNA methylase CDSs. the first (WP_228426257.1) is much longer, and is annotated as SNF2-related protein, as it has a SNF2 C-terminus domain.
                        #     ()
                        # three site-specific integrase CDSs, but each in the opposite strand than its nearest neighbor or neighbors (i.e., not a RIT).
                        'locus_definition': r'$\mathit{Chryseobacterium}$ $\mathit{carnipullorum}$',
                        'phylum': 'Bacteroidetes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((2778297, 2784881), 'WP_228426257.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_228426257.1
                        'ir_pair_region_with_margins': (2783281, 2786471),
                        'alignment_region': (2763000, 2807000),
                        #     repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0           26         0  2784186  2784211  2785582  2785607  5.610000e-09        1370
                        # 1           58        13  2783974  2784031  2785721  2785778  6.320000e-05        1689
                        # 2           17         0  2783781  2783797  2785955  2785971  1.000000e-03        2157
                        # 3           26         4  2784345  2784370  2785423  2785448  4.900000e-02        1052
                        # 4           20         2  2785829  2785848  2786119  2786138  4.900000e-02         270
                        # 5           16         1  2785018  2785033  2785082  2785097  1.900000e-01          48
                        # 6           16         2  2784599  2784614  2785855  2785870  1.000000e+01        1240
                        # 7           16         2  2785257  2785272  2786405  2786420  1.000000e+01        1132
                        # 8           18         3  2783916  2783933  2785355  2785372  3.900000e+01        1421
                        # 9           39        10  2783938  2783976  2785085  2785123  3.900000e+01        1108
                        # 10          23         5  2784407  2784429  2785364  2785386  1.470000e+02         934
                        'sra_accession_to_variants_and_reads_info': {

                            'ERR832413': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'ERR832413.1.138204.1': {(2784186, 2784211, 2785582, 2785607)},
                                    'ERR832413.1.157662.1': {(2784186, 2784211, 2785582, 2785607)},
                                    'ERR832413.1.157662.3': {(2784186, 2784211, 2785582, 2785607)},
                                    'ERR832413.1.157662.5': {(2784186, 2784211, 2785582, 2785607)},
                                    'ERR832413.1.157662.7': {(2784186, 2784211, 2785582, 2785607)},
                                    'ERR832413.1.157662.9': {(2784186, 2784211, 2785582, 2785607)},
                                    'ERR832413.1.120455.1': {(2784186, 2784211, 2785582, 2785607)},
                                }
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            (2778297, 2784881): 'DNA MTase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_228426257.1&FULL
                            (2774620, 2778258): 'helicase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_128124892.1&FULL
                            # blastp WP_128124893.1, and many of the top scoring homologs which aren't as short as WP_128124893.1 (e.g., WP_170148734.1)
                            # were annotated as DNA MTases.
                            (2784939, 2785532): 'DNA MTase',
                            # blastp WP_128124894.1, and many of the top scoring homologs which aren't as short as WP_128124894.1 (e.g., WP_035593149.1)
                            # were annotated as DNA MTases.
                            (2785666, 2785971): 'DNA MTase',
                        },

                        'describe_in_the_paper': True,
                    },
                ],
            },

            'MTase: Class 1 DISARM': {
                'NZ_CP010519.1': [ # Streptomyces albus strain DSM 41398
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP010519.1?report=graph&v=3705033:3720540
                        # 'presumably_relevant_cds_region': (3706964, 3720540), # start at DrmA
                        'presumably_relevant_cds_region': (3705033, 3720540), # start at DUF1998

                        # blastp WP_052482302.1 to WP_198036332.1 (the drmMI gene mentioned in Aparicio-Maldonado, Cristian, et al.
                        # "Class I DISARM provides anti-phage and anti-conjugation activity by unmethylated DNA recognition." bioRxiv (2021))
                        # results in an alignment with evalue=0, from aa 7 to 1339 out of 1359 (the length of WP_198036332.1)
                        'target_gene_product_description': 'Class 1 DISARM DrmMI',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_052482303.1&FULL
                            'DrmD, which is a SNF2 (COG0553, helicase) domain-containing protein'
                        ),

                        'presumably_associated_downstream_gene_product_descriptions': (
                            'DrmA; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_107071315.1&FULL
                            'DUF1998 (pfam09369) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_040250235.1&FULL
                            'DrmC'
                        ),

                        'locus_description_for_table_3': 'Class 1 DISARM',

                        'locus_definition': r'$\mathit{Streptomyces}$ $\mathit{albus}$',
                        'phylum': 'Actinobacteria',
                        'longest_linked_repeat_cds_region_and_protein_id': ((3712964, 3717292), 'WP_052482302.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_052482302.1

                        'locus_description': 'Class 1 DISARM',
                        # 'ir_pair_region_with_margins': (3711451, 3714486),
                        # 'ir_pair_region_with_margins': (3711901, 3714036), # mauve failed for this one...
                        'ir_pair_region_with_margins': (3711901, 3715000),
                        'alignment_region': (3691000, 3735000),
                        'sra_accession_to_variants_and_reads_info': {
                            # 'ERR1055237': { # nothing here. (only two reads aligned to the ir pair region with margins, both matching ref, but one with inverted dup.
                            # 'ERR1055236': { # nothing here. (given the thresholds of all covered etc.)
                            # 'ERR1055225': { # nothing here. (given the thresholds of all covered etc.)
                            # 'ERR1055232': { # nothing here. (given the thresholds of all covered etc.)
                            # 'ERR2125767': { # nothing here. (given the thresholds of all covered etc.)
                            'ERR1055234': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # This read is not-bad evidence, I think. If we assume there is no inverted duplication here, then simplest thing seems to be that the genome
                                    # that was sequenced had the region flanked by the following repeat inverted.
                                    # here is the lines of the read from the fastq file extracted using fastq-dump
                                    # (IMPORTANT: this is not the same thing you would find in https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR1055234, which contains,
                                    # IIUC, a more raw (i.e., unprocessed) version of the read):
                                    # >ERR1055234.1.139400.1 length=3119
                                    # GAAGTATCGCCATCGTCGAATAGAACGTGAAACCAGAACGACTCAAGAAACACCACCAGTCAACGATCGCCGAATCTGGG
                                    # CGGTCAACATGACGACCCCCAATGGACCTTACTTGTGCCACCCGACCTTGGACGCGATTCGTATGTTCTGGTGTATCGCT
                                    # GTCTATCCCCGAACGTCTCATTTTGTTGTCCACGTCGTGTACACTCAGCCGCCAAGTCCTTCTCAATAAATTGGGAGTCT
                                    # TTCGGCTACGGCCCGAGGTAGCGACCTCGCCCTACTCTCGAGTATTCCACTCGATTTGGCCAGGCGCAATTCTAGCGAAT
                                    # CCGATGAAGGGGATCTCAATTCTACTCTCCGCGTGCCGAGGCTGTGTGACTGAGACATCCCTATCACCCAGGTCACCAAC
                                    # TGCGAAGATCGAAACACATGGGGAAATTTCTCGACAGGTGAGATTTTGATAGGTAATGATTGACAGATACGTAGAACCAC
                                    # AAGGCTATAGCAACATGGTCCAGACCTCCGTGGAAGGAATATGACATCGAGTCCTGCGCGAGAACCAACCCGAAAAAGAT
                                    # CCGGCCCGATCTGTACTTGAGTCCTACAATGGGCTAGCACCGCGACCTCGCTCACGTATACTATTCCAAATTCGACAAAG
                                    # TACGCGATTCGCGTCGCTGCATCGCTTCTCAGATTGAACAATCCTAGACAAGTCTTCCTTTGAACTAACCATGCTCAGTC
                                    # AGAAAAAACTCACCAAAATTCTCCTCAAACGCAATGGACCGCGTGGCGATTGCGCAGAACGGCATTCATATCAGGAACAC
                                    # CCTAACTCCGGGACTTCACGGGAACCTTGTTCTGAGAATTACGTCAAACAAGTATCCCGGGAACTGCGACGACTCCCTTC
                                    # GACGTCGCGCATGGTGTTGCACGCAGTCGTTACATAAGGCTTATTTACCGCGACGCTTCTCGACTTGAAAGCGCGTAAGT
                                    # TGAGACTCAAGAAGTCCCTGTCCAGGATCGCCATTTTGCAACCACTGAGGGCGACCGGTGACAGCGGACCGGCCATTCGA
                                    # GTTTCATGAACCCATGCCGGAGTCAAGGCGTCGCCGACCATGCGTTAAGCCTTCAGCCCCCGCCTCATCGACACTTGTAT
                                    # GGGCACCACGCGAGCTACTTATAAATCGCTAAGCGAAACCTGTGGATTATGACGCAATTATAAAGCTCGTAGAGGCCCGC
                                    # ACCTCAGTGTGCACTCGGTTCCGCCATGATTGCAAAGTTCGGTTCGTATTAGTCTTCCGGAAATTCCCAAGAAGAGAGTG
                                    # AGATTAGCGGCATGGAAATGTTCAAAGCACACGACTCGGAGTTTACCGAAGGTCAGATTCAAACTCCGCTCGAATATTTC
                                    # CAATCCCATTAATACGTGAATAACTTGATCAGCAGGGCCATGTACCAACACGGTCACTGGCAAAAAGAACGATCTTCCGT
                                    # TCAGACTTGACCACAGGCACGAGTCGCATGGCTTCGCTCACGCCGCTGCTATCCACCAACACCGACGAGGATCCCCGCCA
                                    # ATCGCCCGTATAGCCAGGACGGCTCATCTCTGTAGAGGCCACCGAGCTTCTCGGGCATTCTTGCTGTTGTAACGCGAAGG
                                    # CAGATCGGTTCACCTCTTTCGACGATGGCCGAATACTTCGGTACCCGCCCGCTTCCTCGCCCAAGTCATGGAAGTTGACA
                                    # GCCAGCGCTTGGCTGAGCAGTCTGGCGGAGTTAAGTCCTCCGCCGTGAGAGTTAGAGGGAAACGAGACGTCTGTTCGAAG
                                    # GTACGCTGATGCAGCGTGGGCCTGTTCGGGATGGATGATAATCCCTTCCTCAAGCACGTGGTACCCCTGGAACGACTGGC
                                    # CCCTATGTTCCTACAGACGGACGGCCGCCCCGAAACGCGGACCGCGATCAACGACGAATGATACACAGGACCGACGACGT
                                    # CGTCCTAGTACCCGAAGCTGGCTTCCTGCACCCAATACTCGAGCTTCGTGTCCACACAGCGCAGTACTCCCAGCGCCGCA
                                    # ACCGCGAAGGCCAGCGAGCTCATCCTTTTAACTGCTGCGGACGTACATCAAAATGCCGTCGCAAGTAGCTTCGTCCACCC
                                    # GACCTCCCGTGCGCTTTGGGCAACGGTGGGTACGATCCAAGCCCTCTGGCGCGCTCGCTGAGCAACCGGTGGACCCGAGC
                                    # TCAAGTAGCGACGAGGTCCGCGCTACCTATCGCTGTCTCGGCCCACCGGCACGAGGTACTCGCGATGCCTGCCCGAGCGC
                                    # TACCTACCAGTTTCTGGCCCACCGAGAAAGGAGGGGTTACCGATGATCGCAATTAGAACCGCCCTGGTCCGGCCGAAGAC
                                    # TTCGGGGAAACGAGGGCGCTCAGTGCACGGGAGGCAACGAAGCCCTGTGAGGGAAAACCTGCTCAGCCAGATCCAGGGCG
                                    # CGCGCTTTCGCGCAATCGGAATCAACGATCCGAGCCGCGGACCGCACGCGAGCAATCATCTCCGCCAGTCCGCACCTGGA
                                    # GTAAGGCTAAGCGTTGCACGAGTCCCAGCCGCTGCAGAAGCCAGGCACCCCGTCGCGATGTCGCGACAGCCTCAGGCGTG
                                    # TGCCATTGGTGTCGTTGTGCCTACGCGAGTGCGACCGCTTCATGAGGCGGCGATGTGTCGCTGTTATCCGCGATGCCGCC
                                    # GCTTTCGCGGCACCTGGACGCGAGCCGCCATCCGCTCGTCACCCCTCATCGAAAGGCCCTTGGGGTGGATTTCGGCCCGC
                                    # CTTCCGATCGAGATGCACACGCTCGATCTGTCCAATTGGCATCCCCAGCAACGAATCCCGCCCACCTAACCGTCGTCCAG
                                    # AAGGTCAAGGCCGTTCCGCGTCCATGAGACAGCCACAAGACACTGCCACTCTCCACGGCCATAGGATTGATGTCGACCCC
                                    # ATAAGGCATGCTCGATGATCAGGTGCCCGGCGCGGGACCATCCACGGTCGCTTTCCGCGGCCCGCCGCCTTCATGGCGCC
                                    # CGCTGGCCGCGTGGCTGCCGCGGAAGGCTAGTGGCCTCCTACGTCTCCGGCCTCGCACCCCAGGCGTCGACCAACGGTCC
                                    # CGAAGGCTAGCAGCGGCCGACGAGAAGCGCCGAACCCGCACGCGATGTCGGCGACCTTGAAGCTTCAAAATGCATCACG
                                    #
                                    # blasting this read to NZ_CP010519.1 shows very clearly that it aligns to our region of interest, as we show in the paper
                                    'ERR1055234.1.139400.1': {(3711951, 3712216, 3713721, 3713986)}, # inversion
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'rna_seq_sra_accessions': {
                            'SRR10883644',
                            'SRR10883645',
                            # 'SRR10883646', # not a control
                            # 'SRR10883647', # not a control
                        },

                        'cds_start_and_end_to_curated_product_class': {
                            (3712964, 3717292): 'DNA MTase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_052482302.1&FULL

                            # blastp WP_159393001.1, and the non-hypothetical results (out of the top 100)
                            # were annotated as 'N-6 DNA methylase', 'SAM-dependent DNA methyltransferase' and 'restriction enzyme and modification methylase'.
                            (3711936, 3712955): 'DNA MTase',
                        },

                        'describe_in_the_paper': True,
                    },
                ],
            },

            'MTase: Class 1 DISARM-like': {
                'NZ_CP061344.1': [ # Microbacterium hominis strain 01094
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP061344.1?report=graph&v=2376723:2392871
                        # 'presumably_relevant_cds_region': (2376723, 2392871), # without WP_191008233.1, just because it is shorter...
                        'presumably_relevant_cds_region': (2374520, 2392871), # with WP_191008233.1, the 'AAA family ATPase' helicase

                        # homology claim is based on blasting WP_191009655.1 to WP_052482302.1 (two alignments with evalue=1e-20 and 0.002,
                        # and of lengths 546 and 27, respectively.
                        # Type II restriction-modification enzyme claim is based on http://rebase.neb.com/cgi-bin/seqget?Mho1094ORF11160P (and homology to DrmMI)
                        'target_gene_product_description': 'Type II restriction-modification enzyme, partially homologous to Class 1 DISARM DrmMI',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            # actually, we have pretty nearby upstream a YprA domain-containing protein and a DUF1998 domain-containing protein...
                            # but they are on the opposite strand...

                            # blastp WP_191008237.1 to WP_052482303.1 (the DrmA encoded in NZ_CP010519.1) gives an alignment with evalue=4e-57
                            # from aa 112 to 560 out of 945 (the length of WP_191008237.1).
                            'SNF2 (COG0553, helicase) domain-containing protein' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_191008237.1&FULL
                        ),

                        'presumably_associated_downstream_gene_product_descriptions': (
                            # blastp WP_191008234.1 to WP_107071315.1 (the DrmA encoded in NZ_CP010519.1) gives a 40bp alignment with evalue=2e-4
                            # blastp WP_191008234.1 to WP_040250235.1 (the DUF1998 domain-containing protein encoded in NZ_CP010519.1) resulted in no alignments

                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_191008234.1&FULL
                            'A protein containing a long YprA (COG1205, helicase) domain and a DUF1998 (pfam09369) domain; '
                            'UvrD (COG0210, helicase) domain-containing protein' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_191008233.1&FULL
                        ),

                        'locus_description_for_table_3': 'Similar to Class 1 DISARM in terms of some encoded protein domains and gene order, '
                                                         'and with a single CDS encoding a long YprA (COG1205, helicase) domain and '
                                                         'a DUF1998 (pfam09369) domain',

                        # 'locus_definition': r'$\mathit{Microbacterium}$ $\mathit{hominis}$ strain 01094',
                        'locus_definition': r'$\mathit{Microbacterium}$ $\mathit{hominis}$',
                        'phylum': 'Actinobacteria',
                        'longest_linked_repeat_cds_region_and_protein_id': ((2385445, 2390031), 'WP_191009655.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_191009655.1
                        # One of the paperBLAST results for WP_191009655.1 was:
                        # A Comparative Study of the Outer Membrane Proteome from an Atypical and a Typical Enteropathogenic -
                        # https://openmicrobiologyjournal.com/VOLUME/5/PAGE/83/FULLTEXT/ (2011), which mentioned ECs5262, i.e., NP_313289.1, from
                        # https://ncbi.nlm.nih.gov/nuccore/NC_002695.2?report=graph, which is "Escherichia coli O157:H7 str. Sakai". A pretty famous fellow, i think.
                        # Anyway, it is a hypoth in Sakai (though blastp (of NP_313289.1) results in many homologs annotated as
                        # 'class I SAM-dependent DNA methyltransferase', and there are two upstream helicases, that seem like the two upstream helicases
                        # in 'MTase: two upstream helicases' (with regard to conserved domains).
                        # This sounds important, i guess? There is also right downstream another helicase with DUF1998, and downstream to that, a UvrD helicase.
                        #
                        # blastp of WP_191009655.1 to NP_313289.1 results in an alignment with evalue=0.
                        # the alignment is to aa 159 to 1468 out of 1528 (the length of WP_191009655.1).


                        # the upstream helicase is indeed a SNF2 helicase, but there aren't any drmD homologs in the first 100 hits of blastp.
                        # the downstream helicase has a DUF1998 close to its C-terminus, but there aren't any drmA homologs in the first 100 hits of blastp.
                        # but it seems like we don't have a PLD here.
                        # the MTase has an identified PglX region, which sounds more like DISARM, i guess, but this region is pretty small relative to the CDS length.
                        # there is also a cytosine methyltransferase a bit downstream, reminiscent of Class 2 DISARM.
                        # there is also another DUF1998 CDS a bit upstream, but on the other strand...
                        # overall, maybe the cds context name should be 'Class 1 DISARM-like', rather than 'Class 1 DISARM'

                        # blasted WP_191008237.1 (the helicase upstream to the MTase) to QHN28767.1 (which is homologous to DISARM DrmD), and got a 51% coverage and 28.5% identity.
                        # blasted WP_191008234.1 (the helicase downstream to the MTase) to QHN27295.1 (which is homologous to DISARM DrmA), and got a 8% coverage and 32.7% identity.


                        #    repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0         642         5  2384162  2384803  2386093  2386734  0.000000e+00        1289
                        # 1          59         3  2383968  2384026  2386870  2386928  1.190000e-22        2843
                        # 2          51         9  2385185  2385235  2385667  2385717  1.330000e-07         431
                        # 3          32         3  2384051  2384082  2386814  2386845  5.060000e-07        2731
                        # 4          34         6  2384912  2384945  2385957  2385990  6.000000e-03        1011
                        # 5          20         2  2384825  2384844  2386052  2386071  8.200000e-02        1207
                        # 6          20         3  2386644  2386663  2386974  2386993  4.500000e+00         310
                        # do we have another merging, this time blue green and cyan?
                        # 2384162  2384803  2386093  2386734 - blue
                        # 2384051  2384082  2386814  2386845 - cyan
                        # 2383968  2384026  2386870  2386928 - green
                        # ok. cyan and green look like they can be merged. yes. also blue and cyan. so let's make the merged of their three:
                        # (2383968, 2384803, 2386093, 2386928)
                        # oops. it went also to gold. let's see if we can merge it also:
                        # 2384825  2384844  2386052  2386071. indeed. so let's merge all four:
                        # (2383968, 2384844, 2386052, 2386928)
                        # checked, and pink can't be merged with gold. alright then.
                        #
                        'ir_pair_region_with_margins': (2383400, 2387500),
                        # maybe it makes sense to have this one while 'MTase: Class 1 DISARM' has until DUF1998, because here DUF1998 is in the downstream helicase...
                        'alignment_region': (2363000, 2408000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR12502384': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR12502384.1.100286.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.10132.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.103152.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. only one side reaches the gold one, while the other stops in blue.
                                    'SRR12502384.1.112026.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. only one side reaches the gold one, while the other stops in blue.
                                    'SRR12502384.1.129052.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. no side has both alignments overlapping gold, and for the outer overlaps there is asymmetry. ugh.
                                    'SRR12502384.1.134135.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. perfect green to blue, i.e., (2383968, 2384803, 2386093, 2386928)
                                    'SRR12502384.1.134464.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.135824.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. perfect green to blue, i.e., (2383968, 2384803, 2386093, 2386928)
                                    'SRR12502384.1.13759.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. not perfect
                                    'SRR12502384.1.17787.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    # 'SRR12502384.1.22303.1' # unclear. maybe this is nothing?
                                    'SRR12502384.1.26384.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR12502384.1.33168.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. not perfect

                                    'SRR12502384.1.36256.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.36675.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. not perfect
                                    'SRR12502384.1.38789.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.48188.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.49992.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.53674.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion
                                    'SRR12502384.1.60626.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. not perfect
                                    # 'SRR12502384.1.60979.1' # unclear. maybe this is nothing?
                                    'SRR12502384.1.64485.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. perfect green to blue, i.e., (2383968, 2384803, 2386093, 2386928)
                                    'SRR12502384.1.89802.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. perfect green to blue, i.e., (2383968, 2384803, 2386093, 2386928)


                                    # added late (on 220304)
                                    'SRR12502384.1.59616.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. perfect green to blue, i.e., (2383968, 2384803, 2386093, 2386928)
                                    'SRR12502384.1.78497.1': {(2383968, 2384844, 2386052, 2386928)}, # inversion. ugh. perfect green to blue, i.e., (2383968, 2384803, 2386093, 2386928)



                                    'SRR12502384.1.110525.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.115429.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.115626.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.119356.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.127875.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.16231.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.17164.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.20447.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.20721.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.29937.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.32271.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.35093.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.48182.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.52519.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.59837.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.813.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.86589.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.87269.1': {(2383968, 2384844, 2386052, 2386928)},
                                    'SRR12502384.1.9287.1': {(2383968, 2384844, 2386052, 2386928)},
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            # blastp WP_191008236.1, and the top scoring homologs which aren't as short as WP_191008236.1 (e.g., WP_219879556.1)
                            # were annotated as 'class I SAM-dependent DNA methyltransferase'
                            (2383934, 2385463): 'DNA MTase',

                            (2374520, 2376721): 'helicase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_191008233.1&FULL
                        },

                        'describe_in_the_paper': True,
                    },
                ],
                # 'CP022685.1': [ # Streptomyces formicae strain KY5
                #             'SRR6029562': {
                # 'AP022565.1': [ # Mycolicibacterium alvei JCM 12272
                #             'DRR161258': {
                # 'CP045806.1': [ # Gordonia pseudoamarae strain BEN371
                #             'SRR11613597': {
                # 'CP045809.1': [ # Gordonia pseudoamarae strain CON9
                #             'SRR11613601': {
            },


            'MTase: solitary': {
                # 'CP068086.1': [ # Sphingobacterium multivorum strain FDAARGOS_1143
                #     {
                #         # QQT44630.1 has no identified conserved domains, but blastp reveals it is homologous to
                #         # 'Eco57I restriction-modification methylase domain-containing protein' and 'N-6 DNA methylase' CDSs, and even one Pglx CDS.
                #         'ir_pair_region_with_margins': (5391262, 5394744),
                #         'alignment_region': (5371000, 5415000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR12938600': {
                #                 # ugh. seems like there is some mistake in the assembly...
                #                 # (double checked, and it doesn't seem like there is something especially interesting in this presumably wrong (or not matching) assembly)
                #                 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                #                     'SRR12938600.1.111879.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.111880.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.122789.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.122790.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.122791.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.122792.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.137845.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.137846.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.137847.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.137848.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.137849.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.151663.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.156024.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.160099.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.169179.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.179241.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.186791.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.192112.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.25157.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.25158.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.43853.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.43854.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.43855.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.4454.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.4455.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.4456.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.50281.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.50282.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.50976.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.50977.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.5354.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.5355.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.68406.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.68407.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.81153.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.95512.1', # inversion, though not the whole read is aligned
                #                     'SRR12938600.1.95513.1', # inversion, though not the whole read is aligned
                #                 },
                #             },
                #         },
                #         'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
                'NZ_CP082886.1': [ # Bacteroides nordii strain FDAARGOS_1461
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP082886.1?report=graph&v=2881747:2887030
                        'presumably_relevant_cds_region': (2881747, 2887030),

                        # homology claim is based on blasting WP_223381307.1 to YP_002343503.1 (Cj0031 of C. jejuni)
                        'target_gene_product_description': 'Type II restriction-modification enzyme, homologous to Cj0031 of C. jejuni',

                        'locus_description_for_table_3': 'Solitary Type II restriction-modification CDS',

                        # 'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{nordii}$ strain FDAARGOS_1461',
                        'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{nordii}$',
                        'phylum': 'Bacteroidetes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((2883305, 2887030), 'WP_223381307.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_223381307.1
                        #    repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0         208        37  2882543  2882750  2883864  2884071  8.010000e-50        1113
                        # 1          98         7  2882653  2882750  2883864  2883961  3.020000e-38        1113
                        # 2          38         2  2882543  2882580  2884034  2884071  3.350000e-12        1453
                        # 3          27         0  2882901  2882927  2883675  2883701  2.620000e-09         747
                        # 4          14         0  2882448  2882461  2884156  2884169  8.800000e-02        1694
                        # 5          15         1  2881824  2881838  2882386  2882400  1.300000e+00         547
                        # 6          29         6  2882787  2882815  2883811  2883839  4.800000e+00         995
                        # 7          14         1  2883649  2883662  2884289  2884302  4.800000e+00         626
                        # 8          14         1  2883706  2883719  2885780  2885793  4.800000e+00        2060

                        # 2882448  2882461  2884156  2884169
                        # 2882543  2882750  2883864  2884071
                        # 'ir_pair_region_with_margins': (2881800, 2886044),
                        'ir_pair_region_with_margins': (2881800, 2884800),
                        'alignment_region': (2861000, 2907000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR16259013': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # 'SRR16259013.1.117966.1', # inversion, but the alignment is not symmetric, which is a bit weird. doesn't span the whole region.
                                    # 'SRR16259013.1.117967.1', # inversion, but the alignment is not symmetric, which is a bit weird. doesn't span the whole region.
                                    # 'SRR16259013.1.117968.1', # inversion. doesn't span the whole region.
                                    # 'SRR16259013.1.117969.1', # inversion, but the alignment is not symmetric, which is a bit weird. doesn't span the whole region.
                                    # 'SRR16259013.1.127913.1', # inversion. doesn't cover the whole region.
                                    'SRR16259013.1.127914.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.127915.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.127916.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.129943.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    # 'SRR16259013.1.172699.1', # inversion. doesn't span the whole region.
                                    # 'SRR16259013.1.178468.1', # inversion. doesn't cover the whole region.
                                    'SRR16259013.1.18510.1': {(2882901, 2882927, 2883675, 2883701)}, # inversion
                                    'SRR16259013.1.18511.1': {(2882901, 2882927, 2883675, 2883701)}, # inversion

                                    'SRR16259013.1.2205.1': {(2882901, 2882927, 2883675, 2883701)}, # inversion
                                    'SRR16259013.1.2206.1': {(2882901, 2882927, 2883675, 2883701)}, # inversion
                                    'SRR16259013.1.45972.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.45973.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.45975.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.46415.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion. not perfect.
                                    'SRR16259013.1.6939.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    # 'SRR16259013.1.6946.1', # inversion. doesn't cover the whole region.
                                    'SRR16259013.1.83930.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.83931.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion. not perfect

                                    # added late (on 220304)
                                    'SRR16259013.1.117964.1': {(2882448, 2882461, 2884156, 2884169)}, # inversion
                                    'SRR16259013.1.117965.1': {(2882448, 2882461, 2884156, 2884169)}, # inversion
                                    'SRR16259013.1.117966.1': {(2882448, 2882461, 2884156, 2884169)}, # inversion
                                    'SRR16259013.1.117967.1': {(2882448, 2882461, 2884156, 2884169)}, # inversion
                                    'SRR16259013.1.117968.1': {(2882448, 2882461, 2884156, 2884169)}, # inversion
                                    'SRR16259013.1.127913.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.172699.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion
                                    'SRR16259013.1.6946.1': {(2882543, 2882750, 2883864, 2884071)}, # inversion

                                    'SRR16259013.1.103896.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.103897.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.103898.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.178467.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.178470.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.200467.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.200468.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.200469.1': {(2882543, 2882750, 2883864, 2884071)},

                                    # Ugh. mauve fails to identify the cassette switch here. I guess mauve is wrong, but for the sake of time,
                                    # I grudgingly mark these reads as not_non_ref_variant. ugh. (can't just mark it as
                                    # 'inaccurate_or_not_beautiful_mauve_alignment', because it is the only read of that variant...)
                                    # 'SRR16259013.1.202032.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},
                                    # 'SRR16259013.1.171471.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},
                                    # 'SRR16259013.1.202026.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},
                                    # 'SRR16259013.1.202027.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},
                                    # 'SRR16259013.1.202028.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},
                                    # 'SRR16259013.1.202029.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},
                                    # 'SRR16259013.1.202030.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},
                                    # 'SRR16259013.1.202031.1': {(2882543, 2882750, 2883864, 2884071), (2882901, 2882927, 2883675, 2883701)},

                                    'SRR16259013.1.206127.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.206128.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.259901.1': {(2882901, 2882927, 2883675, 2883701)},
                                    'SRR16259013.1.40147.1': {(2882901, 2882927, 2883675, 2883701)},
                                    'SRR16259013.1.40148.1': {(2882901, 2882927, 2883675, 2883701)},
                                    'SRR16259013.1.40149.1': {(2882901, 2882927, 2883675, 2883701)},
                                    'SRR16259013.1.45970.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.45971.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.45974.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.45976.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.6938.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.6941.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.6942.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.6943.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.6945.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.6947.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.6948.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.83925.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.83926.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.83927.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.83928.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.83929.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.83932.1': {(2882543, 2882750, 2883864, 2884071)},
                                    'SRR16259013.1.93101.1': {(2882901, 2882927, 2883675, 2883701)},
                                    'SRR16259013.1.97331.1': {(2882543, 2882750, 2883864, 2884071)},
                                },
                                'not_non_ref_variant_read_name_to_anomaly_description': {
                                    # I guess these reads are perfectly fine, but they are here because mauve failed for them. see above.
                                    'SRR16259013.1.202032.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                    'SRR16259013.1.171471.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                    'SRR16259013.1.202026.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                    'SRR16259013.1.202027.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                    'SRR16259013.1.202028.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                    'SRR16259013.1.202029.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                    'SRR16259013.1.202030.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                    'SRR16259013.1.202031.1': TYPE_II_RM_SRR16259013_NOT_NON_REF_EXPLANATION,
                                },
                                # 'inaccurate_or_not_beautiful_mauve_alignment': {
                                #     'SRR16259013.1.171471.1',
                                #     'SRR16259013.1.202026.1',
                                #     'SRR16259013.1.202027.1',
                                #     'SRR16259013.1.202028.1',
                                #     'SRR16259013.1.202029.1',
                                #     'SRR16259013.1.202030.1',
                                #     'SRR16259013.1.202031.1',
                                # },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        # 'cds_start_and_end_to_curated_product_class': {
                        #     # blastp WP_223381306.1, and the top scoring homologs which aren't as short as WP_223381306.1 (e.g., WP_071057709.1)
                        #     # were annotated as 'Eco57I restriction-modification methylase domain-containing protein'
                        #     # (2882473, 2883321): 'DNA MTase'
                        # },

                        'describe_in_the_paper': True,
                    },
                ],
            },
            'MTase: upstream DUF1016': {
                # 'NZ_QENX01000001.1': [ # Flavobacterium sp. 103 Ga0213544_11 # maybe that was 'MTase: solitary'. no evidence there, so no reason to check.
                #             'SRR6976419': {
                'CP033760.1': [ # Chryseobacterium indologenes strain FDAARGOS_537
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP033760.1?report=graph&v=4018068:4024437
                        'presumably_relevant_cds_region': (4018068, 4024437),

                        # homology claim is based on blasting AYY86337.1 to YP_002343503.1 (Cj0031 of C. jejuni)
                        'target_gene_product_description': 'Type II restriction-modification enzyme, homologous to Cj0031 of C. jejuni',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'DUF1016 (pfam06250) domain-containing protein' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=AYY87217.1&FULL
                        ),

                        'locus_description_for_table_3': 'Type II restriction-modification CDS with an immediately upstream CDS encoding a '
                                                         'DUF1016 (pfam06250) domain',

                        # 'locus_definition': r'$\mathit{Chryseobacterium}$ $\mathit{indologenes}$ strain FDAARGOS_537',
                        'locus_definition': r'$\mathit{Chryseobacterium}$ $\mathit{indologenes}$',
                        'phylum': 'Bacteroidetes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((4019614, 4023351), 'AYY86337.1'),
                        # https://ncbi.nlm.nih.gov/protein/AYY86337.1
                        #    repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0          24         0  4018753  4018776  4020474  4020497  5.990000e-08        1697
                        # 1          29         5  4018827  4018855  4020395  4020423  3.700000e-02        1539
                        # 2          16         1  4019448  4019463  4019730  4019745  1.400000e-01         266
                        # 3          19         2  4019956  4019974  4020303  4020321  1.400000e-01         328
                        # 4          21         3  4018920  4018940  4020310  4020330  5.300000e-01        1369
                        # 5          15         1  4018985  4018999  4020251  4020265  5.300000e-01        1251
                        # 6          15         1  4019931  4019945  4020518  4020532  5.300000e-01         572
                        # 7          75        21  4018985  4019059  4020191  4020265  5.300000e-01        1131
                        # 8          17         2  4018758  4018774  4020008  4020024  2.000000e+00        1233
                        'ir_pair_region_with_margins': (4018250, 4021000),
                        'alignment_region': (3998000, 4041000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR8180785': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # 'SRR8180785.1.129585.1', # inversion between IR pairs i haven't listed. doesn't cover the whole region
                                    # 'SRR8180785.1.115221.1', # weak evidence for inversion at the N-terminus of the shorter MTase. doesn't span the whole region
                                    # 'SRR8180785.1.130001.1', # weak evidence for inversion at the N-terminus of the shorter MTase. doesn't span the whole region
                                    # 'SRR8180785.1.13209.1', # weak evidence for inversion at the N-terminus of the shorter MTase. doesn't span the whole region
                                    # 'SRR8180785.1.134632.1', # a cassette switch, though alignments aren't perfect (with regard to read coverage). doesn't cover the whole region
                                    'SRR8180785.1.11857.1': {(4018753, 4018776, 4020474, 4020497)}, # inversion
                                    'SRR8180785.1.139353.1': {(4018753, 4018776, 4020474, 4020497)}, # inversion
                                    'SRR8180785.1.141015.1': {(4018985, 4018999, 4020251, 4020265)}, # inversion
                                    'SRR8180785.1.141131.1': {(4019216, 4019229, 4020015, 4020028)}, # inversion
                                    # 'SRR8180785.1.141131.5', # inversion between IR pairs i haven't listed. doesn't span the whole region
                                    'SRR8180785.1.141787.1': {(4018985, 4018999, 4020251, 4020265)}, # inversion
                                    'SRR8180785.1.141787.3': {(4018985, 4018999, 4020251, 4020265)}, # inversion
                                    'SRR8180785.1.149429.1': {(4018753, 4018776, 4020474, 4020497)}, # inversion
                                    # 'SRR8180785.1.151100.1', # inversion at the N-terminus of the shorter MTase. doesn't cover the whole region
                                    'SRR8180785.1.159135.1': {(4019216, 4019229, 4020015, 4020028)}, # inversion
                                    # 'SRR8180785.1.47326.1', # inversion at the N-terminus of the shorter MTase. doesn't cover the whole region
                                    'SRR8180785.1.67355.1': {(4018753, 4018776, 4020474, 4020497)}, # inversion
                                    'SRR8180785.1.83445.5': {(4019216, 4019229, 4020015, 4020028)}, # inversion
                                    # 'SRR8180785.1.87337.1', # inversion between IR pairs i haven't listed. doesn't span the whole region
                                    'SRR8180785.1.87570.1': {(4018753, 4018776, 4020474, 4020497)}, # inversion
                                    'SRR8180785.1.90637.1': {(4018985, 4018999, 4020251, 4020265)}, # inversion
                                    'SRR8180785.1.94990.1': {(4018753, 4018776, 4020474, 4020497)}, # inversion


                                    # added late (on 220304)
                                    'SRR8180785.1.87337.1': {(4018985, 4018999, 4020251, 4020265)}, # inversion


                                    'SRR8180785.1.11346.1': {(4018985, 4018999, 4020251, 4020265)},
                                    'SRR8180785.1.44230.1': {(4019216, 4019229, 4020015, 4020028)},
                                },
                                'not_non_ref_variant_read_name_to_anomaly_description': {
                                    'SRR8180785.1.71140.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,
                                }
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        # 'cds_start_and_end_to_curated_product_class': {
                        #     # (4018068, 4018664): 'recombinase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=AYY86335.1&FULL
                        # },


                        'describe_in_the_paper': True,
                    },
                ],
            },

            'MTase: upstream PLD&SNF2 helicase': {
                # unlike in DrmD (IIUC) and in the YprA-YprA-DUF1998-HepA system, in the helicase here the identified HepA domain doesn't contain the SNF2 C-terminus domain.
                # i guess the identified HepA domain contains the SNF2 N-terminus domain, though here, for example, the SNF2 domain wasn't identified:
                # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_004312793.1&FULL
                # here it is as i guessed it should be (with the HepA overlapping only the N-terminus SNF2 domain):
                # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_063968665.1&FULL

                # 'NZ_CP012938.1': [ # Bacteroides ovatus strain ATCC 8483
                #             # 'SRR2637631': {
                #             # 'SRR2637689': {
                'CP083813.1': [ # Bacteroides ovatus strain FDAARGOS_1516
                    # this is a nice 'standard' one:
                    # UBN58240.1 is the PLD and SNF2 helicase, then we have the usual big MTase-small MTase-recombinase trio.
                    # both before and after these 4 CDSs, we have CDSs on the other strand (maybe suggesting that they are not related to our 4 CDSs). upstream to the 4 we have
                    # UBN58239.1, a site specific integrase, and downstream to the 4 we have UBN58244.1, a hypoth without identified conserved domains and only hypoth close
                    # homologs.
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP083813.1?report=graph&v=5322008:5329826
                        'presumably_relevant_cds_region': (5322008, 5329826),

                        # homology claim is based on blasting UBN58241.1 to YP_002343503.1 (Cj0031 of C. jejuni)
                        # (the homology is not as high as the homology between AYY86337.1/WP_223381307.1 and YP_002343503.1, but still very high)
                        'target_gene_product_description': 'Type II restriction-modification enzyme, homologous to Cj0031 of C. jejuni',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBN58240.1&FULL
                            # see https://www.ncbi.nlm.nih.gov/Structure/cdd/cddsrv.cgi?uid=cl15239&spf=1 (the superfamily of cd09178) if it is
                            # unclear from https://www.ncbi.nlm.nih.gov/Structure/cdd/cddsrv.cgi?uid=cd09178 that it is a phospholipase D domain.
                            # blastp UBN58240.1 to WP_052482303.1 (DrmD), and got one alignment with evalue=3e-12, from aa 565 to 734 in UBN58240.1,
                            # which seems to overlap a large part of the SNF (cd18793) domain of UBN58240.1, and another alignment with evalue=2e-6
                            # from aa 165 to 219 of UBN58240.1, which seems to overlap part of the DEXDc domain (smart00487, DEAD-helicases) and
                            # the SNF2 N-terminal domain (pfam00176).
                            # blastp UBN58240.1 to WP_040250232.1 (DrmC, which contains a phospholipase D domain), and got no alignments.

                            # blastp QFG44042.1 to WP_052482303.1 (DrmD), and got one alignment with evalue=5e-17, from aa 292 to 821 in QFG44042.1,
                            # which seems to overlap a large part of the HepA (COG0553) domain of QFG44042.1.
                            # blastp QFG44042.1 to WP_040250232.1 (DrmC, which contains a phospholipase D domain), and got no alignments.

                            # blastp UBN58240.1 to QFG44042.1 (the similar protein which is downstream to the MTase in CP044495.1), and got
                            # an alignment with evalue=1e-24 from aa 2 to 702 out of 990 (the length of UBN58240.1).
                            'A protein containing a phospholipase D (cd09178) domain and a SNF (cd18793, helicase) domain'
                        ),

                        'locus_description_for_table_3': 'Type II restriction-modification CDS with an immediately upstream CDS encoding a '
                                                         'phospholipase D (cd09178) domain and a SNF (cd18793, helicase) domain',

                        # 'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{ovatus}$ strain FDAARGOS_1516',
                        'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{ovatus}$',
                        'phylum': 'Bacteroidetes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((5324991, 5328359), 'UBN58241.1'),
                        # https://ncbi.nlm.nih.gov/protein/UBN58241.1
                        #    repeat_len  mismatch    left1   right1    left2   right2         evalue  spacer_len
                        # 0         237        12  5327435  5327671  5328980  5329216  2.200000e-110        1308
                        # 1          68         6  5327679  5327746  5328905  5328972   5.480000e-23        1158
                        # 2          24         4  5327945  5327968  5328716  5328739   5.400000e-01         747
                        # 3          16         2  5328389  5328404  5328414  5328429   7.800000e+00           9
                        'ir_pair_region_with_margins': (5326935, 5329716),
                        'alignment_region': (5306000, 5350000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR16259014': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # 'SRR16259014.1.121263.1', # inversion. doesn't span the whole region
                                    'SRR16259014.1.135948.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.123822.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.123825.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.139567.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.139568.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.139569.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.139570.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    # 'SRR16259014.1.159567.1', # inversion - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8
                                    # 'SRR16259014.1.159568.1', # inversion - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8
                                    # 'SRR16259014.1.159569.1', # inversion - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8
                                    # 'SRR16259014.1.159570.1', # inversion - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR16259014.1.159866.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion

                                    'SRR16259014.1.214507.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    # 'SRR16259014.1.214508.1', # inversion. doesn't cover the whole region
                                    'SRR16259014.1.217392.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.263500.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.263504.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.268492.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.276066.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.284439.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.284440.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.295581.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    # 'SRR16259014.1.295582.1', # inversion. doesn't cover the whole region
                                    'SRR16259014.1.35003.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.55867.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.58775.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.58776.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.71852.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    # 'SRR16259014.1.76872.1', # inversion. doesn't span the whole region
                                    # 'SRR16259014.1.76873.1', # inversion. doesn't span the whole region

                                    # added late (on 220304)
                                    'SRR16259014.1.121263.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.135944.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.203140.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.203142.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.203144.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.263506.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.295582.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.310546.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.310549.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.6201.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.6202.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.76872.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.76873.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.77309.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.77311.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.77312.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.77313.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion
                                    'SRR16259014.1.77316.1': {(5327435, 5327746, 5328905, 5329216)}, # inversion


                                    'SRR16259014.1.100051.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.100052.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.121259.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.121260.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.121261.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.121262.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.121264.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.121265.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.121266.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.123823.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.123824.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.135945.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.135946.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.135947.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.139566.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.143580.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.146952.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.154432.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.159861.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.159862.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.159863.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.159864.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.159865.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.170585.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.170586.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.170587.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.176951.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.176952.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.176955.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.176956.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183978.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183979.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183980.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183981.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183983.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183984.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183985.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.183986.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.185189.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.185191.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.194373.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.194374.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.202464.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.202465.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.202467.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.203136.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.203138.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.203141.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.203143.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.208695.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.208696.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.208697.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.214506.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.214509.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.217389.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.217390.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.217393.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.22388.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.22389.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.22390.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.22391.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.22392.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.223976.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.223977.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.223978.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.223980.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.241798.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.24925.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.256229.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.257235.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.257236.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.259514.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.261679.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.263501.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.263502.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.263503.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.263505.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.265605.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.265606.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.268490.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.268491.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.268493.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.268494.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.269456.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.269457.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.274765.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.274766.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.274767.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.274770.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.279968.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.279969.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.279970.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.281232.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.281233.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.284438.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.284441.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.284442.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.284443.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.284444.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.284445.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.284446.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28764.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28765.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28766.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28767.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28768.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28769.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28770.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.28771.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.295580.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.296147.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.296148.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.296149.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.297530.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.302206.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.302207.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.310547.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.35002.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.42484.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.42485.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.4916.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55859.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55860.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55861.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55862.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55863.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55864.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55865.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.55868.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.58777.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.58778.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.58779.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.6199.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.6200.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.6203.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.6204.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.6205.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.71753.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.71754.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.71756.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.71758.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.71759.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.76874.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.77310.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.77314.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.77315.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.7902.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.7903.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.7904.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.80519.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.80521.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.80522.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.80523.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.80524.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.80525.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.84010.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.84011.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.84012.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.84013.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.85070.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.87847.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.91105.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.91107.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.91111.1': {(5327435, 5327746, 5328905, 5329216)},
                                    'SRR16259014.1.91113.1': {(5327435, 5327746, 5328905, 5329216)},
                                },
                                'not_non_ref_variant_read_name_to_anomaly_description': {
                                    'SRR16259014.1.82263.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,
                                    'SRR16259014.1.82264.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,
                                    'SRR16259014.1.82265.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'rna_seq_sra_accessions': {
                            'SRR8867157',
                            'SRR8867158',
                        },

                        'cds_start_and_end_to_curated_product_class': {
                            # (5328334, 5329224): 'DNA MTase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBN58242.1&FULL
                            (5322008, 5324980): 'helicase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBN58240.1&FULL
                        },

                        'describe_in_the_paper': True,
                    },
                ],
                # 'NZ_CP016766.1': [ # Ligilactobacillus agilis strain La3
                #     {
                #         'ir_pairs': (
                #             (564833, 565142, 567264, 567573),
                #         ),
                #         'alignment_region': (544000, 588000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR8101013': {
                #
                #                 'SRR8101013.1.104571.1', # inversion between the bigger CDSs
                #                 'SRR8101013.1.112341.1', # inversion with the right small CDS
                #                 'SRR8101013.1.121030.1', # inversion with the right small CDS
                #                 'SRR8101013.1.130762.1', # inversion with the right small CDS
                #                 'SRR8101013.1.131181.1', # inversion between the bigger CDSs, though the alignments are pretty bad
                #                 'SRR8101013.1.27344.1', # inversion with the left small CDS
                #                 'SRR8101013.1.27344.3', # inversion with the left small CDS (i guess this read came from the same bacterium as 27344.1)
                #                 'SRR8101013.1.27344.5', # inversion with the left small CDS (i guess this read came from the same bacterium as 27344.1)
                #                 'SRR8101013.1.52815.1', # inversion between the bigger CDSs
                #                 'SRR8101013.1.63580.1', # inversion with the right small CDS
                #                 'SRR8101013.1.66517.1', # inversion between the bigger CDSs, though the alignment in the middle isn't perfect
                #             },
                #         },
                        # 'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
                # 'NZ_CP040468.1': [ # Parabacteroides distasonis strain CavFT-hAR46
                #     {
                #         'ir_pairs': (
                #             (3288888, 3288958, 3289995, 3290065),
                #         ),
                #         'alignment_region': (3268000, 3311000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR9221602': {
                #                 'SRR9221602.1.14349.1', # a beautiful rearrangement (could be achieved with two inversions, IIUC.
                #             },
                #         },
                        # 'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
            },

            # 'RM subunit S in M-S-invertibleSs-R': {
            #     # 'AP025571.1': [ # [Clostridium] symbiosum CE91-St65
            #     #             'DRR307123': {
            'MTase: downstream PLD&SNF2 helicase': {
                'CP044495.1': [ # Streptococcus mutans strain UA140
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP044495.1?report=graph&v=1253046:1261063
                        'presumably_relevant_cds_region': (1253046, 1261063),

                        # homology claim is based on blasting QFG45918.1 to YP_002343503.1 (Cj0031 of C. jejuni)
                        # (the homology is not as high as the homology between AYY86337.1/WP_223381307.1 and YP_002343503.1, but still pretty high)
                        'target_gene_product_description': 'Type II restriction-modification enzyme, homologous to Cj0031 of C. jejuni',

                        'presumably_associated_downstream_gene_product_descriptions': (
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QFG44042.1&FULL
                            # see https://www.ncbi.nlm.nih.gov/Structure/cdd/cddsrv.cgi?uid=cl15239&spf=1 (the superfamily of cd09178) if it is
                            # unclear from https://www.ncbi.nlm.nih.gov/Structure/cdd/cddsrv.cgi?uid=cd09178 that it is a phospholipase D domain.
                            'A protein containing a phospholipase D (cd09178) domain and a SNF (cd18793, helicase) domain'
                        ),

                        'locus_description_for_table_3': 'Type II restriction-modification CDS with a downstream CDS encoding a '
                                                         'phospholipase D (cd09178) domain and a SNF (cd18793, helicase) domain',

                        # 'locus_definition': r'$\mathit{Streptococcus}$ $\mathit{mutans}$ strain UA140',
                        'locus_definition': r'$\mathit{Streptococcus}$ $\mathit{mutans}$',
                        'phylum': 'Firmicutes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((1257875, 1261063), 'QFG45918.1'),
                        # https://ncbi.nlm.nih.gov/protein/QFG45918.1
                        #    repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0          27         0  1257614  1257640  1258173  1258199  1.390000e-09         532
                        # 1         146        40  1257177  1257322  1258494  1258639  5.260000e-09        1171
                        # 2          23         0  1257125  1257147  1258669  1258691  2.860000e-07        1521
                        # 3          21         1  1257381  1257401  1258433  1258453  2.240000e-04        1031
                        # 4          17         2  1258789  1258805  1259474  1259490  2.500000e+00         668
                        'ir_pair_region_with_margins': (1256457, 1259545),
                        'alignment_region': (1236000, 1280000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR11812841': {
                                # seems like the assembly is wrong (with an inversion (exactly where i would expect) relative to the SRA) - or just doesn't match the SRA.
                                # (double checked, and it doesn't seem like there is something especially interesting in this presumably wrong (or not matching) assembly)
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR11812841.1.10061.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.10062.1', # inversion. doesn't cover the whole region
                                    # 'SRR11812841.1.11462.1', # inversion. doesn't cover the whole region

                                    'SRR11812841.1.11465.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.1295.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.1296.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.13260.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.13610.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.15895.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.15896.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.15966.1', # inversion. doesn't cover the whole region

                                    'SRR11812841.1.16705.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.16706.1', # inversion. doesn't cover the whole region

                                    'SRR11812841.1.17241.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.17242.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.17243.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.17966.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.17967.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.17968.1', # inversion. doesn't cover the whole region
                                    'SRR11812841.1.18447.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    'SRR11812841.1.18449.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.18450.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.18451.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.18452.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    'SRR11812841.1.18454.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    'SRR11812841.1.18456.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.19561.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.19562.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.20352.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    # 'SRR11812841.1.20354.1', # inversion. doesn't span the whole region

                                    # 'SRR11812841.1.21609.1', # inversion. doesn't span the whole region
                                    'SRR11812841.1.22289.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.22290.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.22292.1', # inversion. doesn't cover the whole region
                                    'SRR11812841.1.22525.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    'SRR11812841.1.23772.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23773.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23819.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23820.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23821.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23822.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.24328.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.26687.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.26688.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.26689.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.26690.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28480.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28483.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28485.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28896.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28897.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28898.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28899.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28900.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28901.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28902.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28903.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion


                                    'SRR11812841.1.29829.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.30303.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.30848.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.33705.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.33706.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.415.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.5897.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion


                                    'SRR11812841.1.8170.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.8171.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.9220.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.9595.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.9597.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    # (assuming my code has no bugs) the alignments of the following reads that cover the required region in ref don't cover a region in the
                                    # read without gaps, so these reads don't count...
                                    # 'SRR11812841.1.17240.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.28481.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.29490.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.20353.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.18455.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.9598.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.11463.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.23689.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.20799.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.15967.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.18453.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.29828.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.8168.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.18448.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    # 'SRR11812841.1.6837.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    # added late (on 220304)
                                    'SRR11812841.1.10062.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.11462.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.15966.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.15968.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.16704.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.20354.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.21609.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.22292.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23818.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23824.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.23825.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.24460.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.24461.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    # for this read, alignment to the whole genome covered more bases (while discarding alignments that overlap the
                                    # IRR) than alignment to the alignment region. From the fasta file (extracted from the SRA entry):
                                    # >SRR11812841.1.24913.1 24913 length=34388
                                    # so it seems (also according to a manual BLAST i did online of the read I took from the fasta file) that
                                    # the read simply continues also outside the alignment region, and so the alignment region alone covered less bases...
                                    # usually this is not a problem, because in this step we first truncate the read (to keep only the part that aligned
                                    # to the alignment region) and only then align it to the whole genome.
                                    # here the truncation failed to remove the parts that didn't align to the alignment region, because there is another
                                    # inversion in this read, around 25kbps left - at ~1123800!
                                    # As this is such a rare case (the only case I saw up to 220327), I think it doesn't make sense for us specifically
                                    # to change our algorithm to not discard such reads...
                                    # 'SRR11812841.1.24913.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion

                                    'SRR11812841.1.26686.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.26691.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.26692.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.28482.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.29371.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.33084.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.6040.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.6041.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.6042.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.6043.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.6044.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.6045.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion
                                    'SRR11812841.1.9596.1': {(1257614, 1257640, 1258173, 1258199)}, # inversion


                                    'SRR11812841.1.11461.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.11464.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.11466.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.20798.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.21608.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.22288.1': {(1257614, 1257640, 1258173, 1258199)},
                                    # seems like we also have an intriguing rearrangement here. blasted to find relevant inverted repeats, but found nothing... So i guess
                                    # this isn't something programmed?
                                    'SRR11812841.1.29370.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.29830.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.30302.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.33083.1': {(1257614, 1257640, 1258173, 1258199)},
                                    'SRR11812841.1.8169.1': {(1257614, 1257640, 1258173, 1258199)},
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            (1256482, 1257009): 'recombinase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QFG44822.1&FULL
                            # (1257875, 1261063): 'DNA MTase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QFG45918.1&FULL
                            (1253046, 1256495): 'helicase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QFG44042.1&FULL

                            # blastp QFG44487.1, and the top scoring homologs which aren't as short as QFG44487.1 (e.g., QZS43349.1)
                            # were annotated as 'Eco57I restriction-modification methylase domain-containing protein'
                            (1257015, 1257944): 'DNA MTase',
                        },

                        'describe_in_the_paper': True,
                    },
                ],
                # 'CP066294.2': [ # Streptococcus mutans strain 27-3
                #             'SRR13617002': {
            },

            'MTase: BREX type 1': {
                'NC_013198.1': [ # Lacticaseibacillus rhamnosus GG (Firmicutes)
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NC_013198.1?report=graph&v=2156046:2169193
                        # 'presumably_relevant_cds_region': (2154002, 2169193),
                        'presumably_relevant_cds_region': (2154002, 2170387), # the complete BREX type 1 - from BrxA to BrxL
                        # 'presumably_relevant_cds_region': (2156046, 2169193),

                        # PglX is a Type II restriction-modification enzyme - http://rebase.neb.com/cgi-bin/seqget?LrhORF2097P (this is WP_015765014.1)
                        'target_gene_product_description': 'BREX type 1 PglX',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'BrxC'
                        ),
                        'presumably_associated_downstream_gene_product_descriptions': (
                            'PglZ; '
                            'BrxL'
                        ),

                        'locus_description_for_table_3': 'BREX type 1',

                        # annoyingly, GCF_000026505.1, the assembly to which NC_013198.1 belongs, is not marked as "reference genome" and not even
                        # "representative genome" in ftp://ftp.ncbi.nih.gov/genomes/refseq/bacteria/assembly_summary.txt. Thus, our pipeline doesn't
                        # identify this specific locus...


                        # 'locus_definition': r'$\mathit{Lacticaseibacillus}$ $\mathit{rhamnosus}$ GG',
                        'locus_definition': r'$\mathit{Lacticaseibacillus}$ $\mathit{rhamnosus}$',
                        'phylum': 'Firmicutes',
                        'longest_linked_repeat_cds_region_and_protein_id': ((2161973, 2165527), 'WP_015765014.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_015765014.1
                        'locus_description': 'BREX type 1',

                        'ir_pair_region_with_margins': (2158000, 2165653),
                        'alignment_region': (2138000, 2186000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR9952487': {
                                # the SRA is not linked to NC_013198.1. I didn't find strong evidence for the sequencing to be of a single cell colony.
                                # what I did find:
                                #   https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR9952487
                                #   https://www.ncbi.nlm.nih.gov/biosample/SAMN12559155 - seems like the submitter's name is "Witold Tomaszewski"?
                                #   https://www.ncbi.nlm.nih.gov/biosample/docs/packages/MIGS.ba.human-gut.5.0/
                                #   the isolation source is infant's feces, and the "analysis" section in
                                #   https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR9952487 says 97% of the reads belong to L rhamnosus, so I guess there must
                                #   have been a step in which a single L rhamnosus was isolated? hopefully.
                                #   another L. rhamnosus sequencing, maybe by the same person?
                                #       https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR11080692
                                #       https://www.ncbi.nlm.nih.gov/biosample/SAMN14090328
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # except for two specific cases, that only half distinguished between
                                    # (2159995, 2160324, 2163308, 2163637) and (2160376, 2160407, 2163225, 2163256), in all cases it seemed like they form a single IR pair
                                    # together. an so i think i will join them together, and later in my code i make sure that the new IR pair indeed formed by merging these two.
                                    #
                                    'SRR9952487.1.101097.1': {(2160721, 2160749, 2162886, 2162914)}, # inversion
                                    'SRR9952487.1.101241.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.102154.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.107988.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.11598.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.118450.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.127518.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.129292.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.132894.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.13657.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.139605.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.144518.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    # 'SRR9952487.1.15153.1', # a beautiful cassette switch. doesn't span the whole ir_pair_region_with_margins (it doesn't reach 2158000)

                                    # actually, the alignments of the read that cover the required region in ref don't cover a region in the read without gaps, so
                                    # this read doesn't count...
                                    # 'SRR9952487.1.19677.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch

                                    'SRR9952487.1.23124.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    # 'SRR9952487.1.2436.1', # a beautiful cassette switch - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8
                                    'SRR9952487.1.276.1': {(2158636, 2158760, 2164818, 2164942)}, # inversion
                                    'SRR9952487.1.27683.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.28272.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.29391.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.37423.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.37568.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.3887.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    # 'SRR9952487.1.39150.1', # a beautiful cassette switch - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8
                                    'SRR9952487.1.42737.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.44169.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.461.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.52975.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.55422.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch

                                    # on the right here there are two different alignments for the red and blue, though on the left there is a single one for both, so still not
                                    # a strong reason to not merge them...
                                    'SRR9952487.1.59069.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch

                                    # 'SRR9952487.1.63323.1', # a beautiful cassette switch, though it doesn't cover the whole ir_pair_region_with_margins
                                    'SRR9952487.1.6707.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.68718.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.70378.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.78121.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch . ditto SRR9952487.1.59069.1.
                                    'SRR9952487.1.81691.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.82617.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.86692.1': {(2159995, 2160407, 2163225, 2163637)}, # inversion
                                    'SRR9952487.1.92435.1': {(2159995, 2160407, 2163225, 2163637)}, # inversion
                                    'SRR9952487.1.92847.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.97784.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch

                                    # added late (on 220304)
                                    'SRR9952487.1.122893.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch
                                    'SRR9952487.1.128099.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.138938.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.25104.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)}, # a beautiful cassette switch
                                    'SRR9952487.1.82872.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)}, # a beautiful cassette switch

                                    # added even later (on 220408)
                                    'SRR9952487.1.103103.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.103693.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.107680.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.107938.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.10849.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.109565.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.109851.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.113559.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.115144.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.117528.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.121901.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.124401.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.124437.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.133808.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.138457.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.139718.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.144745.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.17420.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.23653.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.23859.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.2436.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.25130.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.2928.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.32010.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.32076.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.39150.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.39354.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.42967.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.4584.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.50853.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.52688.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.54252.1': {(2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.57796.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.6106.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.64245.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.64777.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.65223.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.67599.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.70874.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.70898.1': {(2159995, 2160407, 2163225, 2163637), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.71034.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.78140.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                    'SRR9952487.1.85795.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.95498.1': {(2158636, 2158760, 2164818, 2164942), (2159995, 2160407, 2163225, 2163637)},
                                    'SRR9952487.1.98042.1': {(2158636, 2158760, 2164818, 2164942), (2160721, 2160749, 2162886, 2162914)},
                                }
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'rna_seq_sra_accessions': {
                            'SRR6322568',
                            'SRR6322569',
                            'SRR6322570',
                        },

                        'describe_in_the_paper': True,
                    },
                ],
            },

            'MTase: BREX type 1, downstream extra short PglX': { # good
                # 'CP045806.1': [ # Gordonia pseudoamarae strain BEN371
                #             'SRR11613597': {
                # 'CP045809.1': [ # Gordonia pseudoamarae strain CON9
                #             'SRR11613601': {
                'NZ_CP068173.1': [ # Brevibacterium casei strain FDAARGOS_1100 (Actinobacteria)
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP068173.1?report=graph&v=2187164:2199385
                        'presumably_relevant_cds_region': (2184967, 2200609), # the complete BREX type 1 - from BrxA to BrxL
                        # 'presumably_relevant_cds_region': (2187164, 2199385),

                        # blastp of WP_201670672.1 to WP_015765014.1 (the PglX of Lacticaseibacillus rhamnosus GG) results in an alignment with evalue=2e-146,
                        # from aa 1 to 859 out of 862 of WP_201670672.1, and from aa 1 to 878 out of 1184 of WP_015765014.1.
                        'target_gene_product_description': 'N-terminus and middle parts of BREX type 1 PglX',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'BrxC'
                        ),
                        'presumably_associated_downstream_gene_product_descriptions': (
                            # blastp of WP_236586838.1 to WP_015765014.1 (the PglX of Lacticaseibacillus rhamnosus GG) results in an alignment with evalue=2e-36,
                            # from aa 3 to 299 out of 315 of WP_236586838.1, and from aa 885 to 1172 out of 1184 of WP_015765014.1.
                            'C-terminus part of PglX; ' 
                            'PglZ; '
                            'BrxL'
                        ),

                        'locus_description_for_table_3': 'BREX type 1, with one CDS encoding the N-terminus and middle parts of PglX '
                                                         '(targeted by programmed inversions), and a downstream CDS '
                                                         'encoding the C-terminus part of PglX (not targeted by programmed inversions)',

                        # blastp of WP_201670668.1 (the integrase of Brevibacterium casei) to WP_014569897.1 (the integrase of Lacticaseibacillus rhamnosus GG)
                        # gives an alignment with evalue=5e-12, from aa 191 to 340 out of 360 of WP_014569897.1


                        # 'locus_definition': r'$\mathit{Brevibacterium}$ $\mathit{casei}$ strain FDAARGOS_1100',
                        'locus_definition': r'$\mathit{Brevibacterium}$ $\mathit{casei}$',
                        'phylum': 'Actinobacteria',
                        'longest_linked_repeat_cds_region_and_protein_id': ((2193315, 2195903), 'WP_201670672.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_201670672.1
                        # https://ncbi.nlm.nih.gov/protein/WP_236586838.1

                        'locus_description': 'BREX type 1',
                           #     repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                           #  0          80         9  2192782  2192861  2193736  2193815  1.530000e-24         874
                           #  1          49         5  2192571  2192619  2193978  2194026  1.520000e-13        1358
                           #  2          26         0  2192953  2192978  2193655  2193680  6.490000e-09         676
                           #  3          22         0  2192183  2192204  2194429  2194450  1.340000e-06        2224
                           #  4          52        12  2193228  2193279  2193357  2193408  4.000000e-03          77
                           #  5          17         1  2192704  2192720  2193877  2193893  5.700000e-02        1156
                           #  6          18         2  2192416  2192433  2194491  2194508  8.200000e-01        2057
                        'ir_pair_region_with_margins': (2191587, 2195018),
                        'alignment_region': (2171000, 2216000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR12935271': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR12935271.1.100947.1': {(2192587, 2192619, 2193978, 2194010)}, # inversion
                                    'SRR12935271.1.121954.1': {(2192953, 2192978, 2193655, 2193680)}, # inversion
                                    'SRR12935271.1.129400.1': {(2192587, 2192619, 2193978, 2194010)}, # inversion
                                    'SRR12935271.1.131311.1': {(2192183, 2192204, 2194429, 2194450), (2192953, 2192978, 2193655, 2193680)}, # a beautiful cassette switch.
                                    'SRR12935271.1.132135.1': {(2192953, 2192978, 2193655, 2193680)}, # inversion
                                    'SRR12935271.1.149101.1': {(2192183, 2192204, 2194429, 2194450)}, # inversion
                                    # 'SRR12935271.1.153719.1', # inversion at the N-terminus of the hypoth, but with a small gap. i.e., doesn't cover the whole region.
                                    'SRR12935271.1.161578.1': {(2192183, 2192204, 2194429, 2194450)}, # inversion
                                    'SRR12935271.1.18410.1': {(2192587, 2192619, 2193978, 2194010)}, # inversion
                                    'SRR12935271.1.34740.1': {(2192183, 2192204, 2194429, 2194450)}, # inversion
                                    'SRR12935271.1.37929.1': {(2192587, 2192619, 2193978, 2194010), (2192953, 2192978, 2193655, 2193680)}, # cassette switch.
                                    'SRR12935271.1.37929.3': {(2192587, 2192619, 2193978, 2194010), (2192953, 2192978, 2193655, 2193680)}, # cassette switch.

                                    # added late (on 220304)
                                    'SRR12935271.1.55345.1': {(2192953, 2192978, 2193655, 2193680)}, # inversion
                                    'SRR12935271.1.72714.1': {(2192953, 2192978, 2193655, 2193680)}, # inversion


                                    'SRR12935271.1.127405.1': {(2192587, 2192619, 2193978, 2194010)}, # inversion
                                    'SRR12935271.1.160386.1': {(2192183, 2192204, 2194429, 2194450)}, # inversion
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            (2192191, 2193318): 'DNA MTase', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_201670670.1&FULL
                        },

                        'describe_in_the_paper': True,
                    },
                ],

            },

            'DUF4965': {
                'CP065872.1': [ # Bacteroides thetaiotaomicron strain FDAARGOS_935
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP065872.1?report=graph&v=6172495:6187445
                        # 'presumably_relevant_cds_region': (6170362, 6189525), # it is really even longer than that...
                        'presumably_relevant_cds_region': (6172495, 6187445),
                        # 'presumably_relevant_cds_region': (6175634, 6184292),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QQA08364.1&FULL
                        'target_gene_product_description': 'A protein of unknown function containing 4 domains of unknown function (DUFs): '
                                                           'DUF4964 (pfam16334), DUF5127 (pfam17168), DUF4965 (pfam16335) and DUF1793 (pfam08760)',

                        'presumably_associated_downstream_gene_product_descriptions': (
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QQA10773.1&FULL
                            'A protein containing a IPT PCSR (cd00603) domain and a NHL repeat (cd14953) domain; '
                            'SusC; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QQA08363.1&FULL
                            'SusD; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QQA08362.1&FULL
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QQA08361.1&FULL
                            'A protein containing a DUF4973 (pfam16343) domain and DUF4361 (pfam14274) domain'
                        ),

                        # 6182745 - 6182667 - 1 == 77 bps between the short CDS and the CDS downstream to it.
                        # 6177248 - 6177202 - 1 == 45 bps between the long CDS and the CDS downstream to it.
                        # 6197344 - 6197302 - 1 == 41 bps between the long CDS and the CDS downstream to it.
                        'locus_description_for_table_3': 'A protein of unknown function (targeted by programmed inversions) with a downstream presumed '
                                                         'operon (according to short distances between CDSs) containing, among others, CDSs encoding '
                                                         'outer membrane proteins SusC and SusD',

                        # blasting QQA08364.1 to QQA08366.1 results in a pretty good, but not perfect, alignment from aa 1 to 169 out of 172 (the length of QQA08366.1).
                        # so in addition to the final 3 or 2 amino acids (in QQA08366.1 and QQA08364.1, respectively), which are different, there are also some
                        # aa substitutions throughout the alignment, which might be important?
                        # it could also be that the inversion is mainly meant for switching the rest of the operon, though (on a second thought, even though the rest of the
                        # operon might be the important part, the DUF1793 of each variant has to fit the rest of the operon, so it seems reasonable that the DUF1793 of each
                        # variant is optimized for that variant, even though there are only small changes. in fact, i would guess that the inverted repeat spans almost the
                        # DUF1793 exactly because minor changes in it are required to fit the rest of the operon. otherwise, it wouldn't be necessary to have inverted
                        # repeats that long...
                        # blasting QQA08367.1 to QQA10773.1 (the next genes in the operons) gives a long alignment, but very far from perfect (41% identity).
                        # blasting QQA08368.1 to QQA08363.1 (the next genes in the operons) gives a long pretty good alignment (67% identity).
                        # blasting QQA08369.1 to QQA08362.1 (the next genes in the operons) gives a long not bad alignment (55% identity).
                        'longest_linked_repeat_cds_region_and_protein_id': ((6177248, 6179761), 'QQA08364.1'),
                        # https://ncbi.nlm.nih.gov/protein/QQA08364.1

                        'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{thetaiotaomicron}$',
                        'phylum': 'Bacteroidetes',

                        'ir_pair_region_with_margins': (6176750, 6183160),
                        # 'ir_pair_region_with_margins': (6168781, 6191038),
                        # 'ir_pair_region_with_margins': (6176750, 6197802),
                        # 'alignment_region': (6156000, 6204000),
                        'alignment_region': (6148000, 6212000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR12047085': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # this one is covered better by the rest of the assembly
                                    # 'SRR12047085.1.35495.1': {(6177261, 6177848, 6182064, 6182651)}, # inversion

                                    'SRR12047085.1.105653.1': {(6177261, 6177848, 6182064, 6182651)}, # inversion
                                    'SRR12047085.1.105653.3': {(6177261, 6177848, 6182064, 6182651)}, # inversion
                                    'SRR12047085.1.20249.1': {(6177261, 6177848, 6182064, 6182651)}, # inversion
                                    'SRR12047085.1.20249.3': {(6177261, 6177848, 6182064, 6182651)}, # inversion

                                    # the following reads seem like not bad evidence for another, much larger inversion, with the alleged frameshifted CDS at
                                    # 6196782-6197302, but none of them satisfies all of our requirements and thresholds:
                                    # 'SRR12047085.1.122984.1',
                                    # 'SRR12047085.1.123081.1',
                                    # 'SRR12047085.1.2247.1',
                                    # 'SRR12047085.1.57685.1',
                                    # 'SRR12047085.1.65701.1',
                                    # 'SRR12047085.1.82574.1',
                                    # 'SRR12047085.1.83257.1',
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,
                    },
                ],
            },

            'PilV and phage tail collar': {
                'CP084655.1': [ # Pectobacterium brasiliense strain SR10
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP084655.1?report=graph&v=813503:828430
                        # 'presumably_relevant_cds_region': (813503, 820358),
                        'presumably_relevant_cds_region': (813503, 828430),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91044.1&FULL
                        'target_gene_product_description': 'A protein containing a Shufflon PilV N-terminus (pfam04917) domain '
                                                           'and a phage tail collar (pfam07484) domain',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'PilL (TIGR03748) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91054.1&FULL
                            'PilM (pfam07419) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91053.1&FULL
                            'PilN (TIGR02520) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91052.1&FULL
                            'PilO (pfam06864) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91051.1&FULL
                            'PilP (TIGR03021) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91050.1&FULL
                            # blastp UCP91049.1 to Q70M91.2 (A PilQ domain-containing protein) resulted in no alignments.  
                            'PulE (COG2804) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91049.1&FULL
                            'PulF (COG1459) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91048.1&FULL
                            'PilS (pfam08805) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91047.1&FULL
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91046.1&FULL
                            'Lytic transglycosylase (cd13400) domain-containing protein; '
                            
                            # interestingly, it seems like WP_240342859.1 (a protein in NZ_CP059957.1, which is another assembled chromosome of 
                            # Pectobacterium brasiliense) is a fusion-protein of UCP91045.1 and UCP91046.1. (according to blastp of WP_240342859.1
                            # to UCP91045.1 and UCP91046.1).
                            # blastp UCP91045.1 to nr gave mostly alignments to prepilin peptidase proteins.
                            'prepilin peptidase'
                        ),

                        'locus_description_for_table_3': 'Two adjacent presumed operons (according to short distances between CDSs) '
                                                         'containing multiple pilus associated CDSs',

                        # left opreon: 816997 to 820358. right operon: 820415 to 828430.

                         # CDS             complement(816997..818571)
                         #                 /product="shufflon system plasmid conjugative transfer
                         #                 pilus tip adhesin PilV"
                         #                 /protein_id="UCP91044.1"
                         # CDS             complement(818585..819238)
                         #                 /product="prepilin peptidase"
                         #                 /protein_id="UCP91045.1"
                         # CDS             complement(819235..819753)
                         #                 /product="lytic transglycosylase domain-containing
                         #                 protein"
                         #                 /protein_id="UCP91046.1"
                         # CDS             complement(819762..820358)
                         #                 /product="pilus assembly protein PilX"
                         #                 /protein_id="UCP91047.1"
                         # CDS             complement(820415..821581)
                         #                 /product="type II secretion system F family protein"
                         #                 /protein_id="UCP91048.1"
                         # CDS             complement(821584..823143)
                         #                 /product="Flp pilus assembly complex ATPase component
                         #                 TadA"
                         #                 /protein_id="UCP91049.1"
                         # CDS             complement(823143..823652)
                         #                 /product="type IV pilus biogenesis protein PilP"
                         #                 /protein_id="UCP91050.1"
                         # CDS             complement(823649..824956)
                         #                 /product="type 4b pilus protein PilO2"
                         #                 /protein_id="UCP91051.1"
                         # CDS             complement(824967..826631)
                         #                 /product="PilN family type IVB pilus formation outer
                         #                 membrane protein"
                         #                 /protein_id="UCP91052.1"
                         # CDS             complement(826641..827087)
                         #                 /product="type IV pilus biogenesis protein PilM"
                         #                 /protein_id="UCP91053.1"
                         # CDS             complement(827087..828430)
                         #                 /product="TcpQ domain-containing protein"
                         #                 /protein_id="UCP91054.1"

                        # 220403: the current conserved domain page of WP_000002046.1
                        # (https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_000002046.1&FULL, which is the 'phage tail protein' whose
                        # gene was mentioned in the paper 'Cloning and Sequencing of a Genomic Island Found in the Brazilian Purpuric Fever
                        # Clone of Haemophilus influenzae Biogroup Aegyptius' (https://journals.asm.org/doi/10.1128/IAI.73.4.1927-1938.2005,
                        # https://www.ncbi.nlm.nih.gov/nuccore/NC_003198.1?report=graph 1569k)) describes
                        # the protein as "phage tail protein is part of a multi-protein structure that mediates the attachment, digestion and penetration
                        # of the cell wall and genome ejection"
                        # this made me think: maybe the specificity domain of this conjugative pili and the specificity domain of the phage tail
                        # have a common ancestor???
                        #
                        # Phase and Antigenic Variation in Bacteria (https://journals.asm.org/doi/full/10.1128/CMR.17.3.581-611.2004) says:
                        # "Type IV pili function as adhesins and include conjugative pili. Phase variation, antigenic variation of the structural
                        # subunits, and phase-variable modification have been described. Sequence variation in the type IV conjugative pili encoded
                        # by plasmids R64, R721, and ColI-P9 occurs as a result of incorporation of only one of a set of distinct C termini in the
                        # PilV tip proteins of the pilus in an individual cell. This sequence variation is associated with different receptor
                        # specificity, thereby dictating the species that will be preferred as a DNA recipient in a conjugation reaction"
                        # So maybe the (phase-variable) phage tail collar domain was repurposed by the bacteria to have a pili with variable
                        # specificity??
                        #
                        # blasting WP_001389385.1, the product of the long PilV in Salmonella enterica subsp. enterica serovar Typhimurium plasmid R64
                        # (https://www.ncbi.nlm.nih.gov/nuccore/AP005147.1?report=graph&v=101000-109000) to UCP91044.1 resulted in an alignment only
                        # of the Shufflon_N domain, and not the downstream amino acids.
                        # blasting WP_011117612.1, a shorter gene in Salmonella enterica subsp. enterica serovar Typhimurium plasmid R64 (targeted by inversions),
                        # to UCP91044.1 results in no alignments.
                        # blasting WP_011117613.1, a shorter gene in Salmonella enterica subsp. enterica serovar Typhimurium plasmid R64 (targeted by inversions),
                        # to UCP91044.1 results in a very short alignment to the end of the Shufflon_N domain and 2 aa downstream to the Shufflon_N
                        # domain, which is 9 aa upstream to the phage tail collar domain.
                        # blasting WP_010891273.1, a shorter gene in Salmonella enterica subsp. enterica serovar Typhimurium plasmid R64 (targeted by inversions),
                        # to UCP91044.1 results in no alignments.
                        # blasting WP_010891272.1, a shorter gene in Salmonella enterica subsp. enterica serovar Typhimurium plasmid R64 (targeted by inversions),
                        # to UCP91044.1 results in no alignments.
                        # blasting WP_001389393.1, a shorter gene in Salmonella enterica subsp. enterica serovar Typhimurium plasmid R64 (targeted by inversions),
                        # to UCP91044.1 results in no alignments.

                        # OOPS.
                        # blasting WP_010891274.1, a shorter gene in Salmonella enterica subsp. enterica serovar Typhimurium plasmid R64 (targeted by inversions),
                        # to UCP91044.1 results in a very short alignment (evalue=0.048) to the phage tail collar domain.

                        # WP_001389385.1 is also the long PilV that appears in Shigella sonnei plasmid P9
                        # (https://www.ncbi.nlm.nih.gov/nuccore/NC_002122.1?report=graph&v=74000-82000)
                        # blasting WP_001213803.1, a shorter gene in Shigella sonnei plasmid P9 (targeted by inversions, i guess), to UCP91044.1
                        # results in a short alignment to the end of the Shufflon_N domain and 7 aa downstream to the Shufflon_N domain, which is 4 aa
                        # upstream to the phage tail collar domain.
                        # blasting WP_001302705.1, a shorter gene in Shigella sonnei plasmid P9 (targeted by inversions, i guess),
                        # to UCP91044.1 results in no alignments.
                        #
                        # blasting WP_010895887.1, the product of the long PilV in Escherichia coli plasmid R721
                        # (https://www.ncbi.nlm.nih.gov/nuccore/NC_002525.1?report=graph&v=40000-50000) to UCP91044.1 resulted in an alignment only of the
                        # Shufflon_N domain, and 9 aa downstream to the Shufflon_N domain, which is 2 aa upstream to the phage tail collar domain.
                        # blasting WP_001141249.1, a shorter gene in Escherichia coli plasmid R721 (targeted by inversions, i guess), to UCP91044.1
                        # results in no alignments.

                        # blasting AAO71709.1, the product of the long PilV in Salmonella enterica subsp. enterica serovar Typhi Ty2
                        # (https://www.ncbi.nlm.nih.gov/nuccore/AE014613.1?report=graph&v=4416000-4422000) to UCP91044.1 resulted in an alignment
                        # only of the Shufflon_N domain, and 4 aa downstream to the Shufflon_N domain, which is 7 aa upstream to the phage tail collar domain.
                        # blasting WP_001141249.1, a shorter gene in Escherichia coli plasmid R721 (targeted by inversions, i guess), to UCP91044.1
                        # results in no alignments.

                        # and all alignments together:
                        # blast
                        # UCP91044.1
                        # UCP91043.1
                        # UCP91042.1
                        # UCP92603.1
                        # to
                        # WP_001389385.1
                        # WP_011117612.1
                        # WP_011117613.1
                        # WP_010891273.1
                        # WP_010891272.1
                        # WP_001389393.1
                        # WP_010891274.1
                        # WP_001213803.1
                        # WP_001302705.1
                        # WP_010895887.1
                        # WP_001141249.1
                        # AAO71709.1
                        # WP_001141249.1
                        # results in no alignments for the shorter target CDSs (UCP91043.1, UCP91042.1, UCP92603.1), and only alignments upstream
                        # to the phage tail collar domain of the longer target CDS (UCP91044.1). (the alignment from earlier to the phage tail collar
                        # domain had a two high evalue, I guess, which is kind of a correction for multiple hypotheses...)
                        # and alignments only of longer CDSs:
                        # UCP91044.1
                        # to
                        # WP_001389385.1
                        # WP_001389385.1
                        # WP_010895887.1
                        # AAO71709.1
                        #
                        'longest_linked_repeat_cds_region_and_protein_id': ((816997, 818571), 'UCP91044.1'),
                        # https://ncbi.nlm.nih.gov/protein/UCP91044.1

                        'locus_definition': r'$\mathit{Pectobacterium}$ $\mathit{brasiliense}$',
                        'ir_pair_region_with_margins': (814300, 817950),
                        'phylum': 'Proteobacteria',

                        'alignment_region': (794000, 838000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR16683342': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR16683342.1.10048.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.102047.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.102769.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.104256.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.106076.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.106474.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.107155.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.107390.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.107942.1': {(816119, 816322, 817235, 817438), (814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.107944.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.109880.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.110008.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.110210.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.11258.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.112741.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.112753.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.113106.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.113464.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.114001.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.114435.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.114990.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.116664.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.117859.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.118712.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.118880.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.119238.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.121102.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.122259.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.124591.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.12469.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.124997.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.125184.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.126782.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.127373.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.127661.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.129349.1': {(815888, 815910, 817417, 817439)},
                                    'SRR16683342.1.131017.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.131351.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.131399.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.132292.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.132501.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.13332.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.135665.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.135853.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.136290.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.140119.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.141374.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.144305.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.144365.1': {(815888, 815910, 817417, 817439), (816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.144783.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.145451.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.146714.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.148142.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.148479.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.148532.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.148669.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.11709.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.125372.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.132193.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.142106.1': {(815188, 815207, 817420, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.149574.1': {(815888, 815910, 817417, 817439)},
                                    'SRR16683342.1.150021.1': {(814683, 814907, 817214, 817438), (815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.150361.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.15060.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.150741.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.151795.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.151860.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.152799.1': {(815406, 815614, 817231, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.153169.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.153910.1': {(815406, 815614, 817231, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.157429.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.157832.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.157895.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.158081.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.15819.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.158804.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.158971.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.160056.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.160561.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.161068.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.161241.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.164912.1': {(815406, 815614, 817231, 817439)},
                                    'SRR16683342.1.166541.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.166760.1': {(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.168143.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.168196.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.16936.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.170368.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.170775.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.170785.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.171739.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.172373.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.172576.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.173297.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.18519.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.19406.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.20494.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.20885.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.21262.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.21331.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.21376.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.21677.1': {(815406, 815614, 817231, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.22792.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.23492.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.23746.1': {(815188, 815207, 817420, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.25203.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.25769.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.26163.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.28365.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.29832.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.29960.1': {(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.30967.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.31129.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.31499.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.32478.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.33311.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.34298.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.3434.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.34459.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.35892.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.36175.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.36209.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.36513.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.37571.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.38354.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.38551.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.39312.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.39344.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.39640.1': {(815888, 815910, 817417, 817439)},
                                    'SRR16683342.1.39697.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.41109.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.41886.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.43777.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.43785.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.44581.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.44696.1': {(815888, 815910, 817417, 817439), (816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.44727.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.4718.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.48345.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.49169.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.49362.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.49989.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.51159.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.55593.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.5595.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.56392.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.5679.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.57557.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.57768.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.59130.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.59585.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.6101.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.61077.1': {(815188, 815207, 817420, 817439), (815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.62217.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.62571.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.63899.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.6524.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.65962.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.66833.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.67606.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.68688.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.69501.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.69707.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.70269.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.71700.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.71857.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.72602.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.74433.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.74501.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.74523.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.74884.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.75932.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.7614.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.7628.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.77779.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.78797.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.80678.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.81654.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.82727.1': {(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.83329.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.83734.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.84886.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.85538.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.86208.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.87079.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.88886.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.92754.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.93124.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.95782.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.97352.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.97359.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.97371.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.98161.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.98683.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.99321.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},


                                    'SRR16683342.1.100198.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.100343.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.102058.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.104537.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.104605.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.104785.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.104861.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.105571.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.105634.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.106235.1': {(815406, 815614, 817231, 817439)},
                                    'SRR16683342.1.107761.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.108001.1': {(814683, 814907, 817214, 817438), (815406, 815614, 817231, 817439),
                                                              (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.108340.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.108390.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.108901.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.110005.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.111188.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.112425.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.112663.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.11289.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.113452.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.113838.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.115034.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.115407.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.115507.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.116581.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.117107.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.117740.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.117741.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.119073.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.12133.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.122557.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.124353.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.12436.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.126806.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.127803.1': {(815888, 815910, 817417, 817439)},
                                    'SRR16683342.1.128319.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.128535.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.129189.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.130252.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.131082.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.131302.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.131987.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.133245.1': {(815188, 815207, 817420, 817439)},
                                    'SRR16683342.1.135824.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.138655.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.138685.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.139046.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.14220.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.142469.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.14432.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.147720.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.155167.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.155352.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.155683.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.158918.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.159459.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.161332.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.161540.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.162217.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.163242.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.164906.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.16528.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.166638.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.168128.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.170984.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.171713.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.172347.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.1771.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.18651.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.19350.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.19444.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.19621.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.20838.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.21320.1': {(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.22725.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.25424.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.25745.1': {(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.26090.1': {(815406, 815614, 817231, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.2790.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.28965.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.29144.1': {(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.29520.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.30376.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.34556.1': {(815188, 815207, 817420, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.3685.1': {(815406, 815614, 817231, 817439)},
                                    'SRR16683342.1.37180.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.37186.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.38580.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.42727.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.433.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.43968.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.4511.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.47883.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.48117.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.49184.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.4988.1': {(815888, 815910, 817417, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.50626.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.51141.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.52545.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.52648.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.52761.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.53397.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.53916.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.54263.1': {(814683, 814907, 817214, 817438), (815888, 815910, 817417, 817439),
                                                              (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.54825.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.54945.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.55760.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.5580.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.56382.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.56588.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.5755.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.58303.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.58569.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.58642.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.60075.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.62395.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.62933.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.64186.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.64283.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.6575.1': {(815406, 815614, 817231, 817439), (816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.66330.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.66370.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.67053.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.67589.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.68021.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.69135.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.69724.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.71936.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.7484.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.74910.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.75564.1': {(815406, 815614, 817231, 817439)},
                                    'SRR16683342.1.7565.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.76430.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.76735.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.7790.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.79120.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.7947.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.79919.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.80968.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.81125.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.81375.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.81519.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.82217.1': {(814683, 814907, 817214, 817438)},
                                    'SRR16683342.1.85444.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.85630.1': {(814683, 814907, 817214, 817438), (815888, 815910, 817417, 817439)},
                                    'SRR16683342.1.8564.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.85889.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.86121.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.86395.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.86701.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.8671.1': {(814683, 814907, 817214, 817438), (815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.86756.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.87008.1': {(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.87403.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.87411.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},

                                    # Ugh. mauve fails to identify the cassette switch here. I guess mauve is wrong, but for the sake of time,
                                    # I grudgingly mark this read as a not_non_ref_variant. ugh. (can't just mark it as
                                    # 'inaccurate_or_not_beautiful_mauve_alignment', because it is the only read of that variant...)
                                    # 'SRR16683342.1.89718.1': {(815406, 815614, 817231, 817439), (815888, 815910, 817417, 817439),
                                    #                           (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},

                                    'SRR16683342.1.90648.1': {(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.91418.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.92026.1': {(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.92059.1': {(814683, 814907, 817214, 817438), (815188, 815207, 817420, 817439)},
                                    'SRR16683342.1.93126.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.96208.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.96722.1': {(816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.97012.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},
                                    'SRR16683342.1.97013.1': {(816119, 816322, 817235, 817438)},
                                    'SRR16683342.1.98079.1': {(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)},



                                },
                                'not_non_ref_variant_read_name_to_anomaly_description': {
                                    'SRR16683342.1.125915.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,
                                    'SRR16683342.1.41436.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,
                                    'SRR16683342.1.65946.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,

                                    # I guess this read is perfectly fine, but it is here because mauve failed for it. see above.
                                    'SRR16683342.1.89718.1': get_mauve_failed_explanation({
                                        (815406, 815614, 817231, 817439), (815888, 815910, 817417, 817439),
                                        (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}),
                                },
                                'complex_variant_ir_pairs_to_variant_regions_and_types': {
                                    frozenset({(816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((816322, 816634), {'moved', 'inverted'}), ((816634, 817235), {'moved'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((814907, 816634), {'moved', 'inverted'}), ((816634, 817214), {'moved'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (815188, 815207, 817420, 817439)}): (
                                        ((814907, 815207), {'moved', 'inverted'}), ((815207, 817214), {'moved'}),
                                    ),
                                    frozenset({(815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)}): (
                                        ((815614, 816634), {'moved', 'inverted'}), ((816634, 817231), {'moved'}),
                                    ),
                                    frozenset({(815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)}): (
                                        ((815910, 816634), {'moved', 'inverted'}), ((816634, 817417), {'moved'}),
                                    ),
                                    frozenset({(815888, 815910, 817417, 817439), (816119, 816322, 817235, 817438)}): (
                                        ((815910, 816322), {'moved', 'inverted'}), ((816322, 817235), {'moved'}),
                                    ),
                                    frozenset({(815188, 815207, 817420, 817439), (816612, 816634, 817417, 817439)}): (
                                        ((815207, 816634), {'moved', 'inverted'}), ((816634, 817417), {'moved'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438)}): (
                                        ((814907, 816322), {'moved', 'inverted'}), ((816322, 817214), {'moved'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (815888, 815910, 817417, 817439)}): (
                                        ((814907, 815910), {'moved', 'inverted'}), ((815910, 817214), {'moved'}),
                                    ),
                                    frozenset({(815406, 815614, 817231, 817439), (816119, 816322, 817235, 817438)}): (
                                        ((815614, 816322), {'moved', 'inverted'}), ((816322, 817231), {'moved'}),
                                    ),

                                    frozenset({(815888, 815910, 817417, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((815910, 816322), {'moved', 'inverted'}), ((816322, 816634), {'moved'}), ((816634, 817235), {'moved', 'inverted'}),
                                    ),
                                    frozenset({(815188, 815207, 817420, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((815207, 816322), {'moved', 'inverted'}), ((816322, 816634), {'moved'}), ((816634, 817235), {'moved', 'inverted'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)}): (
                                        ((814907, 815910), {'moved', 'inverted'}), ((815910, 816634), {'moved'}), ((816634, 817214), {'moved', 'inverted'}),
                                    ),
                                    frozenset({(815188, 815207, 817420, 817439), (815888, 815910, 817417, 817439), (816612, 816634, 817417, 817439)}): (
                                        ((815207, 815910), {'moved', 'inverted'}), ((815910, 816634), {'moved'}), ((816634, 817417), {'moved', 'inverted'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((814907, 816322), {'moved', 'inverted'}), ((816322, 816634), {'moved'}), ((816634, 817214), {'moved', 'inverted'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (815406, 815614, 817231, 817439), (816612, 816634, 817417, 817439)}): (
                                        ((814907, 815614), {'moved', 'inverted'}), ((815614, 816634), {'moved'}), ((816634, 817214), {'moved', 'inverted'}),
                                    ),
                                    frozenset({(815406, 815614, 817231, 817439), (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((815614, 816322), {'moved', 'inverted'}), ((816322, 816634), {'moved'}), ((816634, 817231), {'moved', 'inverted'}),
                                    ),

                                    frozenset({(814683, 814907, 817214, 817438), (815406, 815614, 817231, 817439),
                                               (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((814907, 815614), {'moved', 'inverted'}), ((815614, 816322), {'moved'}),
                                        ((816322, 816634), {'moved', 'inverted'}), ((816634, 817214), {'moved'}),
                                    ),
                                    frozenset({(814683, 814907, 817214, 817438), (815888, 815910, 817417, 817439),
                                               (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((814907, 815910), {'moved', 'inverted'}), ((815910, 816322), {'moved'}),
                                        ((816322, 816634), {'moved', 'inverted'}), ((816634, 817214), {'moved'}),
                                    ),
                                    frozenset({(815406, 815614, 817231, 817439), (815888, 815910, 817417, 817439),
                                               (816119, 816322, 817235, 817438), (816612, 816634, 817417, 817439)}): (
                                        ((815614, 815910), {'moved', 'inverted'}), ((815910, 816322), {'moved'}),
                                        ((816322, 816634), {'moved', 'inverted'}), ((816634, 817231), {'moved'}),
                                    ),
                                    # {(-817439, -817417, -816634, -816612), (-817438, -817235, -816322, -816119), (-817438, -817214, -814907, -814683), (-817439, -817417, -815910, -815888)}

                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            # blastp UCP92603.1, and almost all homologs were annotated as phage tail protein or tail fiber protein.
                            (814921, 815172): 'phage',
                            # blastp UCP91048.1 to nr results in many alignments to proteins annotated as 'pilus assembly protein PilR'.
                            # not strong enough, I think? also, from Type IV Pilin Proteins: Versatile Molecular Modules
                            # (https://journals.asm.org/doi/10.1128/MMBR.00035-12) it doesn't sound like PilR and PulF could be different
                            # names for the same domain, though that's just a wild guess.
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UCP91048.1&FULL
                            # (820415, 821581): 'pili',
                        },

                        'describe_in_the_paper': True,
                    },
                ],
            },



            'RM specificity: R-M-S': {
                'NZ_CP022464.2': [ # Enterocloster bolteae strain ATCC BAA-613
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP022464.2?report=graph&v=6449455:6459229
                        'presumably_relevant_cds_region': (6449455, 6459229),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_007037442.1&FULL
                        'target_gene_product_description': 'Type I restriction-modification HsdS',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'HsdR; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_007037444.1&FULL
                            'HsdM' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_007037443.1&FULL
                        ),

                        'locus_description_for_table_3': 'Type I restriction-modification, with core CDS order HsdR-HsdM-HsdS',

                        'longest_linked_repeat_cds_region_and_protein_id': ((6453804, 6455096), 'WP_007037442.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_007037442.1

                        # 'locus_definition': r'$\mathit{Enterocloster}$ $\mathit{bolteae}$ strain ATCC BAA-613',
                        'locus_definition': r'$\mathit{Enterocloster}$ $\mathit{bolteae}$',
                        'phylum': 'Firmicutes',
                        # 'ir_pair_region_with_margins': (6453300, 6458750), # moved away from this one because the other repeats didn't seem to cause inversions in our data.
                        'ir_pair_region_with_margins': (6453300, 6456808),
                        'alignment_region': (6423000, 6479000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR5817725': {
                                # only 2 ref_variant reads here, but they are good enough, I think.
                                # double checked to see whether there is something especially interesting here, due to the low amount of reads supporting the
                                # ref variant, but didn't spot anything weird.
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR5817725.3.25390.1': {(6453817, 6453878, 6456247, 6456308)}, # inversion
                                    # 'SRR5817725.3.25797.1', # weak evidence for inversion. doesn't span the whole region
                                    # 'SRR5817725.3.32015.1', # weak evidence for inversion. doesn't span the whole region
                                    # 'SRR5817725.3.33400.1', # inversion at the N-terminus of the presumably functional CDS. slightly weak because the first alignment is short. doesn't span the whole region

                                    # added late (on 220304)
                                    'SRR5817725.3.26669.1': {(6454426, 6454447, 6455705, 6455726)}, # inversion
                                    'SRR5817725.3.32015.1': {(6453817, 6453878, 6456247, 6456308)}, # inversion
                                    'SRR5817725.3.33400.1': {(6453817, 6453878, 6456247, 6456308)}, # inversion
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,
                    },
                ],
                # didn't try: https://www.ncbi.nlm.nih.gov/nuccore/CP069520.1?report=graph 3218000 - it is an FDAARGOS and looks like a perfect example...
                #   (https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR13182360)
            },

            'RM specificity: R-M-DUF1016-S': { # a better name would be: 'RM specificity: R-M-DUF1016-S and more upstream M and R subunits': {
                # just in case, used blastp to blast ALJ57861.1 (the DUF1016 in CP012801.1) to both AUR48659.1 (the DOC in CP025931.1)
                # and AUR47984.1 (the DNA-binding protein in CP025931.1), and got no alignments.

                'CP046428.1': [ # Phocaeicola dorei strain JR05
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP046428.1?report=graph&v=3453614:3464013
                        'presumably_relevant_cds_region': (3453614, 3464013),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QJR55863.1&FULL
                        'target_gene_product_description': 'Type I restriction-modification HsdS',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'HsdR; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QJR55859.1&FULL
                            'HsdM; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QJR55860.1&FULL
                            'DUF1016 (pfam06250) domain-containing protein; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QJR55861.1&FULL
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QJR55862.1&FULL - no conserved domains identified, and
                            # blastp QJR55862.1 to nr resulted in only alignments to hypothetical proteins.
                            'A hypothetical protein'
                        ),

                        'locus_description_for_table_3': 'Type I restriction-modification, with core CDS order HsdR-HsdM-HsdS, and with '
                                                         'a CDS encoding a DUF1016 (pfam06250) domain between core CDSs',

                        'longest_linked_repeat_cds_region_and_protein_id': ((3459162, 3460715), 'QJR55863.1'),
                        # https://ncbi.nlm.nih.gov/protein/QJR55863.1
                        # 'locus_definition': r'$\mathit{Phocaeicola}$ $\mathit{dorei}$ strain JR05',
                        'locus_definition': r'$\mathit{Phocaeicola}$ $\mathit{dorei}$',
                        'phylum': 'Bacteroidetes',
                        #     repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0           96        10  3459353  3459448  3463046  3463141  9.360000e-32        3597
                        # 1          125        20  3459993  3460117  3462371  3462495  3.550000e-31        2253
                        # 2           40         3  3461612  3461651  3462767  3462806  1.690000e-11        1115
                        # 3           28         2  3461537  3461564  3462854  3462881  2.740000e-06        1289
                        # 4           21         0  3461283  3461303  3462314  3462334  1.040000e-05        1010
                        # 5           45         8  3461728  3461772  3462646  3462690  1.040000e-05         873
                        # 6           19         0  3460143  3460161  3462327  3462345  1.490000e-04        2165
                        # 7           36         6  3460650  3460685  3462598  3462633  5.660000e-04        1912
                        # 8           54        12  3461356  3461409  3463009  3463062  5.660000e-04        1599
                        # 9           21         1  3461481  3461501  3462917  3462937  5.660000e-04        1415
                        # 10          23         2  3460696  3460718  3462315  3462337  2.000000e-03        1596
                        # 11          37         8  3461242  3461278  3462586  3462622  4.400000e-01        1307
                        # 12          19         2  3462134  3462152  3463315  3463333  4.400000e-01        1162
                        # 13          15         1  3460004  3460018  3462134  3462148  1.700000e+00        2115
                        # 14          20         3  3461501  3461520  3462010  3462029  6.400000e+00         489
                        # these two can be merged.
                        # 3459993  3460117  3462371  3462495
                        # 3460143  3460161  3462327  3462345
                        # the merged one:
                        # 3459993  3460161  3462327  3462495

                        'ir_pair_region_with_margins': (3458800, 3463700),
                        'alignment_region': (3438000, 3484000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR10666959': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR10666959.1.105015.1': {(3459353, 3459448, 3463046, 3463141)}, # inversion
                                    'SRR10666959.1.10583.1': {(3459353, 3459448, 3463046, 3463141)}, # inversion
                                    'SRR10666959.1.116005.1': {(3459353, 3459448, 3463046, 3463141)}, # inversion
                                    # 'SRR10666959.1.12010.1', # inversion. doesn't span the whole region
                                    'SRR10666959.1.133492.1': {(3460696, 3460718, 3462315, 3462337)}, # another inversion
                                    'SRR10666959.1.134221.1': {(3459353, 3459448, 3463046, 3463141)}, # inversion
                                    'SRR10666959.1.134501.1': {(3459353, 3459448, 3463046, 3463141)}, # inversion
                                    # 'SRR10666959.1.137476.1', # inversion. doesn't span the whole region
                                    'SRR10666959.1.156899.1': {(3459353, 3459448, 3463046, 3463141), (3459993, 3460161, 3462327, 3462495)}, # a beautiful cassette switch
                                    # 'SRR10666959.1.36504.1', # inversion. doesn't span the whole region
                                    'SRR10666959.1.49742.1': {(3459353, 3459448, 3463046, 3463141)}, # inversion
                                    'SRR10666959.1.56424.1': {(3459993, 3460161, 3462327, 3462495), (3461283, 3461303, 3462314, 3462334)}, # a nice rearrangement that might be achieved by 2 inversions
                                    'SRR10666959.1.86178.1': {(3459353, 3459448, 3463046, 3463141)}, # inversion

                                    # added late (on 220304)
                                    'SRR10666959.1.12010.1': {(3459993, 3460161, 3462327, 3462495)}, # inversion
                                    'SRR10666959.1.137476.1': {(3461283, 3461303, 3462314, 3462334)}, # inversion
                                    'SRR10666959.1.36504.1': {(3459993, 3460161, 3462327, 3462495)}, # inversion


                                    'SRR10666959.1.140776.1': {(3459353, 3459448, 3463046, 3463141)},
                                },
                                'complex_variant_ir_pairs_to_variant_regions_and_types': {
                                    frozenset({(3459993, 3460161, 3462327, 3462495), (3461283, 3461303, 3462314, 3462334)}): (
                                        ((3460161, 3461303), {'moved', 'inverted'}), ((3461303, 3462314), {'moved'}),
                                    ),
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            (3461281, 3461850): 'HsdS', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QJR55865.1&FULL
                            (3453614, 3456010): 'HsdR', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QJR55859.1&FULL
                        },

                        'describe_in_the_paper': True,
                    },
                ],
                # 'CP042282.1': [ # Bacteroides xylanisolvens strain APCS1/XY
                #     {
                #         'ir_pair_region_with_margins': (3250780, 3255043),
                #         'alignment_region': (3230000, 3276000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR13664849': {
                #                 # no alignment spanned the whole IR pair region, suggesting that the assembly is wrong?? ha. maybe the nuccore was updated???
                #                 # this means that what i need to find here is evidence for inversions relative to the cassette switch i see, which i guess was the dominant
                #                 # variant in the sequenced sample. alright. not reporting any "cassette switches"
                #
                #                 # (double checked, and it doesn't seem like there is something especially interesting in this presumably wrong (or not matching) assembly)
                #
                #                 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                #                     'SRR13664849.1.104293.1', # a pretty nice cassette switch, though the read doesn't fully align (though I guess this is really good enough)
                #                     'SRR13664849.1.109165.1', # a beautiful cassette switch
                #                     'SRR13664849.1.12030.1', # a beautiful cassette switch
                #                     'SRR13664849.1.122331.1', # a beautiful cassette switch
                #                     'SRR13664849.1.152699.1', # inversion (relative both to the dominant variant (cassette switch) and to the reference genome)
                #                     'SRR13664849.1.82244.1', # weak evidence (or not so week) for inversion (or inversions) relative to the dominant (cassette switch) variant.
                #                 },
                #             },
                #         },
                #         'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
                # 'CP081920.1': [ # Bacteroides caccae strain BFG-100
                #             'SRR15521847': {
                # 'CP012801.1': [ # Bacteroides cellulosilyticus strain WH2
                #     {
                #         'ir_pairs': (
                #             (591450, 591629, 595329, 595508),
                #             (592065, 592319, 594642, 594896),
                #             (592265, 592298, 593202, 593235),
                #         ),
                #         'alignment_region': (571000, 616000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR2531247': {
                #                 'SRR2531247.1.149488.1', # inversion at the N-terminus of the small RM specificity gene.
                #                 'SRR2531247.1.157490.1', # a beautiful cassette switch.
                #             },
                #             # 'SRR2531765': {
                #             # 'SRR2531772': {
                #             #     'SRR2531772.1.11892.1', # kind of weak evidence for inversion at the N-terminus of the bigger RM specificity genes...
                #             #     # I guess these two came from the same bacterium...
                #             #     'SRR2531772.1.11892.3', # kind of weak evidence for inversion at the N-terminus of the bigger RM specificity genes...
                #             # },
                #             # 'SRR2531775': {
                #         },
                        # 'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
            },

            'RM specificity: R-M-RhuM-S': {
                'CP081899.1': [ # Bacteroides salyersiae strain BFG-288
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP081899.1?report=graph&v=2594728:2605840
                        'presumably_relevant_cds_region': (2594728, 2605840),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBD18070.1&FULL
                        'target_gene_product_description': 'Type I restriction-modification HsdS',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'HsdR; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBD18066.1&FULL
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBD18067.1&FULL
                            'GIY-YIG nuclease (cl15257) domain-containing protein; ' 
                            'HsdM; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBD18068.1&FULL
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=UBD18069.1&FULL
                            'A protein containing a dinD (PRK11525) domain and a RhuM (pfam13310) domain'
                        ),

                        'locus_description_for_table_3': 'Type I restriction-modification, with core CDS order HsdR-HsdM-HsdS, and with '
                                                         'a CDS encoding a dinD (PRK11525) domain and a RhuM (pfam13310) domain, as well as '
                                                         'a CDS encoding a GIY-YIG nuclease (cl15257) domain, between core CDSs',


                        # 'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{salyersiae}$ strain BFG-288',
                        'longest_linked_repeat_cds_region_and_protein_id': ((2601241, 2602485), 'UBD18070.1'),
                        # https://ncbi.nlm.nih.gov/protein/UBD18070.1
                        'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{salyersiae}$',
                        'phylum': 'Bacteroidetes',
                        #    repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0          45         0  2601276  2601320  2604824  2604868  1.190000e-19        3503
                        # 1          38         5  2601796  2601833  2604323  2604360  6.410000e-07        2489
                        # 2          17         0  2601844  2601860  2603085  2603101  2.000000e-03        1224
                        # 3          19         1  2603113  2603131  2604290  2604308  7.000000e-03        1158
                        # 4          20         2  2603525  2603544  2604048  2604067  1.000000e-01         503
                        # 5          16         1  2603468  2603483  2604730  2604745  3.900000e-01        1246
                        # 6          15         1  2601486  2601500  2604589  2604603  1.500000e+00        3088
                        # 7          15         1  2601973  2601987  2604853  2604867  1.500000e+00        2865
                        # 'ir_pair_region_with_margins': (2600764, 2605380),
                        'ir_pair_region_with_margins': (2600764, 2605999),
                        'alignment_region': (2580000, 2626000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR15521845': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # 'SRR15521845.1.10924.1', # a nice inversion at the N-terminus of the left small CDS. doesn't cover the whole region.
                                    'SRR15521845.1.10925.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    'SRR15521845.1.10926.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    'SRR15521845.1.10927.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    'SRR15521845.1.10928.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    'SRR15521845.1.134713.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    # 'SRR15521845.1.134714.1', # a nice inversion at the N-terminus of the left small CDS. is it the same one as 134713? doesn't cover the whole region.
                                    'SRR15521845.1.134715.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    # 'SRR15521845.1.134716.1', # a nice inversion at the N-terminus of the left small CDS. is it the same one as 134713? doesn't cover the whole region.
                                    'SRR15521845.1.184528.1': {(2601276, 2601320, 2604824, 2604868), (2601796, 2601833, 2604323, 2604360), (2601844, 2601860, 2603085, 2603101)}, # cassette switch + inversion
                                    'SRR15521845.1.184529.1': {(2601276, 2601320, 2604824, 2604868), (2601796, 2601833, 2604323, 2604360), (2601844, 2601860, 2603085, 2603101)}, # cassette switch + inversion
                                    'SRR15521845.1.194488.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    'SRR15521845.1.2979.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    'SRR15521845.1.2980.1': {(2601844, 2601860, 2603085, 2603101)}, # inversion
                                    # 'SRR15521845.1.57829.1', # a nice inversion at the N-terminus of the left small CDS. doesn't cover the whole region.
                                    # 'SRR15521845.1.57831.1', # a nice inversion at the N-terminus of the left small CDS. is it the same one as 57829? doesn't span the whole region.
                                    # 'SRR15521845.1.57833.1', # a nice inversion at the N-terminus of the left small CDS. is it the same one as 57829? - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8

                                    # added late (on 220304)
                                    # Ugh. mauve fails to identify the cassette switch here. I guess mauve is wrong, but for the sake of time, I grudgingly mark this read as a
                                    # not_non_ref_variant. ugh.
                                    # 'SRR15521845.1.100732.1': {(2601276, 2601320, 2604824, 2604868)}, # inversion


                                    'SRR15521845.1.100731.1': {(2601276, 2601320, 2604824, 2604868)},
                                    'SRR15521845.1.70849.1': {(2601844, 2601860, 2603085, 2603101)},
                                },
                                'not_non_ref_variant_read_name_to_anomaly_description': {
                                    # I guess this read is perfectly fine, but it is here because mauve failed for it. see above.
                                    'SRR15521845.1.100732.1': get_mauve_failed_explanation({(2601276, 2601320, 2604824, 2604868)}),
                                },
                                'complex_variant_ir_pairs_to_variant_regions_and_types': {
                                    frozenset({(2601276, 2601320, 2604824, 2604868), (2601796, 2601833, 2604323, 2604360), (2601844, 2601860, 2603085, 2603101)}): (
                                        ((2601320, 2601860), {'moved', 'inverted'}), ((2601860, 2603085), {'inverted'}),
                                        ((2603085, 2604323), set()), ((2604323, 2604824), {'moved', 'inverted'}),
                                    ),
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,
                    },
                ],
            },

            # https://www.uniprot.org/uniprot/A0A0S7BXK1: seems like DOC and RhuM are of the same family. oh well.
            # However, I have used blastp to blast AUR48659.1 (the DOC in CP025931.1) to UBD18069.1 (the RhuM in CP081899.1), and got no alignments.
            # I have also blasted AUR47984.1 (the DNA-binding protein in CP025931.1) to UBD18069.1 (the RhuM in CP081899.1), and got a single pretty
            # short and bad-looking alignment. So maybe they aren't that similar?
            'RM specificity: R-M-Fic/DOC-S': {
                'NZ_CP082886.1': [ # Bacteroides nordii strain FDAARGOS_1461
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP082886.1?report=graph&v=4364190:4373966
                        'presumably_relevant_cds_region': (4364190, 4373966),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_223381563.1&FULL
                        'target_gene_product_description': 'Type I restriction-modification HsdS',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'HsdR; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_025867068.1&FULL
                            'HsdM; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_002558267.1&FULL
                            # blastp of WP_002558268.1 to nr results in many 'DNA-binding protein' CDSs
                            'DNA-binding domain-containing protein; '
                            'Fic/DOC (pfam02661) domain-containing protein' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_002558269.1&FULL
                        ),

                        'locus_description_for_table_3': 'Type I restriction-modification, with core CDS order HsdR-HsdM-HsdS, and with '
                                                         'a CDS encoding a hypothetical protein, as well as '
                                                         'a CDS encoding a Fic/DOC (pfam02661) domain, between core CDSs',

                        # WP_002558268.1 is homologous to many 'DNA-binding protein' CDSs (according to first 100 hits of blastp)

                        'longest_linked_repeat_cds_region_and_protein_id': ((4369547, 4371175), 'WP_223381563.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_223381563.1

                        # 'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{nordii}$ strain FDAARGOS_1461',
                        'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{nordii}$',
                        'phylum': 'Bacteroidetes',
                        #    repeat_len  mismatch    left1   right1    left2   right2    evalue  spacer_len
                        # 0         843         0  4369738  4370580  4372256  4373098  0.000000        1675
                        # 1          21         0  4370575  4370595  4371663  4371683  0.000008        1067
                        # 2          39         7  4370624  4370662  4372174  4372212  0.000456        1511
                        # 3          20         2  4369790  4369809  4372196  4372215  0.094000        2386
                        # 4          19         2  4369335  4369353  4372216  4372234  0.360000        2862
                        # 5          15         1  4370827  4370841  4373141  4373155  1.400000        2299
                        # 6          15         1  4373468  4373482  4373504  4373518  1.400000          21
                        # 7          17         2  4369589  4369605  4372397  4372413  5.100000        2791
                        # 8          17         2  4370820  4370836  4372218  4372234  5.100000        1381
                        # 9          17         2  4371489  4371505  4373184  4373200  5.100000        1678
                        'ir_pair_region_with_margins': (4369200, 4373600),
                        'alignment_region': (4349000, 4394000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR16259013': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    # 'SRR16259013.1.173912.1', # inversion, but kind of weak evidence. doesn't span the whole region
                                    # 'SRR16259013.1.173913.1', # inversion, but kind of weak evidence. doesn't span the whole region
                                    # 'SRR16259013.1.173915.1', # inversion, but kind of weak evidence. doesn't span the whole region
                                    # 'SRR16259013.1.199607.1', # inversion, but kind of weak evidence. doesn't span the whole region
                                    # 'SRR16259013.1.215242.1', # inversion, but kind of weak evidence. doesn't span the whole region
                                    'SRR16259013.1.231616.1': {(4369738, 4370580, 4372256, 4373098), (4370575, 4370595, 4371663, 4371683)}, # beautiful multiple rearrangements
                                    'SRR16259013.1.253615.1': {(4370575, 4370595, 4371663, 4371683)}, # inversion
                                    'SRR16259013.1.29850.1': {(4369738, 4370580, 4372256, 4373098)}, # inversion


                                    # added late (on 220304)
                                    'SRR16259013.1.173912.1': {(4369738, 4370580, 4372256, 4373098)}, # inversion
                                    'SRR16259013.1.173913.1': {(4369738, 4370580, 4372256, 4373098)}, # inversion
                                    'SRR16259013.1.173915.1': {(4369738, 4370580, 4372256, 4373098)}, # inversion


                                    'SRR16259013.1.132245.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.163220.1': {(4370575, 4370595, 4371663, 4371683)},
                                    'SRR16259013.1.163222.1': {(4370575, 4370595, 4371663, 4371683)},
                                    'SRR16259013.1.163223.1': {(4370575, 4370595, 4371663, 4371683)},
                                    'SRR16259013.1.163224.1': {(4370575, 4370595, 4371663, 4371683)},
                                    'SRR16259013.1.163225.1': {(4370575, 4370595, 4371663, 4371683)},
                                    'SRR16259013.1.163226.1': {(4370575, 4370595, 4371663, 4371683)},
                                    'SRR16259013.1.164467.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.164468.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.173911.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.173914.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.173916.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.199605.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.199606.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.199608.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.199609.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.199610.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.199611.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.199612.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.51059.1': {(4369738, 4370580, 4372256, 4373098)},
                                    'SRR16259013.1.89959.1': {(4370575, 4370595, 4371663, 4371683)},
                                },
                                # 'not_non_ref_variant_read_name_to_anomaly_description': {
                                #     'SRR16259013.1.223609.1', # inverted duplication
                                # },
                                'inaccurate_or_not_beautiful_mauve_alignment': {
                                    'SRR16259013.1.223609.1',
                                    'SRR16259013.1.177592.1',
                                },
                                'complex_variant_ir_pairs_to_variant_regions_and_types': {
                                    frozenset({(4369738, 4370580, 4372256, 4373098), (4370575, 4370595, 4371663, 4371683)}): (
                                        ((4370595, 4371663), {'moved'}), ((4371663, 4372256), {'moved', 'inverted'}),
                                    ),
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'cds_start_and_end_to_curated_product_class': {
                            (4364190, 4366496): 'HsdR', # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_025867068.1&FULL
                        },

                        'describe_in_the_paper': True,
                    },
                ],
                # 'CP025931.1': [ # Porphyromonas gingivalis strain TDC 60
                #     {
                #         'ir_pairs': (
                #             (634882, 634971, 637462, 637551),
                #             (635509, 635701, 636726, 636918),
                #         ),
                #         'alignment_region': (614000, 658000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR5810500': { # also, there seemed to be a huge number of reads with inverted duplications...
                #                 'SRR5810500.1.9923.1', # the read isn't very nice, but clearly there is an inversion in the middle of the CDS.
                #                 'SRR5810500.1.9924.1', # the read isn't very nice, but clearly there is an inversion in the middle of the CDS.
                #             },
                #         },
                        # 'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
            },

            'RM specificity: M-invertibleSs-R': {
                'NZ_CP059830.1': [ # Lactobacillus ultunensis strain Kx293C1 plasmid unnamed
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP059830.1?report=graph&v=10032:18186
                        'presumably_relevant_cds_region': (10032, 18186),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_201247869.1&FULL
                        'target_gene_product_description': 'Type I restriction-modification HsdS',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'HsdM' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_040529399.1&FULL
                        ),
                        'presumably_associated_downstream_gene_product_descriptions': (
                            'HsdR' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_007125626.1&FULL
                        ),

                        'locus_description_for_table_3': 'Type I restriction-modification, with core CDS order HsdM-HsdS-HsdR',

                        'longest_linked_repeat_cds_region_and_protein_id': ((15345, 16556), 'WP_201247869.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_201247869.1

                        # 'locus_definition': r'$\mathit{Lactobacillus}$ $\mathit{ultunensis}$ strain Kx293C1',
                        'locus_definition': r'$\mathit{Lactobacillus}$ $\mathit{ultunensis}$',
                        'phylum': 'Firmicutes',
                        'ir_pair_region_with_margins': (12579, 17027),
                        'alignment_region': (1, 38000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR14480597': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR14480597.1.24647.1': {(13146, 13662, 15981, 16497)}, # inversion
                                    # 'SRR14480597.1.26670.1', # weak evidence for inversion. doesn't span the whole region
                                    'SRR14480597.1.50426.1': {(13146, 13662, 15981, 16497)}, # inversion

                                    # added late (on 220304)
                                    'SRR14480597.1.26670.1': {(13146, 13662, 15981, 16497)}, # inversion
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,
                    },
                ],
            },

            'phage tail: upstream phage tail protein I and downstream tail fiber assembly': {
                # 'NZ_CP057125.1': [ # Citrobacter sp. RHB36-C18
                #     {
                #         # this case might be somewhat special, as the shorter phage tail CDS doesn't have a downstream tail fiber assembly CDS.
                #         # so maybe only a cassette switch makes sense in this case? though maybe the CDSs are simply not in an operon. they are
                #         # 63 bps apart...
                #         'ir_pairs': (
                #             (3113030, 3113109, 3114674, 3114753),
                #             (3113188, 3113229, 3114554, 3114595),
                #         ),
                #         'alignment_region': (3093000, 3135000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR12299098': {
                #                 'SRR12299098.1.152949.1', # inversion at the N-terminus of the phage tail protein
                #                 'SRR12299098.1.15600.1', # inversion at the N-terminus of the phage tail protein
                #                 'SRR12299098.1.206609.1', # inversion at the N-terminus of the phage tail protein
                #                 'SRR12299098.1.231691.1', # inversion at the N-terminus of the phage tail protein, though part of the read doesn't align.
                #                 'SRR12299098.1.248529.1', # inversion at the N-terminus of the phage tail protein
                #                 'SRR12299098.1.257586.1', # inversion at the N-terminus of the phage tail protein
                #                 'SRR12299098.1.67726.1', # inversion at the N-terminus of the phage tail protein
                #                 'SRR12299098.1.70867.1', # inversion at the N-terminus of the phage tail protein
                #             },
                #         },
                #         'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
                'CP056267.1': [ # Citrobacter sp. RHBSTW-00887
                    # here we do have a downstream tail fiber assembly CDS for both the long and short phage tail CDSs.
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP056267.1?report=graph&v=5090687:5097011
                        'presumably_relevant_cds_region': (5090687, 5097011),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_181546899.1&FULL
                        'target_gene_product_description': 'Phage tail fiber (COG5301) domain-containing protein',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'Multiple phage proteins'
                        ),

                        'presumably_associated_downstream_gene_product_descriptions': (
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_181546896.1&FULL
                            'Tail fiber assembly (pfam02413) domain-containing protein'
                        ),

                        'locus_description_for_table_3': 'Prophage',

                        # WP_181546901.1 - 'phage tail protein I'
                        # WP_181546903.1 - 'baseplate assembly protein' right upstream to WP_181546901.1

                        # blasted WP_181546899.1 to QWT41050.1, and got a single alignment, evalue=4e-143, from first to last aa of both proteins.

                        # 220403: two domains are currently identified in WP_181546899.1:
                        # DUF3751 (described as "Phage tail-collar fibre protein. [...]"), and
                        # COG5301 (described as "Phage-related tail fibre protein [Mobilome: prophages, transposons];")
                        # 'longest_linked_repeat_cds_region_and_protein_id': ((5092989, 5094518), 'WP_181546899.1'),
                        'longest_linked_repeat_cds_region_and_protein_id': ((5092989, 5094518), 'QLS57046.1'),
                        # https://ncbi.nlm.nih.gov/protein/QLS57046.1
                        # blastp of QLS57046.1 to P03744 (T4 gp37) resulted in an alignment with evalue=2e-15.
                        # blastp of QLS57046.1 to P18771 (T4 gp34) resulted in no significant alignments.

                        'locus_definition': r'$\mathit{Citrobacter}$ sp. RHBSTW-00887',
                        'phylum': 'Proteobacteria',
                        #    repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0         211        23  5091335  5091545  5093260  5093470  3.950000e-76        1714
                        # 1         228        42  5091770  5091997  5092802  5093029  5.590000e-53         804
                        # 2          51        11  5092020  5092070  5092729  5092779  2.320000e-04         658
                        # 3          16         2  5092118  5092133  5093934  5093949  9.900000e+00        1800
                        'ir_pair_region_with_margins': (5090832, 5093973),
                        'alignment_region': (5070000, 5114000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR12299129': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR12299129.1.138850.1': {(5091335, 5091545, 5093260, 5093470)}, # inversion
                                    'SRR12299129.1.151265.1': {(5091335, 5091545, 5093260, 5093470)}, # inversion
                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,
                    },
                ],
                # 'NZ_CP074159.1': [ # Enterobacter sp. JBIWA005
                #     # here we do have a downstream tail fiber assembly CDS for both the long and short phage tail CDSs.
                #             'SRR14684531': {
            },
            'phage tail: downstream transporter and endonuclease': {
                'CP076386.1': [ # Dickeya dadantii strain S3-1
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP076386.1?report=graph&v=90637:101047
                        'presumably_relevant_cds_region': (90637, 101047),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QWT41050.1&FULL
                        'target_gene_product_description': 'Phage tail fiber (COG5301) domain-containing protein',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'Multiple phage proteins'
                        ),

                        'presumably_associated_downstream_gene_product_descriptions': (
                            'DNA endonuclease SmrA; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QWT42989.1&FULL
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QWT41051.1&FULL
                            'MFS transporter domain-containing protein; '
                            'Phytase (pfam13449) domain-containing protein' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QWT41052.1&FULL
                        ),

                        'locus_description_for_table_3': 'Prophage, with a CDS encoding DNA endonuclease SmrA, '
                                                         'a CDS encoding a MFS transporter domain, '
                                                         'and a CDS encoding a Phytase (pfam13449) domain',


                        # blasted WP_181546899.1 to QWT41050.1, and got a single alignment, evalue=4e-143, from first to last aa of both proteins.
                        # blasted QWT41050.1 to QQB93820.1, and got two relatively short alignments, evlaues=4e-6,3e-5

                        # 220403: three domains are currently identified in QWT41050.1:
                        # DUF3751 (described as "Phage tail-collar fibre protein. [...]"),
                        # Collar (described as "Phage Tail Collar Domain. [...]"), and
                        # COG5301 (described as "Phage-related tail fibre protein [Mobilome: prophages, transposons];")
                        'longest_linked_repeat_cds_region_and_protein_id': ((93499, 94659), 'QWT41050.1'),
                        # https://ncbi.nlm.nih.gov/protein/QWT41050.1
                        # blastp of QWT41050.1 to P03744 (T4 gp37) resulted in two alignments, with evalues 2e-19 and 2e-5.
                        # blastp of QWT41050.1 to P18771 (T4 gp34) resulted in no significant alignments.

                        # 'locus_definition': r'$\mathit{Dickeya}$ $\mathit{dadantii}$ strain S3-1',
                        'locus_definition': r'$\mathit{Dickeya}$ $\mathit{dadantii}$',
                        'phylum': 'Proteobacteria',
                        #    repeat_len  mismatch  left1  right1   left2  right2         evalue  spacer_len
                        # 0         241         0  94191   94431   98782   99022  1.050000e-132        4350
                        # 1         178         5  94609   94786   98460   98637   1.470000e-87        3673
                        # 2          38         0  98813   98850  100092  100129   3.280000e-15        1241
                        # 3          73        12  98903   98975   99967  100039   1.250000e-14         991
                        # 4          32         2  98600   98631  100389  100420   2.900000e-08        1757
                        # 5          15         1  95836   95850   98933   98947   3.700000e+00        3082
                        # 6          15         1  95925   95939   99161   99175   3.700000e+00        3221
                        # 'ir_pair_region_with_margins': (93691, 100926), # moved away from this one because the other repeats didn't seem to cause inversions in our data.
                        # 'ir_pair_region_with_margins': (93691, 99522), # moved to the one below in hope to get a better looking mauve alignment.
                        'ir_pair_region_with_margins': (93291, 99922),
                        'alignment_region': (73000, 121000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR14725125': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR14725125.1.1023276.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1028200.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1036833.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.108400.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1089318.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    # 'SRR14725125.1.1143613.1', # inversion - oops. seems like i had a typo here in the read name... or maybe it wasn't identified with seed_len == 8
                                    'SRR14725125.1.1165442.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1188942.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1196256.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.119873.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1212880.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1226441.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    # 'SRR14725125.1.1226933.1', # inversion. doesn't cover the whole region.
                                    'SRR14725125.1.1282795.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1318922.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1342411.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1344581.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1367382.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1404846.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1439899.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1486764.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1492125.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1525299.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1529084.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    # 'SRR14725125.1.1538390.1', # inversion. doesn't cover the whole region.
                                    'SRR14725125.1.1543851.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1548375.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1559404.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.157156.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.158195.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1585455.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    # 'SRR14725125.1.1606803.1', # inversion. doesn't span the whole region.
                                    'SRR14725125.1.173877.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.182313.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.183270.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.18658.1': {(94191, 94431, 98782, 99022)}, # inversion

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR14725125.1.220516.1': {(94191, 94431, 98782, 99022)}, # inversion + inverted duplication, but with one copy much further from ref.

                                    'SRR14725125.1.228042.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.25161.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.273563.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.321226.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.325978.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.376043.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.39169.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.393766.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.432191.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.526582.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.527001.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.531022.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.583557.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.609965.1': {(94191, 94431, 98782, 99022)}, # inversion

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR14725125.1.61151.1': {(94191, 94431, 98782, 99022)}, # inversion + inverted duplication,
                                    # but with one copy further from ref in a somewhat peculiar manner.

                                    'SRR14725125.1.612271.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.627894.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.641299.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.65083.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.676805.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.686488.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.691420.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.710318.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.732890.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.761575.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.763509.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.778577.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.781392.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.792465.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.797433.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.808267.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.808477.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.819267.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    # inversion. at first i was alarmed that 30k bases are missing, but i guess it is ok, because what are the chances to have such a huge homology
                                    # with another region in the genome.
                                    'SRR14725125.1.827086.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    # 'SRR14725125.1.829324.1', # inversion. doesn't span the whole region.
                                    'SRR14725125.1.839437.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.854382.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.860542.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.860641.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.865312.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.867091.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.87333.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.874554.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.880851.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.884298.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.898139.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.902246.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.915083.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.941352.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.942370.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.966296.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.972185.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.998247.1': {(94191, 94431, 98782, 99022)}, # inversion

                                    # added late (on 220304)
                                    'SRR14725125.1.1128112.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1226933.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.1606803.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.260587.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.617980.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.767868.1': {(94191, 94431, 98782, 99022)}, # inversion
                                    'SRR14725125.1.829324.1': {(94191, 94431, 98782, 99022)}, # inversion


                                    'SRR14725125.1.583580.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1250034.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.818728.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.152409.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.427020.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.275499.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.928633.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.205945.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1379155.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1105240.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1512935.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.43277.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.772217.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.336895.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.319918.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.735054.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1369094.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1094796.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1241104.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1306832.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.393388.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.954111.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.273939.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.884689.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1181603.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.155028.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.105900.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.354372.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.993346.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.373395.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.471339.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.388830.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.931738.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1374173.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.359744.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1520280.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1436729.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.944086.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.598743.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1043354.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1053616.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1209931.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.174126.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.64247.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.37415.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.420377.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1055736.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.715688.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.339216.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.394277.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.517512.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.324405.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1230623.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1491345.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1221636.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.675482.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.174935.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.538976.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1267300.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.749417.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1444639.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1009641.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.525768.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.154885.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1425058.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.6841.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.952658.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1328791.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1245089.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1143613.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1253398.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1263628.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.498555.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.323859.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.329185.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.454826.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.915550.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.336982.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.346081.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1476354.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.510844.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.609813.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.115324.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.944171.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1026424.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1150284.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1620469.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.215349.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.546050.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.700909.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1242969.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.252155.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.786605.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.98495.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.173587.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1495184.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1439399.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1011954.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.982081.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.163313.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1554579.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.419105.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1338427.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.953583.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.421610.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.924116.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.42830.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.856853.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1046770.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1117361.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.306294.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1251342.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.560045.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.910941.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.593163.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.470769.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.952497.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.968429.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.109792.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.791711.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.335328.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.196088.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1216227.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.529080.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1253532.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1412287.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.32114.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.239994.1': {(94191, 94431, 98782, 99022)},
                                    'SRR14725125.1.1227566.1': {(94191, 94431, 98782, 99022)},
                                },
                                'not_non_ref_variant_read_name_to_anomaly_description': {
                                    'SRR14725125.1.540615.1': READ_DOESNT_MATCH_ANY_VARIANT_THAT_CAN_BE_REACHED_FROM_REFERENCE_EXPLANATION,
                                    # 'SRR14725125.1.1034700.1',
                                    # 'SRR14725125.1.1058400.1',
                                    # 'SRR14725125.1.1098264.1',
                                    # 'SRR14725125.1.1104963.1',
                                }
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,
                    },
                ],
                # 'CP031560.1': [ # Dickeya dianthicola strain ME23
                #     {
                #         'ir_pairs': (
                #             (2700670, 2700895, 2705190, 2705415),
                #             (2701076, 2701159, 2704959, 2705042),
                #         ),
                #         'alignment_region': (2680000, 2726000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR7866019': {
                #                 'SRR7866019.1.10317.3', # inversion
                #                 'SRR7866019.1.10317.5', # inversion
                #                 'SRR7866019.1.105055.5', # inversion
                #                 'SRR7866019.1.107210.3', # inversion
                #                 'SRR7866019.1.41124.1', # inversion
                #                 'SRR7866019.1.41124.3', # inversion
                #                 'SRR7866019.1.48347.1', # inversion
                #                 'SRR7866019.1.55806.3', # inversion
                #             },
                #         },
                        # 'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
                # 'CP020872.1': [ # Dickeya fangzhongdai strain PA1
                #     {
                #         'ir_pairs': (
                #             (2837675, 2837907, 2842334, 2842566),
                #             (2838324, 2838364, 2842021, 2842061),
                #             (2838195, 2838250, 2842124, 2842179),
                #             (2842139, 2842179, 2843881, 2843921),
                #             (2842342, 2842524, 2843506, 2843688),
                #         ),
                #         'alignment_region': (2817000, 2864000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR5456972': {
                #                 'SRR5456972.1.10125.3', # weak evidence for inversion
                #                 'SRR5456972.1.104331.1', # weak evidence for inversion
                #                 'SRR5456972.1.104331.3', # weak evidence for inversion (i guess this read came from the same bacterium as 104331.1)
                #                 'SRR5456972.1.10630.1', # weak evidence for inversion
                #                 'SRR5456972.1.10630.3', # weak evidence for inversion (i guess this read came from the same bacterium as 10630.1)
                #                 'SRR5456972.1.107226.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.107634.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.107634.3', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.112283.1', # weak evidence for inversion
                #                 'SRR5456972.1.112283.3', # weak evidence for inversion
                #                 'SRR5456972.1.120991.5', # weak evidence for inversion
                #                 'SRR5456972.1.126080.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.126080.3', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.126809.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.131215.1', # weak evidence for inversion
                #                 'SRR5456972.1.132044.1', # weak evidence for inversion
                #                 'SRR5456972.1.138276.1', # weak evidence for inversion. ok, no more reporting of weak evidence.
                #                 'SRR5456972.1.141821.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.141821.3', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.146852.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.147364.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.150686.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.150686.3', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.159062.1', # inversion at the N-terminus of the shortest phage tail protein CDS
                #                 'SRR5456972.1.17064.5', # inversion
                #                 'SRR5456972.1.18247.3', # inversion
                #                 'SRR5456972.1.18348.1', # inversion
                #                 'SRR5456972.1.27261.3', # inversion
                #                 'SRR5456972.1.28143.1', # inversion
                #                 'SRR5456972.1.29339.1', # inversion
                #                 'SRR5456972.1.36674.1', # inversion
                #                 'SRR5456972.1.38095.1', # inversion
                #                 'SRR5456972.1.38176.1', # inversion
                #                 'SRR5456972.1.49354.1', # inversion
                #                 'SRR5456972.1.52970.1', # inversion
                #                 'SRR5456972.1.52970.3', # inversion
                #                 'SRR5456972.1.62775.1', # inversion
                #                 'SRR5456972.1.65926.1', # inversion
                #                 'SRR5456972.1.76875.3', # inversion
                #                 'SRR5456972.1.87205.1', # inversion
                #                 'SRR5456972.1.88231.1', # inversion
                #                 'SRR5456972.1.9499.1', # inversion
                #                 'SRR5456972.1.95435.3', # inversion
                #             },
                #         },
                        # 'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
            },

            'phage tail: upstream DUF2313 and downstream tail fiber assembly': {
                'CP066032.1': [ # Escherichia coli strain FDAARGOS_1059
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/CP066032.1?report=graph&v=4020231:4027133
                        'presumably_relevant_cds_region': (4020231, 4027133),

                        # blastp QQB93820.1 to NP_050653.1 (the famous tail fiber targeted by programmed inversions in phage Mu) shows high sequence similarity.
                        # Involvement of the invertible G segment in bacteriophage Mu tail fiber biosynthesis - ScienceDirect -
                        # https://www.sciencedirect.com/science/article/pii/004268228490299X (1984)
                        'target_gene_product_description': 'Phage tail fiber protein, homologous to the variable tail fiber protein of phage Mu (NP_050653.1)',

                        'presumably_associated_upstream_gene_product_descriptions': (
                            'Multiple phage proteins'
                        ),

                        'presumably_associated_downstream_gene_product_descriptions': (
                            # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=QQB93819.1&FULL
                            'Tail fiber assembly (pfam02413) domain-containing protein'
                        ),

                        'locus_description_for_table_3': 'Prophage',

                        # QQB93821.1 - DUF2313 domain-containing protein
                        # blasted WP_181546901.1 ('phage tail protein I' from NZ_CP056267.1, which is in the same position as QQB93821.1 here) to QQB93821.1, and
                        #   got no alignments. so I guess it is ok to keep these two distinct?

                        # QQB93822.1 - 'baseplate J/gp47 family protein' right upstream to QQB93821.1

                        # blasted WP_181546903.1 ('baseplate assembly protein' right upstream to WP_181546901.1) to QQB93822.1, and got no alignments.

                        # QQB93823.1 - phage GP46 family protein
                        # blasted WP_000108899.1 ('GPW/gp25 family protein' right upstream to WP_181546903.1) to QQB93823.1, and got no alignments

                        # QQB93824.1 - phage baseplate assembly protein V
                        # blasted WP_023276962.1 ('phage baseplate assembly protein V' right upstream to WP_000108899.1) to QQB93824.1, and got an alignment with
                        # evalue=8e-6. so I guess it does make sense to consider them as different systems???

                        # blasted WP_181546899.1 to QQB93820.1, and got no alignments. what.
                        # blasted QWT41050.1 to QQB93820.1, and got two relatively short alignments, evlaues=4e-6,3e-5

                        # 220403: only one short domain is currently identified in QQB93820.1: Phage_fiber_2 (described as "Phage tail fibre repeat; [...]")
                        'longest_linked_repeat_cds_region_and_protein_id': ((4023024, 4024490), 'QQB93820.1'),
                        # https://ncbi.nlm.nih.gov/protein/QQB93820.1
                        # blastp of QQB93820.1 to P03744 (T4 gp37) resulted in no significant alignments.
                        # blastp of QQB93820.1 to P18771 (T4 gp34) resulted in no significant alignments.

                        # 'locus_definition': r'$\mathit{Escherichia}$ $\mathit{coli}$ strain FDAARGOS_1059',
                        'locus_definition': r'$\mathit{Escherichia}$ $\mathit{coli}$',
                        'phylum': 'Proteobacteria',
                        #    repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
                        # 0          74        12  4022154  4022227  4022726  4022799  1.060000e-15         498
                        # 1          41         4  4020879  4020919  4023954  4023994  1.710000e-10        3034
                        # 2          56         9  4022013  4022068  4022885  4022940  1.710000e-10         816
                        # 3          43         8  4021931  4021973  4022980  4023022  1.050000e-04        1006
                        # 4          16         1  4024284  4024299  4024350  4024365  3.100000e-01          50
                        # 5          17         2  4021898  4021914  4023040  4023056  4.500000e+00        1125
                        'ir_pair_region_with_margins': (4020379, 4024494),
                        'alignment_region': (4000000, 4045000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR12825962': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR12825962.1.100759.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.109017.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.110992.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.112726.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    # 'SRR12825962.1.110992.3', # inversion at the N-terminus of the phage tail protein. doesn't cover the whole region.
                                    'SRR12825962.1.114047.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR12825962.1.116925.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion

                                    'SRR12825962.1.121800.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.123711.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.124054.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.125121.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.139250.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.149426.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.151971.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.159426.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.161042.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.20770.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    # 'SRR12825962.1.23138.1', # inversion. doesn't cover the whole region.
                                    'SRR12825962.1.26063.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.27177.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.34401.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.37143.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.43663.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    # 'SRR12825962.1.48622.1', # inversion. doesn't cover the whole region.

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR12825962.1.50809.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion

                                    'SRR12825962.1.50809.3': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    # 'SRR12825962.1.53505.1', # inversion. doesn't cover the whole region.
                                    'SRR12825962.1.53991.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.57320.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.57320.3': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.62733.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.68825.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.70739.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR12825962.1.72180.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion

                                    'SRR12825962.1.77004.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.7842.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.92314.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    # 'SRR12825962.1.77229.1', # inversion. doesn't cover the whole region.
                                    # 'SRR12825962.1.78190.1', # inversion. doesn't cover the whole region.

                                    # added late (on 220304)
                                    'SRR12825962.1.111579.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.11894.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.162071.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.43170.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.49379.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.73389.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.77229.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.78190.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.84716.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion


                                    'SRR12825962.1.130993.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion
                                    'SRR12825962.1.18912.1': {(4020879, 4020919, 4023954, 4023994)}, # inversion

                                },
                            },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,

                        # 'cds_start_and_end_to_curated_product_class': {
                        #     # https://www.ncbi.nlm.nih.gov/Structure/cdd/pfam10076 says: "Uncharacterized protein conserved in bacteria (DUF2313)
                        #     #    Members of this family of proteins comprise various hypothetical and putative bacteriophage tail proteins."
                        #     (4024490, 4025032): 'phage',
                        # },
                    },
                ],
            },

            'OM receptor: downstream SusD/RagB': {
                'NZ_CP012938.1': [ # Bacteroides ovatus strain ATCC 8483 (Bacteroidetes)
                    {
                        # https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP012938.1?report=graph&v=2845801:2857149
                        'presumably_relevant_cds_region': (2845801, 2857149),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_004299186.1&FULL
                        'target_gene_product_description': 'SusC',

                        'presumably_associated_downstream_gene_product_descriptions': (
                            'SusD; ' # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_004299187.1&FULL
                        ),

                        # https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi?SEQUENCE=WP_004299188.1&FULL
                        # it seems like the 'GTP-binding protein HflX' is not strictly associated... it appears only on one side, and not
                        # close enough to look like part of the operon...

                        'locus_description_for_table_3': 'A presumed operon (according to short distances between CDSs) composed of CDSs encoding '
                                                         'outer membrane proteins SusC and SusD',

                        'longest_linked_repeat_cds_region_and_protein_id': ((2847275, 2850391), 'WP_004299186.1'),
                        # https://ncbi.nlm.nih.gov/protein/WP_004299186.1

                        # 'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{ovatus}$ strain ATCC 8483',
                        'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{ovatus}$',
                        'phylum': 'Bacteroidetes',
                        'ir_pair_region_with_margins': (2849000, 2854000),
                        'alignment_region': (2825000, 2879000),
                        'sra_accession_to_variants_and_reads_info': {
                            'SRR2637689': {
                                'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                                    'SRR2637689.1.113473.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    # 'SRR2637689.1.114464.1', # inversion. doesn't span the whole region.
                                    'SRR2637689.1.118283.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.133846.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    # 'SRR2637689.1.13506.1', # inversion. doesn't span the whole region.
                                    # 'SRR2637689.1.13506.3', # inversion. doesn't span the whole region.
                                    # 'SRR2637689.1.13506.5', # inversion. doesn't span the whole region.
                                    'SRR2637689.1.137114.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    # 'SRR2637689.1.137114.3', # inversion. doesn't cover the whole region.
                                    'SRR2637689.1.149730.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    # 'SRR2637689.1.152969.1', # inversion. doesn't cover the whole region.

                                    # (assuming my code has no bugs) the alignments of the read that cover the required region in ref don't cover a region in the
                                    # read without gaps, so this read doesn't count...
                                    # 'SRR2637689.1.152969.3': {(2850098, 2850190, 2852883, 2852975)}, # inversion

                                    'SRR2637689.1.152969.5': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    # 'SRR2637689.1.27139.1', # inversion. doesn't span the whole region.
                                    # 'SRR2637689.1.27139.3', # inversion. doesn't span the whole region.
                                    # 'SRR2637689.1.31008.1', # inversion. doesn't span the whole region.

                                    # added late (on 220304)
                                    'SRR2637689.1.114464.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.13506.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.13506.3': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.13506.5': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.152969.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.27139.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.27139.3': {(2850098, 2850190, 2852883, 2852975)}, # inversion


                                    'SRR2637689.1.133633.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                    'SRR2637689.1.75104.1': {(2850098, 2850190, 2852883, 2852975)}, # inversion
                                },
                                'inaccurate_or_not_beautiful_mauve_alignment': {
                                    'SRR2637689.1.137990.1', # not sure it is inaccurate, but it doesn't look nice, and there are many more ref reads...
                                }
                            },
                            # 'SRR2637631': {
                            #     'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                            #         'SRR2637631.1.115788.1', # inversion
                            #         'SRR2637631.1.115788.3', # inversion
                            #         'SRR2637631.1.127277.3', # inversion
                            #         'SRR2637631.1.131079.3', # inversion
                            #         'SRR2637631.1.13862.3', # inversion
                            #         'SRR2637631.1.23914.1', # inversion
                            #         'SRR2637631.1.23914.3', # inversion
                            #         'SRR2637631.1.27983.1', # inversion
                            #         'SRR2637631.1.37276.1', # inversion
                            #         'SRR2637631.1.39719.1', # inversion
                            #         'SRR2637631.1.50663.1', # inversion
                            #         'SRR2637631.1.54222.1', # inversion
                            #         'SRR2637631.1.55138.1', # inversion
                            #         'SRR2637631.1.61607.1', # inversion
                            #         'SRR2637631.1.87166.1', # inversion
                            #     },
                            # },
                        },
                        'align_to_whole_genome_to_verify_evidence_reads': True,
                        # 'align_to_whole_genome_to_verify_evidence_reads': False,

                        'describe_in_the_paper': True,
                    },
                ],
                # 'CP065889.1': [ # Bacteroides uniformis strain FDAARGOS_901
                #             # 'SRR12045544': {
                #             #    # after more than 90h of blast, decided to give up on it, as i already found 2 other good SRAs.
                # 'CP081913.1': [ # Bacteroides stercoris strain BFG-121
                #     {
                #         'ir_pair_region_with_margins': (2039200, 2058000),
                #         'alignment_region': (2019000, 2079000),
                #         'sra_accession_to_variants_and_reads_info': {
                #             'SRR15521841': {
                #                 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
                #                     'SRR15521841.1.273513.1', # inversion at N-terminus plus cassette switch??
                #                     'SRR15521841.1.134851.1', # inversion at N-terminus plus cassette switch??
                #                     'SRR15521841.1.134853.1', # inversion at N-terminus plus cassette switch?? i guess this one is identical to the one before?
                #                     'SRR15521841.1.461809.1', # two cassette switches?
                #                     'SRR15521841.1.461811.1', # two cassette switches? i guess this one is identical to the one before???
                #                     'SRR15521841.1.461812.1', # two cassette switches?
                #                 },
                #                 # also, many more reads with what i would call "weak evidence", though maybe they aren't so weak...
                #             },
                #         },
                #         'align_to_whole_genome_to_verify_evidence_reads': True,
                #         # 'align_to_whole_genome_to_verify_evidence_reads': False,
                #     },
                # ],
            },

            # 'porin with GIY-YIG nuclease': {
            #     'CP034453.1': [ # Mesorhizobium sp. M7D.F.Ca.US.005.01.1.1
            #                 'SRR8143895': {
            # 'porin without GIY-YIG nuclease': {
            #     'NZ_CP051772.1': [ # Mesorhizobium japonicum R7A
            #                 'SRR11579120': {
            # 'porin with DUF423': {
            #     'CP023974.1': [ # Brucella canis strain FDAARGOS_420
            #                 'SRR5879320': {
            #     'CP066175.1': [ # Brucella abortus strain 68
            #                 'SRR13298070': {
            # didn't try (i think): (porin) https://www.ncbi.nlm.nih.gov/nuccore/NZ_CP011479.1?report=graph, 1208k, https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR1948173


            # 'hypoth with DUF3869': {
            #     'NZ_CP082886.1': [ # Bacteroides nordii strain FDAARGOS_1461
            #                 'SRR16259013': {





            # 'DUF4393': {
            #     'CP082844.1': [ #
            #         {
            #             'longest_linked_repeat_cds_region_and_protein_id': ((2923588, 2924481), 'UAK34719.1'),
            #
            #             # 'locus_definition': r'$\mathit{Bacteroides}$ $\mathit{thetaiotaomicron}$',
            #             # 'phylum': 'Bacteroidetes',
            #
            #             'ir_pair_region_with_margins': (2923380, 2927100),
            #             'presumably_relevant_cds_region': (2923588, 2926865),
            #             'alignment_region': (2903000, 2948000),
            #             'sra_accession_to_variants_and_reads_info': {
            #                 'SRR16263439': {
            #                     # 'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
            #                     #     'SRR12047085.1.35495.1': {(6177261, 6177848, 6182064, 6182651)}, # inversion
            #                     #     'SRR12047085.1.105653.1': {(6177261, 6177848, 6182064, 6182651)}, # inversion
            #                     #     'SRR12047085.1.105653.3': {(6177261, 6177848, 6182064, 6182651)}, # inversion
            #                     #     'SRR12047085.1.20249.3': {(6177261, 6177848, 6182064, 6182651)}, # inversion
            #                     # },
            #                 },
            #             },
            #             # 'align_to_whole_genome_to_verify_evidence_reads': True,
            #             'align_to_whole_genome_to_verify_evidence_reads': False,
            #
            #             'describe_in_the_paper': True,
            #         },
            #     ],
            # },

            # 'DUF3688': {
            #     'CP047426.1': [ # Spiroplasma citri strain C189
            #         {
            #             'longest_linked_repeat_cds_region_and_protein_id': ((1464414, 1466588), 'QIA75684.1'),
            #
            #             # 'locus_definition': r'$\mathit{Citrobacter}$ sp. RHBSTW-00887',
            #             # 'phylum': 'Proteobacteria',
            #
            #             'ir_pair_region_with_margins': (1463600, 1466550),
            #             'presumably_relevant_cds_region': (1462851, 1468422),
            #             'alignment_region': (1443000, 1487000),
            #             'sra_accession_to_variants_and_reads_info': {
            #                 # 'SRR10843927': {
            #             },
            #             # 'align_to_whole_genome_to_verify_evidence_reads': True,
            #             'align_to_whole_genome_to_verify_evidence_reads': False,
            #
            #             'describe_in_the_paper': True,
            #         },
            #     ],
            # },

            # 'pilin': {
            #     'CP068176.1': [ # Acinetobacter ursingii strain FDAARGOS_1096
            #         {
            #             'longest_linked_repeat_cds_region_and_protein_id': ((635054, 635542), 'QQT86871.1'),
            #
            #             # 'locus_definition': r'$\mathit{Citrobacter}$ sp. RHBSTW-00887',
            #             # 'phylum': 'Proteobacteria',
            #
            #             'ir_pair_region_with_margins': (633700, 635670),
            #             'presumably_relevant_cds_region': (633873, 635542),
            #             'alignment_region': (613000, 666000),
            #             'sra_accession_to_variants_and_reads_info': {
            #                 # 'SRR12935056': {
            #                 # 'SRR12935057': {
            #                 # 'SRR12935058': {
            #             },
            #             # 'align_to_whole_genome_to_verify_evidence_reads': True,
            #             'align_to_whole_genome_to_verify_evidence_reads': False,
            #
            #             'describe_in_the_paper': True,
            #         },
            #     ],
            # },

            # 'helix-turn-helix': {
            #     'NZ_FRDE01000003.1': [ #
            #         {
            #             # WP_015765014.1 is the longer PglX protein coded here.
            #             'phylum': 'Firmicutes',
            #             # 'longest_linked_repeat_cds_region_and_protein_id': ((2161973, 2165527), 'WP_015765014.1'),
            #
            #             'ir_pair_region_with_margins': (2158000, 2165653),
            #             # 'presumably_relevant_cds_region': (2154002, 2169193),
            #             # 'presumably_relevant_cds_region': (2154002, 2170387),
            #             'presumably_relevant_cds_region': (2156046, 2169193),
            #             'alignment_region': (2138000, 2186000),
            #             'sra_accession_to_variants_and_reads_info': {
            #                 'ERR1735878': { # IMPORTANT: didn't try it.
            #                     'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
            #                         # 'SRR9952487.1.101097.1': {(2160721, 2160749, 2162886, 2162914)}, # inversion
            #                     }
            #                 },
            #             },
            #             # 'align_to_whole_genome_to_verify_evidence_reads': True,
            #             'align_to_whole_genome_to_verify_evidence_reads': False,
            #
            #             'describe_in_the_paper': False,
            #         },
            #     ],
            # },

            # 'WecA-like: solitary': {
            #     'NZ_LN907858.1': [ # Helicobacter typhlonius strain MIT 97-6810
            #                 'SRR12825962': { # looks like the SRA doesnt fit the nuccore (only ~50 short alignments)...

            # 'aldo/keto reductase': {
            #     'NZ_PPFB01000001.1': [ # Streptomyces sp. DH-12 scaffold1_size7418681
            #                 'SRR11853261': {

            # 'tyrosine-type recombinase/integrase RM chimera': {
            #     'NZ_CP037440.1': [ # Bacteroides fragilis strain DCMOUH0085B
            #                 'SRR8961732': {
            #                 'SRR8961731': {
            #                 'SRR8961729': {

            # 'RM specificity: M-S-R': {
            #     # 'NZ_CP013688.1': [ # Streptococcus gallolyticus strain ICDDRB-NRC-S1
            #     #             'SRR3031661': {
            #     # 'CP015403.2': [ # Burkholderiales bacterium YL45 # first tried to use SRR3371363 on it, but then figured out that CP015403.1 fits much better...
            #     #             'SRR3371363': {
            #     'CP015403.1': [ # Burkholderiales bacterium YL45
            #         # it seems to me that CP015403.1 fits SRR3371363 better than CP015403.2, at least in the alignment region.
            #         {
            #             'locus_definition': r'$\mathit{Burkholderiales}$ bacterium YL45',
            #             'phylum': 'Proteobacteria',
            #             #     repeat_len  mismatch    left1   right1    left2   right2        evalue  spacer_len
            #             #  0         167         1  2871719  2871885  2877140  2877306  3.320000e-88        5254
            #             #  1          92         0  2871977  2872068  2876953  2877044  1.560000e-46        4884
            #             #  2          81         0  2871892  2871972  2877051  2877131  3.630000e-40        5078
            #             #  3          53         0  2871668  2871720  2877306  2877358  5.840000e-24        5585
            #             #  4          38         0  2872076  2872113  2876906  2876943  2.810000e-15        4792
            #             #  5          24         0  2872112  2872135  2876883  2876906  3.560000e-07        4747
            #             #  6          19         1  2871668  2871686  2876758  2876776  1.500000e-02        5071
            #             #  7          14         0  2872125  2872138  2876270  2876283  2.200000e-01        4131
            #             'ir_pair_region_with_margins': (2871168, 2877858),
            #             'presumably_relevant_cds_region': (2871744, 2879008),
            #             'alignment_region': (2851000, 2898000),
            #             'sra_accession_to_variants_and_reads_info': {
            #                 'SRR3371363': {
            #                     'non_ref_variant_read_name_to_possible_ir_pairs_used_to_reach_from_ref': {
            #                         # 'SRR3371363.2.102858.1', # cassette switch, i think. doesn't cover the whole region
            #                         'SRR3371363.2.10835.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         # 'SRR3371363.2.11279.1', # cassette switch, i think. doesn't cover the whole region
            #                         'SRR3371363.2.11279.3': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.116113.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.116332.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.120943.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.12759.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.14547.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.14547.3': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.157500.3': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         # 'SRR3371363.2.16802.1', # cassette switch, i think. doesn't cover the whole region
            #                         # 'SRR3371363.2.22577.1', # cassette switch, i think. doesn't cover the whole region
            #                         'SRR3371363.2.47140.3': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.55166.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.60972.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.60972.3': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         'SRR3371363.2.76423.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #                         # 'SRR3371363.2.78714.1', # cassette switch, i think. doesn't cover the whole region
            #                         'SRR3371363.2.79936.1': {(2872125, 2872138, 2876270, 2876283), (2871668, 2871686, 2876758, 2876776)}, # cassette switch, i think
            #
            #                         # inversion. note that not all of the read is aligned. it seems like we have only the repeat1s (or only the repeat2s?). maybe that's why
            #                         # the bacteria didn't have an even number of inversions? maybe they got stuck in this odd variant?
            #                         # i guess something like that is more suggestive of an actual regulation mechanism to make sure the variant has an even number
            #                         # of inversions, rather than autoimmunity and natural selection being the mechanism?
            #                         # 'SRR3371363.2.84380.3', # doesn't cover the whole region
            #
            #                         # 'SRR3371363.2.128628.1', # ditto 84380.3. doesn't cover the whole region
            #
            #                     },
            #                 },
            #             },
            #             'align_to_whole_genome_to_verify_evidence_reads': True,
            #             # 'align_to_whole_genome_to_verify_evidence_reads': False,
            #
            #             # 'describe_in_the_paper': False, # one of the hsdS CDSs is truncated, or it actually isn't an invertible CDS...
            #         },
            #     ],
            # },
        },

        'nuccore_accession_to_assembly_accesion': {
            'NC_013198.1': 'GCF_000026505.1',
            'NZ_CP068173.1': 'GCF_016726245.1',
            'CP083813.1': 'GCF_020149745.1',
            'NZ_CP061344.1': 'GCF_014725695.1',
            'CP033760.1': 'GCF_003812605.1',
            'NZ_CP082886.1': 'GCF_019930665.1',
            'NZ_UFVQ01000003.1': 'GCF_900446825.1',
            'NZ_CP012938.1': 'GCF_001314995.1',
            'CP081913.1': 'GCF_020091485.1',
            'NZ_CP057125.1': 'GCF_013814485.1',
            'CP076386.1': 'GCF_018904205.1',
            'CP066032.1': 'GCF_016127695.1',
            'NZ_CP022464.2': 'GCF_002234575.2',
            'CP046428.1': 'GCF_013010365.1',
            'CP081899.1': 'GCF_020091345.1',
            'CP044495.1': 'GCF_008831365.1',
            'CP066294.2': 'GCF_014842815.3',
            'NZ_CP056267.1': 'GCF_013728095.1',
            'CP056267.1': 'GCF_013728095.1',
            'NZ_CP068294.1': 'GCF_013112255.2',
            'CP042282.1': 'GCF_018279805.1',
            'CP068086.1': 'GCF_016725865.1',
            'NZ_LN907858.1': 'GCF_001460635.1',
            'NZ_CP010519.1': 'GCF_000827005.1',
            'NZ_CP059830.1': 'GCF_016647595.1',
            'NZ_UGYW01000002.1': 'GCF_900457365.1',
            'CP047426.1': 'GCF_010587175.1',
            'CP084655.1': 'GCF_020423105.1',
            'CP068176.1': 'GCF_016726345.1',
            'CP065872.1': 'GCF_016117715.1',
            'CP082844.1': 'GCF_019930625.1',

            # 'CP015403.2': 'GCA_001688905.2',
            # 'CP015403.1': 'GCF_001688905.1',
            # 'CP015403.2': None, # in https://www.ncbi.nlm.nih.gov/assembly/GCF_001688905.2 it seems like CP015403.2 is the only nuccore comprising the genome...
            # 'CP015403.1': None, # in https://www.ncbi.nlm.nih.gov/assembly/GCF_001688905.1 it seems like CP015403.1 is the only nuccore comprising the genome...
        },
        'nuccore_accession_to_name_in_assembly': {
            # 'NC_013198.1': 'NC_013198.1',
        },

        'sra_accession_to_bioproject_accession': {
            'SRR6322570': 'PRJNA419802',
            'SRR6322568': 'PRJNA419802',
            'SRR6322569': 'PRJNA419802',
            'SRR10883644': 'PRJNA601093',
            'SRR10883645': 'PRJNA601093',
            'SRR8867157': 'PRJNA531520',
            'SRR8867158': 'PRJNA531520',
        },
        'sra_accession_to_type_and_sra_file_name': {

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR768077
            # curl -o ERR768077.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-2/ERR768077/ERR768077.1
            'ERR768077': ('long_reads', 'ERR768077.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR832413
            # curl -o ERR832413.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-2/ERR832413/ERR832413.1
            'ERR832413': ('long_reads', 'ERR832413.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR11613597
            # curl -o SRR11613597.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR11613597/SRR11613597.1
            'SRR11613597': ('long_reads', 'SRR11613597.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR11613601
            # curl -o SRR11613601.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR11613601/SRR11613601.1
            'SRR11613601': ('long_reads', 'SRR11613601.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12935271
            # curl -o SRR12935271.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR12935271/SRR12935271.1
            'SRR12935271': ('long_reads', 'SRR12935271.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR8143895
            # curl -o SRR8143895.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-1/SRR8143895/SRR8143895.1
            'SRR8143895': ('long_reads', 'SRR8143895.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR11579120
            # curl -o SRR11579120.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR11579120/SRR11579120.1
            'SRR11579120': ('long_reads', 'SRR11579120.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR8639574
            # curl -o SRR8639574.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR8639574/SRR8639574.1
            'SRR8639574': ('long_reads', 'SRR8639574.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12276161
            # curl -o SRR12276161.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR12276161/SRR12276161.1
            'SRR12276161': ('long_reads', 'SRR12276161.1'),



            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR16259013
            # curl -o SRR16259013.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra62/SRR/015877/SRR16259013
            'SRR16259013': ('long_reads', 'SRR16259013.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR9952487 # 1.2G
            # curl -o SRR9952487.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR9952487/SRR9952487.1
            'SRR9952487': ('long_reads', 'SRR9952487.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR5879320 # 7.7G
            # curl -o SRR5879320.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-11/SRR5879320/SRR5879320.1
            'SRR5879320': ('long_reads', 'SRR5879320.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=DRR161258 # 950M
            # curl -o DRR161258.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/DRR161258/DRR161258.1
            'DRR161258': ('long_reads', 'DRR161258.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR13298070 # 10.5G
            # curl -o SRR13298070.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR13298070/SRR13298070.1
            'SRR13298070': ('long_reads', 'SRR13298070.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR2531247
            # curl -o SRR2531247.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-1/SRR2531247/SRR2531247.1
            'SRR2531247': ('long_reads', 'SRR2531247.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR2531765
            # curl -o SRR2531765.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-1/SRR2531765/SRR2531765.1
            'SRR2531765': ('long_reads', 'SRR2531765.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR2531772
            # curl -o SRR2531772.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-1/SRR2531772/SRR2531772.1
            'SRR2531772': ('long_reads', 'SRR2531772.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR2531775
            # curl -o SRR2531775.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-1/SRR2531775/SRR2531775.1
            'SRR2531775': ('long_reads', 'SRR2531775.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR15521841
            # curl -o SRR15521841.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra10/SRR/015158/SRR15521841
            'SRR15521841': ('long_reads', 'SRR15521841.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR5810500
            # curl -o SRR5810500.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-11/SRR5810500/SRR5810500.1
            'SRR5810500': ('long_reads', 'SRR5810500.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR15521845
            # curl -o SRR15521845.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra19/SRR/015158/SRR15521845
            'SRR15521845': ('long_reads', 'SRR15521845.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12045544
            # curl -o SRR12045544.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR12045544/SRR12045544.1
            'SRR12045544': ('long_reads', 'SRR12045544.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR6029562
            # curl -o SRR6029562.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-11/SRR6029562/SRR6029562.1
            'SRR6029562': ('long_reads', 'SRR6029562.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR5817725
            # curl -o SRR5817725.3 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-13/SRR5817725/SRR5817725.3
            'SRR5817725': ('long_reads', 'SRR5817725.3'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR8961732
            # curl -o SRR8961732.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR8961732/SRR8961732.1
            'SRR8961732': ('long_reads', 'SRR8961732.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR8961731
            # curl -o SRR8961731.3 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR8961731/SRR8961731.3
            'SRR8961731': ('long_reads', 'SRR8961731.3'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR8961729
            # curl -o SRR8961729.2 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR8961729/SRR8961729.2
            'SRR8961729': ('long_reads', 'SRR8961729.2'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR15521847
            # curl -o SRR15521847.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra39/SRR/015158/SRR15521847
            'SRR15521847': ('long_reads', 'SRR15521847.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR8101013
            # curl -o SRR8101013.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR8101013/SRR8101013.1
            'SRR8101013': ('long_reads', 'SRR8101013.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR9221602
            # curl -o SRR9221602.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR9221602/SRR9221602.1
            'SRR9221602': ('long_reads', 'SRR9221602.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12299098
            # curl -o SRR12299098.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR12299098/SRR12299098.1
            'SRR12299098': ('long_reads', 'SRR12299098.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12299129
            # curl -o SRR12299129.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR12299129/SRR12299129.1
            'SRR12299129': ('long_reads', 'SRR12299129.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR14684531
            # curl -o SRR14684531.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra69/SRR/014340/SRR14684531
            'SRR14684531': ('long_reads', 'SRR14684531.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR14725125
            # curl -o SRR14725125.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos4/sra-pub-run-25/SRR14725125/SRR14725125.1
            'SRR14725125': ('long_reads', 'SRR14725125.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR7866019
            # curl -o SRR7866019.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-2/SRR7866019/SRR7866019.1
            'SRR7866019': ('long_reads', 'SRR7866019.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR5456972
            # curl -o SRR5456972.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-7/SRR5456972/SRR5456972.1
            'SRR5456972': ('long_reads', 'SRR5456972.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12502384
            # curl -o SRR12502384.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR12502384/SRR12502384.1
            'SRR12502384': ('long_reads', 'SRR12502384.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR6976419
            # curl -o SRR6976419.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-13/SRR6976419/SRR6976419.1
            'SRR6976419': ('long_reads', 'SRR6976419.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12825962
            # curl -o SRR12825962.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR12825962/SRR12825962.1
            'SRR12825962': ('long_reads', 'SRR12825962.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR11853261
            # curl -o SRR11853261.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR11853261/SRR11853261.1
            'SRR11853261': ('long_reads', 'SRR11853261.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR16259014
            # curl -o SRR16259014.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra73/SRR/015877/SRR16259014
            'SRR16259014': ('long_reads', 'SRR16259014.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12938600
            # curl -o SRR12938600.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR12938600/SRR12938600.1
            'SRR12938600': ('long_reads', 'SRR12938600.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR2637631
            # curl -o SRR2637631.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-5/SRR2637631/SRR2637631.1
            'SRR2637631': ('long_reads', 'SRR2637631.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR2637689
            # curl -o SRR2637689.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-5/SRR2637689/SRR2637689.1
            'SRR2637689': ('long_reads', 'SRR2637689.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR8180785
            # curl -o SRR8180785.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-1/SRR8180785/SRR8180785.1
            'SRR8180785': ('long_reads', 'SRR8180785.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR13664849
            # curl -o SRR13664849.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR13664849/SRR13664849.1
            'SRR13664849': ('long_reads', 'SRR13664849.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR10666959
            # curl -o SRR10666959.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR10666959/SRR10666959.1
            'SRR10666959': ('long_reads', 'SRR10666959.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR3031661
            # curl -o SRR3031661.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-5/SRR3031661/SRR3031661.1
            'SRR3031661': ('long_reads', 'SRR3031661.1'),


            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR14480597
            # curl -o SRR14480597.1 https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR14480597/SRR14480597
            'SRR14480597': ('long_reads', 'SRR14480597.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=DRR307123
            # curl -o DRR307123.1 https://sra-download.ncbi.nlm.nih.gov/traces/dra2/DRR/000299/DRR307123
            'DRR307123': ('long_reads', 'DRR307123.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR11812841
            # curl -o SRR11812841.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR11812841/SRR11812841.1
            'SRR11812841': ('long_reads', 'SRR11812841.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR13617002
            # curl -o SRR13617002.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR13617002/SRR13617002.1
            'SRR13617002': ('long_reads', 'SRR13617002.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR3371363
            # curl -o SRR3371363.2 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-9/SRR3371363/SRR3371363.2
            'SRR3371363': ('long_reads', 'SRR3371363.2'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR11301546
            # curl -o SRR11301546.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR11301546/SRR11301546.1
            'SRR11301546': ('long_reads', 'SRR11301546.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR985538
            # curl -o ERR985538.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos1/sra-pub-run-5/ERR985538/ERR985538.1
            'ERR985538': ('long_reads', 'ERR985538.1'),

            # did not download
            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR13182360
            # curl -o SRR13182360.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR13182360/SRR13182360.1
            # 'SRR13182360': ('long_reads', 'SRR13182360.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR1055237
            # curl -o ERR1055237.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-7/ERR1055237/ERR1055237.1
            'ERR1055237': ('long_reads', 'ERR1055237.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR1055236
            # curl -o ERR1055236.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-7/ERR1055236/ERR1055236.1
            'ERR1055236': ('long_reads', 'ERR1055236.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR1055234
            # curl -o ERR1055234.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-7/ERR1055234/ERR1055234.1
            'ERR1055234': ('long_reads', 'ERR1055234.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR1055232
            # curl -o ERR1055232.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-7/ERR1055232/ERR1055232.1
            'ERR1055232': ('long_reads', 'ERR1055232.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR1055225
            # curl -o ERR1055225.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos2/sra-pub-run-7/ERR1055225/ERR1055225.1
            'ERR1055225': ('long_reads', 'ERR1055225.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=ERR2125767
            # curl -o ERR2125767.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/ERR2125767/ERR2125767.1
            'ERR2125767': ('long_reads', 'ERR2125767.1'),


            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12935056
            # curl -o SRR12935056.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR12935056/SRR12935056.1
            'SRR12935056': ('long_reads', 'SRR12935056.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12935057
            # curl -o SRR12935057.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-20/SRR12935057/SRR12935057.1
            'SRR12935057': ('long_reads', 'SRR12935057.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12935058
            # curl -o SRR12935058.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR12935058/SRR12935058.1
            'SRR12935058': ('long_reads', 'SRR12935058.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR12047085
            # curl -o SRR12047085.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR12047085/SRR12047085.1
            'SRR12047085': ('long_reads', 'SRR12047085.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR16683342
            # curl -o SRR16683342.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra63/SRR/016292/SRR16683342
            'SRR16683342': ('long_reads', 'SRR16683342.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR10843927
            # curl -o SRR10843927.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-21/SRR10843927/SRR10843927.1
            'SRR10843927': ('long_reads', 'SRR10843927.1'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR16263439
            # curl -o SRR16263439.1 https://sra-download.ncbi.nlm.nih.gov/traces/sra62/SRR/015882/SRR16263439
            'SRR16263439': ('long_reads', 'SRR16263439.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...


            # https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR6322570
            'SRR6322570': ('paired_rna_seq', 'SRR6322570.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR6322568
            'SRR6322568': ('paired_rna_seq', 'SRR6322568.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR6322569
            'SRR6322569': ('paired_rna_seq', 'SRR6322569.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/sra?run=SRR10883644
            'SRR10883644': ('paired_rna_seq', 'SRR10883644.1'), # added the '.1' because otherwise the name of the SRA entry dir is the same...

            # https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR10883645
            'SRR10883645': ('paired_rna_seq', 'SRR10883645.man'),

            # # https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR10883646
            # 'SRR10883646': ('paired_rna_seq', 'SRR10883646.man'), # not a control
            #
            # # https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR10883647
            # 'SRR10883647': ('paired_rna_seq', 'SRR10883647.man'), # not a control

            # https://trace.ncbi.nlm.nih.gov/Traces/sra?run=SRR8867157
            'SRR8867157': ('paired_rna_seq', 'SRR8867157.man'),

            # https://trace.ncbi.nlm.nih.gov/Traces/sra?run=SRR8867158
            'SRR8867158': ('paired_rna_seq', 'SRR8867158.man'),



        },
    },


    'debug___taxon_uid_to_forced_best_assembly_accession': None,

    'debug___num_of_taxa_to_go_over': None,
    # 'debug___num_of_taxa_to_go_over': 2,
    # 'debug___num_of_taxa_to_go_over': 5,
    # 'debug___num_of_taxa_to_go_over': 100,
    # 'debug___num_of_taxa_to_go_over': 1387,
    # 'debug___num_of_taxa_to_go_over': 5000,
    # 'debug___num_of_taxa_to_go_over': 15000,
    # 'debug___num_of_taxa_to_go_over': 15662,
    # 'debug___num_of_taxa_to_go_over': 34692,
    # 'debug___num_of_taxa_to_go_over': 35366,

    'debug___num_of_nuccore_entries_to_go_over': None,

    'debug___taxon_uids': None,

    # the first 5 taxa each having at least one presumable programmed inversion on 220422, including 1888 which is the one in which we found a PI targeting DISARM,
    # and the 6th (2721173) contains a homolog of the PI targeting DISARM in which IRs were not identified.
    # 'debug___taxon_uids': [1888, 1355477, 1511761, 1618207, 1778540, 2721173],

    # all 114 taxa each having at least one presumable programmed inversion on 220422
    # 'debug___taxon_uids': [213, 250, 256, 258, 476, 562, 715, 779, 817, 818, 821, 823, 1160, 1264, 1307, 1313, 1339, 1351, 1512, 1582, 1588, 1590, 1622, 1624, 1888, 2115, 2130, 13373, 28111, 28116, 28129, 28137, 28139, 28251, 28450, 28454, 28901, 29459, 35623, 40216, 42235, 42862, 45361, 46503, 47678, 53344, 53417, 59737, 60519, 68892, 71451, 75985, 84112, 85831, 110321, 114090, 120577, 161895, 162426, 166486, 204038, 204039, 208479, 208962, 214856, 239935, 246787, 291112, 310298, 310300, 310514, 328812, 328813, 329854, 338188, 357276, 371601, 387661, 398555, 446660, 469591, 481722, 501571, 544645, 574930, 626929, 671267, 674529, 683124, 712710, 744515, 938155, 1064539, 1082704, 1089444, 1124835, 1249999, 1355477, 1511761, 1618207, 1778540, 1796635, 1805473, 1841857, 1843235, 2044587, 2507160, 2599607, 2650158, 2777781, 2836161, 2854757, 2854759, 2854763],
}

# PREFIX_TO_ADD_TO_DEBUG_OUTPUT_DIR_NAME = 'debug_'
PREFIX_TO_ADD_TO_DEBUG_OUTPUT_DIR_NAME = ''

if SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT['debug___taxon_uids'] is not None:
    assert SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT['debug___num_of_taxa_to_go_over'] is None
    for stage in (
        'stage1',
        'stage2',
        'stage3',
        'stage4',
        'stage5',
        'stage6',
        'enrichment_analysis',
    ):
        SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT[stage]['output_dir_path'] = os.path.join(
            os.path.dirname(SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT[stage]['output_dir_path']),
            PREFIX_TO_ADD_TO_DEBUG_OUTPUT_DIR_NAME + os.path.basename(SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT[stage]['output_dir_path']),
        )


if SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT['debug_local_blast_database_path'] is None:
    pathlib.Path(SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT['stage5']['output_dir_path']).mkdir(parents=True, exist_ok=True)
    dummy_empty_file_path = os.path.join(SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT['stage5']['output_dir_path'], 'dummy_empty_file')
    generic_utils.write_empty_file(dummy_empty_file_path)
    SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT['debug_local_blast_database_path'] = dummy_empty_file_path

if 0:
    SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT['stage6']['cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args'] = {
        'MTase: Class 1 DISARM': SEARCH_FOR_PROGRAMMED_INVERSIONS_ARGS_DICT[
            'stage6']['cds_context_name_to_nuccore_accession_to_alignment_regions_raw_read_alignment_args']['MTase: Class 1 DISARM']
    }

