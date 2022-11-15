import io
import pickle

import numpy as np
import pandas as pd

from generic import generic_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

# according to https://ftp.ncbi.nlm.nih.gov/genomes/README_assembly_summary.txt
ALL_VERSION_STATUS_VALUES = {
    'latest',
    'replaced',
    'suppressed',
    np.nan,
    # 'GCA_000650635.1', # 220110: looks like an error in the current assembly_summary.txt
    # 'Research Center for Medicine & Biology', # 220110: looks like an error in the current assembly_summary.txt
}

ALL_GENOME_REP_VALUES = {
    'Full',
    'Partial',
    # np.nan,
}

ALL_ASSEMBLY_LEVEL_VALUES = {
    'Complete Genome',
    'Chromosome',
    'Scaffold',
    'Contig',
    # np.nan,
}

ALL_REFSEQ_CATEGORY_VALUES = {
    'reference genome',
    'representative genome',
    'na',
}

def read_assemblies_df_from_csv(assembly_summary_file_path):
    assembly_summary_first_two_lines = generic_utils.run_cmd_and_check_ret_code_and_return_stdout(['head', '-n', '2', assembly_summary_file_path], verbose=False)
    assembly_summary_second_line = assembly_summary_first_two_lines.splitlines()[-1]
    assembly_summary_column_names = assembly_summary_second_line[1:].strip().split()
    print(assembly_summary_column_names)

    assembly_summary_without_comment_lines = io.StringIO(generic_utils.run_cmd_and_check_ret_code_and_return_stdout(['grep', '-v', '^#', assembly_summary_file_path],
                                                                                                                    verbose=False))
    # print("'\n'.join(assembly_summary_without_comment_lines.readlines()[:5])")
    # print('\n'.join(assembly_summary_without_comment_lines.readlines()[:5]))
    # exit()
    return pd.read_csv(assembly_summary_without_comment_lines, sep='\t', low_memory=False, names=assembly_summary_column_names)

def get_filtered_assemblies_df(
        assembly_summary_file_path,
        allowed_assembly_level_values,
        allowed_version_status_values=('latest',),
        allowed_genome_rep_values=('Full',),
):
    assert set(allowed_assembly_level_values) <= ALL_ASSEMBLY_LEVEL_VALUES

    assemblies_df = read_assemblies_df_from_csv(assembly_summary_file_path)
    print('len(assemblies_df)')
    print(len(assemblies_df))

    print("\n\n\nassemblies_df['version_status'].value_counts()")
    print(assemblies_df['version_status'].value_counts())

    # unfortunately, due to what seems like errors in the assembly summary file, this doesn't hold:
    # assert set(assemblies_df['version_status'].unique()) <= ALL_VERSION_STATUS_VALUES

    assemblies_df.drop(
        assemblies_df[~(assemblies_df['version_status'].isin(allowed_version_status_values))].index,
        inplace=True,
    )
    print('len(assemblies_df) after filter by allowed_version_statuses')
    print(len(assemblies_df))
    assert not assemblies_df.empty


    print("\n\n\nassemblies_df['genome_rep'].value_counts()")
    print(assemblies_df['genome_rep'].value_counts())

    # unfortunately, due to what seems like errors in the assembly summary file, this doesn't hold:
    # assert set(assemblies_df['genome_rep'].unique()) <= ALL_GENOME_REP_VALUES

    assemblies_df.drop(
        assemblies_df[~(assemblies_df['genome_rep'].isin(allowed_genome_rep_values))].index,
        inplace=True,
    )
    print('len(assemblies_df) after filter by allowed_genome_rep_values')
    print(len(assemblies_df))
    assert not assemblies_df.empty

    assert set(assemblies_df['assembly_level'].unique()) <= ALL_ASSEMBLY_LEVEL_VALUES
    assemblies_df.drop(
        assemblies_df[~(assemblies_df['assembly_level'].isin(allowed_assembly_level_values))].index,
        inplace=True,
    )
    print('len(assemblies_df) after filter by allowed_assembly_level_values')
    print(len(assemblies_df))

    return assemblies_df

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_species_taxon_uid_to_best_assembly_accessions_pickle(
        input_file_path_refseq_assembly_summary,
        allowed_assembly_level_values_sorted_by_preference,
        max_num_of_assemblies_per_species,
        output_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        allowed_version_status_values,
        allowed_genome_rep_values,
        debug___taxon_uid_to_forced_best_assembly_accession,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    with generic_utils.timing_context_manager('cached_write_species_taxon_uid_to_best_assembly_accessions_pickle'):
        assemblies_df = get_filtered_assemblies_df(
            assembly_summary_file_path=input_file_path_refseq_assembly_summary,
            allowed_assembly_level_values=allowed_assembly_level_values_sorted_by_preference,
            allowed_version_status_values=allowed_version_status_values,
            allowed_genome_rep_values=allowed_genome_rep_values,
        )

        species_taxon_uid_to_best_assembly_accessions = {}
        # Importantly, https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html says: Groupby preserves the order of rows within each group.
        for i, (species_taxon_uid, species_assemblies_df) in enumerate(assemblies_df.groupby('species_taxid', sort=False)):
            species_taxon_uid = int(species_taxon_uid)
            if debug___taxon_uid_to_forced_best_assembly_accession and (species_taxon_uid in debug___taxon_uid_to_forced_best_assembly_accession):
                species_assemblies_df = species_assemblies_df[species_assemblies_df['assembly_accession'] ==
                                                              debug___taxon_uid_to_forced_best_assembly_accession[species_taxon_uid]].copy()

            best_assembly_accessions = []

            for assembly_level in allowed_assembly_level_values_sorted_by_preference:
                curr_level_assemblies_df = species_assemblies_df[species_assemblies_df['assembly_level'] == assembly_level]

                # according to https://ftp.ncbi.nlm.nih.gov/genomes/README_assembly_summary.txt, 'reference' is preffered over 'representative'.
                reference_assembly_row = curr_level_assemblies_df[curr_level_assemblies_df['refseq_category'] == 'reference genome']
                if not reference_assembly_row.empty:
                    reference_assembly_accession = reference_assembly_row['assembly_accession'].iloc[0]
                    best_assembly_accessions.append(reference_assembly_accession)
                    curr_level_assemblies_df = curr_level_assemblies_df[curr_level_assemblies_df['refseq_category'] != 'reference genome']

                representative_assembly_row = curr_level_assemblies_df[curr_level_assemblies_df['refseq_category'] == 'representative genome']
                if not representative_assembly_row.empty:
                    representative_assembly_accession = representative_assembly_row['assembly_accession'].iloc[0]
                    best_assembly_accessions.append(representative_assembly_accession)
                    curr_level_assemblies_df = curr_level_assemblies_df[curr_level_assemblies_df['refseq_category'] != 'representative genome']

                if not curr_level_assemblies_df.empty:
                    best_assembly_accessions.extend(curr_level_assemblies_df['assembly_accession'])

            best_assembly_accessions = best_assembly_accessions[:max_num_of_assemblies_per_species]




            species_taxon_uid_to_best_assembly_accessions[species_taxon_uid] = best_assembly_accessions

        # print()
        # print('species_taxon_uid_to_best_assembly_accessions[47715]')
        # print(species_taxon_uid_to_best_assembly_accessions[47715])
        # print()
        # print('species_taxon_uid_to_best_assembly_accessions[1313]')
        # print(species_taxon_uid_to_best_assembly_accessions[1313])
        # print()
        # print('species_taxon_uid_to_best_assembly_accessions[562]')
        # print(species_taxon_uid_to_best_assembly_accessions[562])
        # print()

        print(f'len(species_taxon_uid_to_best_assembly_accessions): {len(species_taxon_uid_to_best_assembly_accessions)}')

        with open(output_file_path_species_taxon_uid_to_best_assembly_accessions_pickle, 'wb') as f:
            pickle.dump(species_taxon_uid_to_best_assembly_accessions, f, protocol=4)


def write_species_taxon_uid_to_best_assembly_accessions_pickle(
        input_file_path_refseq_assembly_summary,
        allowed_assembly_level_values_sorted_by_preference,
        max_num_of_assemblies_per_species,
        output_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        allowed_version_status_values=('latest',),
        allowed_genome_rep_values=('Full',),
        debug___taxon_uid_to_forced_best_assembly_accession=None,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_species_taxon_uid_to_best_assembly_accessions_pickle(
        input_file_path_refseq_assembly_summary=input_file_path_refseq_assembly_summary,
        allowed_assembly_level_values_sorted_by_preference=allowed_assembly_level_values_sorted_by_preference,
        max_num_of_assemblies_per_species=max_num_of_assemblies_per_species,
        output_file_path_species_taxon_uid_to_best_assembly_accessions_pickle=output_file_path_species_taxon_uid_to_best_assembly_accessions_pickle,
        allowed_version_status_values=allowed_version_status_values,
        allowed_genome_rep_values=allowed_genome_rep_values,
        debug___taxon_uid_to_forced_best_assembly_accession=debug___taxon_uid_to_forced_best_assembly_accession,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=2,
    )



