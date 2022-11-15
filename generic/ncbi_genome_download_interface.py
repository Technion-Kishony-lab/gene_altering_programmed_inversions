import collections
import gzip
import os
import os.path
import pickle
import shutil
import subprocess

import numpy as np
import pandas as pd

from generic import bio_utils
from generic import generic_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

class NcbiGenomeDownloadSubprocessError(Exception):
    pass

def check_ncbi_genome_download_version():
    # bowtie2_version_output, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['bowtie2', '--version'], raise_exception_if_subproc_returned_non_zero=False)
    genome_download_version_output, _ = generic_utils.run_cmd_and_get_stdout_and_stderr(['ncbi-genome-download', '--version'])
    if '0.3.1' not in genome_download_version_output:
        raise NotImplementedError(f'sorry. a wrapper for this version ({genome_download_version_output}) hasnt been implemented yet.')

def get_assembly_accession_to_nuccore_accession_to_gbff_file_path_of_downloaded_assemblies_and_ungz_if_needed(assemblies_accessions, output_dir_path):
    assert isinstance(assemblies_accessions, (set, list, tuple))

    assemblies_output_dir_path = os.path.join(output_dir_path, 'refseq', 'bacteria')
    assembly_accession_to_gbff_file_paths_before_splitting = collections.defaultdict(set)
    for assembly_accession in assemblies_accessions:
        assembly_output_dir_path = os.path.join(assemblies_output_dir_path, assembly_accession)
        if not os.path.isdir(assembly_output_dir_path):
            continue
        names_of_files_in_assembly_output_dir = os.listdir(assembly_output_dir_path)

        names_of_md5sums_files_in_assembly_output_dir = [x for x in names_of_files_in_assembly_output_dir if 'MD5SUM' in x]
        assert len(names_of_md5sums_files_in_assembly_output_dir) <= 1

        names_of_gbff_files_in_assembly_output_dir = [x for x in names_of_files_in_assembly_output_dir if x.endswith('.gbff')]
        for gbff_file_name in names_of_gbff_files_in_assembly_output_dir:
            assembly_accession_to_gbff_file_paths_before_splitting[assembly_accession].add(os.path.join(assembly_output_dir_path, gbff_file_name))

        names_of_gbff_gz_files_in_assembly_output_dir = [x for x in names_of_files_in_assembly_output_dir if x.endswith('.gbff.gz')]
        for gbff_gz_file_name in names_of_gbff_gz_files_in_assembly_output_dir:
            gbff_gz_file_path = os.path.join(assembly_output_dir_path, gbff_gz_file_name)
            gbff_file_path = gbff_gz_file_path[:-3]
            assert gbff_file_path.endswith('.gbff')
            if not os.path.isfile(gbff_file_path):
                # https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python/44712152#44712152
                with gzip.open(gbff_gz_file_path, 'rb') as f_in:
                    with open(gbff_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                assembly_accession_to_gbff_file_paths_before_splitting[assembly_accession].add(gbff_file_path)

    assembly_accession_to_gbff_file_paths_before_splitting = dict(assembly_accession_to_gbff_file_paths_before_splitting) # I don't want a defaultdict moving around.

    assembly_accession_to_nuccore_accession_to_gbff_file_path = {}
    for assembly_accession, gbff_file_paths in assembly_accession_to_gbff_file_paths_before_splitting.items():
        assembly_accession_to_nuccore_accession_to_gbff_file_path[assembly_accession] = {}
        for gbff_file_path in gbff_file_paths:
            gb_records = list(bio_utils.get_gb_records(gbff_file_path))
            if len(gb_records) > 1:
                for gb_record in gb_records:
                    nuccore_accession = gb_record.id
                    nuccore_gb_file_path = f'{gbff_file_path}.{nuccore_accession}.gbff'
                    bio_utils.write_records_to_fasta_or_gb_file(gb_record, nuccore_gb_file_path, 'gb')
                    assembly_accession_to_nuccore_accession_to_gbff_file_path[assembly_accession][nuccore_accession] = nuccore_gb_file_path
            else:
                assert len(gb_records) == 1
                nuccore_accession = gb_records[0].id
                nuccore_gb_file_path = gbff_file_path

                assembly_accession_to_nuccore_accession_to_gbff_file_path[assembly_accession][nuccore_accession] = nuccore_gb_file_path

    return assembly_accession_to_nuccore_accession_to_gbff_file_path


@generic_utils.execute_if_output_doesnt_exist_already
def cached_download_assembly(
        assembly_accession,
        output_file_path_gb,
        output_dir_path,
):
    cmd_line_words = [
        'ncbi-genome-download',
        '--verbose',
        '--output-folder', output_dir_path,
        '--assembly-accessions', assembly_accession,
        'bacteria', # "groups" argument
    ]
    generic_utils.run_cmd_and_get_stdout_and_stderr(cmd_line_words)

    download_output_dir_path = os.path.join(output_dir_path, 'refseq', 'bacteria', assembly_accession)
    assert os.path.isdir(download_output_dir_path)

    names_of_gbff_gz_files_in_download_output_dir = [x for x in os.listdir(download_output_dir_path) if x.endswith('.gbff.gz')]
    assert len(names_of_gbff_gz_files_in_download_output_dir) == 1
    gbff_gz_file_path = os.path.join(download_output_dir_path, names_of_gbff_gz_files_in_download_output_dir[0])

    # https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python/44712152#44712152
    with gzip.open(gbff_gz_file_path, 'rb') as f_in:
        with open(output_file_path_gb, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)



def download_assembly(
        assembly_accession,
        output_file_path_gb,
        output_dir_path,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_download_assembly(
        assembly_accession=assembly_accession,
        output_file_path_gb=output_file_path_gb,
        output_dir_path=output_dir_path,
    )


@generic_utils.execute_if_output_doesnt_exist_already
def cached_download_bacteria_assemblies_and_ungz_and_get_assembly_accession_to_nuccore_accession_to_gbff_file_path(
        assemblies_accessions,
        output_dir_path,
        output_file_path_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle,
):
    assert isinstance(assemblies_accessions, (set, list, tuple))

    check_ncbi_genome_download_version()

    assembly_accession_to_nuccore_accession_to_gbff_file_path = get_assembly_accession_to_nuccore_accession_to_gbff_file_path_of_downloaded_assemblies_and_ungz_if_needed(
        assemblies_accessions, output_dir_path)
    assembly_accessions_to_download = set(assemblies_accessions) - set(assembly_accession_to_nuccore_accession_to_gbff_file_path)

    # ncbi-genome-download --verbose --assembly-accessions GCF_000011045.1,GCF_000026525.1 bacteria
    if assembly_accessions_to_download:
        cmd_line_words = [
            'ncbi-genome-download', # https://github.com/kblin/ncbi-genome-download
            '--verbose',
            '--output-folder', output_dir_path,
            '--assembly-accessions', ','.join(assembly_accessions_to_download),
            'bacteria', # "groups" argument
         ]

        # subproc_stdout, subproc_stderr, subproc_ret_code = generic_utils.run_cmd_and_get_stdout_and_stderr(
        #     cmd_line_words,
        #     verbose=True,
        #     also_return_return_code=True,
        # )
        # if subproc_ret_code != 0:
        #     pass
        try:
            generic_utils.run_cmd_and_get_stdout_and_stderr(cmd_line_words)
        except subprocess.SubprocessError:
            raise NcbiGenomeDownloadSubprocessError()

        assembly_accession_to_nuccore_accession_to_gbff_file_path = get_assembly_accession_to_nuccore_accession_to_gbff_file_path_of_downloaded_assemblies_and_ungz_if_needed(
            assemblies_accessions, output_dir_path)
        assert set(assembly_accession_to_nuccore_accession_to_gbff_file_path) == set(assemblies_accessions)

    with open(output_file_path_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle, 'wb') as f:
        pickle.dump(assembly_accession_to_nuccore_accession_to_gbff_file_path, f, protocol=4)

def download_bacteria_assemblies_and_ungz_and_get_assembly_accession_to_nuccore_accession_to_gbff_file_path(
        assemblies_accessions,
        output_dir_path,
        output_file_path_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_download_bacteria_assemblies_and_ungz_and_get_assembly_accession_to_nuccore_accession_to_gbff_file_path(
        assemblies_accessions=assemblies_accessions,
        output_dir_path=output_dir_path,
        output_file_path_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle=output_file_path_assembly_accession_to_nuccore_accession_to_gbff_file_path_pickle,
    )