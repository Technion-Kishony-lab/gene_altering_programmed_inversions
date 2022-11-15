import os
import os.path
import pickle
import re

import numpy as np

from generic import bio_utils
from generic import generic_utils

# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')


@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_taxon_wgs_nuccore_uids(
        taxon_uid,
        min_wgs_nuccore_entry_len,
        max_num_of_wgs_nuccore_entries,
        output_file_path_wgs_nuccore_uids_pickle,
):
    search_term = f'(txid{taxon_uid}[orgn:exp] AND "wgs"[properties] AND ("{min_wgs_nuccore_entry_len}"[SLEN] : "100000000"[SLEN])) NOT "wgs master"[properties]'

    try:
        wgs_nuccore_uids = bio_utils.run_entrez_esearch_and_return_uids(
            db_name='nuccore',
            search_term=search_term,
            sort='SLEN',
            retmax=max_num_of_wgs_nuccore_entries,
        )
    except bio_utils.EntrezEsearchPhraseNotFound as err:
        assert f'txid{taxon_uid}[orgn:exp]' in str(err)
        wgs_nuccore_uids = None
    else:
        assert len(wgs_nuccore_uids) <= max_num_of_wgs_nuccore_entries

    with open(output_file_path_wgs_nuccore_uids_pickle, 'wb') as f:
        pickle.dump(wgs_nuccore_uids, f, protocol=4)

def write_taxon_wgs_nuccore_uids(
        taxon_uid,
        min_wgs_nuccore_entry_len,
        max_num_of_wgs_nuccore_entries,
        output_file_path_wgs_nuccore_uids_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_taxon_wgs_nuccore_uids(
        taxon_uid=taxon_uid,
        min_wgs_nuccore_entry_len=min_wgs_nuccore_entry_len,
        max_num_of_wgs_nuccore_entries=max_num_of_wgs_nuccore_entries,
        output_file_path_wgs_nuccore_uids_pickle=output_file_path_wgs_nuccore_uids_pickle,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_nuccore_accession_to_nuccore_entry_len_and_nuccore_accession_to_nuccore_uid(
        input_file_path_wgs_nuccore_uids_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_len_pickle,
        output_file_path_nuccore_accession_to_nuccore_uid_pickle,
):
    with open(input_file_path_wgs_nuccore_uids_pickle, 'rb') as f:
        wgs_nuccore_uids = pickle.load(f)

    if wgs_nuccore_uids is None:
        nuccore_accession_to_nuccore_entry_len = None
        nuccore_accession_to_nuccore_uid = None
    else:
        nuccore_uid_to_summary_record = bio_utils.run_entrez_esummary_and_return_input_uid_to_summary_record(
            db_name='nuccore',
            # uids=random.sample(wgs_nuccore_uids, k=20),
            uids=wgs_nuccore_uids,
        )

        nuccore_accession_to_nuccore_entry_len = {}
        nuccore_accession_to_nuccore_uid = {}
        for nuccore_uid, summary_record in nuccore_uid_to_summary_record.items():
            nuccore_accession = str(summary_record['AccessionVersion'])
            # according to https://www.ncbi.nlm.nih.gov/books/NBK21091/table/ch18.T.refseq_accession_numbers_and_mole/
            if bool(re.match(r'AC_|NC_|NG_|NT_|NW_|NZ_', nuccore_accession)):
                nuccore_accession_without_prefix = nuccore_accession[3:]
                if nuccore_accession_without_prefix in nuccore_accession_to_nuccore_entry_len:
                    nuccore_accession_to_nuccore_entry_len.pop(nuccore_accession_without_prefix)
                    nuccore_accession_to_nuccore_uid.pop(nuccore_accession_without_prefix)

            entry_len = int(summary_record['Length'])
            nuccore_accession_to_nuccore_entry_len[nuccore_accession] = entry_len
            nuccore_accession_to_nuccore_uid[nuccore_accession] = nuccore_uid

    with open(output_file_path_nuccore_accession_to_nuccore_entry_len_pickle, 'wb') as f:
        pickle.dump(nuccore_accession_to_nuccore_entry_len, f, protocol=4)
    with open(output_file_path_nuccore_accession_to_nuccore_uid_pickle, 'wb') as f:
        pickle.dump(nuccore_accession_to_nuccore_uid, f, protocol=4)

def write_nuccore_accession_to_nuccore_entry_len_and_nuccore_accession_to_nuccore_uid(
        input_file_path_wgs_nuccore_uids_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_len_pickle,
        output_file_path_nuccore_accession_to_nuccore_uid_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_nuccore_accession_to_nuccore_entry_len_and_nuccore_accession_to_nuccore_uid(
        input_file_path_wgs_nuccore_uids_pickle=input_file_path_wgs_nuccore_uids_pickle,
        output_file_path_nuccore_accession_to_nuccore_entry_len_pickle=output_file_path_nuccore_accession_to_nuccore_entry_len_pickle,
        output_file_path_nuccore_accession_to_nuccore_uid_pickle=output_file_path_nuccore_accession_to_nuccore_uid_pickle,
    )

@generic_utils.execute_if_output_doesnt_exist_already
def cached_write_taxon_wgs_nuccore_entries_info(
        taxon_uid,
        taxon_primary_assembly_nuccore_total_len,
        min_wgs_nuccore_entry_len,
        max_num_of_wgs_nuccore_entries,
        taxon_out_dir_path,
        output_file_path_taxon_wgs_nuccore_entries_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution,
):
    wgs_nuccore_uids_pickle_file_path = os.path.join(taxon_out_dir_path, 'wgs_nuccore_uids.pickle')
    write_taxon_wgs_nuccore_uids(
        taxon_uid=taxon_uid,
        min_wgs_nuccore_entry_len=min_wgs_nuccore_entry_len,
        max_num_of_wgs_nuccore_entries=max_num_of_wgs_nuccore_entries,
        output_file_path_wgs_nuccore_uids_pickle=wgs_nuccore_uids_pickle_file_path,
    )

    nuccore_accession_to_nuccore_entry_len_pickle_file_path = os.path.join(taxon_out_dir_path, 'wgs_nuccore_accession_to_entry_len.pickle')
    nuccore_accession_to_nuccore_uid_pickle_file_path = os.path.join(taxon_out_dir_path, 'wgs_nuccore_accession_to_nuccore_uid.pickle')
    write_nuccore_accession_to_nuccore_entry_len_and_nuccore_accession_to_nuccore_uid(
        input_file_path_wgs_nuccore_uids_pickle=wgs_nuccore_uids_pickle_file_path,
        output_file_path_nuccore_accession_to_nuccore_entry_len_pickle=nuccore_accession_to_nuccore_entry_len_pickle_file_path,
        output_file_path_nuccore_accession_to_nuccore_uid_pickle=nuccore_accession_to_nuccore_uid_pickle_file_path,
    )

    with open(nuccore_accession_to_nuccore_entry_len_pickle_file_path, 'rb') as f:
        nuccore_accession_to_nuccore_entry_len = pickle.load(f)

    if nuccore_accession_to_nuccore_entry_len is None:
        total_len_of_wgs_nuccore_entries = None
        estimated_num_of_genomes_in_wgs = None
    else:
        total_len_of_wgs_nuccore_entries = sum(nuccore_accession_to_nuccore_entry_len.values())
        estimated_num_of_genomes_in_wgs = total_len_of_wgs_nuccore_entries / taxon_primary_assembly_nuccore_total_len

    taxon_wgs_nuccore_entries_info = {
        'wgs_nuccore_uids_pickle_file_path': wgs_nuccore_uids_pickle_file_path,
        'nuccore_accession_to_nuccore_entry_len_pickle_file_path': nuccore_accession_to_nuccore_entry_len_pickle_file_path,

        'nuccore_accession_to_nuccore_entry_len': nuccore_accession_to_nuccore_entry_len,
        'nuccore_accession_to_nuccore_uid_pickle_file_path': nuccore_accession_to_nuccore_uid_pickle_file_path,
        'total_len_of_wgs_nuccore_entries': total_len_of_wgs_nuccore_entries,
        'estimated_num_of_genomes_in_wgs': estimated_num_of_genomes_in_wgs,
    }

    with open(output_file_path_taxon_wgs_nuccore_entries_info_pickle, 'wb') as f:
        pickle.dump(taxon_wgs_nuccore_entries_info, f, protocol=4)

def write_taxon_wgs_nuccore_entries_info(
        taxon_uid,
        taxon_primary_assembly_nuccore_total_len,
        min_wgs_nuccore_entry_len,
        max_num_of_wgs_nuccore_entries,
        taxon_out_dir_path,
        output_file_path_taxon_wgs_nuccore_entries_info_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_taxon_wgs_nuccore_entries_info(
        taxon_uid=taxon_uid,
        taxon_primary_assembly_nuccore_total_len=taxon_primary_assembly_nuccore_total_len,
        min_wgs_nuccore_entry_len=min_wgs_nuccore_entry_len,
        max_num_of_wgs_nuccore_entries=max_num_of_wgs_nuccore_entries,
        taxon_out_dir_path=taxon_out_dir_path,
        output_file_path_taxon_wgs_nuccore_entries_info_pickle=output_file_path_taxon_wgs_nuccore_entries_info_pickle,
        dummy_arg_to_make_caching_mechanism_not_skip_execution=1,
    )
