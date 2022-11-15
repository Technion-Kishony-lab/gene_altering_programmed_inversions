import pathlib
import random
import sys
import warnings
import io
import unittest
import hashlib
import math
import os.path
import os
import itertools
import collections
import pandas as pd
import numpy as np
import subprocess
import contextlib
import os
import shutil
import time
import pickle
import logging
import textwrap
import scipy.stats


# 200624: It seems that this is the default (what I get from np.geterr() after importing numpy): {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
# https://numpy.org/doc/stable/reference/generated/numpy.seterr.html says: Underflow: result so close to zero that some precision was lost.
# so I guess it is Ok that I just ignore underflow problems?
# np.seterr(all='raise')
np.seterr(divide='raise', over='raise', invalid='raise')

INPUT_FILE_PATH_ARG_PREFIX = 'input_file_path_'
MULTIPLE_INPUT_FILE_PATH_ARG_PREFIX = 'multiple_input_file_paths_'
OUTPUT_FILE_PATH_ARG_PREFIX = 'output_file_path_'
OUTPUT_FILE_PATH_FOR_CACHING_ONLY_ARG = f'{OUTPUT_FILE_PATH_ARG_PREFIX}for_caching_only' # output_file_path_for_caching_only
HASH_FILE_NAME_SUFFIX = '.kind_of_hash_by_oren.txt'
FUNC_INPUT_ARGS_IN_PREV_CALL_FILE_NAME_SUFFIX = '.input_args_in_prev_call.txt'
INTERNAL_FILES_TO_SKIP_REDUNDANT_CALCULATIONS_DIR_PATH = 'internal___hashes_etc_to_skip_redundant_calculations'

ColumnNameAndValuesAndRanges = collections.namedtuple('ColumnNameAndValuesAndRanges', ['column_name', 'values_and_ranges'])
Range = collections.namedtuple('Range', ['start', 'end'])
Range.__str__ = lambda r: f'Range({r.start}, {r.end})'

WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES = ['word', 'count', 'freq']
WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES_SET = set(WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES)

WORD_COUNTS_AND_FREQS_COLUMN_NAME_TO_DTYPE = {
    'word': 'string',
    'count': int,
    'freq': np.float64,
}


PYTHON_BUILTIN_TYPES = (int, float, complex, bool, str, type(None), list, tuple, dict, set, frozenset)

pathlib.Path(INTERNAL_FILES_TO_SKIP_REDUNDANT_CALCULATIONS_DIR_PATH).mkdir(parents=True, exist_ok=True)

def get_max_finite(num_series):
    return num_series[np.isfinite(num_series)].max()

def get_min_finite(num_series):
    return num_series[np.isfinite(num_series)].min()

def get_min_strictly_positive(num_series):
    return num_series[num_series > 0].min()

def get_range_tuple_from_edges(edges):
    assert pd.Series(edges).is_monotonic_increasing
    return tuple(Range(x,y) for x,y in zip(edges[:-1], edges[1:]))

def print_and_write_to_log(msg):
    print(msg)
    logging.debug(msg)

def read_text_file(file_path, if_file_doesnt_exist_return_empty_str=False):
    if if_file_doesnt_exist_return_empty_str and (not os.path.isfile(file_path)):
        return ''
    with open(file_path) as f:
        return f.read()

def get_file_size(file_path):
    return os.stat(file_path).st_size

def is_file_empty(file_path):
    return get_file_size(file_path) == 0

def is_text_file_empty_or_containing_only_whitespaces(file_path):
    return not read_text_file(file_path).strip()

def read_bin_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read()


def write_text_file(file_path, file_contents):
    with open(file_path, 'w') as f:
        return f.write(file_contents)

def write_empty_file(file_path):
    with open(file_path, 'w') as f:
        pass

def add_suffix_to_file_name_while_keeping_extension(file_path, suffix):
    file_path_without_extension, extension = os.path.splitext(file_path)
    return file_path_without_extension + suffix + extension

def get_num_of_lines_in_text_file(file_path):
    return int(subprocess.check_output(['wc', '-l', file_path]).split()[0])

def get_len_of_longest_line_in_text_file(file_path):
    return int(subprocess.check_output(['wc', '-L', file_path]).split()[0])

def get_num_of_false_values(values):
    return sum(x == False for x in values)

def does_any_str_in_strs1_contain_any_str_in_strs2(strs1, strs2):
    # print('\nstrs1')
    # print(strs1)
    return any(str2 in str1 for (str1, str2) in itertools.product(strs1, strs2))

def does_any_str_in_strs1_ends_with_any_str_in_strs2(strs1, strs2):
    # print('\nstrs1')
    # print(strs1)
    return any(str1.endswith(str2) for (str1, str2) in itertools.product(strs1, strs2))

def get_public_attribute_names_set(obj):
    return set(attr for attr in dir(obj) if not attr.startswith('_'))


def get_file_sha256_in_hex(file_path):
    return hashlib.sha256(read_bin_file(file_path)).hexdigest()


def get_str_sha256_in_hex(str_to_hash):
    return hashlib.sha256(str_to_hash.encode()).hexdigest()


def print_dict(dicti, file=None, dict_name=None):
    print(file=file)
    if dict_name:
        print(f'{dict_name}:', file=file)
    for k, v in dicti.items():
        print(f'{k}: {v}', file=file)
    print(file=file)


def print_iterable(list_to_print, file=None, iterable_name=None):
    print(file=file)
    if iterable_name:
        print(f'{iterable_name}:', file=file)
    for x in list_to_print:
        print(x, file=file)
    print(file=file)

class KeysStep:
    pass

def get_values_in_nested_obj(obj, path_to_values, predicate_func=None, return_whole_paths=False, curr_path=()):
    assert type(path_to_values) == tuple
    if not path_to_values:
        if predicate_func:
            try:
                predicate_satisfied = predicate_func(obj)
            except Exception as err:
                print(f'predicate_func(obj) raised an error when called by get_values_in_nested_obj().')
                print(f'err:\n{err}')
                print(f'predicate_func(obj):\n{predicate_func.__name__}({obj})')
                raise
            if not predicate_satisfied:
                return []
        if return_whole_paths:
            return [(obj, curr_path)]
        return [obj]

    if obj is None:
        return []

    curr_step = path_to_values[0]
    next_steps = path_to_values[1:]
    if type(curr_step) == str:
        assert type(obj) == dict
        if curr_step not in obj:
            return []
        return get_values_in_nested_obj(obj[curr_step], next_steps, predicate_func=predicate_func, return_whole_paths=return_whole_paths, curr_path=(curr_path + (curr_step,)))
    elif type(curr_step) in {set, frozenset}:
        assert type(obj) == dict
        curr_steps = curr_step
        curr_step = None # just playing it safe.
        values = []
        for curr_step in curr_steps:
            if curr_step in obj:
                values.extend(get_values_in_nested_obj(obj[curr_step], next_steps, predicate_func=predicate_func, return_whole_paths=return_whole_paths,
                                                       curr_path=(curr_path + (curr_step,))))
        return values
    elif curr_step == KeysStep:
        values = []
        assert type(obj) == dict
        for k in obj:
            values.extend(get_values_in_nested_obj(k, next_steps, predicate_func=predicate_func, return_whole_paths=return_whole_paths, curr_path=(curr_path + (KeysStep,))))
        return values
    elif curr_step == Ellipsis:
        values = []
        if type(obj) == dict:
            for k, v in obj.items():
                values.extend(get_values_in_nested_obj(v, next_steps, predicate_func=predicate_func, return_whole_paths=return_whole_paths, curr_path=(curr_path + (k,))))
        elif type(obj) in (tuple, list, set, frozenset):
            for x in obj:
                values.extend(get_values_in_nested_obj(x, next_steps, predicate_func=predicate_func, return_whole_paths=return_whole_paths,
                                                       curr_path=(curr_path + ('element_in_iterable',))))
        else:
            print('obj')
            print(obj)
            print('type(obj)')
            print(type(obj))
            raise NotImplementedError
        return values
    else:
        print('curr_step')
        print(curr_step)
        print('type(curr_step)')
        print(type(curr_step))
        raise NotImplementedError

# def get_paths_to_values_in_nested_obj_that_satisfy_predicate(obj, path_to_values, predicate_func):


def get_substr_indices_including_overlapping(a_str, substr):
    # We trust str.find to do things efficiently, so we just try until it returns -1.
    substr_indices = []
    search_start_index = 0
    while True:
        found_index = a_str.find(substr, search_start_index)
        if found_index == -1:
            return substr_indices
        substr_indices.append(found_index)
        search_start_index = found_index + 1


assert get_substr_indices_including_overlapping('GATATATGCATATACTT', 'ATAT') == [1, 3, 9]
assert get_substr_indices_including_overlapping('.12.456.', '.') == [0, 3, 7]
assert get_substr_indices_including_overlapping('...', '.') == [0, 1, 2]
assert get_substr_indices_including_overlapping('...a', '.') == [0, 1, 2]
assert get_substr_indices_including_overlapping('a...', '.') == [1, 2, 3]
assert get_substr_indices_including_overlapping('...', '..') == [0, 1]
assert get_substr_indices_including_overlapping('...', '...') == [0]


def get_defaultdict_counter_using_numpy_unique___probably_faster_than_Counter_if_np_or_pd_obj(values):

    # Kind of equivalent to the following, but hopefully faster:
    # return collections.Counter(values)

    # When I tried this on a very long list of numbers, Counter was (very roughly) 4 times slower if I gave it a np.array or a Series, and the np.unique based implementation was
    # (very roughly) 1.5 times slower if I gave it a plain python list.
    unique_values, counts = np.unique(values, return_counts=True)
    return collections.defaultdict(int, zip(unique_values, counts))


@contextlib.contextmanager
def chdir_context_manager(new_dir):
    orig_dir = os.getcwd()
    os.chdir(os.path.expanduser(new_dir))
    try:
        yield
    finally:
        os.chdir(orig_dir)


@contextlib.contextmanager
def timing_context_manager(block_name):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        running_time_in_seconds = end_time - start_time
        print(f'block {block_name} running time (in seconds):\n{running_time_in_seconds:.3f}')


def rmtree_silent(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def remove_files_silently(file_paths):
    for file_path in file_paths:
        remove_file_silently(file_path)


def remove_file_silently(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    assert not os.path.isdir(file_path)


def silently_make_empty_dir(dir_path):
    rmtree_silent(dir_path)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=False)

def run_shell_command(cmd_str, raise_exception_if_subproc_returned_non_zero=True, capture_output=True):
    # proc = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = subprocess.run(cmd_str, shell=True, capture_output=capture_output)
    if proc.returncode:
        print(f'cmd_str:\n{cmd_str}\n')
        stdout_str = proc.stdout.decode() if proc.stdout is not None else None
        if stdout_str:
            print(f'stdout_str:\n{stdout_str}\n')

        stderr_str = proc.stderr.decode() if proc.stderr is not None else None
        if stderr_str:
            print(f'stderr_str:\n{stderr_str}\n')

        if raise_exception_if_subproc_returned_non_zero or stderr_str:
            raise RuntimeError(proc)

    return proc

def run_cmd_and_get_stdout_and_stderr(cmd_line_words, raise_exception_if_subproc_returned_non_zero=True, also_return_return_code=False, verbose=False):
    assert raise_exception_if_subproc_returned_non_zero or also_return_return_code
    cmd_as_str = ' '.join(cmd_line_words)
    if verbose:
        print('\nrunning cmd: ' + cmd_as_str)

    subproc = subprocess.run(cmd_line_words, check=False, capture_output=True)
    subproc_stdout = subproc.stdout.decode()
    subproc_stderr = subproc.stderr.decode()
    if raise_exception_if_subproc_returned_non_zero and subproc.returncode:
        print(f'cmd that returned non-zero:\n{cmd_as_str}')
        print(f'cmd result stderr:\n{subproc_stderr}')
        print(f'cmd result stdout:\n{subproc_stdout}')
        print(f'cmd result ret_code: {subproc.returncode}')
        raise subprocess.SubprocessError(f'Process failed with return code {subproc.returncode}')
    if also_return_return_code:
        return subproc_stdout, subproc_stderr, subproc.returncode
    return subproc_stdout, subproc_stderr


def run_cmd_and_write_stdout_and_stderr_to_files(cmd_line_words, stdout_file_path=None, stderr_file_path=None, raise_exception_if_subproc_returned_non_zero=True, verbose=True):
    if verbose:
        print('\nrunning cmd: ' + ' '.join(cmd_line_words))

    subproc_stdout, subproc_stderr = run_cmd_and_get_stdout_and_stderr(
        cmd_line_words=cmd_line_words,
        raise_exception_if_subproc_returned_non_zero=raise_exception_if_subproc_returned_non_zero,
    )

    if stdout_file_path:
        with open(stdout_file_path, 'w') as f:
            f.write(subproc_stdout)
    if stderr_file_path:
        with open(stderr_file_path, 'w') as f:
            f.write(subproc_stderr)


def run_cmd_and_assert_stdout_and_stderr_are_empty(cmd_line_words):
    subproc_stdout, subproc_stderr, subproc_ret_code = run_cmd_and_get_stdout_and_stderr(cmd_line_words, raise_exception_if_subproc_returned_non_zero=False,
                                                                                         also_return_return_code=True, verbose=True)
    if (subproc_ret_code != 0) or subproc_stdout or subproc_stderr:
        if subproc_stderr:
            print(f'cmd result stderr:\n{subproc_stderr}')
        if subproc_stdout:
            print(f'cmd result stdout:\n{subproc_stdout}')
        if subproc_ret_code != 0:
            print(f'cmd result ret_code: {subproc_ret_code}')

        assert False


def run_cmd_and_check_ret_code_and_return_stdout(cmd_line_words, stdout_file_path=None, raise_exception_if_stderr_isnt_empty=False, print_stdout_if_verbose=True, verbose=True):
    if verbose:
        print('\nrunning cmd: ' + ' '.join(cmd_line_words))
    # print('\nrunning cmd: ' + ' '.join(cmd_line_words))
    # exit()

    subproc_stdout, subproc_stderr = run_cmd_and_get_stdout_and_stderr(cmd_line_words)

    if stdout_file_path is not None:
        with open(stdout_file_path, 'w') as f:
            f.write(subproc_stdout)
    else:
        if verbose and print_stdout_if_verbose:
            if subproc_stdout:
                print('cmd result stdout:\n' + subproc_stdout)
            else:
                print('cmd result stdout: <empty str>\n')

    if subproc_stderr:
        if raise_exception_if_stderr_isnt_empty:
            raise RuntimeError('cmd result stderr:\n' + subproc_stderr)
        if verbose:
            print('cmd result stderr:\n' + subproc_stderr)
    else:
        if verbose:
            print('cmd result stderr: <empty str>\n')
    return subproc_stdout

def download_file_from_ftp_server(ftp_url, output_file_path, dont_download_if_file_exists=False, verbose=False):
    if (not dont_download_if_file_exists) or (not os.path.isfile(output_file_path)):
        run_cmd_and_check_ret_code_and_return_stdout(['curl', '-o', output_file_path, ftp_url], verbose=verbose)

def get_file_hash_file_path(file_path):
    # 200522: --------ATTENTION--------
    # Emabarrassingly, until today, this function was returning the same file path for two different files with the same file name (but in different dirs). An epic fail.
    # This comment is here to make sure you don't change the code to what it was earlier, and thus repeat this epic fail. Thank you.
    return file_path + HASH_FILE_NAME_SUFFIX


def get_file_path_with_func_name_and_input_args_hash(func_name, args_hash, suffix):
    func_dir_path = os.path.join(INTERNAL_FILES_TO_SKIP_REDUNDANT_CALCULATIONS_DIR_PATH, func_name)
    assert type(args_hash) == str
    args_hash_start = args_hash[:8]
    # args_hash_end = args_hash[8:]
    curr_hash_dir_path = os.path.join(func_dir_path, *textwrap.wrap(args_hash_start, 2))
    pathlib.Path(curr_hash_dir_path).mkdir(parents=True, exist_ok=True)
    # print(f'curr_hash_dir_path: {curr_hash_dir_path}')
    # exit()
    return os.path.join(curr_hash_dir_path, f'{args_hash}{suffix}')


def get_func_input_args_file_path(func_name, args_hash):
    return get_file_path_with_func_name_and_input_args_hash(func_name, args_hash, FUNC_INPUT_ARGS_IN_PREV_CALL_FILE_NAME_SUFFIX)


def remove_files_and_their_file_hash_files(file_paths):
    for file_path in file_paths:
        file_hash_file_path = get_file_hash_file_path(file_path)
        remove_file_silently(file_hash_file_path)
        remove_file_silently(file_path)

def get_file_last_modification_time(file_path):
    assert os.path.isfile(file_path)
    return os.path.getmtime(file_path)

def get_file_or_dir_last_modification_time(file_or_dir_path):
    if os.path.isfile(file_or_dir_path):
        return get_file_last_modification_time(file_or_dir_path)

    assert os.path.isdir(file_or_dir_path)
    # thanks https://stackoverflow.com/questions/29685069/get-the-last-modified-date-of-a-directory-including-subdirectories-using-pytho/29685234#29685234
    return max(os.path.getmtime(root) for root, _, _ in os.walk(file_or_dir_path))


def do_output_files_exist(output_file_paths):
    if not output_file_paths:
        return True
    hashes = set()
    for file_path in output_file_paths:
        file_hash_file_path = get_file_hash_file_path(file_path)
        if (not os.path.exists(file_path)) or (not os.path.isfile(file_hash_file_path)) or (os.path.getmtime(file_hash_file_path) < os.path.getmtime(file_path)):
            return False
        hashes.add(read_text_file(file_hash_file_path))
    return len(hashes) == 1

def get_object_deterministic_str_repr_lines_by_recursive_sort(obj):
    if isinstance(obj, (int, float, complex, bool, str, type(None))):
        return [str(obj)]
    if isinstance(obj, (set, frozenset)):
        return [str(sorted(obj))]

    deterministic_str_repr_lines = []
    if isinstance(obj, (list, tuple)):
        for x in obj:
            deterministic_str_repr_lines += get_object_deterministic_str_repr_lines_by_recursive_sort(x)
    elif isinstance(obj, dict):
        # print('\n\nprint each key')
        # for k in obj:
        #     print(k)
        for k in sorted(obj):
            deterministic_str_repr_lines.append(f'{k}:')
            deterministic_str_repr_lines += get_object_deterministic_str_repr_lines_by_recursive_sort(obj[k])
    else:
        raise NotImplementedError(f'obj is of type that is not supported yet (type(obj): {type(obj)})')
    return deterministic_str_repr_lines

def get_input_file_path_arg_text_repr_lines_and_file_last_modification_time_and_name(input_arg_name, input_arg_val):
    if not isinstance(input_arg_val, str):
        print(f'input_arg_val should be a str. ({input_arg_name}, {input_arg_val})')
    assert isinstance(input_arg_val, str)
    input_file_path = input_arg_val

    # I don't want to receive a directory. Too much added complexity, and it isn't really needed.
    assert not os.path.isdir(input_file_path)
    assert os.path.isfile(input_file_path)


    input_file_hash_file_path = get_file_hash_file_path(input_file_path)
    input_file_path_last_modification_time = get_file_last_modification_time(input_file_path)
    if os.path.isfile(input_file_hash_file_path) and (os.path.getmtime(input_file_hash_file_path) >= input_file_path_last_modification_time):
        input_file_hash = read_text_file(input_file_hash_file_path)
    else:
        # useful_debug_str = (f'input_file_path: {input_file_path}\n'
        #                     f'os.path.getmtime(input_file_hash_file_path): {os.path.getmtime(input_file_hash_file_path)}\n'
        #                     f'input_file_path_last_modification_time:      {input_file_path_last_modification_time}\n')

        # print(useful_debug_str)
        # # if os.path.isfile(input_file_hash_file_path):
        # #     raise RuntimeError(f'seems like someone overwrote a file without updating its hash file. I think this shouldnt happen silently...\n'
        # #                        f'{useful_debug_str}')
        # print('######################## recalculating input_file_hash')
        input_file_hash = get_file_sha256_in_hex(input_file_path)
        write_text_file(input_file_hash_file_path, input_file_hash)

    input_file_name = os.path.basename(input_file_path)
    input_arg_text_repr_lines = [f'{input_arg_name}: {input_file_path}: {input_file_hash}']
    return (input_arg_text_repr_lines, input_file_path_last_modification_time, input_file_name)

def execute_if_output_doesnt_exist_already(func):
    func_name = func.__name__

    def new_func(*args, **kwargs):
        # print(f'skipping execution of ############################################ func_name: {func_name}')

        assert args == ()  # this is just for my convenience (and laziness) - this way i don't need to use reflection to retrieve the names of the arguments...
        output_args_dict = {arg_name: arg_val for arg_name, arg_val in kwargs.items() if arg_name.startswith(OUTPUT_FILE_PATH_ARG_PREFIX)}
        output_file_for_caching_only_path = kwargs.get(OUTPUT_FILE_PATH_FOR_CACHING_ONLY_ARG)
        output_file_paths = [output_path for output_path in output_args_dict.values() if output_path is not None]
        for output_file_path in output_file_paths:
            assert isinstance(output_file_path, str)
            # I am afraid of removing important stuff...
            # assert not os.path.isabs(output_file_path)
            assert not os.path.isdir(output_file_path)
        input_args_dict = {arg_name: arg_val for arg_name, arg_val in kwargs.items() if not arg_name.startswith(OUTPUT_FILE_PATH_ARG_PREFIX)}

        # add the hash of the inputs without their hashes to the file names. nah. you use dirs instead.

        input_files_names = []
        func_curr_input_args_lines = []
        input_files_last_modification_times = []
        for input_arg_name, input_arg_val in sorted(input_args_dict.items()):
            if not isinstance(input_arg_val, PYTHON_BUILTIN_TYPES):
                print(f'input_arg_name: {input_arg_name}')
                print(f'input_arg_val: {input_arg_val}')
                print(f'type(input_arg_val): {type(input_arg_val)}')
                assert isinstance(input_arg_val, PYTHON_BUILTIN_TYPES)
            if input_arg_name.startswith(INPUT_FILE_PATH_ARG_PREFIX):
                input_arg_text_repr_lines, input_file_path_last_modification_time, input_file_name = (
                    get_input_file_path_arg_text_repr_lines_and_file_last_modification_time_and_name(input_arg_name, input_arg_val))

                input_files_names.append(input_file_name)
                input_files_last_modification_times.append(input_file_path_last_modification_time)
            elif input_arg_name.startswith(MULTIPLE_INPUT_FILE_PATH_ARG_PREFIX):
                if (not isinstance(input_arg_val, (list, tuple, set, frozenset))) or (not (len(input_arg_val) == len(set(input_arg_val)))):
                    print(f'input_arg_name: {input_arg_name}')
                    print(f'input_arg_val: {input_arg_val}')
                    print(f'type(input_arg_val): {type(input_arg_val)}')
                    assert isinstance(input_arg_val, (list, tuple, set, frozenset))
                    assert len(input_arg_val) == len(set(input_arg_val))
                input_arg_text_repr_lines = []
                for i, input_file_path in enumerate(sorted(input_arg_val)):
                    curr_input_arg_text_repr_lines, input_file_path_last_modification_time, input_file_name = (
                        get_input_file_path_arg_text_repr_lines_and_file_last_modification_time_and_name(f'{input_arg_name}{i}', input_file_path))

                    input_files_names.append(input_file_name)
                    input_files_last_modification_times.append(input_file_path_last_modification_time)
                    input_arg_text_repr_lines += curr_input_arg_text_repr_lines
            else:
                input_arg_val_text_repr_lines = get_object_deterministic_str_repr_lines_by_recursive_sort(input_arg_val)
                input_arg_text_repr_lines = [f'{input_arg_name}:'] + input_arg_val_text_repr_lines

            func_curr_input_args_lines += input_arg_text_repr_lines

        func_curr_input_args = '\n'.join(func_curr_input_args_lines)
        func_curr_input_args_hash = get_str_sha256_in_hex(func_curr_input_args)
        func_curr_input_args_file_path = get_func_input_args_file_path(func_name, func_curr_input_args_hash)
        need_to_execute_func = (
            # I suspect that I could remove func_curr_input_args_file_path and still make the caching work, ***BUT*** I think that would be a bad idea. Currently, a convenient
            # way to make my code execute the cached function anyway is to remove all func_curr_input_args_file_path in internal___hashes_etc_to_skip_redundant_calculations.
            # Removing func_curr_input_args_file_path would make it harder to make the function execute anyway, I think.
                (not os.path.isfile(func_curr_input_args_file_path)) or
                any(not os.path.isfile(x) for x in output_file_paths)
        )
        if need_to_execute_func:
            # print(f'skipping execution of ############################################ need_to_execute_func1: {need_to_execute_func} {func_name}')
            # print(f'skipping execution of ############################################ (not os.path.isfile(func_curr_input_args_file_path)): {(not os.path.isfile(func_curr_input_args_file_path))}')
            # print(f'skipping execution of ############################################ (not do_output_files_and_potentially_valid_hash_files_exist(output_file_paths)): {(not do_output_files_and_potentially_valid_hash_files_exist(output_file_paths))}')
            pass
        # print(f'skipping execution of ############################################')
        # print(f'skipping execution of ############################################ {func_name}')
        # print(f'skipping execution of ############################################ (not os.path.isfile(func_curr_input_args_file_path)): {(not os.path.isfile(func_curr_input_args_file_path))}')

        if (not need_to_execute_func) and output_file_paths:
            output_files_last_modification_times = [get_file_or_dir_last_modification_time(x) for x in output_file_paths]

            if 1: # 220104: added this only now. so if some bug arises, it might be the culprit.
                if not need_to_execute_func:
                    if max(output_files_last_modification_times) > get_file_or_dir_last_modification_time(func_curr_input_args_file_path):
                        need_to_execute_func = True
                        # print(f'skipping execution of ############################################ need_to_execute_func3: {need_to_execute_func}')

        # print(f'skipping execution of ############################################ need_to_execute_func4: {need_to_execute_func}')

        if need_to_execute_func:
            remove_file_silently(func_curr_input_args_file_path)
            if output_file_for_caching_only_path in output_file_paths:
                remove_file_silently(output_file_for_caching_only_path)


            func_ret = func(*args, **kwargs)
            assert func_ret is None

            write_text_file(func_curr_input_args_file_path, func_curr_input_args)
            for output_file_path in output_file_paths:
                if output_file_path == output_file_for_caching_only_path:
                    # Why is this needed? For things like make_blast_nucleotide_db and execute_blast_nucleotide - we don't know exactly what makeblastdb creates, so we create this
                    # file. then, when we call execute_blast_nucleotide, the arguments include the path of the blast db, and we also give this file's path as an argument.
                    assert not os.path.isfile(output_file_path)
                    # this was problematic (IIUC) when less than a second passed between the two calls! use a random number instead!
                    # write_text_file(output_file_path, time.strftime('%Y_%m_%d_%H_%M_%S') + str(time.time()))
                    write_text_file(output_file_path, str(random.random()))
                elif not os.path.isfile(output_file_path):
                    raise RuntimeError(f'Func {func_name} did not create the output file {output_file_path}. Shame.')

        else:
            comma_separated_input_file_names_as_str_literals = ','.join(f"'{file_name}'" for file_name in input_files_names)
            print(f'skipping execution of {func_name}({comma_separated_input_file_names_as_str_literals})')

    new_func.__name__ = f'execute_if_output_doesnt_exist_already___{func_name}'
    return new_func

@execute_if_output_doesnt_exist_already
def cached_unzip_gz_file(
        input_file_path_gz,
        output_file_path_unzipped,
):
    # if gunzip is too slow, consider using pigz?
    # https://unix.stackexchange.com/questions/623881/how-to-gzip-100-gb-files-faster-with-high-compression/623883#623883
    # https://superuser.com/questions/599329/why-is-gzip-slow-despite-cpu-and-hard-drive-performance-not-being-maxed-out
    assert input_file_path_gz.endswith('.gz')
    assert output_file_path_unzipped == input_file_path_gz[:-3]
    run_cmd_and_check_ret_code_and_return_stdout(['gunzip', '-k', input_file_path_gz], verbose=False)

def unzip_gz_file(
        input_file_path_gz,
        output_file_path_unzipped,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_unzip_gz_file(
        input_file_path_gz=input_file_path_gz,
        output_file_path_unzipped=output_file_path_unzipped,
    )

@contextlib.contextmanager
def redirect_stdout_and_verify_that_each_line_starts_with_skipping_execution_of():
    orig_stdout = sys.stdout
    captured_output_stringio = io.StringIO()
    sys.stdout = captured_output_stringio
    try:
        yield
    finally:
        sys.stdout = orig_stdout
        captured_output = captured_output_stringio.getvalue()
        captured_output_lines = captured_output.splitlines()
        for line in captured_output_lines:
            if not line.startswith('skipping execution of '):
                raise RuntimeError(f'Expected all output lines to start with "skipping execution of ", but this line does not:\n{line}')


def feature_scale_pd_series(pd_series):
    series_max = pd_series.max()
    series_min = pd_series.min()
    series_range_size = series_max - series_min
    if series_range_size == 0:
        return pd.Series(np.ones(len(pd_series)))
    return pd_series.apply(lambda x: (x - series_min) / series_range_size)


def np_array_contains_only_zeroes(arr):
    return not arr.any()


def get_hamming_dist_between_same_len_strs(str1, str2):
    hamming_dist = 0
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            hamming_dist += 1
    return hamming_dist


def is_hamming_dist_at_most_x(str1, str2, max_hamming_dist):
    hamming_dist = 0
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            hamming_dist += 1
            if hamming_dist > max_hamming_dist:
                return False
    return True


def get_nrows_and_ncols(num_of_axes):
    # nrows_ncols_possible_pairs = [(nrows, int(np.ceil(num_of_axes / nrows))) for nrows in range(1, num_of_axes + 1)]
    nrows_ncols_possible_pairs = [(nrows, num_of_axes // nrows) for nrows in range(1, num_of_axes + 1) if num_of_axes % nrows == 0]
    return nrows_ncols_possible_pairs[np.argmin([abs(nrows - ncols) for nrows, ncols in nrows_ncols_possible_pairs])]


def get_indices_of_mismatches_of_same_len_strs(str1, str2):
    assert len(str1) == len(str2)
    indices_of_mismatches = []
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            indices_of_mismatches.append(i)

    # print(str1)
    # print(str2)
    # print(indices_of_mismatches)
    return indices_of_mismatches


class MyTestCase(unittest.TestCase):
    @contextlib.contextmanager
    def assertPrints(self, expected_output):
        orig_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            yield
        finally:
            sys.stdout = orig_stdout
            self.assertEqual(captured_output.getvalue(), expected_output)

def replace_file_name_in_path(file_path, new_file_name):
    file_dir_path, file_name = os.path.split(file_path)
    return os.path.join(file_dir_path, new_file_name)


def does_str_contain_any_whitespace(s):
    splitted_s = s.split()
    num_of_splitted_parts = len(splitted_s)
    assert num_of_splitted_parts >= 1
    return num_of_splitted_parts > 1


def read_pickled_object_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def strip_special_chars_on_edges_and_replace_others_with_underscores(orig_str):
    str_with_special_replaced_with_underscores = ''.join(char if char.isalnum() else '_' for char in orig_str)
    return str_with_special_replaced_with_underscores.strip('_')

def str_contains_only_alnum_and_underscores(s):
    return (s.replace('_', '')).isalnum()

def filter_df_by_column_name_and_value_or_values_or_range(df, column_name, value_or_values_or_range):
    # if df.empty:
    #     return df.copy()

    if isinstance(value_or_values_or_range, Range):
        # if column_name not in list(df):
        #     print('df')
        #     print(df)
        rang = value_or_values_or_range
        return df[(df[column_name] >= rang.start) & (df[column_name] <= rang.end)]
    elif isinstance(value_or_values_or_range, tuple):
        values = value_or_values_or_range
        any_value_filter = pd.Series([False] * len(df), index=df.index)
        for value in values:
            any_value_filter = any_value_filter | (df[column_name] == value)
        return df[any_value_filter]
    elif (value_or_values_or_range is None) or pd.isna(value_or_values_or_range):
        return df[df[column_name].isna()]
    else:
        assert isinstance(value_or_values_or_range, (int, float, complex, bool, str))
        # print(f'df: {df}')
        # print(f'value_or_values_or_range: {value_or_values_or_range}')
        value = value_or_values_or_range
        return df[df[column_name] == value]

@execute_if_output_doesnt_exist_already
def cached_filter_csv_by_column_name_and_values_range(
        input_file_path_csv,
        column_name,
        values_range,
        output_file_path_filtered_csv,
        csv_missing_column_names,
):
    df = pd.read_csv(input_file_path_csv, sep='\t', names=csv_missing_column_names)
    filtered_df = filter_df_by_column_name_and_value_or_values_or_range(df, column_name, Range(*values_range))
    filtered_df.to_csv(output_file_path_filtered_csv, sep='\t', index=False)

def filter_csv_by_column_name_and_values_range(
        input_file_path_csv,
        column_name,
        values_range,
        output_file_path_filtered_csv,
        csv_missing_column_names=None,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_filter_csv_by_column_name_and_values_range(
        input_file_path_csv=input_file_path_csv,
        column_name=column_name,
        values_range=values_range,
        output_file_path_filtered_csv=output_file_path_filtered_csv,
        csv_missing_column_names=csv_missing_column_names,
    )

def filter_df_to_keep_only_rows_with_extreme_val_for_each_group(
        df,
        group_by_column_name,
        extreme_val_column_name,
        max_or_min,
):
    assert max_or_min in ('max', 'min')

    filtered_df = df.groupby(group_by_column_name, as_index=False, sort=False).apply(
        lambda group_df: group_df.loc[(group_df[extreme_val_column_name].idxmax() if max_or_min == 'max' else group_df[extreme_val_column_name].idxmin()),:]
    )
    return filtered_df

@execute_if_output_doesnt_exist_already
def cached_filter_csv_to_keep_only_rows_with_extreme_val_for_each_group(
        input_file_path_csv,
        group_by_column_name,
        extreme_val_column_name,
        max_or_min,
        output_file_path_filtered_csv,
        csv_missing_column_names,
):
    df = pd.read_csv(input_file_path_csv, sep='\t', names=csv_missing_column_names)

    filtered_df = filter_df_to_keep_only_rows_with_extreme_val_for_each_group(
        df=df,
        group_by_column_name=group_by_column_name,
        extreme_val_column_name=extreme_val_column_name,
        max_or_min=max_or_min,
    )

    filtered_df.to_csv(output_file_path_filtered_csv, sep='\t', index=False)

def filter_csv_to_keep_only_rows_with_extreme_val_for_each_group(
        input_file_path_csv,
        group_by_column_name,
        extreme_val_column_name,
        max_or_min,
        output_file_path_filtered_csv,
        csv_missing_column_names=None,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_filter_csv_to_keep_only_rows_with_extreme_val_for_each_group(
        input_file_path_csv=input_file_path_csv,
        group_by_column_name=group_by_column_name,
        extreme_val_column_name=extreme_val_column_name,
        max_or_min=max_or_min,
        output_file_path_filtered_csv=output_file_path_filtered_csv,
        csv_missing_column_names=csv_missing_column_names,
    )

def get_filtered_df_by_fixed_column_names_and_values_or_ranges(df, fixed_column_names_and_values_or_ranges):
    fixed_columns_str_reprs = []
    for column_name, value_or_values_or_range_single_elem_list in fixed_column_names_and_values_or_ranges:
        assert len(value_or_values_or_range_single_elem_list) == 1
        value_or_values_or_range = value_or_values_or_range_single_elem_list[0]
        df = filter_df_by_column_name_and_value_or_values_or_range(df, column_name, value_or_values_or_range)
        fixed_columns_str_reprs.append(f'{column_name}: {value_or_values_or_range}')
    return (df, fixed_columns_str_reprs)


def load_np_array_from_single_array_npz_file(npz_file_path):
    loaded_npz = np.load(npz_file_path)
    assert len(loaded_npz.files) == 1
    return loaded_npz[loaded_npz.files[0]]

def write_word_counts_and_freqs_df_to_csv(word_counts_and_freqs_df, csv_file_path):
    if word_counts_and_freqs_df.empty:
       word_counts_and_freqs_df = pd.DataFrame(columns=WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES)

    assert set(word_counts_and_freqs_df) == WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES_SET
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        word_counts_and_freqs_df.to_csv(csv_file_path, sep='\t', index=False)

def read_word_counts_and_freqs_from_csv(csv_file_path):
    word_counts_and_freqs_df = pd.read_csv(csv_file_path, sep='\t', dtype=WORD_COUNTS_AND_FREQS_COLUMN_NAME_TO_DTYPE)
    assert set(word_counts_and_freqs_df) == WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES_SET
    return word_counts_and_freqs_df

def add_freq_column_and_write_word_counts_and_freqs_to_csv_and_total_word_count_to_text_file(word_counts_and_freqs_df, word_counts_and_freqs_csv_file_path,
                                                                                             total_word_count_txt_file_path):
    total_count = word_counts_and_freqs_df['count'].sum()
    if total_count == 0:
        word_counts_and_freqs_df.loc[:, 'freq'] = np.nan
    else:
        word_counts_and_freqs_df.loc[:, 'freq'] = word_counts_and_freqs_df['count'] / total_count

    write_word_counts_and_freqs_df_to_csv(word_counts_and_freqs_df, word_counts_and_freqs_csv_file_path)
    write_text_file(total_word_count_txt_file_path, str(total_count))

def write_word_counts_and_freqs_to_csv_and_total_word_count_to_text_file(text_series, word_counts_and_freqs_csv_file_path, total_word_count_txt_file_path):
    # print(f'text_series:\n{text_series}')
    if text_series.empty:
        word_counts_and_freqs_df = pd.DataFrame(columns=WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES)
    else:
        # Counting the number of rows for each word would be much less efficient, i think, because i don't know how to do that with the nice split(' ', expand=True).
        # So let's just do this. It seems to me like it won't change much (there is a difference only if a word appears more than once in the same row).
        word_counts_and_freqs_df = text_series.str.lower().str.split(' ', expand=True).stack().value_counts().reset_index(name='count').rename(columns={'index': 'word'})

    add_freq_column_and_write_word_counts_and_freqs_to_csv_and_total_word_count_to_text_file(
        word_counts_and_freqs_df=word_counts_and_freqs_df,
        word_counts_and_freqs_csv_file_path=word_counts_and_freqs_csv_file_path,
        total_word_count_txt_file_path=total_word_count_txt_file_path,
    )

def merge_word_counts_and_freqs_dfs_and_write_to_csv_and_write_total_word_count_to_text_file(word_counts_and_freqs_dfs_list, merged_word_counts_and_freqs_csv_file_path,
                                                                                             merged_total_word_count_txt_file_path):
    if not word_counts_and_freqs_dfs_list:
        merged_word_counts_and_freqs_df = pd.DataFrame(columns=WORD_COUNTS_AND_FREQS_DF_COLUMN_NAMES)
    else:
        merged_word_counts_and_freqs_df = pd.concat(word_counts_and_freqs_dfs_list)
        merged_word_counts_and_freqs_df = merged_word_counts_and_freqs_df.groupby('word', as_index=False, sort=False).agg({'count': np.sum})

    add_freq_column_and_write_word_counts_and_freqs_to_csv_and_total_word_count_to_text_file(
        word_counts_and_freqs_df=merged_word_counts_and_freqs_df,
        word_counts_and_freqs_csv_file_path=merged_word_counts_and_freqs_csv_file_path,
        total_word_count_txt_file_path=merged_total_word_count_txt_file_path,
    )

def get_num_of_strs_in_which_substr_appears(strs, substr):
    return sum((substr in s) for s in strs)

def get_paths_of_files_and_dirs_in_dir(dir_path):
    return [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]

def get_df_filtered_by_allowed_column_values(df, column_name, allowed_values):
    filter_column = pd.Series(np.zeros(len(df)), dtype='bool', index=df.index)
    for allowed_val in allowed_values:
        filter_column.loc[df[column_name] == allowed_val] = True
    return df[filter_column]

def get_df_filtered_by_unallowed_column_values(df, column_name, unallowed_values):
    filter_column = pd.Series(np.ones(len(df)), dtype='bool', index=df.index)
    for unallowed_val in unallowed_values:
        filter_column.loc[df[column_name] == unallowed_val] = False
    return df[filter_column]


@execute_if_output_doesnt_exist_already
def cached_write_filtered_df_to_csv(
        input_file_path_csv,
        column_name,
        column_value,
        output_file_path_filtered_csv,
        csv_separator,
):
    df = pd.read_csv(input_file_path_csv, sep=csv_separator)
    filtered_df = df[df[column_name] == column_value]
    filtered_df.to_csv(output_file_path_filtered_csv, sep=csv_separator, index=False)
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore', DeprecationWarning)
    #     filtered_df.to_csv(output_file_path_filtered_csv, sep=csv_separator, index=False)

def write_filtered_df_to_csv(
        input_file_path_csv,
        column_name,
        column_value,
        output_file_path_filtered_csv,
        csv_separator='\t',
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_filtered_df_to_csv(
        input_file_path_csv=input_file_path_csv,
        column_name=column_name,
        column_value=column_value,
        output_file_path_filtered_csv=output_file_path_filtered_csv,
        csv_separator=csv_separator,
    )

def is_point_touching_interval(point, interval):
    return interval[0] <= point <= interval[1]

@execute_if_output_doesnt_exist_already
def cached_write_index_to_is_covered_by_any_interval_array(
        intervals_csv_file_path,
        array_len,
        output_file_path_index_to_is_covered_by_any_interval_array_npz,
):
    intervals_df = pd.read_csv(intervals_csv_file_path, sep='\t')
    index_to_is_covered_by_any_interval_array = np.zeros(array_len)
    for _, row in intervals_df.iterrows():
        index_to_is_covered_by_any_interval_array[(row['start'] - 1):row['end']] = 1

    np.savez_compressed(output_file_path_index_to_is_covered_by_any_interval_array_npz,
                        index_to_is_covered_by_any_interval_array=index_to_is_covered_by_any_interval_array)

def write_index_to_is_covered_by_any_interval_array(
        intervals_csv_file_path,
        array_len,
        output_file_path_index_to_is_covered_by_any_interval_array_npz,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_index_to_is_covered_by_any_interval_array(
        intervals_csv_file_path=intervals_csv_file_path,
        array_len=array_len,
        output_file_path_index_to_is_covered_by_any_interval_array_npz=output_file_path_index_to_is_covered_by_any_interval_array_npz,
    )

@execute_if_output_doesnt_exist_already
def cached_write_index_to_num_of_intervals_covering_it_array(
        intervals_csv_file_path,
        array_len,
        output_file_path_index_to_num_of_intervals_covering_it_array_npz,
):
    intervals_df = pd.read_csv(intervals_csv_file_path, sep='\t')
    index_to_num_of_intervals_covering_it_array = np.zeros(array_len)
    for _, row in intervals_df.iterrows():
        index_to_num_of_intervals_covering_it_array[(row['start'] - 1):row['end']] += 1

    np.savez_compressed(output_file_path_index_to_num_of_intervals_covering_it_array_npz,
                        index_to_num_of_intervals_covering_it_array=index_to_num_of_intervals_covering_it_array)

def write_index_to_num_of_intervals_covering_it_array(
        intervals_csv_file_path,
        array_len,
        output_file_path_index_to_num_of_intervals_covering_it_array_npz,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_write_index_to_num_of_intervals_covering_it_array(
        intervals_csv_file_path=intervals_csv_file_path,
        array_len=array_len,
        output_file_path_index_to_num_of_intervals_covering_it_array_npz=output_file_path_index_to_num_of_intervals_covering_it_array_npz,
    )

def get_max_num_of_overlapping_intervals(intervals_df):
    assert (intervals_df.iloc[:,0] <= intervals_df.iloc[:,1]).all()
    start_to_count = intervals_df.iloc[:,0].value_counts().to_dict()
    end_to_count = intervals_df.iloc[:,1].value_counts().to_dict()
    starts_and_ends_sorted = sorted(set(start_to_count) | set(end_to_count))
    curr_num_of_overlapping_intervals = 0
    max_num_of_overlapping_intervals = 0
    for pos in starts_and_ends_sorted:
        if pos in start_to_count:
            curr_num_of_overlapping_intervals += start_to_count[pos]
        if pos in end_to_count:
            max_num_of_overlapping_intervals = max(max_num_of_overlapping_intervals, curr_num_of_overlapping_intervals)
            curr_num_of_overlapping_intervals -= end_to_count[pos]
    return max(max_num_of_overlapping_intervals, curr_num_of_overlapping_intervals)

assert get_max_num_of_overlapping_intervals(pd.DataFrame([(1,1),(1,24),(3,5)])) == 2
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(1,1),(1,24),(3,5),(4,8)])) == 3
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(1,1),(1,4),(3,5),(4,8)])) == 3
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(1,1),(1,3),(3,5),(4,8)])) == 2
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(1,1),(1,3),(3,5),(3,8)])) == 3
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(1,3),(1,3),(3,5),(3,8)])) == 4
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(1,3),(1,2),(3,5),(3,8)])) == 3
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(2,2),(1,2),(3,5),(7,8)])) == 2
assert get_max_num_of_overlapping_intervals(pd.DataFrame([(2,2),(1,1),(3,5),(7,8)])) == 1

def is_interval(interval):
    return (type(interval) == tuple) and (len(interval) == 2) and (interval[0] <= interval[1])

def is_interval1_contained_in_interval2(interval1, interval2):
    assert is_interval(interval1)
    assert is_interval(interval2)
    return (interval2[0] <= interval1[0]) and (interval1[1] <= interval2[1])

def is_iterable_of_intervals(intervals):
    return all(is_interval(interval) for interval in intervals)

def old_slow_very_naive_get_merged_intervals(intervals):
    assert is_iterable_of_intervals(intervals)

    merged_intervals = set(intervals)
    while True:
        merged_something_this_cycle = False

        for interval1, interval2 in itertools.permutations(merged_intervals, 2):
            if not (
                (interval1[1] < interval2[0]) or
                (interval2[1] < interval1[0])
            ):
                both_intervals_edges = interval1 + interval2
                assert len(both_intervals_edges) == 4
                new_interval = (min(both_intervals_edges), max(both_intervals_edges))
                merged_intervals.remove(interval1)
                merged_intervals.remove(interval2)
                merged_intervals.add(new_interval)
                merged_something_this_cycle = True
                break

        if not merged_something_this_cycle:
            return merged_intervals

def get_merged_intervals(intervals):
    assert is_iterable_of_intervals(intervals)

    if not intervals:
        return set()

    sorted_intervals = sorted(intervals)
    merged_intervals = set()
    curr_merged_interval_start, curr_merged_interval_end = sorted_intervals[0]
    for interval in sorted_intervals[1:]:
        if interval[0] > curr_merged_interval_end:
            merged_intervals.add((curr_merged_interval_start, curr_merged_interval_end))
            curr_merged_interval_start, curr_merged_interval_end = interval
        else:
            curr_merged_interval_end = max(curr_merged_interval_end, interval[1])
    merged_intervals.add((curr_merged_interval_start, curr_merged_interval_end))

    return merged_intervals

def naive_get_interval_to_merged_interval(intervals):
    intervals = set(intervals)
    merged_intervals = get_merged_intervals(intervals)
    interval_to_merged_interval = {}
    for interval in intervals:
        for merged_interval in merged_intervals:
            if is_interval1_contained_in_interval2(interval, merged_interval):
                assert interval not in interval_to_merged_interval
                interval_to_merged_interval[interval] = merged_interval
                # break # not breaking makes things slower but more safe - it is unnecessary if we trust get_merged_intervals() completely.
    return interval_to_merged_interval

def naive_get_interval_to_merged_interval_and_merged_interval_to_intervals(intervals):
    interval_to_merged_interval = naive_get_interval_to_merged_interval(intervals)

    merged_interval_to_intervals = collections.defaultdict(set)
    for interval, merged_interval in interval_to_merged_interval.items():
        merged_interval_to_intervals[merged_interval].add(interval)
    merged_interval_to_intervals = dict(merged_interval_to_intervals) # I don't want a defaultdict moving around.

    return interval_to_merged_interval, merged_interval_to_intervals

def naive_get_merged_interval_to_intervals(intervals):
    return naive_get_interval_to_merged_interval_and_merged_interval_to_intervals(intervals)[1]

def do_intervals_overlap(interval1, interval2):
    # an alternative would be: is the distance between the interval middles smaller or equal to the sum of half their lengths?
    return len(get_merged_intervals((interval1, interval2))) == 1

# https://en.wikipedia.org/wiki/Partition_of_an_interval
def find_partition_to_subintervals_and_return_subinterval_to_containing_given_intervals(given_intervals, region_to_partition_start=None, region_to_partition_end=None):
    assert given_intervals
    assert is_iterable_of_intervals(given_intervals)
    if region_to_partition_start is None:
        region_to_partition_start = min(start for start, _ in given_intervals)
    if region_to_partition_end is None:
        region_to_partition_end = max(end for _, end in given_intervals)
    assert min(start for start, _ in given_intervals) >= region_to_partition_start
    assert max(end for _, end in given_intervals) <= region_to_partition_end

    ordered_partition_subintervals = []
    subinterval_to_containing_given_intervals = {}

    given_intervals_edges = set(itertools.chain.from_iterable(given_intervals))
    given_interval_edge_to_edge_type_to_given_intervals = {
        edge: collections.defaultdict(set)
        for edge in given_intervals_edges
    }
    for given_interval in given_intervals:
        start, end = given_interval
        given_interval_edge_to_edge_type_to_given_intervals[start]['start'].add(given_interval)
        given_interval_edge_to_edge_type_to_given_intervals[end]['end'].add(given_interval)
    given_interval_edge_to_edge_type_to_given_intervals = {
        edge: dict(edge_type_to_given_intervals) # I don't want a defaultdict moving around.
        for edge, edge_type_to_given_intervals in given_interval_edge_to_edge_type_to_given_intervals.items()
    }

    # print(f'given_interval_edge_to_edge_type_to_given_intervals: {given_interval_edge_to_edge_type_to_given_intervals}')
    # print()

    curr_containing_given_intervals = set()
    curr_subinterval_start = region_to_partition_start
    for given_interval_edge in sorted(given_interval_edge_to_edge_type_to_given_intervals):
        assert curr_subinterval_start <= given_interval_edge
        given_interval_edge_type_to_given_intervals = given_interval_edge_to_edge_type_to_given_intervals[given_interval_edge]
        if 'start' in given_interval_edge_type_to_given_intervals:
            if curr_subinterval_start < given_interval_edge:
                new_subinterval = (curr_subinterval_start, given_interval_edge - 1)
                ordered_partition_subintervals.append(new_subinterval)
                subinterval_to_containing_given_intervals[new_subinterval] = frozenset(curr_containing_given_intervals) # use frozenset to prevent mutability bugs...

            assert (not ordered_partition_subintervals) or (ordered_partition_subintervals[-1][1] == given_interval_edge - 1)
            curr_subinterval_start = given_interval_edge

            curr_containing_given_intervals = curr_containing_given_intervals | given_interval_edge_type_to_given_intervals['start']
            
        if 'end' in given_interval_edge_type_to_given_intervals:
            assert curr_containing_given_intervals
            assert curr_subinterval_start <= given_interval_edge
            new_subinterval = (curr_subinterval_start, given_interval_edge)
            ordered_partition_subintervals.append(new_subinterval)
            subinterval_to_containing_given_intervals[new_subinterval] = frozenset(curr_containing_given_intervals) # use frozenset to prevent mutability bugs...

            curr_subinterval_start = given_interval_edge + 1

            curr_containing_given_intervals = curr_containing_given_intervals - given_interval_edge_type_to_given_intervals['end']

    assert not curr_containing_given_intervals
    assert curr_subinterval_start <= region_to_partition_end + 1
    if curr_subinterval_start < region_to_partition_end + 1:
        assert ordered_partition_subintervals[-1][1] < region_to_partition_end
        new_subinterval = (curr_subinterval_start, region_to_partition_end)
        ordered_partition_subintervals.append(new_subinterval)
        subinterval_to_containing_given_intervals[new_subinterval] = frozenset() # use frozenset to prevent mutability bugs...

    assert sorted(subinterval_to_containing_given_intervals) == ordered_partition_subintervals
    assert is_iterable_an_ordered_partition_to_subintervals(ordered_partition_subintervals)
    # return (ordered_partition_subintervals, subinterval_to_containing_given_intervals)
    return subinterval_to_containing_given_intervals

def find_partition_to_subintervals_and_return_subinterval_to_num_of_containing_given_intervals(given_intervals, region_to_partition_start=None, region_to_partition_end=None):
    assert is_iterable_of_intervals(given_intervals)

    given_intervals_counter = collections.Counter(given_intervals)

    subinterval_to_containing_given_intervals = find_partition_to_subintervals_and_return_subinterval_to_containing_given_intervals(
        given_intervals=given_intervals,
        region_to_partition_start=region_to_partition_start,
        region_to_partition_end=region_to_partition_end,
    )
    return {
        subinterval: sum(given_intervals_counter[given_interval] for given_interval in containing_given_intervals)
        for subinterval, containing_given_intervals in subinterval_to_containing_given_intervals.items()
    }

def naive_get_n_positions_that_touch_all_intervals(intervals, n):
    assert intervals
    assert is_iterable_of_intervals(intervals)
    assert n > 0

    intervals = set(intervals) # remove duplicates
    intervals_edges = set(itertools.chain.from_iterable(intervals))
    for curr_edges in itertools.permutations(intervals_edges, n):
        curr_edges_touch_all_intervals = True
        for interval in intervals:
            if all(not is_point_touching_interval(edge, interval) for edge in curr_edges):
                curr_edges_touch_all_intervals = False
                break
        if curr_edges_touch_all_intervals:
            return set(curr_edges)

def is_iterable_an_ordered_partition_to_subintervals(intervals):
    if not is_iterable_of_intervals(intervals):
        return False

    for curr_interval, next_interval in zip(intervals[:-1], intervals[1:]):
        if curr_interval[1] + 1 != next_interval[0]:
            return False

    return True

def get_intersection_of_2_intervals(interval1, interval2):
    assert is_interval(interval1)
    assert is_interval(interval2)

    intersection_interval = (max(interval1[0], interval2[0]),
                             min(interval1[1], interval2[1]))
    if intersection_interval[0] > intersection_interval[1]:
        return None

    return intersection_interval

def get_intersections_of_intervals_with_interval(intervals, interval):
    intersections = {get_intersection_of_2_intervals(x, interval) for x in intervals}
    return {x for x in intersections if x is not None}

def replace_chars_with_char(s, chars_to_replace, new_char):
    chars_in_s = set(s)
    new_str = s
    for char_to_replace in chars_in_s & set(chars_to_replace):
        new_str = new_str.replace(char_to_replace, new_char)
    return new_str

def get_count_dict_with_low_counts_merged(count_dict, low_count_max_fraction_of_total_count):
    total_count = sum(count_dict.values())
    low_counts = [x for x in count_dict.values() if (x / total_count <= low_count_max_fraction_of_total_count)]
    total_low_count = sum(low_counts)
    new_dict = {k: count for k, count in count_dict.items() if (count / total_count > low_count_max_fraction_of_total_count)}
    new_dict['other'] = total_low_count
    assert sum(new_dict.values()) == total_count
    return new_dict

def remove_redundant_trailing_zeros(str_repr, unit_str_repr):
    if '.' in str_repr:
        while str_repr.endswith(f'0{unit_str_repr}'):
            str_repr = str_repr.replace(f'0{unit_str_repr}', unit_str_repr)
        if str_repr.endswith(f'.{unit_str_repr}'):
            str_repr = str_repr.replace(f'.{unit_str_repr}', unit_str_repr)
    return str_repr

def get_num_rounded_to_thousands_str_repr(num, num_of_digits_after_decimal_point=0):
    # assert num >= 1e3
    str_repr = str(round(num / 1e3, num_of_digits_after_decimal_point)) + 'K'
    str_repr = remove_redundant_trailing_zeros(str_repr, unit_str_repr='K')
    return str_repr

def get_num_rounded_to_millions_str_repr(num, num_of_digits_after_decimal_point=0):
    # assert num >= 1e6
    str_repr = str(round(num / 1e6, num_of_digits_after_decimal_point)) + 'M'
    str_repr = remove_redundant_trailing_zeros(str_repr, unit_str_repr='M')
    return str_repr

def perform_mw_test(df, test_column_name, bool_column_name, use_continuity, alternative):
    u, pvalue = scipy.stats.mannwhitneyu(
        df[df[bool_column_name]][test_column_name],
        df[~df[bool_column_name]][test_column_name],
        use_continuity=use_continuity,
        alternative=alternative,
    )
    mean_u = df[bool_column_name].sum() * (~(df[bool_column_name])).sum() / 2
    u_minus_mean_u = u - mean_u
    return {
        'pvalue': pvalue,
        'u': u,
        'u_minus_mean_u': u_minus_mean_u,
    }

def perform_g_test_or_fisher_exact_test(df, boolean_column1_name, boolean_column2_name, alternative, return_matrix_in_4_keys=False, use_fisher_exact_anyway=False):
    # Require expected and observed frequencies to be at least 5, according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power_divergence.html:
    # "This test is invalid when the observed or expected frequencies in each category are too small. A typical rule is that all of the observed and expected frequencies
    # should be at least 5."
    MIN_EXPECTED_AND_OBSERVED_FREQ_FOR_G_TEST = 5

    # it seems to me that G-test the way I use it is equivalent to fisher exact with two-sided alternative.
    # scipy.stats.chi2_contingency([[10000,10000],[10000,12000]], lambda_="log-likelihood")
    # scipy.stats.fisher_exact([[10000,10000],[10000,12000]], alternative='two-sided')

    assert not df[boolean_column1_name].isna().any()
    assert not df[boolean_column2_name].isna().any()
    counts_df = df[[boolean_column1_name, boolean_column2_name]].value_counts()
    false_false_count = counts_df[(False, False)] if ((False, False) in counts_df) else 0
    false_true_count = counts_df[(False, True)] if ((False, True) in counts_df) else 0
    true_false_count = counts_df[(True, False)] if ((True, False) in counts_df) else 0
    true_true_count = counts_df[(True, True)] if ((True, True) in counts_df) else 0
    matrix_for_test = [[false_false_count, false_true_count],
                       [true_false_count, true_true_count]]
    matrix_total_count = int(np.array(matrix_for_test).sum())

    # non two-sided requires fisher just because I didn't try to figure out how to perform the test with another type of alternative hypothesis. as mentioned above, as far as I
    # can tell, the default result that the G test gives is for the two-sided alternative hypothesis.
    if use_fisher_exact_anyway or (alternative != 'two-sided') or (not (np.array(matrix_for_test) >= MIN_EXPECTED_AND_OBSERVED_FREQ_FOR_G_TEST).all()):
        use_fisher_exact = True
    else:
        use_fisher_exact = not (scipy.stats.contingency.expected_freq(matrix_for_test) >= MIN_EXPECTED_AND_OBSERVED_FREQ_FOR_G_TEST).all()
        # print(scipy.stats.contingency.expected_freq(matrix_for_test))

    if use_fisher_exact:
        test_performed = 'fisher_exact'
        odds_ratio, pvalue = scipy.stats.fisher_exact(matrix_for_test, alternative=alternative)
    else:
        test_performed = 'g'
        g_test_result = scipy.stats.chi2_contingency(matrix_for_test, lambda_="log-likelihood")
        assert true_false_count > 0
        assert false_true_count > 0
        odds_ratio = true_true_count * false_false_count / (true_false_count * false_true_count)
        pvalue = g_test_result[1]

    if return_matrix_in_4_keys:
        test_result = {
            'matrix_for_test_false_false': false_false_count,
            'matrix_for_test_false_true': false_true_count,
            'matrix_for_test_true_false': true_false_count,
            'matrix_for_test_true_true': true_true_count,
            'matrix_total_count': matrix_total_count,
            'odds_ratio': odds_ratio,
            'pvalue': pvalue,
            'test_performed': test_performed,
        }
    else:
        test_result = {
            'matrix_for_test': matrix_for_test,
            'matrix_total_count': matrix_total_count,
            'odds_ratio': odds_ratio,
            'pvalue': pvalue,
            'test_performed': test_performed,
        }
    return test_result

# test_df = pd.DataFrame([
#     *([(False, False)] * 3500),
#     *([(False, True)] * 70),
#     *([(True, False)] * 400),
#     *([(True, True)] * 4),
# ], columns=['a', 'b'])
# # observed is too low
# print(perform_g_test_or_fisher_exact_test(test_df, 'a', 'b', alternative='two-sided'), scipy.stats.fisher_exact([[3500,70],[400, 4]]))
#
# test_df = pd.DataFrame([
#     *([(False, False)] * 3500),
#     *([(False, True)] * 70),
#     *([(True, False)] * 400),
#     *([(True, True)] * 5),
# ], columns=['a', 'b'])
# print(perform_g_test_or_fisher_exact_test(test_df, 'a', 'b', alternative='two-sided'), scipy.stats.chi2_contingency([[3500,70],[400, 5]], lambda_="log-likelihood"))
#
# test_df = pd.DataFrame([
#     *([(False, False)] * 3500),
#     *([(False, True)] * 30),
#     *([(True, False)] * 400),
#     *([(True, True)] * 5),
# ], columns=['a', 'b'])
# # expected is too low
# print(perform_g_test_or_fisher_exact_test(test_df, 'a', 'b', alternative='two-sided'), scipy.stats.fisher_exact([[3500,30],[400, 5]]))

@execute_if_output_doesnt_exist_already
def cached_perform_linear_fit_of_column_histogram(
        input_file_path_df_csv,
        column_name,
        bins,
        fit_log10_of_hist,
        output_file_path_linear_fit_result_pickle,
):
    df = pd.read_csv(input_file_path_df_csv, sep='\t', low_memory=False)
    column = df[column_name]
    assert not column.isna().any()
    hist, _ = np.histogram(column, bins=bins)

    hist_sum = hist.sum()
    hist_median = np.median(hist)

    std_err_of_estimated_gradient = r_value = p_value = intercept = slope = np.nan
    # r_value = p_value = slope_std_err = intercept_std_err = intercept = slope = np.nan
    reason_for_not_performing_fit = None

    if fit_log10_of_hist:
        if (hist > 0).all():
            ys = np.log10(hist)
        else:
            reason_for_not_performing_fit = 'fit_log10_of_hist=True but one of the hist values was zero'
    else:
        ys = hist

    if reason_for_not_performing_fit is None:
        bins = np.array(bins)
        xs = (bins[:-1] + bins[1:]) / 2
        # only numpy 1.7 has the alternative argument. If I understand correctly, the default until then is the same as alternative='two-sided' in newer versions.
        slope, intercept, r_value, p_value, std_err_of_estimated_gradient = scipy.stats.linregress(
            xs, ys,
            # alternative='two-sided',
        )

        # wrote this (but didn't test it) before finding out about scipy.stats.linregress.
        # coefficients, (residual_sum_of_squares, _, _, _) = np.polynomial.Polynomial.fit(xs, ys, deg=1, full=True)
        # intercept, slope = coefficients

    linear_fit_result = {
        'intercept': intercept,
        'slope': slope,
        'r_value': r_value,
        'p_value': p_value,
        # the following two are only available in newer versions, while the version i use instead gives std_err_of_estimated_gradient, if I understand correctly.
        # 'slope_std_err': slope_std_err,
        # 'intercept_std_err': intercept_std_err,
        'std_err_of_estimated_gradient': std_err_of_estimated_gradient,
        'reason_for_not_performing_fit': reason_for_not_performing_fit,
        'hist_sum': hist_sum,
        'hist_median': hist_median,
    }

    with open(output_file_path_linear_fit_result_pickle, 'wb') as f:
        pickle.dump(linear_fit_result, f, protocol=4)

def perform_linear_fit_of_column_histogram(
        input_file_path_df_csv,
        column_name,
        bins,
        fit_log10_of_hist,
        output_file_path_linear_fit_result_pickle,
):
    # the extra level is needed so that the cached functions will always have all arguments.
    # otherwise, if some of the optional arguments aren't specified, my caching algorithm would think the arguments are
    # different, and run the function again.
    return cached_perform_linear_fit_of_column_histogram(
        input_file_path_df_csv=input_file_path_df_csv,
        column_name=column_name,
        bins=bins,
        fit_log10_of_hist=fit_log10_of_hist,
        output_file_path_linear_fit_result_pickle=output_file_path_linear_fit_result_pickle,
    )

def get_vertex_to_connected_component_index_for_undirected_graph(vertices, edges):
    assert isinstance(vertices, set)
    assert isinstance(edges, set)

    vertex_index_to_vertex = {i: v for i, v in enumerate(vertices)}
    vertex_to_vertex_index = {v: i for i, v in enumerate(vertices)}
    edges_as_sorted_vertex_indices = {tuple(sorted((vertex_to_vertex_index[v1], vertex_to_vertex_index[v2]))) for v1, v2 in edges}

    vertex_index_to_connected_component = {i: frozenset({i}) for i in vertex_index_to_vertex}
    for i1, i2 in edges_as_sorted_vertex_indices:
        new_connected_component = vertex_index_to_connected_component[i1] | vertex_index_to_connected_component[i2]
        for i in new_connected_component:
            vertex_index_to_connected_component[i] = new_connected_component

    index_connected_components = set(vertex_index_to_connected_component.values())
    index_connected_component_to_connected_component_index = {c: i for i, c in enumerate(index_connected_components)}
    vertex_to_connected_component_index = {vertex_index_to_vertex[i]: index_connected_component_to_connected_component_index[c]
                                           for i, c in vertex_index_to_connected_component.items()}
    return vertex_to_connected_component_index

assert len(set(get_vertex_to_connected_component_index_for_undirected_graph(set(range(9)), set()).values())) == 9
assert len(set(get_vertex_to_connected_component_index_for_undirected_graph(set(range(9)), {(1, 2), (2, 8), (6, 7)}).values())) == 6
assert len(set(get_vertex_to_connected_component_index_for_undirected_graph(
    set(range(9)), {(1, 2), (2, 8), (6, 7), (2, 3), (4, 3), (3, 7), (5, 4)}).values())) == 2
assert len(set(get_vertex_to_connected_component_index_for_undirected_graph(
    set(range(9)), {(1, 2), (2, 8), (6, 7), (2, 3), (4, 3), (3, 7), (5, 4), (6, 3)}).values())) == 2
assert len(set(get_vertex_to_connected_component_index_for_undirected_graph(
    set(range(9)), {(1, 2), (2, 8), (6, 7), (2, 3), (4, 3), (3, 7), (5, 4), (6, 0)}).values())) == 1

def uncomment_python_line(line):
    line_before_hash, _, line_after_hash = line.partition('#')
    return line_before_hash + line_after_hash.lstrip()
