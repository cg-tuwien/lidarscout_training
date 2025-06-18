import typing
import collections
import os

import numpy as np


def make_dir_for_file(file):
    file_dir = os.path.dirname(file)
    if file_dir != '':
        if not os.path.exists(file_dir):
            try:
                os.makedirs(os.path.dirname(file), exist_ok=True)
            except FileExistsError as exc:
                pass
            except OSError as exc:  # Guard against race condition
                raise


def call_necessary(file_in: typing.Union[str, typing.Sequence[str]], file_out: typing.Union[str, typing.Sequence[str]],
                   min_file_size=0, verbose=False):
    """
    Check if all input files exist and at least one output file does not exist or is invalid.
    :param file_in: list of str or str
    :param file_out: list of str or str
    :param min_file_size: int
    :param verbose: bool
    :return:
    """

    def check_parameter_types(param):
        if isinstance(param, str):
            return [param]
        elif isinstance(param, list):
            return param
        elif isinstance(param, tuple):
            return param
        else:
            raise ValueError('Wrong input type')

    file_in = check_parameter_types(file_in)
    file_out = check_parameter_types(file_out)

    def print_result(msg: str):
        if verbose:
            print('call_necessary\n {}\n ->\n {}: \n{}'.format(file_in, file_out, msg))

    if len(file_out) == 0:
        print_result('No output')
        return True

    inputs_missing = [f for f in file_in if not os.path.isfile(f)]
    if len(inputs_missing) > 0:
        print_result('WARNING: Input files are missing: {}'.format(inputs_missing))
        return False

    outputs_missing = [f for f in file_out if not os.path.isfile(f)]
    if len(outputs_missing) > 0:
        print_result('Some output files are missing: {}'.format(outputs_missing))
        return True

    min_output_file_size = min([os.path.getsize(f) for f in file_out])
    if min_output_file_size < min_file_size:
        print_result('Output too small')
        return True

    oldest_input_file_mtime = max([os.path.getmtime(f) for f in file_in])
    youngest_output_file_mtime = min([os.path.getmtime(f) for f in file_out])
    if oldest_input_file_mtime >= youngest_output_file_mtime:
        if verbose:
            import time
            import numpy as np
            input_file_mtime_arg_max = np.argmax(np.array([os.path.getmtime(f) for f in file_in]))
            output_file_mtime_arg_min = np.argmin(np.array([os.path.getmtime(f) for f in file_out]))
            input_file_mtime_max = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(oldest_input_file_mtime))
            output_file_mtime_min = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(youngest_output_file_mtime))
            print_result('Input file {} is newer than output file {}: {} >= {}'.format(
                file_in[input_file_mtime_arg_max], file_out[output_file_mtime_arg_min],
                input_file_mtime_max, output_file_mtime_min))
        return True

    return False


def sanitize_file_name(entry: str):
    return entry.replace('.', '_').replace('/', '_').replace(',', '_')


def str_to_consistent_hash(s):
    import hashlib
    # builtin hash() is salted for strings
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:16], 16)


def md5(s):
    import hashlib
    return int(hashlib.md5(s).hexdigest()[:16], 16)
