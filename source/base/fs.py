import typing
import collections
import os

import numpy as np


def load_csv_points_cached(
        csv_path: str, npy_path: typing.Optional[str] = None,
        mmap_mode: typing.Optional[str] = None) -> np.ndarray:
    """
    Loads a comma-delimited points file (e.g. chunkPoints.csv, 'x, y, z, r, g, b' rows) as a
    float64 (n, k) array, transparently caching it as a binary .npy file next to it so repeat
    reads skip text parsing entirely.

    np.loadtxt on a large points file is slow (line-by-line text parsing) and memory-heavy
    (several times the final array size, transiently, while parsing) compared to a binary
    np.load -- for a 1.6 GB chunkPoints.csv (35M points) this is the difference between a
    multi-minute parse and a near-instant memory-mapped-speed read. First read still has to
    parse the CSV once; every read after that (until the CSV changes) hits the .npy cache.

    mmap_mode='r' opens the cached .npy read-only via memory-mapping instead of loading it fully
    -- this is what actually avoids N-way RAM duplication across separate DataLoader worker
    processes (spawn-based multiprocessing on Windows doesn't share memory the way fork's
    copy-on-write does, but independent mmaps of the *same file* still share physical pages via
    the OS page cache, no explicit shared-memory API needed). Only takes effect on a cache hit;
    the first-time parse still needs the full array in memory to write it out.
    """
    if npy_path is None:
        npy_path = os.path.splitext(csv_path)[0] + '.npy'

    if not call_necessary(file_in=csv_path, file_out=npy_path):
        try:
            return np.load(npy_path, mmap_mode=mmap_mode)
        except Exception:
            pass  # fall through and reparse if the cache is corrupt/truncated

    chunk_pts = np.loadtxt(csv_path, dtype=np.float64, delimiter=',')
    make_dir_for_file(npy_path)
    tmp_path = f'{npy_path}.tmp.{os.getpid()}.npy'
    np.save(tmp_path, chunk_pts)
    os.replace(tmp_path, npy_path)
    return chunk_pts


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


def _unit_test_load_csv_points_cached():
    import tempfile
    import time

    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1000.0, size=(200, 6))

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, 'points.csv')
        npy_path = os.path.splitext(csv_path)[0] + '.npy'
        np.savetxt(csv_path, pts, delimiter=',')

        assert not os.path.exists(npy_path)
        loaded = load_csv_points_cached(csv_path)
        assert os.path.exists(npy_path), 'cache file was not created on first read'
        assert np.allclose(loaded, pts, atol=1e-4), 'first (CSV-parsing) read does not match the source data'

        # second read should hit the cache and still match, byte-for-byte this time (no reparse)
        loaded_cached = load_csv_points_cached(csv_path)
        assert np.array_equal(loaded, loaded_cached), 'cached read differs from the first read'

        # touching the CSV after the cache was written must invalidate it
        time.sleep(0.05)
        new_pts = rng.uniform(0.0, 1000.0, size=(50, 6))
        np.savetxt(csv_path, new_pts, delimiter=',')
        loaded_after_update = load_csv_points_cached(csv_path)
        assert loaded_after_update.shape[0] == 50, 'stale .npy cache was used instead of reparsing the updated CSV'
        assert np.allclose(loaded_after_update, new_pts, atol=1e-4)
    print('_unit_test_load_csv_points_cached: OK')


if __name__ == '__main__':
    _unit_test_load_csv_points_cached()
