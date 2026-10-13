"""Loading generator files (function generate_map(seed) -> str) and generating maps in parallel. No torch needed."""
import atexit
import importlib.util
import multiprocessing as mp
import os

_generators = {}


def load_generator(path):
    path = os.path.abspath(path)
    if path not in _generators:
        spec = importlib.util.spec_from_file_location(f'generator_{len(_generators)}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, 'generate_map'):
            raise ValueError(f'{path} has no function generate_map(seed)')
        _generators[path] = module
    return _generators[path]


def _generate_one(args):
    path, seed = args
    return load_generator(path).generate_map(seed)


_pool = None


def generate(path, seeds, workers):
    """One process pool is kept for the whole run: starting worker processes is slow on Windows."""
    global _pool
    if _pool is None:
        _pool = mp.Pool(workers)
        atexit.register(_pool.terminate)
    return list(_pool.imap(_generate_one, [(path, s) for s in seeds], chunksize=16))
