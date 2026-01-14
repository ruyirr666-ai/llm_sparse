import os

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_dict(args):
    # If args is already a dict, return it.
    if isinstance(args, dict):
        return args
    # If args has __dict__, return it (e.g. argparse.Namespace)
    if hasattr(args, '__dict__'):
        return args.__dict__
    return args
