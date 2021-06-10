import carla_utils as cu

parse_yaml_file_unsafe = cu.parse_yaml_file_unsafe
create_dir = cu.basic.create_dir

setup_seed = cu.basic.setup_seed



def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')
    argparser.add_argument('-m', '--method', default='None', type=str, help='[Method] Method to use.')
    argparser.add_argument('--eval', action='store_true', help='[Method] Eval mode (default: False)')

    argparser.add_argument('--load-model', action='store_true', help='[Model] Load model (default: False)')
    argparser.add_argument('--model-dir', default='None', type=str, help='[Model] dir contains model (default: False)')
    argparser.add_argument('--model-num', default=-1, type=str, help='[Model] model-num to use.')

    args = argparser.parse_args()
    return args

