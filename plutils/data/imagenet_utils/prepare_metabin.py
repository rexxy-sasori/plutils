import argparse

from pl_bolts.datasets import UnlabeledImagenet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devkit-path', required=True)
    command_args = parser.parse_args()

    UnlabeledImagenet.generate_meta_bins(command_args.devkit_path)
