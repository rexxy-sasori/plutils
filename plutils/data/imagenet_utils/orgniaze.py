import argparse
import os

import wget

SHELL_SCRIPT_LINK = "http://randomsite.com/file.gz"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-dir', required=True)
    cmd_args = parser.parse_args()

    current_dir = os.getcwd()

    os.chdir(cmd_args.working_dir)
    filepath = wget.download(SHELL_SCRIPT_LINK, cmd_args.working_dir)
    os.system("bash imgnet_organize.sh")
    os.chdir(current_dir)
