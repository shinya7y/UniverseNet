import argparse
import subprocess
from collections import OrderedDict

import torch


def convert(in_file, out_file):
    """Convert keys in checkpoints."""
    in_state_dict = torch.load(in_file, map_location='cpu')
    out_state_dict = OrderedDict()

    for key, val in in_state_dict.items():
        new_key = key

        if key[:5] == 'conv1':
            new_key = 'stem' + key[5:]
        if key[:3] == 'bn1':
            new_key = 'stem.7' + key[3:]

        if key != new_key:
            print(f'{key} -> {new_key}')

        out_state_dict[new_key] = val
    torch.save(out_state_dict, out_file)

    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
