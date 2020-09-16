import argparse
from collections import OrderedDict

import torch


def convert(in_file, out_file):
    """Convert values in checkpoints."""
    # to ('pedestrian', 'bicycledriver', 'motorbikedriver')
    # from ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')
    indices = [1, 2, 2]

    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()

    for key, val in in_state_dict.items():
        new_val = val

        if key == 'bbox_head.atss_cls.weight':
            print(key)
            print(val.shape)
            splitted_weights = torch.split(new_val.data, 1, dim=0)
            rearranged_weights = [splitted_weights[idx] for idx in indices]
            new_val.data = torch.cat(rearranged_weights, dim=0)
        elif key == 'bbox_head.atss_cls.bias':
            print(key)
            print(val.shape)
            print(new_val)
            splitted_biases = torch.split(new_val.data, 1, dim=0)
            rearranged_biases = [splitted_biases[idx] for idx in indices]
            new_val.data = torch.cat(rearranged_biases, dim=0)
            print(new_val)

        out_state_dict[key] = new_val

    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
