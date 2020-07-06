import torch

from mmdet.datasets import NightOwlsDataset, WaymoOpenDataset


def rearrange_class_weights(model, indices):
    """Rearrange weights and biases of classification layer."""
    target_layers = ['bbox_head.atss_cls', 'module.bbox_head.atss_cls']
    num_found_target_layer = 0
    for name, module in model.named_modules():
        if name in target_layers:
            num_found_target_layer += 1
            conv_cls = module

    if num_found_target_layer != 1:
        raise NotImplementedError('Only ATSS is supported currently.')

    # "index_select" not implemented for 'Half' (fp16) as of PyTorch 1.5.
    # Instead, split and concat weights and biases.

    splitted_weights = torch.split(conv_cls.weight.data, 1, dim=0)
    rearranged_weights = [splitted_weights[idx] for idx in indices]
    conv_cls.weight.data = torch.cat(rearranged_weights, dim=0)

    splitted_biases = torch.split(conv_cls.bias.data, 1, dim=0)
    rearranged_biases = [splitted_biases[idx] for idx in indices]
    conv_cls.bias.data = torch.cat(rearranged_biases, dim=0)

    print('Weights and biases of classification layer were rearranged ' +
          f'in order of {indices}')

    return model


def rearrange_classes(model, classes_to_use, dataset_type):
    if dataset_type not in ['NightOwlsDataset', 'WaymoOpenDataset']:
        raise NotImplementedError

    if dataset_type == 'NightOwlsDataset':
        all_classes = NightOwlsDataset.CLASSES
    if dataset_type == 'WaymoOpenDataset':
        all_classes = WaymoOpenDataset.CLASSES
    print('classes_to_use:', classes_to_use)
    print('all_classes:', all_classes)
    assert set(classes_to_use) <= set(all_classes)

    indices = [all_classes.index(c) for c in classes_to_use]
    idle_class_indices = set(range(len(all_classes))) - set(indices)
    indices.extend(idle_class_indices)

    model = rearrange_class_weights(model, indices)
    return model
