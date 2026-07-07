import numpy as np

def dataloader_wrapper(args, partition):
    if args.network_reference.cue == 'lip':
        from .dataset_lip import get_dataloader_lip as get_dataloader
    else:
        raise NameError('Wrong reference for dataloader selection')
    return get_dataloader(args, partition)
    






















