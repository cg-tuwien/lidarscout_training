import typing

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def flatten_dicts(dicts: typing.Sequence[typing.Dict[typing.Any, typing.Any]]):
    """
    flatten dicts containing other dicts
    :param dicts:
    :return:
    """
    if len(dicts) == 0:
        return dict()
    elif len(dicts) == 1:
        return dicts[0]

    new_dicts = []
    for d in dicts:
        new_dict = {}
        for k in d.keys():
            value = d[k]
            if isinstance(value, typing.Dict):
                new_dict.update(value)
            else:
                new_dict[k] = value
        new_dicts.append(new_dict)

    return new_dicts


def aggregate_dicts(dicts: typing.Sequence[typing.Mapping[typing.Any, 'torch.Tensor']], method: str):
    """

    :param dicts:
    :param method: one of ['mean', 'concat', 'stack']
    :return:
    """
    import torch
    import numbers

    if len(dicts) == 0:
        return dict()
    elif len(dicts) == 1:
        return dicts[0]

    valid_methods = ['mean', 'concat', 'stack']
    if method not in valid_methods:
        raise ValueError('Invalid method {} must be one of {}'.format(method, valid_methods))

    dict_aggregated = dict()
    for k in dicts[0].keys():
        values = [d[k] for d in dicts]
        if isinstance(values[0], numbers.Number):
            values = [torch.as_tensor(v) for v in values]

        if method == 'concat':
            values = torch.cat(values)
        elif method == 'stack':
            if isinstance(values[0], str):
                pass  # keep list of strings
            else:
                values = torch.stack(values)
        elif method == 'mean':
            values = torch.tensor(values)
            if values.dtype == str:
                values = values[0]
            else:
                values = torch.nanmean(values).item()
        else:
            raise ValueError()
        dict_aggregated[k] = values
    return dict_aggregated
