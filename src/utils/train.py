""" Utils for the training loop. Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py """
import os.path
from typing import Mapping, Sequence

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf, ListConfig

from ..self_logger import logger
from ..self_logger.base import is_main_process


def print_config(
        config: DictConfig,
        save_dir: str,
        resolve: bool = True,
        prefix: str = "train",
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_dir:
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    # if is_main_process:
    #     rich.print(tree)
    #     logger.info(OmegaConf.to_yaml(config, resolve=True))

    if os.path.exists(save_dir):
        config_name = "_".join([prefix, "config_tree.txt"])
        with open(os.path.join(save_dir, config_name), "w") as fp:
            fp.write(OmegaConf.to_yaml(config, resolve=True))


def log_optimizer(logger, optimizer, keys):
    """ Log values of particular keys from the optimizer's param groups """
    keys = sorted(keys)
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        logger.info(' | '.join([
                                   f"Optimizer group {i}",
                                   f"{len(g['params'])} tensors",
                               ] + [f"{k} {v}" for k, v in group_hps.items()]))


def process_config(config: OmegaConf) -> DictConfig:
    # because of filter_keys, this is no longer in place
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    # Filter out keys that were used just for interpolation
    # config = dictconfig_filter_keys(config, lambda k: not k.startswith('__'))
    config = omegaconf_filter_keys(config, lambda k: not k.startswith('__'))

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)
    return config


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    return isinstance(x, Mapping)


def omegaconf_filter_keys(d, fn=None):
    """Only keep keys where fn(key) is True. Support nested DictConfig.
    """
    if fn is None:
        fn = lambda _: True
    if is_list(d):
        return ListConfig([omegaconf_filter_keys(v, fn) for v in d])
    elif is_dict(d):
        return DictConfig(
            {k: omegaconf_filter_keys(v, fn) for k, v in d.items() if fn(k)}
        )
    else:
        return d
