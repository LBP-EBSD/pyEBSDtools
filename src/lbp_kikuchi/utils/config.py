from omegaconf import DictConfig, OmegaConf


def cfg_to_dict(cfg: DictConfig) -> dict:
    """Resolve and flatten an OmegaConf config to a plain Python dict for logging."""
    return OmegaConf.to_container(cfg, resolve=True)
