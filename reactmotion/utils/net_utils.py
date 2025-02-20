from easyvolcap.engine import MODELS, cfg
from easyvolcap.utils.net_utils import load_network, load_model


def load_other_network(curr_cfg):
    '''
        Load the network from other config file,
        before that, need to set cfg.norm_file and other global configs
    '''
    model = MODELS.build(curr_cfg.model_cfg)
    model.cuda().eval()
    load_network(
        model=model,  # only loading the network, without recorder
        model_dir=f'data/trained_model/{curr_cfg.exp_name}',
        resume=True,
    )
    return model.network


def load_other_model(curr_cfg):
    model = MODELS.build(curr_cfg.model_cfg)
    model.cuda().eval()
    load_model(
        model=model,  # only loading the network, without recorder
        model_dir=f'data/trained_model/{curr_cfg.exp_name}',
        resume=True,
    )
    return model