from easyvolcap.engine import cfg, call_from_cfg  # need this for initialization?
from easyvolcap.engine import RUNNERS  # controls the optimization loop of a    particular epoch
from reactmotion.runners.motion_base_runner import MotionBaseRunner
from reactmotion.utils.net_utils import load_other_network
from reactmotion.utils.motion_repr_transform import motoken_cfg


@RUNNERS.register_module()
class MoConRunner(MotionBaseRunner):  # a plain and simple object controlling the training loop
    def __init__(self,
                 **kwargs
                 ):
        
        call_from_cfg(super().__init__, kwargs)
        # Motion Tokenizer Network
        cfg.motoken_net = load_other_network(motoken_cfg)