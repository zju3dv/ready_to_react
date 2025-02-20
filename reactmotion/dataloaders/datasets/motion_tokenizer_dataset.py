import torch
import random

from easyvolcap.engine import DATASETS, cfg, dotdict, call_from_cfg
from reactmotion.dataloaders.datasets.motion_base_dataset import MotionBaseDataset
from reactmotion.utils.motion_repr_transform import *


@DATASETS.register_module()
class BoxingMoTokenDataset(MotionBaseDataset):
    def __init__(self,
                # vq settings
                vocab_size: int = cfg.window_cfg.get("vocab_size", 512),
                vqmotion_window_size: int = cfg.window_cfg.get("vqmotion_window_size", 64),
                block_size = cfg.window_cfg.get("block_size", 48),
                infer_gpttoken_window_size = cfg.window_cfg.get("infer_gpttoken_window_size", 1),
                **kwargs,
                ):
        
        call_from_cfg(super().__init__, kwargs)
        # vq settings
        self.vocab_size = vocab_size
        self.vqmotion_window_size = vqmotion_window_size
        self.block_size = block_size
        self.infer_gpttoken_window_size = infer_gpttoken_window_size

        self.pad_token = self.vocab_size # 513
        self.max_motion_length = self.block_size * U
        self.max_token_length = self.block_size


    def __getitem__(self, index):
        index = index % len(self.preprocess_data)
        wholeseq = self.preprocess_data[index]

        if self.split == "TRAIN":
            meta, inputs = wholeseq.meta, wholeseq.pose_series
            ctrl_sid = random.randint(0, len(inputs) - self.vqmotion_window_size)
            ctrl_eid = ctrl_sid + self.vqmotion_window_size
            return dotdict(
                meta = meta,
                inputs = inputs[ctrl_sid : ctrl_eid],
                supervise_MoToken = inputs[ctrl_sid : ctrl_eid],
            )
        
        elif self.split == "VAL":
            meta, motion1, motion2 = wholeseq.meta, wholeseq.motion1, wholeseq.motion2
            prep = getattr(self, f"load_{self.load_prep_type}")(meta, motion1, motion2)
            inputs = prep.pose_series
            ctrl_sid = 0
            ctrl_eid = len(inputs) // U * U
            # ctrl_sid = random.randint(0, len(inputs) - self.vqmotion_window_size)
            # ctrl_eid = ctrl_sid + self.vqmotion_window_size
            return dotdict(
                meta = meta,
                inputs = inputs[ctrl_sid : ctrl_eid],
                supervise_MoToken = inputs[ctrl_sid : ctrl_eid],
            )


    def load_motoken(self, meta, motion1, motion2):
        frames = list(range(motion1.pose_vel.size(1)))
        return dotdict(
            meta = meta,
            pose_series = MoReprTrans.cal_pose_series(motion1, frames)[0],
        )
    

    def cal_norm_data(self, ):
        MoReprTrans.save_norm_data(self.train_data, 'pose_series', cfg.norm_cfg.motoken_input_norm_file)