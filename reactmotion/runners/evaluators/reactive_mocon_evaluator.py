import torch
from sklearn.metrics import pairwise_distances

from easyvolcap.engine import EVALUATORS, cfg, call_from_cfg, log
from easyvolcap.utils.console_utils import *
from reactmotion.runners.evaluators.motion_evaluator import MotionEvaluator
import reactmotion.utils.matrix as matrix
from reactmotion.utils.data_utils import to_numpy
from reactmotion.utils.motion_repr_transform import *
from reactmotion.utils.eval_utils import calculate_fid_between_gt_and_pd
from reactmotion.utils.wis3d_utils import wis3d_add_skeleton, make_wis3d


@EVALUATORS.register_module()
class ReactiveMoConEvaluator(MotionEvaluator):
    '''
    Evaluate generation metrics (FID, Diversity) for Reactive or TwoAgent settings
    '''
    def __init__(self, 
                 inference_func: str = cfg.model_cfg.get('inference_func', 'reactive'),
                 apd_compute: bool = False,
                 feature_compute: bool = False, 
                 feature_types: list = [],
                 feature_metric_names: list = [],
                 vis_vr3joints: bool = False,
                 test_speed: bool = False,
                 **kwargs,
                 ) -> None:
        call_from_cfg(super().__init__, kwargs)
        self.inference_func = inference_func
        log(f"inference setting: {inference_func}")
        self.apd_compute = apd_compute
        self.feature_compute = feature_compute
        self.test_speed = test_speed
        if self.apd_compute: 
            log("compute apd, removing other metrics")
            self.feature_compute = False
            self.metric_names = []
        if self.test_speed:
            log("test speed, removing other metrics")
            self.feature_compute = False
            self.metric_names = ['speed']
        if self.feature_compute: log("compute FID features")
        self.feature_types = feature_types
        self.feature_metric_names = feature_metric_names

        self.perframe_features = dotdict(gt=[], pd=[])
        self.pertrans_features = dotdict(gt=[], pd=[])
        self.perclip_features = dotdict(gt=[], pd=[])

        self.vis_vr3joints = vis_vr3joints
    

    def evaluate(self, output: dotdict, batch: dotdict):
        metrics = super().evaluate(output, batch)
        if self.apd_compute:
            metric_name = 'apd'
            metric = self.__getattribute__(f'compute_{metric_name}')(output, batch)
            for k, v in metric.items():
                metrics[k] = v
                if isinstance(v, (int, float)):
                    if k in self.metrics: self.metrics[k].append(v)
                    else: self.metrics[k] = [v]
                elif isinstance(v, list):
                    if k in self.metrics: self.metrics[k].extend(v)
                    else: self.metrics[k] = v

        if self.feature_compute:
            for feat_type in self.feature_types:
                self.record_feature(output, batch, feat_type)
        return metrics
    

    def update_vis_stats(self, output: dotdict, batch: dotdict):
        '''
            Rarely changed.
        '''
        if self.vis_output:
            gt_stats = dotdict()
            pd_stats = dotdict()
            for key in self.vis_stats_keys:
                gt_stats[key] = batch[key]
                pd_stats[key] = output[key]
            gt_stats.meta = batch.meta
            self.vis_stats.gt.append(gt_stats)
            self.vis_stats.pd.append(pd_stats)
    

    def summarize(self):
        if self.feature_compute:
            for metric_name in self.feature_metric_names:
                getattr(self, f"compute_{metric_name}")()
        summary = super().summarize()
        self.perframe_features = dotdict(gt=[], pd=[])
        self.pertrans_features = dotdict(gt=[], pd=[])
        self.perclip_features = dotdict(gt=[], pd=[])
        return summary
    

    def compute_fid(self, ):
        for feat_type in self.feature_types:
            feature_dict = getattr(self, f"{feat_type}_features")
            pd_features = np.concatenate(feature_dict.pd, axis=1)[0]
            gt_features = np.concatenate(feature_dict.gt, axis=1)[0]
            self.metrics[f"{feat_type}_pd_fid"] = [calculate_fid_between_gt_and_pd(gt_features, pd_features)]
            self.metrics[f"{feat_type}_gt_fid"] = [calculate_fid_between_gt_and_pd(gt_features, gt_features)]

    
    def visualize_pose_pos(self, ):
        for i in range(len(self.vis_stats.gt)): self.vis_pose_i(i, i)
    

    def vis_pose_i(self, idx, vis3d_id):
        gt = self.vis_stats.gt[idx]
        pd = self.vis_stats.pd[idx]
        
        sbj = gt.meta.sbj_name[0]
        name = gt.meta.npy_file[0].split('/')[-1].split('_')[0]
        self.vis3d = make_wis3d(f"{cfg.exp_name}-epoch{self.epoch_name}-file{name}-{self.inference_func}-{sbj}")

        F = pd.agent1.pose_pos.size(1)
        for f in range(F):
            self.vis3d.set_scene_id(f)
            if f < pd.agent1.pose_pos.size(1):
                wis3d_add_skeleton(self.vis3d, f, pd.agent1.pose_pos[0, f], parents, f"agent1-pos-pd")
                wis3d_add_skeleton(self.vis3d, f, pd.agent2.pose_pos[0, f], parents, f"agent2-pos-pd")
                # gt
                if f < gt.agent1.pose_pos.size(1):
                    wis3d_add_skeleton(self.vis3d, f, gt.agent1.pose_pos[0, f], parents, f"agent1-pos-gt")
                    wis3d_add_skeleton(self.vis3d, f, gt.agent2.pose_pos[0, f], parents, f"agent2-pos-gt")
                    if self.vis_vr3joints:
                        self.vis3d.add_point_cloud(gt.agent1.pose_pos[0, f, VR3JOINTS_INDEX], name="vr3joints")
    
    '''
    For Speed Test
    '''
    def compute_speed(self, output, batch):
        error = dotdict()
        error.time_network = batch.time_network
        return error
    
    '''
    Jitter
    '''
    def compute_jitter(self, output, batch):
        """compute jitter of the motion
        Args:
            joints (N, J, 3).
            fps (float).
        Returns:
            jitter (N-3).
        """
        error = dotdict()
        def calc_jitter(joints, fps):
            pred_jitter = torch.norm(
                (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
                dim=2,
            ).mean(dim=-1)
            return pred_jitter.cpu().numpy() / 10.0
        if self.inference_func == 'reactive':
            error.gt_jitter = calc_jitter(batch.agent1.pose_pos[0], fps=30).tolist()
            error.pd_jitter = calc_jitter(output.agent1.pose_pos[0], fps=30).tolist()
        elif self.inference_func == 'twoagent':
            error.gt_jitter = calc_jitter(batch.agent1.pose_pos[0], fps=30).tolist() + calc_jitter(batch.agent2.pose_pos[0], fps=30).tolist()
            error.pd_jitter = calc_jitter(output.agent1.pose_pos[0], fps=30).tolist() + calc_jitter(output.agent2.pose_pos[0], fps=30).tolist()

        return error
    
    '''
    For APD diversity
    '''
    def compute_apd(self, output, batch):
        if self.inference_func == 'reactive':
            feature = self.cal_reactive_perframe_feature(output.agent1, output.agent1)
        elif self.inference_func == 'twoagent':
            feature = self.cal_twoagent_feature(output.agent1, output.agent2, 'perframe')
        feature = to_numpy(feature)
        K, L = feature.shape[0], feature.shape[1]
        dist = 0
        for j in range(L):
            F = feature[:, j, :]
            dist += (pairwise_distances(F, F, metric='l2').sum() / (K * (K - 1)))
        error = dotdict()
        error.apd = dist / L
        return error
        
    
    '''
    For orientation error calculation
    '''
    def compute_twoagent_rootrot(self, output, batch):
        error = dotdict()
        gt_root_dir = matrix.get_mat_BtoA(batch.agent1.root_rot, batch.agent2.root_rot)
        pd_root_dir = matrix.get_mat_BtoA(output.agent1.root_rot, output.agent2.root_rot)
        gt_root_dir_2d = to_numpy(gt_root_dir[..., [0, 2], 2])
        pd_root_dir_2d = to_numpy(pd_root_dir[..., [0, 2], 2])
        threshold = np.pi / 4
        tgt_dir_2d = np.array([[0, -1.]])
        # error.gt_twoagent_rootrot_perseq = (np.arccos(np.sum(gt_root_dir_2d * tgt_dir_2d, axis=-1)) > threshold).mean().item()
        # error.pd_twoagent_rootrot_perseq = (np.arccos(np.sum(pd_root_dir_2d * tgt_dir_2d, axis=-1)) > threshold).mean().item()
        error.gt_twoagent_rootrot_perframe = (np.arccos(np.sum(gt_root_dir_2d * tgt_dir_2d, axis=-1)) > threshold)[0].tolist()
        error.pd_twoagent_rootrot_perframe = (np.arccos(np.sum(pd_root_dir_2d * tgt_dir_2d, axis=-1)) > threshold)[0].tolist()
        return error

    '''
    For foot sliding
    '''
    def compute_foot_slide(self, output, batch):
        error = dotdict()
        toe_idx = SG[skeleton]["toe"]
        threshold = 0.05
        def get_foot_slide(agent):
            toe_pos = agent.pose_pos[:, :, toe_idx]
            ci = torch.nonzero(toe_pos[..., 1] < toe_pos[..., 1].min() + threshold)
            pi = [torch.tensor([i[0], max(i[1]-1, 0), i[2]]) for i in ci]
            pi = torch.stack(pi, dim=0)
            slide = (toe_pos[ci[:, 0], ci[:, 1], ci[:, 2]] - toe_pos[pi[:, 0], pi[:, 1], pi[:, 2]])
            slide = (slide ** 2).sum(-1).sqrt()
            return to_numpy(slide * 100)

        if self.inference_func == 'reactive':
            gt_slide = get_foot_slide(batch.agent1).tolist()
            pd_slide = get_foot_slide(output.agent1).tolist()

        elif self.inference_func == 'twoagent':
            gt_slide = get_foot_slide(batch.agent1).tolist() + get_foot_slide(batch.agent2).tolist()
            pd_slide = get_foot_slide(output.agent1).tolist() + get_foot_slide(output.agent2).tolist()
        
        error.gt_foot_slide = gt_slide
        error.pd_foot_slide = pd_slide
        return error

    '''
    For vr3joints error
    '''
    def compute_vr3joints_error(self, output, batch):
        error = dotdict()
        gt_vr3joints_pos = batch.agent1.pose_pos[:, U:, VR3JOINTS_INDEX]
        pd_vr3joints_pos = output.agent1.pose_pos[:, U:, VR3JOINTS_INDEX]
        assert gt_vr3joints_pos.shape[0] == 1
        pos_error = ((gt_vr3joints_pos[0] - pd_vr3joints_pos[0]) ** 2).sum(-1).sqrt().mean(-1)
        error.pos_error = to_numpy(pos_error * 100).tolist()

        gt_vr3joints_rot = batch.agent1.pose_rot[:, U:, VR3JOINTS_INDEX]
        pd_vr3joints_rot = output.agent1.pose_rot[:, U:, VR3JOINTS_INDEX]
        R1_inv_batch = torch.linalg.inv(gt_vr3joints_rot[0])
        R_relative_batch = torch.matmul(R1_inv_batch, pd_vr3joints_rot[0])
        trace_R_relative_batch = torch.diagonal(R_relative_batch, dim1=-2, dim2=-1).sum(-1)
        theta = torch.acos((trace_R_relative_batch - 1) / 2)
        error.rot_error = to_numpy(torch.rad2deg(theta).flatten()).tolist()
        return error

    '''
    For FID and Diversity Calculation
    '''

    def record_feature(self, output, batch, feat_type):
        if self.inference_func == 'reactive':
            pd_feat: torch.Tensor = getattr(self, f'cal_reactive_{feat_type}_feature')(output.agent1, output.agent1)
            gt_feat: torch.Tensor = getattr(self, f'cal_reactive_{feat_type}_feature')(batch.agent1, batch.agent1)
        elif self.inference_func == 'twoagent':
            pd_feat: torch.Tensor = getattr(self, f'cal_twoagent_feature')(output.agent1, output.agent2, feat_type)
            gt_feat: torch.Tensor = getattr(self, f'cal_twoagent_feature')(batch.agent1, batch.agent2, feat_type)
        else:
            raise NotImplementedError(f"Unknown inference_func: {self.inference_func}")
        getattr(self, f'{feat_type}_features').pd.append(to_numpy(pd_feat))
        getattr(self, f'{feat_type}_features').gt.append(to_numpy(gt_feat))

    
    def cal_reactive_perframe_feature(self, rel_motion, motion):
        B, T, J = motion.pose_rot.shape[:3]
        pose_rot = matrix.get_mat_BtoA(rel_motion.root_rot[:, :, None], motion.pose_rot)
        return matrix_to_rotation_6d(pose_rot).reshape(B, T, -1) # (B, T, J, 6)
    

    def cal_reactive_pertrans_feature(self, rel_motion, motion):
        B, T, J = motion.pose_pos.shape[:3]
        pose_vel = matrix.get_relative_direction_to(motion.pose_vel, rel_motion.root_mat)
        return pose_vel.reshape(B, T, -1)

    
    def cal_reactive_perclip_feature(self, rel_motion, motion):
        B, T, J = motion.pose_rot.shape[:3]
        features = []
        for cid in range(15, T-15):
            root_rot = rel_motion.root_rot[:, cid]
            fids = [cid - 15, cid - 10, cid - 5, cid + 5, cid + 10, cid + 15]
            pose_rot = matrix.get_mat_BtoA(root_rot[:, None, None], motion.pose_rot[:, fids])
            features.append(pose_rot.reshape(1, 1, -1))
        return torch.cat(features, dim=1)
    
    
    def cal_twoagent_feature(self, motion, oppo_motion, feat_type):
        motion_feature = getattr(self, f'cal_reactive_{feat_type}_feature')(motion, motion)
        oppo_feature = getattr(self, f'cal_reactive_{feat_type}_feature')(motion, oppo_motion)
        return torch.cat((motion_feature, oppo_feature), dim=-1)