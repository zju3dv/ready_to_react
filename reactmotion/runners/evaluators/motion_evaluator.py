from easyvolcap.engine import cfg, args
from easyvolcap.utils.console_utils import *


class MotionEvaluator:
    def __init__(self, 
                 metric_names: List = [],
                 vis_types: List = [],
                 vis_stats_keys: List = [],
                 
                 bestn: int = 50,
                 lastn: int = 50,
                 vis_output: int = 0,
                 save_output: bool = False,
                 **kwargs,
                 ) -> None:
        # metrics
        self.metrics = dotdict()
        self.metric_names = metric_names
        
        # visualize
        self.vis_stats = dotdict(gt=[], pd=[])
        self.vis_types = vis_types
        self.vis_stats_keys = vis_stats_keys
        
        self.dpose = 12
        self.bestn = bestn / 100
        self.lastn = lastn / 100
        self.vis_output = (vis_output != 0)
        self.save_output = save_output

    def evaluate(self, output: dotdict, batch: dotdict):
        '''
            Rarely changed.
        '''
        metrics = dotdict()
        self.post_process(output, batch)
        for metric_name in self.metric_names:
            metric = self.__getattribute__(f'compute_{metric_name}')(output, batch)
            for k, v in metric.items():
                metrics[k] = v
                if isinstance(v, (int, float)):
                    if k in self.metrics: self.metrics[k].append(v)
                    else: self.metrics[k] = [v]
                elif isinstance(v, list):
                    if k in self.metrics: self.metrics[k].extend(v)
                    else: self.metrics[k] = v

        return metrics
    
    def post_process(self, output, batch):
        '''
            Always changed.
        '''
        pass
    
    def update_vis_stats(self, output: dotdict, batch: dotdict):
        '''
            Rarely changed.
        '''
        gt_stats = dotdict()
        pd_stats = dotdict()
        for key in self.vis_stats_keys:
            gt_stats[key] = batch[key]
            pd_stats[key] = output[key]
        gt_stats.meta = batch.meta
        self.vis_stats.gt.append(gt_stats)
        self.vis_stats.pd.append(pd_stats)

    def visualize(self, name: str, **kwargs):
        '''
            Rarely changed.
        '''
        self.epoch_name = name
        if args.type == "train":
            ### danger:
            log("remove prev vis3d")
            os.system(f"rm -rf data/vis3d/{cfg.exp_name}-*")
        if self.vis_output:
            for vis_type in self.vis_types:
                self.__getattribute__(f'visualize_{vis_type}')(**kwargs)

    def summarize(self):
        '''
            Rarely changed
        '''
        summary = dotdict()
        if len(self.metrics):
            for key in self.metrics.keys():
                values = self.metrics[key]
                summary[f'{key}_mean'] = np.mean(values)
                # summary[f'{key}_std'] = np.std(values)
        self.metrics.clear()  # clear mean after extracting summary
        self.vis_stats.gt.clear() # clear vis_stats.gt
        self.vis_stats.pd.clear() # clear vis_stats.pd
        log(summary)
        return summary
