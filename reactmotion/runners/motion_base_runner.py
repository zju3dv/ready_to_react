# Default runner, no fancy business here
# Perform the training loop and log stuff out (tensor board etc.)
# Also responsible for saving the model and the optimizer states
# Sometimes performs validation and also writes things to tensorboard

# For type annotation
import time
import torch
import datetime
import random

from easyvolcap.engine import cfg, args  # need this for initialization?

from easyvolcap.engine import RUNNERS, OPTIMIZERS, SCHEDULERS, RECORDERS, EVALUATORS, MODERATORS  # controls the optimization loop of a particular epoch
from easyvolcap.utils.net_utils import save_model, load_model, load_network, setup_deterministic
from easyvolcap.utils.data_utils import add_iter, to_cuda
from easyvolcap.utils.prof_utils import profiler_step
from easyvolcap.utils.dist_utils import get_rank
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *


# The outer most training loop sets lr scheduler, constructs objects etc
# The inner loop call training, logs stuff


@RUNNERS.register_module()
class MotionBaseRunner:  # a plain and simple object controlling the training loop
    def __init__(self,
                 model,  # the network to train
                 dataloader,  # enumerate this
                 val_dataloader,  # enumerate this
                 optimizer_cfg: dotdict = dotdict(),
                 scheduler_cfg: dotdict = dotdict(),

                 moderator_cfg: dotdict = dotdict(),
                 recorder_cfg: dotdict = dotdict(),
                 evaluator_cfg: dotdict = dotdict(),
                 visualizer_cfg: dotdict = dotdict(),

                 epochs: int = 400,  # total: ep_iter * epoch number of iterations
                 ep_iter: int = 1000, # 
                 eval_ep: int = 10,  # report validation stats
                 save_ep: int = 20,  # separately save networks (might be heavy on storage)
                 save_latest_ep: int = 1,  # just in case, save regularly
                 log_interval: int = 1,  # 10ms, tune this if in realtime
                 record_interval: int = 1,  # ?ms, tune this if in realtime
                 strict: bool = True,  # strict loading of network and modules?

                 resume: bool = True,
                 trained_model_dir: str = f'data/trained_model/{cfg.exp_name}',  # MARK: global configuration
                 load_epoch: int = -1,  # load different epoch to start with

                 clip_grad_norm: float = -1,  # 1e-3,
                 clip_grad_value: float = -1,  # 40.0,
                 record_images_to_tb: bool = True,
                 print_test_progress: bool = True,
                 ):
        self.model = model  # possibly already a ddp model?
        self.dataloader = dataloader
        if dataloader: ep_iter = ep_iter
        else: ep_iter = len(val_dataloader.dataset) # set ep_iter
        
        self.optimizer = OPTIMIZERS.build(optimizer_cfg, params=model.parameters())  # requires parameters
        self.scheduler = SCHEDULERS.build(scheduler_cfg, optimizer=self.optimizer, decay_iter=epochs * ep_iter)  # requires parameters
        
        # Used in evaluation
        if not get_rank():  # only build these in main process
            self.val_dataloader = val_dataloader  # different dataloader for validation
            self.recorder = RECORDERS.build(recorder_cfg, resume=resume)
            self.evaluator = EVALUATORS.build(evaluator_cfg)

        # Used for periodically changing runner configs, should apply in all processes
        self.moderator= MODERATORS.build(moderator_cfg, runner=self, total_iter=epochs * ep_iter)  # after dataset init

        self.epochs = epochs
        self.ep_iter = ep_iter
        self.eval_ep = eval_ep
        self.save_ep = save_ep
        self.save_latest_ep = save_latest_ep
        self.log_interval = log_interval
        self.record_interval = record_interval

        self.resume = resume
        self.strict = strict
        self.load_epoch = load_epoch
        self.trained_model_dir = trained_model_dir

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # Use auto mixed precision
        self.gscaler = torch.cuda.amp.GradScaler(enabled=False)

        self.record_images_to_tb = record_images_to_tb
        self.print_test_progress = print_test_progress

        # FIXME: GLOBAL VARIABLE
        cfg.runner = self

    @property
    def total_iter(self):
        return self.epochs * self.ep_iter

    def load_network(self):
        epoch = load_network(model=self.model,  # only loading the network, without recorder?
                             model_dir=self.trained_model_dir,
                             resume=self.resume,
                             epoch=self.load_epoch,
                             strict=self.strict,
                             )  # loads the next epoch to use
        return epoch

    def load_model(self):
        epoch = load_model(model=self.model,
                           optimizer=self.optimizer,
                           scheduler=self.scheduler,
                           moderator=self.moderator,
                           model_dir=self.trained_model_dir,
                           resume=self.resume,
                           epoch=self.load_epoch,
                           strict=self.strict,
                           )  # loads the next epoch to use
        return epoch

    def save_network(self, epoch, latest: bool = True):
        save_model(model=self.model,
                   model_dir=self.trained_model_dir,
                   epoch=epoch,
                   latest=latest,
                   )

    def save_model(self, epoch: int, latest: bool = True):
        save_model(model=self.model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   moderator=self.moderator,
                   model_dir=self.trained_model_dir,
                   epoch=epoch,
                   latest=latest,
                   )

    # Single epoch testing api
    def test(self):  # from begin epoch
        setup_deterministic(fix_random=True, allow_tf32=False, deterministic=True, benchmark=True, seed=66)
        epoch = self.load_network()
        self.test_epoch(epoch)
        setup_deterministic(fix_random=False, allow_tf32=False, deterministic=False, benchmark=False)

    # Epoch based runner
    def train(self):  # from begin epoch
        epoch = self.load_model()

        # The actual training for this epoch
        train_generator = self.train_generator(epoch, self.ep_iter)  # yield every ep iter

        # train the network
        for epoch in range(epoch, self.epochs):

            # MARK: Unintuitive training, should combine epoches and iterations
            # Possible to make this a decorator?
            next(train_generator)  # avoid reconstruction of the dataloader

            # Saving stuff to disk
            if (epoch + 1) % self.save_ep == 0 and not get_rank():
                self.save_model(epoch, latest=False)

            if (epoch + 1) % self.save_latest_ep == 0 and not get_rank():
                self.save_model(epoch, latest=True)

            # Perform validation run if required
            if (epoch + 1) % self.eval_ep == 0 and not get_rank():
                try:
                    # setup deterministic
                    setup_deterministic(fix_random=True, allow_tf32=False, deterministic=True, benchmark=True, seed=66)
                    self.test_epoch(epoch + 1)  # will this provoke a live display?
                    setup_deterministic(fix_random=False, allow_tf32=False, deterministic=False, benchmark=False)
                except Exception as e:
                    log(red('Error in validation pass, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives
                    pass  # for readability

    def train_epoch(self, epoch: int):
        train_generator = self.train_generator(epoch, self.ep_iter)
        for _ in train_generator: pass  # the actual calling

    # Iteration based runner
    def train_generator(self, begin_epoch: int, yield_every: int = 1):
        # Train for one epoch (iterator style)
        # Actual start of the execution
        epoch = begin_epoch  # set starting epoch
        self.model.train()  # set the network (model) to training mode (recursive to all modules)
        start_time = time.perf_counter()
        for index, batch in enumerate(self.dataloader):  # control number of iterations explicitly
            iter = begin_epoch * self.ep_iter + index  # epoch actually is only for logging here
            batch = add_iter(batch, iter, self.total_iter)  # is this bad naming
            batch = to_cuda(batch)  # cpu -> cuda
            data_time = time.perf_counter() - start_time

            # The model's forward function
            # output: random dict storing various forms of output
            # loss: final optimizable loss variable to backward
            # scalar_stats: things to report to logger and recorder (all scalars)
            # image_stats: things to report to recorder (all image tensors (float32))
            with torch.cuda.amp.autocast(enabled=False):
                output: dotdict = self.model(batch)
            loss: torch.Tensor = output.loss
            scalar_stats: dotdict = output.scalar_stats

            self.optimizer.zero_grad(set_to_none=True)
            loss = loss.mean()
            self.gscaler.scale(loss).backward()  # AMP related
            if self.clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            if self.clip_grad_value > 0: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
            self.optimizer.step()
            self.scheduler.step()
            self.moderator.step()

            # Records data and batch forwarding time
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            start_time = end_time  # note that all logging and profiling time are accumuated into data_time
            if (iter + 1) % self.log_interval == 0 and not get_rank():  # MARK: avoid ddp print
                # For recording
                scalar_stats = dotdict({k: v.mean().item() for k, v in scalar_stats.items()})  # MARK: sync (for accurate batch time)

                lr = self.optimizer.param_groups[0]['lr']  # MARK: skechy lr query, only lr of the first param will be saved
                max_mem = torch.cuda.max_memory_allocated() / 2**20
                scalar_stats.data = data_time
                scalar_stats.batch = batch_time
                scalar_stats.lr = lr
                scalar_stats.max_mem = max_mem

                self.recorder.iter = iter
                self.recorder.epoch = epoch
                self.recorder.update_scalar_stats(scalar_stats)

                # For logging
                eta_seconds = self.recorder.scalar_stats.batch.global_avg * (self.total_iter - self.recorder.iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_stats = dotdict()
                log_stats.eta = eta_string
                log_stats.update(self.recorder.log_stats)

                # Render table to screen
                if not getattr(self.recorder, 'log_wandb', False):
                    display_table(log_stats)  # render dict as a table (live, console, table)

                if (iter + 1) % self.record_interval == 0:
                    self.recorder.record(self.dataloader.dataset.split)  # actual writing to the tensorboard logger

            if yield_every > 0 and (iter + 1) % yield_every == 0:
                yield output
                self.model.train()

            if (iter + 1) % self.ep_iter == 0:
                # Actual start of the execution
                epoch = epoch + 1

            profiler_step()  # record a step for the profiler, extracted logic

    def test_epoch(self, epoch: int):
        test_generator = self.test_generator(epoch, -1)  # nevel yield (special logic)
        for _ in test_generator: pass  # the actual calling

    def test_generator(self, epoch: int, yield_every: int = 1):
        # validation for one epoch
        self.model.eval()  # set the network (model) to training mode (recursive to all modules)
        for index, batch in enumerate(tqdm(self.val_dataloader, disable=not self.print_test_progress)):
            iter = epoch * self.ep_iter - 1  # some indexing trick
            batch = add_iter(batch, iter, self.total_iter)  # is this bad naming
            batch = to_cuda(batch)  # cpu -> cuda
            with torch.inference_mode() and torch.cuda.amp.autocast(enabled=False) and torch.no_grad():  # NOTE: tensor still records grad when in inference mode
                output: dotdict = self.model.inference(batch)
                scalar_stats = self.evaluator.evaluate(output, batch)
                self.evaluator.update_vis_stats(output, batch)

            self.recorder.iter = iter
            self.recorder.epoch = epoch
            self.recorder.update_scalar_stats(scalar_stats)
            self.recorder.record(self.val_dataloader.dataset.split)

            if yield_every > 0 and (iter + 1) % yield_every == 0:
                # break  # dataloader could be infinite
                yield output
                self.model.eval()

            profiler_step()  # record a step for the profiler, extracted logic

        self.evaluator.visualize(f"{epoch}") # put it before summarize
        scalar_stats = self.evaluator.summarize()
        self.recorder.update_scalar_stats(scalar_stats)
        self.recorder.record(self.val_dataloader.dataset.split)
