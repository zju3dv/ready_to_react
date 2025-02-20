# fmt: off
from easyvolcap.utils.console_utils import *
import torch  # make early keyboard interrupt possible
import torch.distributed as dist
import tqdm

from typing import Callable
from torch.nn.parallel import DistributedDataParallel as DDP

from easyvolcap.engine import args, cfg  # commandline entrypoint
from easyvolcap.engine import RUNNERS, MODELS, DATALOADERS
from easyvolcap.engine import callable_from_cfg

from reactmotion.utils.engine_utils import discover_modules
discover_modules() # will launch through this interface


from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.dist_utils import synchronize, get_rank, get_distributed
from easyvolcap.utils.net_utils import setup_deterministic, number_of_params
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_stop
# fmt: on


def launch_runner(runner_function: Callable,  # viewer.run or runner.train or runner.test
                  runner_object = None,
                  exp_name='debug',
                  detect_anomaly: bool = False,
                  profiling_cfg: dotdict = dotdict(),  # for debug use only

                  *args,
                  **kwargs
                  ):

    # Perform the actual training
    setup_profiler(**profiling_cfg)  # MARK: overwritting default configs
    prev_anomaly = torch.is_anomaly_enabled()
    torch.set_anomaly_enabled(detect_anomaly)
    profiler_start()  # already setup

    # Give the user some time to save states
    log('Launching experiment:', magenta(exp_name))
    cfg.runner = runner_object  # holds a global reference for hacky usage # FIXME: This is not the best practice
    runner_function()

    profiler_stop()  # already setup
    torch.set_anomaly_enabled(prev_anomaly)


@callable_from_cfg
def data_test(
    dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    val_dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
):
    dataloader = DATALOADERS.build(dataloader_cfg)
    for iter, data in enumerate(dataloader):
        if iter > 10: break
    val_dataloader = DATALOADERS.build(val_dataloader_cfg)
    for iter, data in enumerate(val_dataloader):
        if iter > 10: break


def preflight(
    fix_random: bool = False,
    allow_tf32: bool = True,
    deterministic: bool = False,  # for debug use only
    benchmark: Union[bool, str] = True,  # for static sized input
    ignore_breakpoint: bool = False,
    hide_progress: bool = False,
    less_verbose: bool = False,
    hide_output: bool = False,
    **kwargs,
):
    # Some early on GUI specific configurations
    if ignore_breakpoint: disable_breakpoint()
    if hide_progress: disable_progress()
    if hide_output: disable_console()
    if less_verbose: disable_verbose_log()
    if benchmark == 'train': benchmark = args.type == 'train'  # for static sized input

    # Maybe make this run deterministic?
    setup_deterministic(fix_random, allow_tf32, deterministic, benchmark)  # whether to be deterministic throughout the whole training process?

    # Log the experiment name for later usage
    log(f"Starting experiment: {magenta(cfg.exp_name)}, command: {magenta(args.type)}")  # MARK: GLOBAL


@callable_from_cfg
def test(
    model_cfg: dotdict = dotdict(),
    val_dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    test_dataloader_cfg = None,
    runner_cfg: dotdict = dotdict(),

    # Reproducibility configuration
    base_device: str = 'cuda',

    record_images_to_tb: bool = False,  # MARK: insider config # this is slow
    print_test_progress: bool = True,  # MARK: insider config # this is slow
    dry_run: bool = False,

    **kwargs,
):
    # Maybe make this run deterministic?
    preflight(**kwargs)  # whether to be deterministic throughout the whole training process?

    # Construct other parts of the training process
    if test_dataloader_cfg is not None:
        val_dataloader = DATALOADERS.build(test_dataloader_cfg)
    else:
        val_dataloader = DATALOADERS.build(val_dataloader_cfg)  # reuse the validataion

    # Model building and distributed training related stuff
    model = MODELS.build(model_cfg)
    model = model.to(base_device, non_blocking=True)

    runner = RUNNERS.build(runner_cfg,
                        model=model,
                        dataloader=None,  # no training data loader
                        record_images_to_tb=record_images_to_tb,  # another default
                        print_test_progress=print_test_progress,  # another default
                        val_dataloader=val_dataloader)

    if dry_run: return runner  # just construct everything, then return

    # Just run, no gossip
    launch_runner(**kwargs, runner_function=runner.test, runner_object=runner)


@callable_from_cfg
def train(
    model_cfg: dotdict = dotdict(),
    dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    val_dataloader_cfg: dotdict = dotdict(type="DefaultDataloader"),
    runner_cfg: dotdict = dotdict(),

    # Distributed training
    distributed: bool = False,
    find_unused_parameters: bool = False,

    # Printing configuration
    dry_run: bool = False,  # only print network and exit
    print_model: bool = True,  # since the network is pretty complex, give the option to print

    # Reproducibility configuration
    base_device: str = 'cuda',

    **kwargs,
):
    # Maybe make this run deterministic?
    preflight(**kwargs)

    # Construct other parts of the training process
    dataloader = DATALOADERS.build(dataloader_cfg)
    if not get_rank(): val_dataloader = DATALOADERS.build(val_dataloader_cfg)

    # Handle distributed training
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Handle model placement
    rank = get_rank()
    device = torch.device(f'{base_device}:{rank}')
    torch.cuda.set_device(device)

    # Model building and distributed training related stuff
    model = MODELS.build(model_cfg)  # some components are directly moved to cuda when building
    model.to(device, non_blocking=True)  # move this model to this specific device

    # Handle distributed model
    if get_distributed():
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=find_unused_parameters,
        )

    # Construct the runner (optimization loop controller)
    runner = RUNNERS.build(runner_cfg,
                        model=model,
                        dataloader=dataloader,
                        val_dataloader=val_dataloader if not get_rank() else None)

    if print_model and not get_rank():  # only print once
        # For some methods, both the network and the sampler or even the renderer contains optimizable parameters
        # But the sampler and render both has a reference to the network, which gets printed (not saved, tested)
        # log(model)
        pprint(model)  # with indent guides
        try:
            nop = number_of_params(model)
            log(f'Number of parameters: {nop} ({nop / 1e6:.2f} M)')
        except ValueError as e:
            # Ignore: Attempted to use an uninitialized parameter in <method 'numel' of 'torch._C._TensorBase' objects>
            pass
        return

    if dry_run: return runner  # just construct everything, then return

    # The actual calling, with grace full exit
    launch_runner(**kwargs, runner_function=runner.train, runner_object=runner)

# Should we respect the naming convensions?

def main_entrypoint():
    globals()[args.type](cfg)  # invoke this (call callable_from_cfg -> call_from_cfg)


# Module name == '__main__', this is the outermost commandline entry point
if __name__ == '__main__':
    main_entrypoint()
