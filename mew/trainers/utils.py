import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    ouput_path: str,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
):
    state_dicts = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if lr_scheduler is not None:
        state_dicts["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(state_dicts, ouput_path)


def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> int:
    state_dicts = torch.load(src)
    model.load_state_dict(state_dicts["model"])
    optimizer.load_state_dict(state_dicts["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
    iteration = state_dicts["iteration"]
    return iteration
