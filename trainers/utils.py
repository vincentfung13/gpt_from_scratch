import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    ouput_path: str,
):
    state_dicts = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state_dicts, ouput_path)


def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state_dicts = torch.load(src)
    model.load_state_dict(state_dicts["model"])
    optimizer.load_state_dict(state_dicts["optimizer"])
    iteration = state_dicts["iteration"]
    return iteration
