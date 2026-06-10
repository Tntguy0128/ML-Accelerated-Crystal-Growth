"""
============================================================
  Rollout Fine-Tuning for FNO
  Phase Field Crystal Surrogate — Week 2

  Loads the best checkpoint from Stage 1 training and
  fine-tunes it by unrolling multi-step predictions and
  penalising accumulated error directly.

  Key idea: instead of training on single (frame_t, frame_t+1)
  pairs, we chain the model for `rollout_steps` steps starting
  from a random point in a trajectory, and compute loss against
  the TRUE frames at every step in the chain. This directly
  penalises the error accumulation that one-step training ignores.

  Why we focus on steps 5-20:
  The validation figures show the model loses track specifically
  during the active growth transition (roughly steps 5-15). We
  bias the starting frame sampling toward this region so the
  fine-tuning budget is spent where it matters most.

  NSF IRES Physical AI Design Program
  Ayush Shah & Tobias Li — Georgia Institute of Technology
============================================================

USAGE
-----
    python rollout_finetune.py --config config.yaml \
        --checkpoint runs/fno_baseline/best.pt

Output checkpoint: runs/fno_rollout/best_rollout.pt
Re-evaluate with:  python evaluate_fno.py --config config.yaml \
                       --checkpoint runs/fno_rollout/best_rollout.pt
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import build_datasets, inspect_trajectory
from fno_model import build_model


# ----------------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(requested):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    cfg["model"]["in_channels"] = ckpt["in_channels"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


# ----------------------------------------------------------------------------
#  Rollout fine-tuning config  (separate from train config so you can tune
#  these without touching the original config.yaml)
# ----------------------------------------------------------------------------
ROLLOUT_CFG = {
    # How many steps to unroll per training iteration.
    # 15 targets the 5-20 step region where the model loses track.
    # Reduce to 8 if you get OOM errors on the GPU.
    "rollout_steps"  : 15,

    # Which frame to start rollout from within each trajectory.
    # "transition" = bias toward frames 3-18 (active growth region)
    # "random"     = uniform random start anywhere in trajectory
    # "start"      = always start from frame 0
    "start_mode"     : "transition",

    # Loss weighting: later steps in the rollout get higher weight
    # because they accumulate more error. Linear ramp from 1.0 to
    # final_step_weight over the rollout window.
    # Set to 1.0 for uniform weighting.
    "final_step_weight": 3.0,

    # Also add a one-step MSE term to stop the model forgetting
    # what it learned in Stage 1. Weight relative to rollout loss.
    "onestep_weight" : 0.3,

    # Mass conservation penalty: penalise drift of the predicted
    # spatial mean away from the input mean.
    # This directly targets the 13% mass error we saw in evaluation.
    "mass_penalty_weight": 0.5,

    # Training schedule
    "epochs"         : 30,
    "lr"             : 2e-4,        # lower than Stage 1 — fine-tuning
    "weight_decay"   : 1e-4,
    "grad_clip"      : 0.5,         # tighter clip for stability
    "patience"       : 10,          # early stopping patience

    # Output
    "out_dir"        : "runs/fno_rollout",
    "ckpt_name"      : "best_rollout.pt",
}


# ----------------------------------------------------------------------------
#  Build normalised trajectory tensors from the train split
# ----------------------------------------------------------------------------
def load_trajectories(train_files, norm_mean, norm_std, device):
    """
    Load every training trajectory as a normalised (T, H, W) tensor.
    Returns a list of tensors on `device`.
    """
    trajs = []
    for path in train_files:
        rec    = inspect_trajectory(path)
        # Full trajectory: inputs[0..T-2] + last target = T frames
        frames = np.concatenate(
            [rec["inputs"], rec["targets"][-1:]], axis=0
        ).astype(np.float32)
        frames = (frames - norm_mean) / (norm_std + 1e-8)
        trajs.append(torch.tensor(frames, device=device))   # (T, H, W)
    print(f"  Loaded {len(trajs)} training trajectories onto {device}")
    return trajs


# ----------------------------------------------------------------------------
#  Single rollout training step
# ----------------------------------------------------------------------------
def rollout_step(model, traj, rollout_steps, start_mode, step_weights,
                 mass_penalty_weight, loss_fn):
    """
    Perform one rollout training iteration on a single trajectory.

    Returns the scalar loss (still on graph for backprop).
    """
    T = traj.shape[0]

    # --- choose start frame ---
    max_start = max(T - rollout_steps - 1, 0)
    if max_start == 0:
        start = 0
    elif start_mode == "transition":
        # Bias toward frames 3-18 (active growth phase)
        low  = min(3,  max_start)
        high = min(18, max_start)
        if low >= high:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start = int(torch.randint(low, high + 1, (1,)).item())
    elif start_mode == "start":
        start = 0
    else:  # "random"
        start = int(torch.randint(0, max_start + 1, (1,)).item())

    # --- unroll ---
    # current: (1, 1, H, W)
    current      = traj[start].unsqueeze(0).unsqueeze(0)
    rollout_loss = torch.tensor(0.0, device=traj.device)

    for s in range(rollout_steps):
        target_idx = start + s + 1
        if target_idx >= T:
            break

        predicted = model(current)                          # (1, 1, H, W)
        true_next = traj[target_idx].unsqueeze(0).unsqueeze(0)

        # Weighted MSE — later steps count more
        step_loss = loss_fn(predicted, true_next) * step_weights[s]
        rollout_loss = rollout_loss + step_loss

        # Mass conservation penalty:
        # The spatial mean of the density field should stay constant.
        # Penalise any drift of the predicted mean from the input mean.
        if mass_penalty_weight > 0:
            pred_mean  = predicted.mean()
            input_mean = current.mean()
            mass_loss  = (pred_mean - input_mean).pow(2)
            rollout_loss = rollout_loss + mass_penalty_weight * mass_loss

        # Feed prediction back (this is what causes / reveals error accumulation)
        current = predicted

    return rollout_loss / max(rollout_steps, 1)


# ----------------------------------------------------------------------------
#  One-step MSE on validation set (teacher-forced)
# ----------------------------------------------------------------------------
@torch.no_grad()
def validate_onestep(model, val_trajs, loss_fn):
    model.eval()
    total, n = 0.0, 0
    for traj in val_trajs:
        T = traj.shape[0]
        for t in range(T - 1):
            x = traj[t].unsqueeze(0).unsqueeze(0)
            y = traj[t + 1].unsqueeze(0).unsqueeze(0)
            pred = model(x)
            total += loss_fn(pred, y).item()
            n += 1
    return total / max(n, 1)


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     default="config.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="Stage 1 checkpoint to fine-tune from. "
                         "Defaults to <logging.out_dir>/best.pt")
    args = ap.parse_args()

    cfg    = load_config(args.config)
    device = pick_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")

    # --- load Stage 1 checkpoint ---
    ckpt_path = args.checkpoint or os.path.join(
        cfg["logging"]["out_dir"], cfg["logging"].get("ckpt_name", "best.pt")
    )
    model, ckpt = load_checkpoint(ckpt_path, device)
    print(f"Loaded: {ckpt_path}  (stage1 val_loss={ckpt.get('val_loss'):.3e})")

    norm_mean = ckpt["norm_mean"]
    norm_std  = ckpt["norm_std"]

    # --- load trajectories ---
    print("\nLoading trajectories...")
    train_trajs = load_trajectories(ckpt["train_files"], norm_mean, norm_std, device)
    val_trajs   = load_trajectories(ckpt["val_files"],   norm_mean, norm_std, device)

    # --- build step weight ramp ---
    # Linear ramp from 1.0 at step 0 to final_step_weight at last step
    n_steps     = ROLLOUT_CFG["rollout_steps"]
    fw          = ROLLOUT_CFG["final_step_weight"]
    step_weights = torch.linspace(1.0, fw, n_steps, device=device)
    step_weights = step_weights / step_weights.mean()   # normalise so total ~= 1

    # --- optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ROLLOUT_CFG["lr"],
        weight_decay=ROLLOUT_CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ROLLOUT_CFG["epochs"], eta_min=1e-6
    )
    loss_fn = nn.MSELoss()

    os.makedirs(ROLLOUT_CFG["out_dir"], exist_ok=True)
    ckpt_out = os.path.join(ROLLOUT_CFG["out_dir"], ROLLOUT_CFG["ckpt_name"])

    print(f"\nRollout fine-tuning config:")
    for k, v in ROLLOUT_CFG.items():
        print(f"  {k}: {v}")
    print(f"\nStarting rollout fine-tuning for {ROLLOUT_CFG['epochs']} epochs...\n")
    print("-" * 70)

    best_val    = float("inf")
    no_improve  = 0
    onestep_w   = ROLLOUT_CFG["onestep_weight"]
    mass_w      = ROLLOUT_CFG["mass_penalty_weight"]

    for epoch in range(ROLLOUT_CFG["epochs"]):
        t0 = time.time()
        model.train()

        # Shuffle trajectory order each epoch
        perm         = torch.randperm(len(train_trajs))
        epoch_loss   = 0.0
        epoch_os     = 0.0   # one-step component

        for idx in perm:
            traj = train_trajs[idx]

            # --- rollout loss ---
            r_loss = rollout_step(
                model, traj,
                rollout_steps        = n_steps,
                start_mode           = ROLLOUT_CFG["start_mode"],
                step_weights         = step_weights,
                mass_penalty_weight  = mass_w,
                loss_fn              = loss_fn,
            )

            # --- one-step loss on a random pair from this trajectory ---
            # (keeps the model anchored to what it learned in Stage 1)
            if onestep_w > 0 and traj.shape[0] > 1:
                t_idx  = int(torch.randint(0, traj.shape[0] - 1, (1,)).item())
                x_os   = traj[t_idx].unsqueeze(0).unsqueeze(0)
                y_os   = traj[t_idx + 1].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    pass   # no grad needed for target
                os_loss = loss_fn(model(x_os), y_os)
            else:
                os_loss = torch.tensor(0.0, device=device)

            total_loss = r_loss + onestep_w * os_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), ROLLOUT_CFG["grad_clip"]
            )
            optimizer.step()

            epoch_loss += r_loss.item()
            epoch_os   += os_loss.item()

        epoch_loss /= len(train_trajs)
        epoch_os   /= len(train_trajs)
        scheduler.step()

        # --- validation: one-step MSE (fast, teacher-forced) ---
        val_loss = validate_onestep(model, val_trajs, loss_fn)

        dt      = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        flag    = ""

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : val_loss,
                "config"      : cfg,
                "norm_mean"   : norm_mean,
                "norm_std"    : norm_std,
                "include_conditioning": ckpt["include_conditioning"],
                "cond_stats"  : ckpt["cond_stats"],
                "in_channels" : ckpt["in_channels"],
                "train_files" : ckpt["train_files"],
                "val_files"   : ckpt["val_files"],
                "test_files"  : ckpt["test_files"],
                "stage"       : "rollout_finetune",
                "rollout_cfg" : ROLLOUT_CFG,
            }, ckpt_out)
            flag = "  <- best (saved)"
        else:
            no_improve += 1

        print(f"  epoch {epoch:3d}/{ROLLOUT_CFG['epochs']}  |  "
              f"rollout={epoch_loss:.3e}  onestep={epoch_os:.3e}  "
              f"val={val_loss:.3e}  |  "
              f"lr={lr_now:.1e}  {dt:.1f}s{flag}")

        # --- early stopping ---
        if no_improve >= ROLLOUT_CFG["patience"]:
            print(f"\nEarly stopping: no improvement for "
                  f"{ROLLOUT_CFG['patience']} epochs.")
            break

    print(f"\nRollout fine-tuning complete.")
    print(f"Best val MSE: {best_val:.3e}")
    print(f"Checkpoint:   {ckpt_out}")
    print(f"\nNow evaluate:")
    print(f"  python evaluate_fno.py --config config.yaml "
          f"--checkpoint {ckpt_out}")


if __name__ == "__main__":
    main()
