
import argparse
import dataclasses
import logging
import sys
import os

# Add src and scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__)))

import openpi.training.config as _config
from train import main
import flax.nnx as nnx
import openpi.shared.nnx_utils as nnx_utils

def get_args():
    parser = argparse.ArgumentParser(description="Train script bypassing tyro")
    parser.add_argument("config_name", type=str, help="Name of the config to use")
    parser.add_argument("--exp-name", type=str, help="Experiment name")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-train-steps", type=int, help="Number of training steps")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--fsdp-devices", type=int, help="Number of FSDP devices")
    parser.add_argument("--action-loss-weight", type=float, help="Weight for action loss")
    parser.add_argument("--track-loss-weight", type=float, help="Weight for track loss")
    parser.add_argument("--wandb-enabled", type=str, default="true", help="Enable wandb")
    parser.add_argument("--validation-interval", type=int, default=1000, help="Validation interval in steps")
    parser.add_argument("--visualize-tracks", type=str, default="true", help="Enable track visualization validation")
    parser.add_argument("--train-mode", type=str, default="full", choices=["full", "head"], help="Training mode: full or head (freezes backbone)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    print(f"Loading config: {args.config_name}")
    try:
        config = _config.get_config(args.config_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    # Apply overrides
    updates = {}
    if args.exp_name:
        updates["exp_name"] = args.exp_name
    if args.batch_size:
        updates["batch_size"] = args.batch_size
    if args.num_train_steps:
        updates["num_train_steps"] = args.num_train_steps
    if args.overwrite:
        updates["overwrite"] = True
    if args.resume:
        updates["resume"] = True
    if args.seed:
        updates["seed"] = args.seed
    if args.fsdp_devices:
        updates["fsdp_devices"] = args.fsdp_devices
        
    if args.wandb_enabled.lower() == "false":
        updates["wandb_enabled"] = False

    if args.action_loss_weight is not None or args.track_loss_weight is not None:
        if not hasattr(config.model, "action_loss_weight") or not hasattr(config.model, "track_loss_weight"):
            raise ValueError("Loss weight overrides are only supported for hybrid models.")
        model_updates = {}
        if args.action_loss_weight is not None:
            model_updates["action_loss_weight"] = args.action_loss_weight
        if args.track_loss_weight is not None:
            model_updates["track_loss_weight"] = args.track_loss_weight
        updates["model"] = dataclasses.replace(config.model, **model_updates)

    os.environ["OPENPI_VALIDATION_INTERVAL"] = str(args.validation_interval)
    os.environ["OPENPI_VISUALIZE_TRACKS"] = str(args.visualize_tracks).lower()
        
    # Handle train mode
    if args.train_mode == "head":
        print("Training mode: HEAD (freezing backbone, training only adapter heads)")
        # Freeze everything EXCEPT the head layers
        # Standard Pi0 Track Head components:
        # - vertex_proj (for vertex-based tracks)
        # - track_in_proj (noisy tracks -> embedding)
        # - track_time_mlp_in, track_time_mlp_out (timestep embedding mixing)
        # - track_out_proj (embedding -> velocity)
        #
        # Pi0.5 Hybrid might also have:
        # - track_in_proj, track_time_mlp_in, track_time_mlp_out, track_out_proj (same names in Pi0FASTHybrid)
        
        # Regex to match these components at the top level
        head_regex = "^(vertex_proj|track_in_proj|track_time_mlp_in|track_time_mlp_out|track_out_proj)/.*"
        
        # We use flax.nnx.Not to invert the selection -> matches everything EXCEPT head
        # This sets the _freeze_filter which overrides the model's default freeze filter
        updates["_freeze_filter"] = nnx.Not(nnx_utils.PathRegex(head_regex))
    else:
        print("Training mode: FULL (fine-tuning all layers)")
    
    if updates:
        print(f"Overriding config with: {updates}")
        config = dataclasses.replace(config, **updates)
        
    # Run main
    print("Starting training...")
    main(config)
