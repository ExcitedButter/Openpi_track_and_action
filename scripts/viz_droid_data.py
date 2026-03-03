#!/usr/bin/env python3
"""
Visualization script to verify DROID data loading and model predictions.

This script:
1. Loads a batch of data from DROID TFRecord dataloader (224x224 preprocessed images)
2. Runs the model to get predicted tracks
3. Visualizes input images with overlaid tracks
4. Saves visualization to disk for inspection

Usage:
    python viz_droid_data.py [--droid-data-path /path/to/tfrecords] [--output-dir viz_output]
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import jax
import jax.numpy as jnp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import openpi.training.droid_pi0_pretrain_loader as droid_loader
import openpi.training.config as _config
import openpi.models.model as _model
import openpi.shared.normalize as _normalize

def unnormalize_tracks(tracks, stats_key, norm_stats):
    if norm_stats is None or stats_key not in norm_stats:
        return tracks
    
    s = norm_stats[stats_key]
    # 1. Reverse standardization
    # Reshape s.std/mean to match tracks if needed, but usually broadcasting works for last dim
    # Ensure tracks is numpy array for multiplication
    tracks = np.array(tracks)
    s_std = np.array(s.std)
    s_mean = np.array(s.mean)
    
    x_scaled = tracks * (s_std + 1e-6) + s_mean
    # 2. Reverse scaling [-1, 1] -> [0, 224]
    x_pixel = (x_scaled + 1.0) / 2.0 * 224.0
    return x_pixel

def denormalize_image(img):
    """Convert image from [-1, 1] to [0, 1] for visualization."""
    return (img + 1.0) / 2.0


def visualize_tracks_on_image(ax, img, tracks_2d, title="", color='lime', marker_size=30):
    """
    Visualize 2D tracks overlaid on an image.
    
    Args:
        ax: Matplotlib axis
        img: Image array (H, W, 3) in range [0, 1]
        tracks_2d: Track coordinates (N, 2) or (T, N, 2) where each point is (x, y) in pixel coords
        title: Plot title
        color: Color for track points
        marker_size: Size of track markers
    """
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])  # Set proper extent
    ax.set_title(title)
    
    # Extend axis limits to show out-of-bounds tracks
    ax.set_xlim(-20, img.shape[1] + 20)
    ax.set_ylim(img.shape[0] + 20, -20)
    
    # Draw image boundary
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), img.shape[1], img.shape[0], 
                     linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Handle both single frame and temporal tracks
    if tracks_2d.ndim == 2:
        # Single frame: (N, 2)
        x_coords = tracks_2d[:, 0]
        y_coords = tracks_2d[:, 1]
        ax.scatter(x_coords, y_coords, c=color, s=marker_size, marker='o', 
                  edgecolors='black', linewidths=1.5, alpha=0.9, zorder=10)
        
        # Add point numbers
        for i, (x, y) in enumerate(tracks_2d):
            ax.text(x, y, str(i), fontsize=8, color='white', 
                   ha='center', va='center', weight='bold', zorder=11)
    elif tracks_2d.ndim == 3:
        # Temporal: (T, N, 2) - show trajectory
        T = tracks_2d.shape[0]
        N = tracks_2d.shape[1]
        
        # Check if trajectory has motion
        total_motion = np.abs(tracks_2d[1:] - tracks_2d[:-1]).sum()
        has_motion = total_motion > 1.0  # Threshold for motion detection
        
        if not has_motion:
            # Static trajectory - just show points with warning
            x_coords = tracks_2d[0, :, 0]
            y_coords = tracks_2d[0, :, 1]
            ax.scatter(x_coords, y_coords, c='orange', s=marker_size*2, marker='o', 
                      edgecolors='red', linewidths=2, alpha=0.9, zorder=10, label='STATIC (no motion)')
            # Add point numbers
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax.text(x, y, str(i), fontsize=8, color='white', 
                       ha='center', va='center', weight='bold', zorder=11)
        else:
            # Plot initial points larger
            x_coords = tracks_2d[0, :, 0]
            y_coords = tracks_2d[0, :, 1]
            ax.scatter(x_coords, y_coords, c='red', s=marker_size*2.5, marker='o', 
                      edgecolors='black', linewidths=2, alpha=0.9, label='t=0', zorder=10)
            
            # Plot trajectory lines
            for n in range(N):
                xs = tracks_2d[:, n, 0]
                ys = tracks_2d[:, n, 1]
                ax.plot(xs, ys, c=color, linewidth=2, alpha=0.7, zorder=5)
            
            # Plot final points
            x_coords = tracks_2d[-1, :, 0]
            y_coords = tracks_2d[-1, :, 1]
            ax.scatter(x_coords, y_coords, c='blue', s=marker_size*1.5, marker='x', 
                      linewidths=2, alpha=0.9, label='t=15', zorder=10)
        
        ax.legend(loc='upper right', fontsize=8)


def parse_78dim_tracks(tracks_78dim, img_size=224):
    """
    Parse 78-dim track data into separate components.
    
    Args:
        tracks_78dim: (78,) or (T, 78) array
        img_size: Image size for coordinate scaling (224x224 after preprocessing)
    
    Returns:
        dict with 'agent_mesh', 'eye_mesh', 'eye_uniform' as (7, 2), (7, 2), (25, 2) or temporal versions
    """
    is_temporal = tracks_78dim.ndim == 2
    
    if is_temporal:
        T = tracks_78dim.shape[0]
        agent_mesh = tracks_78dim[:, 0:14].reshape(T, 7, 2)
        eye_mesh = tracks_78dim[:, 14:28].reshape(T, 7, 2)
        eye_uniform = tracks_78dim[:, 28:78].reshape(T, 25, 2)
    else:
        agent_mesh = tracks_78dim[0:14].reshape(7, 2)
        eye_mesh = tracks_78dim[14:28].reshape(7, 2)
        eye_uniform = tracks_78dim[28:78].reshape(25, 2)
    
    # Tracks are already in 224x224 pixel space after preprocessing (center crop + resize)
    # No additional scaling needed
    
    return {
        'agent_mesh': agent_mesh,
        'eye_mesh': eye_mesh,
        'eye_uniform': eye_uniform,
    }


def visualize_batch_sample(obs, tracks_gt, tracks_pred=None, sample_idx=0, output_path=None, instruction=None):
    """
    Visualize a single sample from a batch.
    
    Args:
        obs: Observation object with images and state
        tracks_gt: Ground truth tracks (B, 16, 78)
        tracks_pred: Predicted tracks (B, 16, 78) or None
        sample_idx: Which sample in batch to visualize
        output_path: Where to save the visualization
        instruction: Optional language instruction string
    """
    # Extract sample
    agent_img = np.array(obs.images['base_0_rgb'][sample_idx])
    eye_img = np.array(obs.images['left_wrist_0_rgb'][sample_idx])
    state = np.array(obs.state[sample_idx])
    tracks_gt_sample = np.array(tracks_gt[sample_idx])
    
    # Denormalize images
    agent_img = denormalize_image(agent_img)
    eye_img = denormalize_image(eye_img)
    
    # Parse current state (t=0)
    state_parsed = parse_78dim_tracks(state)
    
    # Parse ground truth tracks (temporal)
    gt_parsed = parse_78dim_tracks(tracks_gt_sample)
    
    # Create figure
    if tracks_pred is not None:
        tracks_pred_sample = np.array(tracks_pred[sample_idx])
        pred_parsed = parse_78dim_tracks(tracks_pred_sample)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    title = 'DROID Data Visualization'
    if instruction:
        # Wrap instruction if too long
        import textwrap
        wrapped_instr = "\n".join(textwrap.wrap(instruction, width=60))
        title += f"\nInstruction: {wrapped_instr}"

    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Row 1: Current state (t=0)
    ax = axes[0, 0] if tracks_pred is not None else axes[0, 0]
    visualize_tracks_on_image(ax, agent_img, state_parsed['agent_mesh'], 
                               title="Agentview - Current State (t=0)\nMesh: 7 joints", 
                               color='red', marker_size=50)
    
    ax = axes[0, 1] if tracks_pred is not None else axes[0, 1]
    # Overlay both mesh and uniform on wrist view
    ax.imshow(eye_img)
    ax.set_title("Wrist View - Current State (t=0)\nMesh: 7 joints (red), Uniform: 25 grid (blue)")
    ax.axis('off')
    ax.scatter(state_parsed['eye_mesh'][:, 0], state_parsed['eye_mesh'][:, 1], 
               c='red', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.8, label='Mesh')
    ax.scatter(state_parsed['eye_uniform'][:, 0], state_parsed['eye_uniform'][:, 1], 
               c='blue', s=30, marker='s', edgecolors='black', linewidths=1, alpha=0.7, label='Uniform')
    ax.legend()
    
    # Row 2: Ground truth trajectories
    ax = axes[1, 0] if tracks_pred is not None else axes[1, 0]
    visualize_tracks_on_image(ax, agent_img, gt_parsed['agent_mesh'], 
                               title="Agentview - GT Trajectory (t=0→15)", 
                               color='lime', marker_size=30)
    
    ax = axes[1, 1] if tracks_pred is not None else axes[1, 1]
    ax.imshow(eye_img, extent=[0, 224, 224, 0])
    ax.set_title("Wrist View - GT Trajectory (t=0→15)")
    ax.set_xlim(-20, 244)
    ax.set_ylim(244, -20)
    
    # Draw boundary
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 224, 224, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Check if mesh has motion
    mesh_motion = np.abs(gt_parsed['eye_mesh'][1:] - gt_parsed['eye_mesh'][:-1]).sum()
    uniform_motion = np.abs(gt_parsed['eye_uniform'][1:] - gt_parsed['eye_uniform'][:-1]).sum()
    
    # Always show mesh points with t=0 and t=15 markers
    if mesh_motion < 1.0:
        # Static mesh - show single point with both markers
        ax.scatter(gt_parsed['eye_mesh'][0, :, 0], gt_parsed['eye_mesh'][0, :, 1], 
                  c='red', s=100, marker='o', edgecolors='black', linewidths=2, 
                  alpha=0.9, label='Mesh (STATIC)', zorder=10)
        # Add point numbers
        for i in range(gt_parsed['eye_mesh'].shape[1]):
            ax.text(gt_parsed['eye_mesh'][0, i, 0], gt_parsed['eye_mesh'][0, i, 1], 
                   str(i), fontsize=9, color='white', ha='center', va='center', 
                   weight='bold', zorder=11)
    else:
        # Plot mesh trajectory lines
        for n in range(gt_parsed['eye_mesh'].shape[1]):
            xs = gt_parsed['eye_mesh'][:, n, 0]
            ys = gt_parsed['eye_mesh'][:, n, 1]
            ax.plot(xs, ys, c='red', linewidth=2, alpha=0.7, zorder=5)
        # t=0 points
        ax.scatter(gt_parsed['eye_mesh'][0, :, 0], gt_parsed['eye_mesh'][0, :, 1],
                  c='red', s=80, marker='o', edgecolors='black', linewidths=2, 
                  alpha=0.9, label='Mesh t=0', zorder=10)
        # t=15 points
        ax.scatter(gt_parsed['eye_mesh'][-1, :, 0], gt_parsed['eye_mesh'][-1, :, 1],
                  c='darkred', s=60, marker='x', linewidths=3, 
                  alpha=0.9, label='Mesh t=15', zorder=10)
    
    # Always show uniform points with t=0 and t=15 markers
    if uniform_motion < 1.0:
        # Static uniform - show single point with both markers
        ax.scatter(gt_parsed['eye_uniform'][0, :, 0], gt_parsed['eye_uniform'][0, :, 1], 
                  c='blue', s=50, marker='s', edgecolors='black', linewidths=1.5, 
                  alpha=0.8, label='Uniform (STATIC)', zorder=9)
    else:
        # Plot uniform trajectory lines
        for n in range(gt_parsed['eye_uniform'].shape[1]):
            xs = gt_parsed['eye_uniform'][:, n, 0]
            ys = gt_parsed['eye_uniform'][:, n, 1]
            ax.plot(xs, ys, c='cyan', linewidth=1.5, alpha=0.6, zorder=5)
        # t=0 points
        ax.scatter(gt_parsed['eye_uniform'][0, :, 0], gt_parsed['eye_uniform'][0, :, 1],
                  c='blue', s=50, marker='s', edgecolors='black', linewidths=1.5, 
                  alpha=0.9, label='Uniform t=0', zorder=9)
        # t=15 points
        ax.scatter(gt_parsed['eye_uniform'][-1, :, 0], gt_parsed['eye_uniform'][-1, :, 1],
                  c='darkblue', s=40, marker='+', linewidths=2.5, 
                  alpha=0.9, label='Uniform t=15', zorder=9)
    
    ax.legend(fontsize=8, loc='upper right')
    
    # Row 3: Predicted trajectories (if available)
    if tracks_pred is not None:
        ax = axes[0, 2]
        visualize_tracks_on_image(ax, agent_img, pred_parsed['agent_mesh'], 
                                   title="Agentview - Predicted (t=0→15)", 
                                   color='yellow', marker_size=30)
        
        ax = axes[1, 2]
        ax.imshow(eye_img)
        ax.set_title("Wrist View - Predicted (t=0→15)")
        ax.axis('off')
        for n in range(pred_parsed['eye_mesh'].shape[1]):
            xs = pred_parsed['eye_mesh'][:, n, 0]
            ys = pred_parsed['eye_mesh'][:, n, 1]
            ax.plot(xs, ys, c='yellow', linewidth=1.5, alpha=0.6)
        for n in range(pred_parsed['eye_uniform'].shape[1]):
            xs = pred_parsed['eye_uniform'][:, n, 0]
            ys = pred_parsed['eye_uniform'][:, n, 1]
            ax.plot(xs, ys, c='cyan', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize DROID data and model predictions')
    parser.add_argument('--droid-data-path', type=str, 
                        default='/mnt/kevin/data/droid_preprocessed_tfrecord',
                        help='Path to DROID TFRecord data directory')
    parser.add_argument('--episode-idx', type=int, default=0,
                        help='Episode index to visualize')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for data loading')
    parser.add_argument('--output-dir', type=str, default='viz_output',
                        help='Directory to save visualizations')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load norm stats
    norm_stats = None
    try:
        assets_path = Path("/mnt/kevin/code/wmrl/howard-branch/code/openpi_fork/assets/droid_pretrain_rlds")
        norm_stats = _normalize.load(assets_path)
        print("Loaded normalization stats")
    except Exception as e:
        print(f"Warning: Could not load norm stats: {e}")
    
    print(f"Loading DROID data from {args.droid_data_path}")
    
    # Create dataloader
    # DroidPi0PretrainLoader requires data_config. We can pass a dummy one for visualization.
    dummy_config = _config.DataConfig(
        droid_data_path=args.droid_data_path
    ) if hasattr(_config, 'DataConfig') else None

    loader = droid_loader.DroidPi0PretrainLoader(
        droid_data_path=args.droid_data_path,
        data_config=dummy_config,
        batch_size=args.batch_size,
        shuffle=False,
        seed=42,
    )
    
    # Get a batch
    data_iter = iter(loader)
    obs, tracks_gt = next(data_iter)
    
    print(f"Loaded batch:")
    print(f"  Images shapes: {[(k, v.shape) for k, v in obs.images.items()]}")
    print(f"  State shape: {obs.state.shape}")
    print(f"  Tracks GT shape: {tracks_gt.shape}")
    print(f"  Image dtype: {obs.images['base_0_rgb'].dtype}")
    print(f"  Image value range: [{obs.images['base_0_rgb'].min():.2f}, {obs.images['base_0_rgb'].max():.2f}]")
    print(f"  Sample pixel values (first image, center): {obs.images['base_0_rgb'][0, 100:105, 100:105, 0]}")
    
    # Check if images are all white
    agent_mean = float(obs.images['base_0_rgb'].mean())
    eye_mean = float(obs.images['left_wrist_0_rgb'].mean())
    print(f"  Agent image mean: {agent_mean:.3f}")
    print(f"  Eye image mean: {eye_mean:.3f}")
    
    if agent_mean > 0.95 or agent_mean < -0.95:
        print("  WARNING: Images appear to be mostly white or black!")
    
    # Un-normalize tracks if stats available
    if norm_stats:
        print("Un-normalizing tracks for visualization...")
        # Un-normalize state (inputs)
        # obs.state is (B, 78)
        state_unnorm = unnormalize_tracks(obs.state, 'state', norm_stats)
        
        # Un-normalize targets (B, 16, 78)
        tracks_gt_unnorm = unnormalize_tracks(tracks_gt, 'actions', norm_stats)
        
        # Create a wrapper object that mimics Observation but with unnormalized state
        class ObsWrapper:
            def __init__(self, original_obs, new_state):
                self.images = original_obs.images
                self.state = new_state
                
        obs_obj = ObsWrapper(obs, state_unnorm)
        tracks_gt = tracks_gt_unnorm
    else:
        obs_obj = obs

    # Visualize samples
    for i in range(min(args.num_samples, args.batch_size)):
        output_path = output_dir / f"droid_sample_{i}.png"
        print(f"\nVisualizing sample {i}...")
        
        # Decode instruction if available
        instruction = "Unknown"
        try:
            if hasattr(obs, 'tokenized_prompt') and obs.tokenized_prompt is not None:
                # tokenized_prompt is (B, L)
                tokens = obs.tokenized_prompt[i]
                # If using Paligemma tokenizer, we need to check how to decode
                # But we can try simple decoding or print tokens
                # If loader.tokenizer has decode, use it
                if hasattr(loader.tokenizer, 'decode'):
                    instruction = loader.tokenizer.decode(tokens)
                else:
                    # Try to reconstruct string if tokens are integers
                    instruction = f"Tokens: {tokens[:10]}..."
        except Exception as e:
            print(f"Warning: Could not decode instruction: {e}")
            
        print(f"  Instruction: {instruction}")
        
        visualize_batch_sample(obs_obj, tracks_gt, tracks_pred=None, sample_idx=i, output_path=output_path, instruction=instruction)
    
    print(f"\nVisualization complete! Check {output_dir}/ for outputs.")


if __name__ == "__main__":
    main()
