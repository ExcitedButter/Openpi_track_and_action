"""
Validation and visualization for track prediction models.
Similar to the PyTorch implementation, visualizes predicted tracks during training.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Dict, Any, Optional
import io
from PIL import Image
from pathlib import Path
import openpi.shared.normalize as _normalize

def validate_and_visualize_tracks(
    model,
    val_data,
    step: int,
    config,
    num_samples: int = 5,
    wandb_enabled: bool = True
) -> Dict[str, Any]:
    """
    Run validation and create dual-view visualizations for track prediction.
    
    Similar to PyTorch's validate_and_visualize but for JAX.
    """
    val_losses = []
    visualizations = []
    
    # Run validation on a few batches
    norm_stats = None
    try:
        assets_path = Path("/mnt/kevin/code/wmrl/howard-branch/code/openpi_fork/assets/droid_pretrain_rlds")
        norm_stats = _normalize.load(assets_path)
    except Exception as e:
        print(f"Warning: Could not load norm stats for validation: {e}")

    for batch_idx, (observation, target_tracks) in enumerate(val_data):
        if batch_idx >= 10:  # Validate on 10 batches
            break
            
        # Compute validation loss
        val_loss = model.compute_loss(
            jax.random.PRNGKey(step + batch_idx),
            observation,
            target_tracks,
            train=False
        )
        val_losses.append(float(jnp.mean(val_loss)))
        
        # Create visualizations for first few batches
        if batch_idx < num_samples:
            viz = create_track_visualization(
                model, observation, target_tracks, step, batch_idx, norm_stats
            )
            if viz is not None:
                visualizations.append(viz)
    
    # Prepare metrics
    metrics = {
        'val_loss': np.mean(val_losses) if val_losses else 0.0
    }
    
    # Log to WandB
    if wandb_enabled and wandb.run is not None:
        wandb_metrics = {'val_loss': metrics['val_loss']}
        
        if visualizations:
            wandb_metrics['val_tracks'] = visualizations
        
        wandb.log(wandb_metrics, step=step)
    
    return metrics


def create_track_visualization(
    model,
    observation,
    target_tracks,
    step: int,
    sample_idx: int,
    norm_stats=None
) -> Optional[wandb.Image]:
    """
    Create dual-view visualization of predicted vs ground truth tracks.
    
    Layout (2x3 grid):
    Row 1: Agentview - Query Points | Predicted | Ground Truth
    Row 2: Eyeinhand - Query Points | Predicted | Ground Truth
    """
    try:
        # Extract data
        img_agent = observation.images['base_0_rgb'][0]  # First in batch
        img_wrist = observation.images['left_wrist_0_rgb'][0]
        
        # Helper to un-normalize: (x * std + mean + 1) / 2 * 224
        def unnormalize_tracks(tracks, stats_key):
            if norm_stats is None or stats_key not in norm_stats:
                return tracks
            
            s = norm_stats[stats_key]
            # Ensure numpy arrays
            tracks = np.array(tracks)
            s_std = np.array(s.std)
            s_mean = np.array(s.mean)
            
            # 1. Reverse standardization
            x_scaled = tracks * (s_std + 1e-6) + s_mean
            # 2. Reverse scaling [-1, 1] -> [0, 224]
            x_pixel = (x_scaled + 1.0) / 2.0 * 224.0
            return x_pixel

        # Handle both TrackObservation (query_points) and regular Observation (state)
        if hasattr(observation, 'query_points'):
            query_points = observation.query_points[0].reshape(-1, 2)  # (64, 2) for tracks
            # Split query points for two views (32 each)
            query_agent = query_points[:32]
            query_wrist = query_points[32:]
            is_vertex = False
        else:
            # Vertex models use state field
            if isinstance(observation, dict):
                query_points = observation['state'][0]
            else:
                query_points = observation.state[0]
            
            # Un-normalize query points (they are inputs, so use 'state' stats)
            # But query points shape varies, unnormalize function needs to handle flat array?
            # Or just unnormalize all if shape matches?
            # 'state' stats are 78-dim. 'query_points' is 78-dim (for asymmetric).
            if norm_stats and 'state' in norm_stats and query_points.shape[-1] == 78:
                query_points = unnormalize_tracks(query_points, 'state')

            if query_points.shape[-1] == 78:  # Asymmetric hybrid (39 points × 2 coords)
                # Order: [agentview_vertices(14), eyeinhand_vertices(14), eyeinhand_uniform(50)]
                query_agent = query_points[:14].reshape(7, 2)  # Agentview vertices only
                query_wrist_vertices = query_points[14:28].reshape(7, 2)  # Eyeinhand vertices
                query_wrist_uniform = query_points[28:].reshape(25, 2)  # Eyeinhand uniform
                # Combine eyeinhand points
                query_wrist = np.vstack([query_wrist_vertices, query_wrist_uniform])  # (32, 2)
            elif query_points.shape[-1] == 128:  # Symmetric hybrid (64 points × 2 coords)
                # Order: [agentview_vertices(14), agentview_uniform(50), eyeinhand_vertices(14), eyeinhand_uniform(50)]
                query_agent_vertices = query_points[:14].reshape(7, 2)
                query_agent_uniform = query_points[14:64].reshape(25, 2)
                query_agent = np.vstack([query_agent_vertices, query_agent_uniform])  # (32, 2)
                
                query_wrist_vertices = query_points[64:78].reshape(7, 2)
                query_wrist_uniform = query_points[78:].reshape(25, 2)
                query_wrist = np.vstack([query_wrist_vertices, query_wrist_uniform])  # (32, 2)
            elif query_points.shape[-1] == 50:  # Uniform-only (25 eyeinhand uniform points)
                # No agentview, only eyeinhand uniform
                query_agent = np.zeros((0, 2))  # No agentview points
                query_wrist = query_points.reshape(25, 2)  # All eyeinhand uniform
            elif query_points.shape[-1] == 7:  # Action-to-uniform (7-dim proprioception input)
                # Proprioception doesn't map to image points directly
                # Use first timestep of target tracks as "query points" for visualization
                if target_tracks.shape[-1] == 50:  # Uniform tracks output
                    query_agent = np.zeros((0, 2))  # No agentview points
                    query_wrist = target_tracks[0, 0].reshape(25, 2)  # First timestep as query
                else:
                    query_agent = np.zeros((0, 2))
                    query_wrist = np.zeros((0, 2))
            elif query_points.shape[-1] == 78:  # Asymmetric hybrid (14 agent + 14 eye + 50 uniform)
                # Split into components
                query_agent = query_points[:14].reshape(7, 2)  # First 14 dims = 7 agentview vertices
                # Wrist has both vertex (14-28) and uniform (28-78)
                # For visualization, let's just show the vertex points for now, or maybe all?
                # Let's show vertex points (7) + uniform points (25)
                wrist_vertex = query_points[14:28].reshape(7, 2)
                wrist_uniform = query_points[28:].reshape(25, 2)
                query_wrist = np.concatenate([wrist_vertex, wrist_uniform], axis=0)
            elif query_points.shape[-1] == 28:  # Vertex-input models: 14 vertex points (7 agentview + 7 eyeinhand)
                # Split into agentview and eyeinhand vertices
                query_agent = query_points[:14].reshape(7, 2)  # First 14 dims = 7 agentview vertices
                query_wrist = query_points[14:].reshape(7, 2)  # Last 14 dims = 7 eyeinhand vertices
            elif query_points.shape[-1] == 14:  # Single view (7 joints × 2 coords)
                query_points_2d = query_points.reshape(7, 2)
                # For selectable models, check which view the query points are for
                if hasattr(model, 'view_type'):
                    if model.view_type == 'eyeinhand':
                        # Eyeinhand model: query points are eyeinhand vertices
                        query_agent = np.zeros((0, 2))  # No agentview points
                        query_wrist = query_points_2d  # Eyeinhand vertices
                    else:
                        # Agentview model: query points are agentview vertices
                        query_agent = query_points_2d  # Agentview vertices
                        query_wrist = np.zeros((0, 2))  # No eyeinhand points
                else:
                    # Default: assume agentview
                    query_agent = query_points_2d  # Agentview vertices
                    query_wrist = np.zeros((0, 2))  # No eyeinhand points
            else:
                raise ValueError(f"Unexpected query_points shape: {query_points.shape}")
            is_vertex = True
        
        # Run inference to get predicted tracks
        # Handle dict observation for tree_map
        if isinstance(observation, dict):
            obs_single = jax.tree_map(lambda x: x[0:1], observation)
        else:
            # Assume it's a structure that tree_map handles (like dataclass)
            obs_single = jax.tree_map(lambda x: x[0:1], observation)

        predicted_tracks = model.sample_actions(
            jax.random.PRNGKey(step),
            obs_single,  # Single sample
            num_steps=10
        )
        
        # [DEBUG] Print types and shapes
        # print(f"DEBUG: predicted_tracks type: {type(predicted_tracks)}")
        # if hasattr(predicted_tracks, 'shape'):
        #     print(f"DEBUG: predicted_tracks shape: {predicted_tracks.shape}")
        
        # If model outputs a dict (like Hybrid model returning token_actions and track_actions), extract track_actions
        if isinstance(predicted_tracks, dict):
            if "track_actions" in predicted_tracks:
                 predicted_tracks = predicted_tracks["track_actions"]
            elif "actions" in predicted_tracks:
                 predicted_tracks = predicted_tracks["actions"]
            else:
                 # Fallback: assume values() contains it? Or keys match spec?
                 # Just take first value?
                 predicted_tracks = list(predicted_tracks.values())[0]

        # Convert predicted_tracks to numpy/array if it's a list or similar
        # But wait, test script output: Output type: <class 'jaxlib.xla_extension.ArrayImpl'>
        # Output shape: (2, 4, 32)
        # So it IS an array, not a dict.
        
        # If target_tracks is a dict (unlikely for Droid loader but possible elsewhere)
        if isinstance(target_tracks, dict):
            if "track_actions" in target_tracks:
                 target_tracks = target_tracks["track_actions"]
            elif "actions" in target_tracks:
                 target_tracks = target_tracks["actions"]
            else:
                 target_tracks = list(target_tracks.values())[0]
        
        # Ensure numpy arrays for unnormalization
        predicted_tracks = np.array(predicted_tracks)
        target_tracks = np.array(target_tracks)

        # Un-normalize predicted and target tracks (using 'actions' stats)
        # Note: predicted_tracks might be (16, 78) or similar.
        # unnormalize_tracks handles broadcasting for std/mean (78-dim)
        if norm_stats and 'actions' in norm_stats:
            predicted_tracks = unnormalize_tracks(predicted_tracks, 'actions')
            target_tracks = unnormalize_tracks(target_tracks, 'actions')

        if is_vertex:
            # Handle different vertex model output formats
            # Check if target has 2 views (selectable model)
            if target_tracks.ndim == 4 and target_tracks.shape[1] == 2:
                # Selectable model: (batch, 2, 16, 14) format
                # Extract both views - use actual timestep count
                timesteps = target_tracks.shape[2] if target_tracks.ndim > 2 else target_tracks.shape[1] // 14
                target_agent = target_tracks[0, 0].reshape(timesteps, 7, 2)  # Agentview
                target_wrist = target_tracks[0, 1].reshape(timesteps, 7, 2)  # Eyeinhand
                
                # Predicted tracks should match this format
                if predicted_tracks.ndim == 2:
                    # Model output: (timesteps, 14) - single view prediction
                    pred_reshaped = predicted_tracks.reshape(timesteps, 7, 2)
                    # Check model's view_type to determine which view was predicted
                    if hasattr(model, 'view_type'):
                        if model.view_type == 'eyeinhand':
                            pred_agent = np.zeros((timesteps, 0, 2))  # No agent predictions
                            pred_wrist = pred_reshaped
                        else:  # agentview or both (default to agentview)
                            pred_agent = pred_reshaped
                            pred_wrist = np.zeros((timesteps, 0, 2))  # No wrist predictions
                    else:
                        # Default: assume agentview
                        pred_agent = pred_reshaped
                        pred_wrist = np.zeros((timesteps, 0, 2))  # No wrist predictions
                else:
                    # Model output shape is not (16, 14), handle other cases
                    # Check if it's (batch, timesteps, 14) format
                    if predicted_tracks.ndim == 3:
                        pred_reshaped = predicted_tracks[0].reshape(timesteps, 7, 2)
                        # Check model's view_type to determine which view was predicted
                        if hasattr(model, 'view_type'):
                            if model.view_type == 'eyeinhand':
                                pred_agent = np.zeros((timesteps, 0, 2))  # No agent predictions
                                pred_wrist = pred_reshaped
                            else:  # agentview or both (default to agentview)
                                pred_agent = pred_reshaped
                                pred_wrist = np.zeros((timesteps, 0, 2))  # No wrist predictions
                        else:
                            # Default: assume agentview
                            pred_agent = pred_reshaped
                            pred_wrist = np.zeros((timesteps, 0, 2))  # No wrist predictions
                    else:
                        # Fallback for other shapes
                        pred_agent = predicted_tracks[0, 0].reshape(timesteps, 7, 2) if predicted_tracks.shape[1] == 2 else predicted_tracks[0].reshape(timesteps, 7, 2)
                        pred_wrist = predicted_tracks[0, 1].reshape(timesteps, 7, 2) if predicted_tracks.shape[1] == 2 else np.zeros((timesteps, 0, 2))
            else:
                # Basic vertex model: check if it's 28-dim (both views) or 14-dim (single view)
                predicted_tracks = predicted_tracks[0] if predicted_tracks.ndim > 2 else predicted_tracks
                target_reshape = target_tracks[0] if target_tracks.ndim > 2 else target_tracks
                
                # Calculate timesteps dynamically from the data
                timesteps = predicted_tracks.shape[0]
                
                print(f"DEBUG: predicted_tracks.shape={predicted_tracks.shape}, target_reshape.shape={target_reshape.shape}")
                
                if predicted_tracks.shape[-1] == 78:  # Asymmetric hybrid
                    # 14 agent + 14 eye + 50 uniform
                    pred_agent = predicted_tracks[:, :14].reshape(timesteps, 7, 2)
                    pred_wrist_vertex = predicted_tracks[:, 14:28].reshape(timesteps, 7, 2)
                    pred_wrist_uniform = predicted_tracks[:, 28:].reshape(timesteps, 25, 2)
                    pred_wrist = np.concatenate([pred_wrist_vertex, pred_wrist_uniform], axis=1) # (16, 32, 2)
                    
                    target_agent = target_reshape[:, :14].reshape(timesteps, 7, 2)
                    target_wrist_vertex = target_reshape[:, 14:28].reshape(timesteps, 7, 2)
                    target_wrist_uniform = target_reshape[:, 28:].reshape(timesteps, 25, 2)
                    target_wrist = np.concatenate([target_wrist_vertex, target_wrist_uniform], axis=1)

                elif predicted_tracks.shape[-1] == 50:  # Uniform-only
                    # Only eyeinhand uniform points
                    pred_agent = np.zeros((timesteps, 0, 2))  # No agentview
                    pred_wrist = predicted_tracks.reshape(timesteps, 25, 2)
                    
                    # Same for targets
                    target_agent = np.zeros((timesteps, 0, 2))  # No agentview
                    target_wrist = target_reshape.reshape(timesteps, 25, 2)
                elif predicted_tracks.shape[-1] == 78:  # Asymmetric hybrid
                    # Order: [agentview_vertices(14), eyeinhand_vertices(14), eyeinhand_uniform(50)]
                    pred_agent = predicted_tracks[:, :14].reshape(timesteps, 7, 2)
                    pred_wrist_vertices = predicted_tracks[:, 14:28].reshape(timesteps, 7, 2)
                    pred_wrist_uniform = predicted_tracks[:, 28:].reshape(timesteps, 25, 2)
                    pred_wrist = np.concatenate([pred_wrist_vertices, pred_wrist_uniform], axis=1)  # (timesteps, 32, 2)
                    
                    # DEBUG: Print target shape and range
                    print(f"[DEBUG] Target reshape shape: {target_reshape.shape}, min: {target_reshape.min():.4f}, max: {target_reshape.max():.4f}")
                    
                    # Same for targets
                    target_agent = target_reshape[:, :14].reshape(timesteps, 7, 2)
                    target_wrist_vertices = target_reshape[:, 14:28].reshape(timesteps, 7, 2)
                    target_wrist_uniform = target_reshape[:, 28:].reshape(timesteps, 25, 2)
                    target_wrist = np.concatenate([target_wrist_vertices, target_wrist_uniform], axis=1)
                    
                    print(f"[DEBUG] Target wrist vertices min: {target_wrist_vertices.min():.4f}, max: {target_wrist_vertices.max():.4f}")
                    print(f"[DEBUG] Target wrist uniform min: {target_wrist_uniform.min():.4f}, max: {target_wrist_uniform.max():.4f}")
                elif predicted_tracks.shape[-1] == 128:  # Symmetric hybrid
                    # Order: [agentview_vertices(14), agentview_uniform(50), eyeinhand_vertices(14), eyeinhand_uniform(50)]
                    pred_agent_vertices = predicted_tracks[:, :14].reshape(timesteps, 7, 2)
                    pred_agent_uniform = predicted_tracks[:, 14:64].reshape(timesteps, 25, 2)
                    pred_agent = np.concatenate([pred_agent_vertices, pred_agent_uniform], axis=1)  # (timesteps, 32, 2)
                    
                    pred_wrist_vertices = predicted_tracks[:, 64:78].reshape(timesteps, 7, 2)
                    pred_wrist_uniform = predicted_tracks[:, 78:].reshape(timesteps, 25, 2)
                    pred_wrist = np.concatenate([pred_wrist_vertices, pred_wrist_uniform], axis=1)  # (timesteps, 32, 2)
                    
                    # Same for targets
                    target_agent_vertices = target_reshape[:, :14].reshape(timesteps, 7, 2)
                    target_agent_uniform = target_reshape[:, 14:64].reshape(timesteps, 25, 2)
                    target_agent = np.concatenate([target_agent_vertices, target_agent_uniform], axis=1)
                    
                    target_wrist_vertices = target_reshape[:, 64:78].reshape(timesteps, 7, 2)
                    target_wrist_uniform = target_reshape[:, 78:].reshape(timesteps, 25, 2)
                    target_wrist = np.concatenate([target_wrist_vertices, target_wrist_uniform], axis=1)
                elif predicted_tracks.shape[-1] == 28:  # Both views concatenated
                    # Split 28-dim into two 14-dim views
                    pred_both = predicted_tracks.reshape(timesteps, 2, 7, 2)  # (timesteps, 2, 7, 2)
                    pred_agent = pred_both[:, 0]  # (timesteps, 7, 2)
                    pred_wrist = pred_both[:, 1]  # (timesteps, 7, 2)
                    
                    target_both = target_reshape.reshape(timesteps, 2, 7, 2)
                    target_agent = target_both[:, 0]
                    target_wrist = target_both[:, 1]
                else:  # 14-dim single view
                    pred_agent = predicted_tracks.reshape(timesteps, 7, 2)  # (timesteps, 7, 2)
                    target_agent = target_reshape.reshape(timesteps, 7, 2)
                    
                    # No eyeinhand predictions/targets for single-view model
                    pred_wrist = np.zeros((timesteps, 0, 2))  # Empty array
                    target_wrist = np.zeros((timesteps, 0, 2))  # Empty array
        else:
            # Track model: get timesteps dynamically
            predicted_tracks_single = predicted_tracks[0] if predicted_tracks.ndim > 2 else predicted_tracks
            target_single = target_tracks[0] if target_tracks.ndim > 2 else target_tracks
            timesteps = predicted_tracks_single.shape[0]
            
            # Reshape to (timesteps, 64, 2)
            predicted_tracks = predicted_tracks_single.reshape(timesteps, -1, 2)
            target_reshape = target_single.reshape(timesteps, -1, 2)
            
            # Split predicted and target tracks
            pred_agent = predicted_tracks[:, :32, :]  # (timesteps, 32, 2)
            pred_wrist = predicted_tracks[:, 32:, :]  # (timesteps, 32, 2)
            target_agent = target_reshape[:, :32, :]
            target_wrist = target_reshape[:, 32:, :]
        
        # Convert images from [-1, 1] to [0, 1]
        img_agent_np = np.array((img_agent + 1.0) / 2.0)
        img_wrist_np = np.array((img_wrist + 1.0) / 2.0)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Track Prediction - Step {step} - Sample {sample_idx}', fontsize=16)
        
        H, W = img_agent_np.shape[:2]

        # Row 1: Agentview
        axes[0, 0].imshow(img_agent_np)
        if len(query_agent) > 0:
            # Query points are now pixels (unnormalized), so no need to multiply by W/H again?
            # Original code: axes[0, 0].scatter(query_agent[:, 0] * W, query_agent[:, 1] * H, c='red', s=20)
            # Wait, if query_agent is pixels, we should NOT multiply by W/H.
            # But the original code was written assuming [0, 1].
            # If I unnormalized to pixels, I should divide by W/H to get back to [0, 1] for plotting?
            # Or remove the W/H multiplication.
            # Let's remove W/H multiplication since we have pixels now.
            axes[0, 0].scatter(query_agent[:, 0], query_agent[:, 1], c='red', s=20)
        axes[0, 0].set_title(f'Agentview: Query Points ({len(query_agent)})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_agent_np)
        num_points_agent = pred_agent.shape[1]  # Get actual number of points
        for j in range(num_points_agent):
            track = np.array(pred_agent[:, j, :])
            axes[0, 1].plot(track[:, 0], track[:, 1], alpha=0.5)
        axes[0, 1].set_title(f'Agentview: Predicted ({num_points_agent} tracks)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_agent_np)
        for j in range(num_points_agent):
            track = np.array(target_agent[:, j, :])
            axes[0, 2].plot(track[:, 0], track[:, 1], alpha=0.5)
        axes[0, 2].set_title(f'Agentview: Ground Truth ({num_points_agent} tracks)')
        axes[0, 2].axis('off')
        
        # Row 2: Eyeinhand
        axes[1, 0].imshow(img_wrist_np)
        if len(query_wrist) > 0:
            axes[1, 0].scatter(query_wrist[:, 0], query_wrist[:, 1], c='red', s=20)
        axes[1, 0].set_title(f'Eyeinhand: Query Points ({len(query_wrist)})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_wrist_np)
        num_points_wrist = pred_wrist.shape[1]  # Get actual number of points
        if num_points_wrist > 0:  # Only plot if there are predictions
            for j in range(num_points_wrist):
                track = np.array(pred_wrist[:, j, :])
                axes[1, 1].plot(track[:, 0], track[:, 1], alpha=0.5)
        axes[1, 1].set_title(f'Eyeinhand: Predicted ({num_points_wrist} tracks)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(img_wrist_np)
        num_gt_wrist = target_wrist.shape[1]
        if num_gt_wrist > 0:  # Only plot if there are ground truth points
            for j in range(num_gt_wrist):
                track = np.array(target_wrist[:, j, :])
                axes[1, 2].plot(track[:, 0], track[:, 1], alpha=0.5)
        axes[1, 2].set_title(f'Eyeinhand: Ground Truth ({num_gt_wrist} tracks)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Convert to image for WandB
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        
        # Save locally too
        save_path = "checkpoints/visualizations/"
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/tracks_step_{step}_sample_{sample_idx}.png", dpi=150)
        
        # Save locally too
        save_path = "checkpoints/visualizations/"
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/tracks_step_{step}_sample_{sample_idx}.png", dpi=150)
        
        # Save locally too
        save_path = "checkpoints/visualizations/"
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/tracks_step_{step}_sample_{sample_idx}.png", dpi=150)
        
        plt.close(fig)
        
        return wandb.Image(img, caption=f"Step {step}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


def log_sample_predictions_wandb(
    model,
    observation, 
    target_tracks,
    predicted_tracks,
    step: int,
    config
):
    """
    Log detailed track predictions to WandB for analysis.
    """
    if not config.wandb_enabled or wandb.run is None:
        return
    
    # Calculate track metrics
    track_error = jnp.mean(jnp.square(predicted_tracks - target_tracks))
    
    # Log track-specific metrics
    wandb.log({
        "track_error_mse": float(track_error),
        "track_error_rmse": float(jnp.sqrt(track_error)),
        "track_error_pixels": float(jnp.sqrt(track_error) * 224),  # Convert to pixel error
    }, step=step)
    
    # Log histogram of track errors per point
    errors_per_point = jnp.mean(jnp.square(predicted_tracks - target_tracks), axis=1)  # Average over time
    wandb.log({
        "track_errors_histogram": wandb.Histogram(np.array(errors_per_point))
    }, step=step)