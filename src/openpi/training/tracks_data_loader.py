"""
Data loader for PI0.5 with mesh point tracks prediction.

Tracks format: 39 points per timestep
- 7 points: agent view
- 25 points: uniform grid eye-in-hand
- 7 points: gripper eye-in-hand

Query points: initial (t=0) positions of the 39 points, used as input to TrackHead.
By default, query_points = tracks[:, 0, :, :] (first frame).

Usage:
    from openpi.training.tracks_data_loader import create_tracks_data_loader

    loader = create_tracks_data_loader(config, framework="pytorch")
    for obs, actions, tracks in loader:
        # obs has query_points from first frame
        # tracks: (B, action_horizon, 39, 3)
"""

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def create_tracks_data_loader(
    config: _config.TrainConfig,
    *,
    shuffle: bool = True,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "pytorch",
):
    """Create a data loader for training with tracks prediction.

    Requires config.model.predict_tracks=True and config.data.tracks_path set.
    Uses the standard create_data_loader which adds TracksSidecarDataset when
    tracks_path is provided. Each batch yields (Observation, actions, tracks)
    where Observation includes query_points (first frame of tracks).
    """
    if not getattr(config.model, "predict_tracks", False):
        raise ValueError("create_tracks_data_loader requires config.model.predict_tracks=True")

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.tracks_path is None:
        raise ValueError(
            "Tracks prediction requires data_config.tracks_path. "
            "Use LeRobotLiberoTracksDataConfig with tracks_path='<path/to/tracks.npy>'"
        )

    return _data_loader.create_data_loader(
        config,
        shuffle=shuffle,
        num_batches=num_batches,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )
