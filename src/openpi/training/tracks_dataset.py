"""
Dataset for training PI0.5 with mesh point tracks prediction.

Tracks format: 39 points per timestep
- 7 points: agent view
- 25 points: uniform grid eye-in-hand
- 7 points: gripper eye-in-hand

Each point has (x, y, z) coordinates. Shape: (action_horizon, 39, 3).

Expected data formats:
1. LeRobot + tracks sidecar: Use TracksSidecarDataset to add tracks from a .npy file
   to an existing LeRobot dataset. tracks.npy shape: (N, action_horizon, 39, 3)

2. HuggingFace dataset: Use TracksLeRobotDataset when your LeRobot dataset
   has an additional "tracks" key in the parquet files.
"""

from pathlib import Path
from typing import SupportsIndex

import numpy as np

import openpi.models.model as _model
import openpi.training.data_loader as _data_loader


class TracksSidecarDataset:
    """
    Wraps a transformed dataset and adds tracks from a sidecar file.

    Use after transform_dataset(). The wrapped dataset returns dict with
    observation + actions. Tracks are loaded from tracks_path and merged by index.

    Args:
        dataset: Transformed dataset (output of transform_dataset)
        tracks_path: Path to .npy file with shape (N, action_horizon, 39, 3)
        action_horizon: Number of action steps
        n_track_points: Number of track points (default 39)
    """

    def __init__(
        self,
        dataset: _data_loader.Dataset,
        tracks_path: str | Path,
        action_horizon: int,
        n_track_points: int = 39,
    ):
        self._dataset = dataset
        self._tracks = np.load(tracks_path).astype(np.float32)
        if self._tracks.ndim != 4:
            raise ValueError(
                f"tracks must be 4D (N, action_horizon, n_points, 3), got {self._tracks.shape}"
            )
        n_samples, ah, np_pts, coords = self._tracks.shape
        if ah != action_horizon or np_pts != n_track_points or coords != 3:
            raise ValueError(
                f"tracks shape mismatch: expected (N, {action_horizon}, {n_track_points}, 3), "
                f"got {self._tracks.shape}"
            )
        if n_samples != len(dataset):
            raise ValueError(
                f"tracks length ({n_samples}) must match dataset length ({len(dataset)})"
            )

    def __getitem__(self, index: SupportsIndex):
        idx = index.__index__()
        sample = self._dataset[idx]
        sample = dict(sample) if not isinstance(sample, dict) else sample.copy()
        tracks = self._tracks[idx].copy()
        sample["tracks"] = tracks
        first_frame = tracks[0]
        n_points = first_frame.shape[0]
        cam_ids = np.zeros(n_points, dtype=np.float32)
        for i, group_size in enumerate((7, 25, 7)):
            start = sum((7, 25, 7)[:i])
            end = min(start + group_size, n_points)
            cam_ids[start:end] = i
        sample["query_points"] = np.concatenate(
            [cam_ids[:, None], first_frame], axis=-1
        ).astype(np.float32)
        return sample

    def __len__(self) -> int:
        return len(self._dataset)


class TracksFakeDataset(_data_loader.FakeDataset):
    """Fake dataset with random tracks for testing."""

    def __init__(
        self,
        model_config: _model.BaseModelConfig,
        num_samples: int,
        n_track_points: int = 39,
    ):
        super().__init__(model_config, num_samples)
        self._n_track_points = n_track_points
        _, action_spec = model_config.inputs_spec()
        self._action_horizon = action_spec.shape[1]

    def __getitem__(self, index: SupportsIndex) -> dict:
        import jax.random as jr

        sample = super().__getitem__(index)
        rng = jr.key(index.__index__())
        rng, data_rng = jr.split(rng)
        tracks = jr.uniform(
            data_rng,
            shape=(self._action_horizon, self._n_track_points, 3),
            minval=-1.0,
            maxval=1.0,
        )
        tracks = np.asarray(tracks)
        sample["tracks"] = tracks
        first_frame = tracks[0]
        cam_ids = np.zeros(self._n_track_points, dtype=np.float32)
        for i, group_size in enumerate((7, 25, 7)):
            start = sum((7, 25, 7)[:i])
            end = min(start + group_size, self._n_track_points)
            cam_ids[start:end] = i
        sample["query_points"] = np.concatenate(
            [cam_ids[:, None], first_frame], axis=-1
        ).astype(np.float32)
        return sample
