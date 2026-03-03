# π₀.5 + TrackHead 模型架构图

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        π₀.5 + TrackHead 双头模型架构                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              PREFIX 输入（共享）                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ SigLIP      │  │ PaliGemma   │  │ Tokenized Prompt (含离散化 state)                    │ │  │
│  │  │ 3 视角图像  │  │ 语言嵌入    │  │ "Task: {prompt}, State: {state};\nAction: "          │ │  │
│  │  │ 224×224    │  │             │  │                                                     │ │  │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────────────┬───────────────────────────────┘ │  │
│  │         │                │                                │                                │  │
│  │         └────────────────┴────────────────────────────────┘                                │  │
│  │                                    │                                                       │  │
│  │                                    ▼                                                       │  │
│  │                    prefix_embeds [B, seq_len, D]  (D=2048 for Gemma 2B)                   │  │
│  └───────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                           │
│                    ┌───────────────┴───────────────┐                                           │
│                    │                               │                                           │
│                    ▼                               ▼                                           │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────────────────────┐   │
│  │     ACTION HEAD（完全不变）      │  │              TRACK HEAD（新增）                     │   │
│  │                                 │  │                                                     │   │
│  │  SUFFIX: [action_tokens]        │  │  Query Points [B, 39, 3]  ← 用户输入的39个点初始位置  │   │
│  │  + adarms_cond (time_emb)       │  │       │                                               │   │
│  │         │                       │  │       ▼                                               │   │
│  │         ▼                       │  │  query_embed MLP → [B, 39, 256]                      │   │
│  │  Gemma Action Expert            │  │       │                                               │   │
│  │  (adaRMSNorm)                   │  │       ▼                                               │   │
│  │         │                       │  │  Cross-Attention (2 layers)                          │   │
│  │         ▼                       │  │  Q = query_embed, K/V = prefix_embeds                │   │
│  │  action_out_proj                │  │       │                                               │   │
│  │         │                       │  │       ▼                                               │   │
│  │         ▼                       │  │  output_proj → [B, action_horizon, 39, 3]            │   │
│  │  actions [B, T, 32]             │  │       │                                               │   │
│  │                                 │  │       ▼                                               │   │
│  │                                 │  │  tracks [B, T, 39, 3]  (每时刻39点3D坐标)              │   │
│  └─────────────────────────────────┘  └─────────────────────────────────────────────────────┘   │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Query Points 输入位置

**Query Points 放在 TrackHead 的输入端**，作为 Cross-Attention 的 Query：

1. **形状**: `[B, 39, 3]` — 39 个点的初始 (t=0) 3D 坐标
2. **来源**: 训练时默认使用 `tracks[:, 0, :, :]`（轨迹第一帧）
3. **推理时**: 用户需提供 39 个点的初始位置

```
39 点分布:
├── 7 points:  agent view (第三人称视角)
├── 25 points: uniform grid eye-in-hand (手眼相机均匀网格)
└── 7 points:  gripper eye-in-hand (夹爪相机)
```

## 数据流

```
训练:
  Dataset → tracks.npy (N, T, 39, 3)
         → query_points = tracks[:, 0, :, :]  (每样本第一帧)
         → Observation(query_points=...)
         → Model(observation, actions, tracks)
         → action_loss + λ * track_loss

推理:
  observation = {images, state, prompt, query_points}
  output = model.sample_actions(observation)
  → {"actions": [B,T,32], "tracks": [B,T,39,3]}
```

## 文件结构

```
src/openpi/
├── models/
│   ├── model.py              # Observation 增加 query_points
│   └── pi0_config.py         # predict_tracks, n_track_points, track_point_groups
├── models_pytorch/
│   ├── pi0_pytorch.py        # 集成 TrackHead，Action Head 不变
│   ├── track_head.py         # TrackHead: Cross-Attention + query_points
│   └── preprocessing_pytorch.py  # 传递 query_points
└── training/
    ├── tracks_dataset.py     # TracksSidecarDataset 添加 query_points
    └── tracks_data_loader.py # create_tracks_data_loader()
```

## 配置示例

```python
TrainConfig(
    name="pi05_libero_tracks",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        predict_tracks=True,
        n_track_points=39,
        track_point_groups=(7, 25, 7),
        tracks_loss_weight=1.0,
    ),
    data=LeRobotLiberoTracksDataConfig(
        tracks_path="path/to/tracks.npy",  # (N, 10, 39, 3)
        ...
    ),
)
```
