"""
EgoDex TFRecord DataLoader for Pi0 Pretraining.
适配 EgoDex 的 14个 Mesh点 + 25个 Uniform Tracks (全部位于 AgentView)。
将数据无缝拼接为 Pi0 所需的 78 维输入/输出结构。
"""
import os
import glob
import random
import struct
from pathlib import Path
from typing import Iterator

import jax
import numpy as np
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize

# =============================================================================
# 1. 纯 Python TFRecord 解析器 (无依赖, 保持不变)
# =============================================================================

def _read_varint(buf, pos):
    value = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80): return value, pos
        shift += 7

def _parse_example_py(serialized):
    """纯 Python 解析 TFRecord Example。"""
    out = {}
    pos = 0
    end = len(serialized)
    while pos < end:
        tag, pos = _read_varint(serialized, pos)
        field_num, wire = tag >> 3, tag & 0x07
        if field_num == 1 and wire == 2:  # features
            feats_len, pos = _read_varint(serialized, pos)
            feats = serialized[pos:pos + feats_len]
            pos += feats_len
            fpos = 0
            while fpos < len(feats):
                ftag, fpos = _read_varint(feats, fpos)
                if (ftag >> 3) != 1 or (ftag & 0x07) != 2: break
                entry_len, fpos = _read_varint(feats, fpos)
                entry = feats[fpos:fpos + entry_len]
                fpos += entry_len

                epos = 0
                key, value_msg = None, None
                while epos < len(entry):
                    etag, epos = _read_varint(entry, epos)
                    ef, ew = etag >> 3, etag & 0x07
                    if ew != 2: break
                    elen, epos = _read_varint(entry, epos)
                    payload = entry[epos:epos + elen]
                    epos += elen
                    if ef == 1: key = payload.decode("utf-8")
                    elif ef == 2: value_msg = payload
                if key is None or value_msg is None: continue

                vpos = 0
                while vpos < len(value_msg):
                    vtag, vpos = _read_varint(value_msg, vpos)
                    vf, vw = vtag >> 3, vtag & 0x07
                    if vf == 1 and vw == 2:  # bytes_list
                        bl_len, vpos = _read_varint(value_msg, vpos)
                        bl = value_msg[vpos:vpos + bl_len]
                        vpos += bl_len
                        bpos = 0
                        while bpos < len(bl):
                            btag, bpos = _read_varint(bl, bpos)
                            if (btag >> 3) == 1 and (btag & 0x07) == 2:
                                val_len, bpos = _read_varint(bl, bpos)
                                out[key] = bl[bpos:bpos + val_len]
                                bpos += val_len
                                break
                        break
                    elif vf == 3 and vw == 2:  # int64_list
                         il_len, vpos = _read_varint(value_msg, vpos)
                         vpos += il_len
                         break
                    elif vw == 0: _, vpos = _read_varint(value_msg, vpos)
                    elif vw == 2:
                        skip_len, vpos = _read_varint(value_msg, vpos)
                        vpos += skip_len
                    else: break
            break
        else:
            if wire == 0: _, pos = _read_varint(serialized, pos)
            elif wire == 2:
                size, pos = _read_varint(serialized, pos)
                pos += size
            elif wire == 5: pos += 4
            elif wire == 1: pos += 8
            else: break
    return out

# =============================================================================
# 2. EgoDex 专用的数据提取与拼接
# =============================================================================

def _decode_example_py(parsed):
    """将 EgoDex 的数据解码并拼接为 78维的 Pi0 标准格式。"""

    # 1. 解码图像 (仅读取 agentview)
    agent_img = np.frombuffer(parsed['observation/agentview_image'], dtype=np.uint8).reshape(3, 224, 224)
    agent_img = np.transpose(agent_img, (1, 2, 0)) # 转为 HWC (224, 224, 3)
    agent_img = agent_img.astype(np.float32) / 255.0 * 2.0 - 1.0 # 归一化至 [-1, 1]

    # 2. 解码轨迹 (直接从 agentview 读取)
    # EgoDex 数据：14个 Mesh 点 + 25个 Uniform 轨迹点
    mesh_agent = np.frombuffer(parsed['observation/mesh/vertex_agentview_2d_seq'], dtype=np.float32).reshape(16, 14, 2)
    uniform_agent = np.frombuffer(parsed['observation/uniform_tracks/agentview'], dtype=np.float32).reshape(16, 25, 2)

    # 轨迹坐标从 [0, 224] 归一化至 [-1, 1]
    mesh_agent = mesh_agent / 224.0 * 2.0 - 1.0
    uniform_agent = uniform_agent / 224.0 * 2.0 - 1.0

    # 3. 拼接状态与轨迹为 78 维
    # 展平单帧：14*2 + 25*2 = 28 + 50 = 78
    vertex_0 = mesh_agent[0].flatten()        # (28,)
    uniform_0 = uniform_agent[0].flatten()    # (50,)
    input_state = np.concatenate([vertex_0, uniform_0], axis=0) # t=0 输入状态 (78,)

    v_seq = mesh_agent.reshape(16, -1)        # (16, 28)
    u_seq = uniform_agent.reshape(16, -1)     # (16, 50)
    tracks = np.concatenate([v_seq, u_seq], axis=1) # t=0~15 预测目标 (16, 78)

    # 4. 指令解析
    instruction = parsed.get('language_instruction', b'move the robot').decode('utf-8')

    return {
        'agent_img': agent_img,
        'input_state': input_state,
        'tracks': tracks,
        'instruction': instruction
    }

def _iter_tfrecord_file(path):
    """迭代读取 TFRecord 文件。"""
    try:
        with open(path, "rb") as f:
            while True:
                length_bytes = f.read(8)
                if not length_bytes or len(length_bytes) != 8: break
                length = struct.unpack("<Q", length_bytes)[0]
                if len(f.read(4)) != 4: break # CRC
                data = f.read(length)
                if len(data) != length: break
                if len(f.read(4)) != 4: break # CRC
                yield data
    except Exception as e:
        print(f"Error reading TFRecord {path}: {e}")

# =============================================================================
# 3. DataLoader 类
# =============================================================================

class EgoDexPi0PretrainLoader:
    def __init__(self, data_path, data_config, batch_size=32, shuffle=True, seed=0):
        self.batch_size = batch_size
        self._data_config = data_config
        self.tokenizer = _tokenizer.PaligemmaTokenizer()
        self.shuffle = shuffle

        # 查找 TFRecord 文件
        path_obj = Path(data_path)
        self.files = sorted(glob.glob(str(path_obj / "*.tfrecord")))
        if not self.files:
            raise ValueError(f"未在 {data_path} 找到 .tfrecord 文件。")

        print(f"[EgoDex Loader] 找到 {len(self.files)} 个 TFRecord 文件。")

        if self.shuffle:
            random.seed(seed)
            random.shuffle(self.files)

        self.file_idx = 0
        self.record_iter = None
        self.buffer = []

        # 尝试加载归一化统计数据
        try:
            assets_path = Path("/mnt/kevin/code/wmrl/howard-branch/code/openpi_fork/assets/egodex_pretrain")
            self.norm_stats = _normalize.load(assets_path)
        except Exception:
            self.norm_stats = None

    def _fill_buffer(self, min_size):
        MAX_BUFFER = 500
        attempts = 0
        while len(self.buffer) < min_size and attempts < len(self.files) + 1:
            if self.record_iter is None:
                if self.file_idx >= len(self.files):
                    if self.shuffle: random.shuffle(self.files)
                    self.file_idx = 0
                self.record_iter = _iter_tfrecord_file(self.files[self.file_idx])
                self.file_idx += 1
                attempts += 1

            try:
                count = 0
                while len(self.buffer) < MAX_BUFFER:
                    self.buffer.append(next(self.record_iter))
                    count += 1
                    if count >= 100: break
            except StopIteration:
                self.record_iter = None

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        self._fill_buffer(self.batch_size)
        if not self.buffer:
            raise StopIteration

        batch_records = []
        for _ in range(self.batch_size):
            if not self.buffer: self._fill_buffer(self.batch_size)
            if self.buffer: batch_records.append(self.buffer.pop(0))
            else: break

        if len(batch_records) < self.batch_size:
            raise StopIteration

        # 组装 Batch
        batch_data = {'agent_img': [], 'input_state': [], 'tracks': [], 'instruction': []}
        for rec in batch_records:
            try:
                decoded = _decode_example_py(_parse_example_py(rec))
                for k, v in decoded.items():
                    batch_data[k].append(v)
            except Exception:
                continue

        if not batch_data['agent_img']: raise StopIteration

        actual_bs = len(batch_data['agent_img'])
        batch = {k: np.stack(v) for k, v in batch_data.items() if k != 'instruction'}

        # 处理 Token
        tokenized = [self.tokenizer.tokenize(inst) for inst in batch_data['instruction']]
        tokens_list, mask_list = zip(*tokenized)
        batch_tokens = np.stack(tokens_list)
        batch_mask = np.stack(mask_list)

        # 应用全局归一化 (如果存在)
        if self.norm_stats is not None:
            stats = self.norm_stats
            def normalize(x, s): return (x - s.mean) / (s.std + 1e-6)
            if 'state' in stats: batch['input_state'] = normalize(batch['input_state'], stats['state'])
            if 'actions' in stats: batch['tracks'] = normalize(batch['tracks'], stats['actions'])

        # 构造 JAX 观测对象 (填充 Dummy 手腕图像以防模型报错)
        dummy_wrist_img = jnp.zeros_like(jnp.array(batch['agent_img']))
        dummy_wrist_mask = jnp.zeros((actual_bs,), dtype=bool)

        obs = _model.Observation(
            images={
                'base_0_rgb': jnp.array(batch['agent_img']),
                'left_wrist_0_rgb': dummy_wrist_img,
                'right_wrist_0_rgb': dummy_wrist_img,
            },
            image_masks={
                'base_0_rgb': jnp.ones((actual_bs,), dtype=bool),
                'left_wrist_0_rgb': dummy_wrist_mask,
                'right_wrist_0_rgb': dummy_wrist_mask,
            },
            state=jnp.array(batch['input_state']),
            tokenized_prompt=jnp.array(batch_tokens),
            tokenized_prompt_mask=jnp.array(batch_mask),
        )

        tracks = jnp.array(batch['tracks'])
        return obs, tracks

def create_egodex_pi0_pretrain_loader(config, **kwargs):
    """入口函数。"""
    # 解析路径逻辑保持不变
    path = getattr(config.data, 'egodex_data_path',
           getattr(getattr(config.data, 'base_config', None), 'egodex_data_path',
           getattr(config, 'egodex_data_path', '/mnt/kevin/data/egodex_preprocessed_tfrecord')))

    process_count = jax.process_count()
    if config.batch_size % process_count != 0:
        raise ValueError(
            f"Global batch size {config.batch_size} must be divisible by process count {process_count}."
        )
    local_batch_size = config.batch_size // process_count

    return EgoDexPi0PretrainLoader(
        data_path=path,
        data_config=config.data.create(config.assets_dirs, config.model),
        batch_size=local_batch_size,
        shuffle=True,
        seed=config.seed,
    )
