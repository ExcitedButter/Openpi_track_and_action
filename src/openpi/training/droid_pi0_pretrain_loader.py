"""
Droid TFRecord DataLoader for Pi0 Pretraining
Loads preprocessed TFRecord data (matching Libero format).
Compatible with pi0 vertex-input asymmetric model (78-dim input/output).
Images are preprocessed to 224x224 (center crop + resize from 180x320).
"""
import os
import glob
import jax
import jax.numpy as jnp
import numpy as np
import struct
import random
from pathlib import Path
import dataclasses
from typing import Iterator
import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize

# Optional TF import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    # Prevent TensorFlow from allocating GPU memory (leave it for JAX)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except:
        pass
except ImportError:
    TF_AVAILABLE = False

# =============================================================================
# Python-only TFRecord Parsing (No TensorFlow dependency)
# =============================================================================

def _read_varint(buf, pos):
    value = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7

def _parse_example_py(serialized):
    """Parse TFRecord example using pure Python."""
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
                if (ftag >> 3) != 1 or (ftag & 0x07) != 2:
                    break
                entry_len, fpos = _read_varint(feats, fpos)
                entry = feats[fpos:fpos + entry_len]
                fpos += entry_len

                epos = 0
                key, value_msg = None, None
                while epos < len(entry):
                    etag, epos = _read_varint(entry, epos)
                    ef, ew = etag >> 3, etag & 0x07
                    if ew != 2:
                        break
                    elen, epos = _read_varint(entry, epos)
                    payload = entry[epos:epos + elen]
                    epos += elen
                    if ef == 1:
                        key = payload.decode("utf-8")
                    elif ef == 2:
                        value_msg = payload
                if key is None or value_msg is None:
                    continue

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
                         vpos += il_len # Skip content
                         break
                    elif vw == 0:
                        _, vpos = _read_varint(value_msg, vpos)
                    elif vw == 2:
                        skip_len, vpos = _read_varint(value_msg, vpos)
                        vpos += skip_len
                    else:
                        break
            break
        else:
             # Skip other fields
            if wire == 0:
                _, pos = _read_varint(serialized, pos)
            elif wire == 2:
                size, pos = _read_varint(serialized, pos)
                pos += size
            elif wire == 5:
                pos += 4
            elif wire == 1:
                pos += 8
            else:
                break
    return out

def _decode_example_py(parsed):
    """Decode parsed features into numpy arrays."""
    # Decode images (CHW -> HWC)
    agent_img = np.frombuffer(parsed['observation/agentview_image'], dtype=np.uint8).reshape(3, 224, 224)
    agent_img = np.transpose(agent_img, (1, 2, 0)) # (224, 224, 3)
    
    eye_img = np.frombuffer(parsed['observation/eyeinhand_image'], dtype=np.uint8).reshape(3, 224, 224)
    eye_img = np.transpose(eye_img, (1, 2, 0))
    
    # Normalize images to [-1, 1]
    agent_img = agent_img.astype(np.float32) / 255.0 * 2.0 - 1.0
    eye_img = eye_img.astype(np.float32) / 255.0 * 2.0 - 1.0
    
    # Decode tracks
    mesh_agent = np.frombuffer(parsed['observation/mesh/vertex_agentview_2d_seq'], dtype=np.float32).reshape(16, 7, 2)
    mesh_eye = np.frombuffer(parsed['observation/mesh/vertex_eyeinhand_2d_seq'], dtype=np.float32).reshape(16, 7, 2)
    uniform_eye = np.frombuffer(parsed['observation/uniform_tracks/eyeinhand'], dtype=np.float32).reshape(16, 25, 2)

    # Normalize tracks from [0, 224] to [-1, 1]
    mesh_agent = mesh_agent / 224.0 * 2.0 - 1.0
    mesh_eye = mesh_eye / 224.0 * 2.0 - 1.0
    uniform_eye = uniform_eye / 224.0 * 2.0 - 1.0
    
    # Prepare inputs (t=0) - 78-dim
    vertex_agentview = mesh_agent[0].flatten() # (14,)
    vertex_eyeinhand = mesh_eye[0].flatten()   # (14,)
    uniform_eyeinhand = uniform_eye[0].flatten() # (50,)
    input_state = np.concatenate([vertex_agentview, vertex_eyeinhand, uniform_eyeinhand], axis=0)
    
    # Output tracks (t=0..15) - 78-dim
    v_agent = mesh_agent.reshape(16, -1)  # (16, 14)
    v_eye = mesh_eye.reshape(16, -1)      # (16, 14)
    u_eye = uniform_eye.reshape(16, -1)   # (16, 50)
    tracks = np.concatenate([v_agent, v_eye, u_eye], axis=1)  # (16, 78)
    
    # Decode Actions if available
    actions = None
    if 'action' in parsed:
        actions = np.frombuffer(parsed['action'], dtype=np.float32)
        # Pad if 7-dim (common in DROID) to 14-dim (Pi0 expectation)
        if actions.shape == (7,):
             actions = np.concatenate([actions, np.zeros(7, dtype=np.float32)])
    else:
        # Dummy actions
        actions = np.zeros(14, dtype=np.float32)

    # Decode optional robot state metadata (stored for future use)
    joint_position = np.frombuffer(parsed.get('observation/joint_position', b''), dtype=np.float32)
    if joint_position.shape != (7,):
        joint_position = np.zeros(7, dtype=np.float32)
    gripper_position = np.frombuffer(parsed.get('observation/gripper_position', b''), dtype=np.float32)
    if gripper_position.shape != (1,):
        gripper_position = np.zeros(1, dtype=np.float32)

    # Decode instruction
    instruction = parsed.get('language_instruction', b'move the robot to manipulate objects').decode('utf-8')

    return {
        'agent_img': agent_img,
        'eye_img': eye_img,
        'input_state': input_state,
        'joint_position': joint_position,
        'gripper_position': gripper_position,
        'tracks': tracks,
        'actions': actions,
        'instruction': instruction
    }

def _iter_tfrecord_file(path):
    """Iterate over TFRecord file records."""
    try:
        with open(path, "rb") as f:
            while True:
                length_bytes = f.read(8)
                if not length_bytes or len(length_bytes) != 8:
                    break
                length = struct.unpack("<Q", length_bytes)[0]
                if len(f.read(4)) != 4: break # CRC
                data = f.read(length)
                if len(data) != length: break
                if len(f.read(4)) != 4: break # CRC
                yield data
    except Exception as e:
        print(f"Error reading TFRecord {path}: {e}")

# =============================================================================
# Dataloader
# =============================================================================

class DroidPi0PretrainLoader:
    def __init__(self, droid_data_path, data_config, batch_size=32, shuffle=True, seed=0, num_workers=8, model_config=None):
        self.batch_size = batch_size
        self._data_config = data_config
        self.model_config = model_config
        self.shuffle = shuffle
        self.seed = seed
        
        # Determine tokenizer based on model_config
        self.is_hybrid = False
        if model_config and hasattr(model_config, "track_dim") and hasattr(model_config, "track_horizon"):
             print("[DROID Loader] Using FASTHybridTokenizer for Pi0 Hybrid")
             self.tokenizer = _tokenizer.FASTHybridTokenizer(max_len=512)
             self.is_hybrid = True
        else:
             self.tokenizer = _tokenizer.PaligemmaTokenizer()


        self.files = []
        
        # Setup paths
        self.files = sorted(glob.glob(str(Path(droid_data_path) / "*.tfrecord")))
        if not self.files:
            self.files = sorted(glob.glob(str(Path(droid_data_path) / "**/*.tfrecord"), recursive=True))
            
        if not self.files:
            # Fallback: check if path itself is a tfrecord
            if Path(droid_data_path).is_file() and str(droid_data_path).endswith(".tfrecord"):
                 self.files = [str(droid_data_path)]
            else:
                 raise ValueError(f"No .tfrecord files found in {droid_data_path}")
            
        print(f"[DROID Loader] Found {len(self.files)} TFRecord files. Using Python-only loader.")
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.files)
            
        # Initialize file iterator
        self.file_idx = 0
        self.record_iter = None
        
        # Buffer for shuffling/batching
        self.buffer = []
        
        # Load norm stats
        try:
            assets_path = Path("/mnt/kevin/code/wmrl/howard-branch/code/openpi_fork/assets/droid_pretrain_rlds")
            print(f"[DROID Loader] Loading norm stats from {assets_path}")
            self.norm_stats = _normalize.load(assets_path)
        except Exception as e:
            print(f"[DROID Loader] Warning: Could not load norm stats: {e}")
            self.norm_stats = None

    def data_config(self):
        return self._data_config

    def _fill_buffer(self, min_size):
        """Fill buffer with records from files."""
        # Limit buffer size to avoid memory issues (e.g. 500 records)
        MAX_BUFFER = 500
        
        attempts = 0
        while len(self.buffer) < min_size and attempts < len(self.files) + 1:
            if self.record_iter is None:
                if self.file_idx >= len(self.files):
                    # Reset epoch
                    if self.shuffle:
                        random.shuffle(self.files)
                    self.file_idx = 0
                    
                self.record_iter = _iter_tfrecord_file(self.files[self.file_idx])
                self.file_idx += 1
                attempts += 1
                
            try:
                # Read chunks from file
                count = 0
                while len(self.buffer) < MAX_BUFFER:
                    record = next(self.record_iter)
                    self.buffer.append(record)
                    count += 1
                    if count >= 100: # Yield control to check buffer size
                        break
            except StopIteration:
                self.record_iter = None
                
    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        # Ensure we have enough data for a batch
        self._fill_buffer(self.batch_size)
        
        # If buffer is still empty after trying to fill (e.g. empty files), try again
        if not self.buffer:
             # Should be handled by _fill_buffer loop unless no files
             raise StopIteration
        
        # Create batch
        batch_records = []
        for _ in range(self.batch_size):
            # If buffer ran out (unlikely with _fill_buffer logic unless EOF edge case)
            if not self.buffer: 
                self._fill_buffer(self.batch_size)
            if self.buffer:
                batch_records.append(self.buffer.pop(0))
            else:
                break # Partial batch if absolutely no data left? Or just use what we have?
                # Drop remainder
        
        if len(batch_records) < self.batch_size:
            # Drop remainder
            # Refill buffer for next call and recurse? Or just skip?
            # For simplicity, if we can't get a full batch, we might be at end of epoch or files broken.
            # But _fill_buffer resets epoch.
            # So this only happens if dataset is empty or corrupted.
            if not batch_records:
                raise StopIteration
            # If partial, maybe return it?
            # But standard is drop_remainder=True usually.
            pass 
            
        # Parse and stack
        batch_data = {
            'agent_img': [],
            'eye_img': [],
            'input_state': [],
            'joint_position': [],
            'gripper_position': [],
            'tracks': [],
            'actions': [],
            'instruction': []
        }
        
        for rec in batch_records:
            try:
                parsed = _parse_example_py(rec)
                decoded = _decode_example_py(parsed)
                for k, v in decoded.items():
                    if k in batch_data:
                        batch_data[k].append(v)
            except Exception as e:
                print(f"Error parsing record: {e}")
                continue

        if not batch_data['agent_img']:
            raise StopIteration
            
        # Stack into arrays
        # Ensure all lists have same length (in case of parse error)
        min_len = min(len(v) for v in batch_data.values())
        batch = {k: np.stack(v[:min_len]) for k, v in batch_data.items() if k != 'instruction'}
        actual_bs = min_len
        
        # Handle instructions separately (list of strings, not numpy array)
        instructions = batch_data['instruction']
        # If batch_data values were popped/appended, they are lists.
        # But we did min_len slicing for numpy arrays. We should do same for instructions.
        instructions = instructions[:actual_bs]
        
        # Apply normalization from norm_stats (if available)
        # We need to normalize 'input_state' (using 'state' stats) and 'tracks' (using 'actions' stats)
        # AND 'actions' (using 'actions' stats, but only first 7 dims usually matches if tracks reuse same stats?)
        # Wait, 'actions' stats usually for robot actions.
        # Tracks might use different normalization or scaled to image size.
        # Tracks are already normalized to [-1, 1] in _decode_example_py.
        # So we should SKIP normalization for tracks if they are already [-1, 1].
        
        if self.norm_stats is not None:
            stats = self.norm_stats
            
            # Helper to normalize: (x - mean) / (std + 1e-6)
            def normalize(x, s):
                return (x - s.mean) / (s.std + 1e-6)
            
            # Normalize inputs if not already normalized
            # input_state contains tracks (pre-normalized) so DO NOT normalize it again with robot state stats!
            # BUT input_state also contains robot state (if we used it).
            # Here input_state IS flattened tracks. So it is already [-1, 1].
            # So we SKIP 'input_state' normalization for DROID tracks.
            
            # Normalize 'actions' (robot actions)
            # The 'actions' field we decoded is raw robot actions? No, usually decoded from 'action' field.
            # DROID 'action' is usually unnormalized joint/gripper or end-effector.
            # So we SHOULD normalize actions.
            
            if 'actions' in stats:
                # Normalize actions
                # batch['actions'] is (B, 14)
                # stats['actions'] is (14,)
                # But stats might be 7-dim or 14-dim.
                # If stats is 14-dim (e.g. 7 arm + 1 gripper + ...), we use it.
                if 'actions' in batch and batch['actions'].shape[-1] == stats['actions'].mean.shape[-1]:
                     batch['actions'] = normalize(batch['actions'], stats['actions'])

        # Tokenize instructions
        # tokenizer.tokenize usually handles single string. For batch, we map.
        # We do this AFTER normalization if actions/states are used in tokens!
        # Pi0FAST uses discretized states/actions in prompt.
        # So we must tokenize AFTER normalization.
        
        tokenized = []
        if self.is_hybrid:
             for i, instr in enumerate(instructions):
                 # q_points are in input_state
                 # They are already normalized to [-1, 1] in _decode_example_py
                 q_points = batch['input_state'][i]
                 
                 # joint_state is dummy
                 joint_state = np.zeros(14, dtype=np.float32)
                 
                 # Expand action to (1, 14) since we only have current step action
                 # Actions are now normalized above
                 actions = batch['actions'][i][None, :]
                 
                 t, m, ar, l = self.tokenizer.tokenize_hybrid(instr, joint_state, q_points, actions)
                 tokenized.append((t, m, ar, l))
             
             tokens_list, mask_list, ar_list, loss_list = zip(*tokenized)
             batch_tokens = np.stack(tokens_list)
             batch_mask = np.stack(mask_list)
             batch_ar = np.stack(ar_list)
             batch_loss = np.stack(loss_list)
        else:
             for instr in instructions:
                  tokens, mask = self.tokenizer.tokenize(instr)
                  tokenized.append((tokens, mask))
                  
             # unzip
             tokens_list, mask_list = zip(*tokenized)
             batch_tokens = np.stack(tokens_list)
             batch_mask = np.stack(mask_list)
             batch_ar = None
             batch_loss = None

        # Convert to JAX/Model format

        # Convert to JAX/Model format
        obs = _model.Observation(
            images={
                'base_0_rgb': jnp.array(batch['agent_img']),
                'left_wrist_0_rgb': jnp.array(batch['eye_img']),
                'right_wrist_0_rgb': jnp.zeros_like(batch['agent_img']) - 1.0, # Dummy black image
            },
            image_masks={
                'base_0_rgb': jnp.ones((actual_bs,), dtype=bool),
                'left_wrist_0_rgb': jnp.ones((actual_bs,), dtype=bool),
                'right_wrist_0_rgb': jnp.zeros((actual_bs,), dtype=bool),
            },
            state=jnp.array(batch['input_state']),
            tokenized_prompt=jnp.array(batch_tokens),
            tokenized_prompt_mask=jnp.array(batch_mask),
            token_ar_mask=jnp.array(batch_ar) if batch_ar is not None else None,
            token_loss_mask=jnp.array(batch_loss) if batch_loss is not None else None,
        )
        
        tracks = jnp.array(batch['tracks'])
        
        # For Hybrid, we return actions too (tuple or dict?)
        # DataLoader expects (Observation, Actions) tuple.
        # Pi0FASTHybrid.compute_loss expects 'actions' arg to be a dict or Actions object.
        # It handles dict: actions={'actions': ..., 'track_actions': ...}
        
        if self.is_hybrid:
             actions_dict = {
                 'actions': jnp.array(batch['actions'][:, None, :]), # (B, 1, 14) - Action Head
                 'track_actions': tracks, # (B, H, 78) - Track Head
             }
             return obs, actions_dict
        else:
             return obs, tracks

def create_droid_pi0_pretrain_loader(config, **kwargs):
    # Get droid_data_path from config
    if hasattr(config, 'data') and hasattr(config.data, 'droid_data_path'):
        droid_path = config.data.droid_data_path
    elif hasattr(config.data, 'base_config') and hasattr(config.data.base_config, 'droid_data_path'):
        droid_path = config.data.base_config.droid_data_path
    elif hasattr(config, 'droid_data_path'):
        droid_path = config.droid_data_path
    else:
        droid_path = ''
    
    if not droid_path:
        # Default to new tfrecord path if not specified
        default_path = '/mnt/kevin/data/droid_preprocessed_tfrecord'
        if os.path.exists(default_path):
            droid_path = default_path
            print(f"[DROID Loader] Using default TFRecord path: {droid_path}")
        else:
            raise ValueError("droid_data_path not found in config and default path missing")
    
    process_count = jax.process_count()
    if config.batch_size % process_count != 0:
        raise ValueError(
            f"Global batch size {config.batch_size} must be divisible by process count {process_count}."
        )
    local_batch_size = config.batch_size // process_count

    return DroidPi0PretrainLoader(
        droid_data_path=droid_path,
        data_config=config.data.create(config.assets_dirs, config.model),
        batch_size=local_batch_size,
        shuffle=True,
        seed=config.seed,
        model_config=config.model,
    )
