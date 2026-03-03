import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import os

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# Import track validation if available
try:
    import openpi.training.track_validation as track_validation
    TRACK_VALIDATION_AVAILABLE = True
except ImportError:
    TRACK_VALIDATION_AVAILABLE = False


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        entity = os.environ.get('WANDB_ENTITY', None)
        wandb.init(entity=entity, id=run_id, resume="must", project=config.project_name)
    else:
        # Default to None if not set, allowing wandb to use the user's default entity
        entity = os.environ.get('WANDB_ENTITY', None)
        wandb.init(
            entity=entity,
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)

    # For track models, we expect partial loading (base Pi0 weights without track-specific layers)
    # So we skip the equality check and just log what was loaded
    loaded_keys = set()
    missing_keys = set()

    flat_loaded = traverse_util.flatten_dict(loaded_params)
    flat_shape = traverse_util.flatten_dict(params_shape)

    for key in flat_shape:
        if key in flat_loaded and not isinstance(flat_loaded[key], jax.ShapeDtypeStruct):
            loaded_keys.add(key[0] if key else 'root')
        else:
            missing_keys.add(key[0] if key else 'root')

    # Log loading status
    logging.info(f"Loaded {len(loaded_keys)} parameter groups from checkpoint")
    if missing_keys:
        track_keys = {'query_proj', 'track_in_proj', 'track_out_proj', 'track_time_mlp_in', 'track_time_mlp_out'}
        missing_track = missing_keys & track_keys
        missing_other = missing_keys - track_keys

        if missing_track:
            logging.info(f"Track-specific parameters will be randomly initialized: {missing_track}")
        if missing_other:
            logging.warning(f"Other parameters missing (will be initialized): {missing_other}")

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in flat_loaded.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        freeze_filter = config.freeze_filter if config.freeze_filter is not None else nnx.Nothing
        params = nnx_utils.state_map(params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[Any, Any],  # Supports array actions and hybrid action dicts.
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: Any, actions: Any
    ):
        if model.__class__.__name__ == "Pi0FASTHybrid":
            total_loss, metrics = model.compute_loss(rng, observation, actions, train=True, return_metrics=True)
            return jnp.mean(total_loss), metrics

        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        mean_loss = jnp.mean(chunked_loss)
        return mean_loss, {
            "loss_total": mean_loss,
            "loss_action": jnp.array(0.0, dtype=mean_loss.dtype),
            "loss_track": jnp.array(0.0, dtype=mean_loss.dtype),
        }

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)

    # Check for gradient explosion
    grad_norm = optax.global_norm(grads)

    # Define thresholds for skipping updates.
    # Early training has high loss by design, so only enable loss-based skipping later.
    GRAD_SKIP_THRESHOLD = 2000.0
    LOSS_SKIP_THRESHOLD = 500.0
    LOSS_FILTER_START_STEP = 10_000

    should_skip = (grad_norm > GRAD_SKIP_THRESHOLD) | (
        (state.step >= LOSS_FILTER_START_STEP) & (loss > LOSS_SKIP_THRESHOLD)
    )

    def skip_update(operand):
        grad_norm, loss, params, grads, opt_state = operand
        return params, opt_state, 1.0  # 1.0 means skipped

    def apply_update(operand):
        grad_norm, loss, params, grads, opt_state = operand
        updates, new_opt_state = state.tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, 0.0  # 0.0 means not skipped

    new_params, new_opt_state, skipped = jax.lax.cond(
        should_skip,
        skip_update,
        apply_update,
        (grad_norm, loss, params, grads, state.opt_state)
    )

    # Diagnostic: Log the actual params norm difference to confirm update happened (or not)
    # We can't log directly inside JIT easily, but we can compute it and return it in info
    # Compute diff norm: norm(new_params - params)
    # Since new_params and params are nnx.State (pytrees), we need to subtract leaves

    # helper to subtract states
    def sub_states(s1, s2):
        return jax.tree.map(lambda x, y: x - y, s1, s2)

    diff_norm = optax.global_norm(sub_states(new_params, params))

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        # We also need to conditionally update EMA
        def update_ema(operand):
             ema_params, new_params = operand
             return jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, ema_params, new_params
            )

        def skip_ema(operand):
            ema_params, new_params = operand
            return ema_params

        new_ema_params = jax.lax.cond(
            should_skip,
            skip_ema,
            update_ema,
            (state.ema_params, new_params)
        )

        new_state = dataclasses.replace(new_state, ema_params=new_ema_params)

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "loss_total": loss_metrics["loss_total"],
        "loss_action": loss_metrics["loss_action"],
        "loss_track": loss_metrics["loss_track"],
        "grad_norm": grad_norm,
        "param_norm": optax.global_norm(kernel_params),
        "skipped": skipped,
        "update_diff_norm": diff_norm,
        # Log mean of the first few elements of state as a proxy "ID" for the batch
        # This helps identify which batch caused the spike (though it's not a true ID)
        "batch_checksum": jnp.mean(observation.state),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    num_processes = int(os.environ.get("JAX_PROCESS_COUNT", 1))

    # 只有当节点数大于 1 时，才初始化分布式集群
    if num_processes > 1:
        coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
        process_id = int(os.environ.get("JAX_PROCESS_ID", 0))

        print(f"Initializing JAX distributed: addr={coordinator_address}, num_nodes={num_processes}, rank={process_id}")
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id
        )

    init_logging()
    is_primary_process = jax.process_index() == 0
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Global device count: {jax.device_count()}") # 应该输出 16
    logging.info(f"Local device count: {jax.local_device_count()}") # 应该输出 8

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled and is_primary_process)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)

    # Create validation loader if we're doing track/vertex prediction
    val_loader = None
    if TRACK_VALIDATION_AVAILABLE and (hasattr(config.model, 'query_dim') or hasattr(config.model, 'track_dim') or hasattr(config.model, 'vertex_dim') or hasattr(config.model, 'vertex_input_dim') or hasattr(config.model, 'uniform_dim') or hasattr(config.model, 'proprioception_dim')):
        # This is a track/vertex model, create validation loader
        val_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            num_workers=config.num_workers,
            shuffle=False,
            split='val'  # Load validation split
        )
        logging.info("Created validation loader for track visualization")
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    validation_interval = int(os.environ.get('OPENPI_VALIDATION_INTERVAL', '1000'))
    visualize_tracks = os.environ.get('OPENPI_VISUALIZE_TRACKS', 'false').lower() == 'true'

    # Run validation at step 0 before training starts
    if val_loader is not None and TRACK_VALIDATION_AVAILABLE and visualize_tracks and start_step == 0:
        if is_primary_process:
            logging.info("Running initial validation at step 0")
        model = nnx.merge(train_state.model_def, train_state.params)
        model.eval()

        with sharding.set_mesh(mesh):
            val_metrics = track_validation.validate_and_visualize_tracks(
                model=model,
                val_data=val_loader,
                step=0,
                config=config,
                num_samples=int(os.environ.get('OPENPI_VISUALIZATION_SAMPLES', '5')),
                wandb_enabled=config.wandb_enabled and is_primary_process
            )

        if val_metrics and is_primary_process:
            val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logging.info(f"Initial validation: {val_str}")

        model.train()

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        # Logging
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            # Count skipped steps in this interval
            if "skipped" in reduced_info:
                # Sum instead of mean for skipped count (since it's 0 or 1 per step)
                skipped_count = int(sum(i["skipped"] for i in infos))
                reduced_info["skipped_count"] = skipped_count

                # If we skipped steps, log a warning and save batch checksums
                if skipped_count > 0:
                    logging.warning(f"Skipped {skipped_count} batches in last {config.log_interval} steps due to gradient explosion")

                    # Log the checksums of skipped batches to a file
                    with open("bad_batches.txt", "a") as f:
                        for idx, info in enumerate(infos):
                            if info["skipped"]:
                                # Calculate absolute step number
                                # step is the END of the interval (e.g., 100)
                                # infos has length 100
                                # idx 0 -> step 1
                                # idx 99 -> step 100
                                abs_step = step - len(infos) + idx + 1
                                checksum = info["batch_checksum"]
                                f.write(f"Step {abs_step}: Skipped due to grad spike. Batch checksum: {checksum}\n")

            # Remove batch_checksum from reduced_info to avoid cluttering logs
            if "batch_checksum" in reduced_info:
                del reduced_info["batch_checksum"]

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if is_primary_process:
                wandb.log(reduced_info, step=step)
            infos = []

        # Validation and visualization for track models
        if (step % validation_interval == 0 and step >= validation_interval and val_loader is not None
            and TRACK_VALIDATION_AVAILABLE and visualize_tracks):
            if is_primary_process:
                logging.info(f"Running validation at step {step}")

            # Get model from train state for validation
            model = nnx.merge(train_state.model_def, train_state.params)
            model.eval()

            # Run validation with visualization
            with sharding.set_mesh(mesh):
                val_metrics = track_validation.validate_and_visualize_tracks(
                    model=model,
                    val_data=val_loader,
                    step=step,
                    config=config,
                    num_samples=int(os.environ.get('OPENPI_VISUALIZATION_SAMPLES', '5')),
                    wandb_enabled=config.wandb_enabled and is_primary_process
                )

            if val_metrics and is_primary_process:
                val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                pbar.write(f"Validation at step {step}: {val_str}")

            model.train()

        batch = next(data_iter)

        # Removed test save at step 0 to avoid immediate memory spike

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
