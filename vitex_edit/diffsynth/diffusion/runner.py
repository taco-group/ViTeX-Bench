import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    # Exclude VAE from DeepSpeed ZeRO-3 wrapping to avoid compatibility issues
    # Store VAE outside the module tree so DeepSpeed doesn't touch it
    vae_module = getattr(model.pipe, 'vae', None)
    if vae_module is not None:
        del model.pipe._modules['vae']
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if vae_module is not None:
        vae_module.to(accelerator.device)
        # Store VAE as a non-module attribute so pipeline code can still use pipe.vae
        pipe = model.module.pipe if hasattr(model, 'module') else model.pipe
        # Use object.__setattr__ to bypass nn.Module's __setattr__ which would register it as a submodule
        object.__setattr__(pipe, 'vae', vae_module)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    # Training log file
    log_path = os.path.join(model_logger.output_path, "training_log.txt")
    if accelerator.is_main_process:
        os.makedirs(model_logger.output_path, exist_ok=True)
        log_file = open(log_path, "a")
        log_file.write(f"Training started. Epochs: {num_epochs}, LR: {learning_rate}, Steps/epoch: {len(dataloader)}\n")
        log_file.flush()
    else:
        log_file = None

    total_target = num_epochs * len(dataloader)
    reached_target = False
    for epoch_id in range(num_epochs):
        if reached_target:
            break
        progress = tqdm(
            total=total_target,
            initial=model_logger.num_steps,
            desc=f"Epoch {epoch_id+1}/{num_epochs}",
        )
        for step_id, data in enumerate(dataloader):
            if model_logger.num_steps >= total_target:
                reached_target = True
                break
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()

                # Log loss
                loss_val = loss.item()
                progress.update(1)
                progress.set_postfix(loss=f"{loss_val:.4f}")
                if accelerator.is_main_process and log_file is not None and (model_logger.num_steps % 10 == 0 or model_logger.num_steps <= 5):
                    log_file.write(f"epoch={epoch_id+1} step={model_logger.num_steps} loss={loss_val:.6f}\n")
                    log_file.flush()
        progress.close()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if accelerator.is_main_process and log_file is not None:
            log_file.write(f"Epoch {epoch_id+1} completed. Checkpoint saved.\n")
            log_file.flush()
    model_logger.on_training_end(accelerator, model, save_steps)
    if log_file is not None:
        log_file.close()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
