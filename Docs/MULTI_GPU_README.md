# Training with Accelerate

This project uses Hugging Face Accelerate for distributed training. **Accelerate is required** - always use `accelerate launch` to run training.

## Installation

```bash
pip install accelerate
```

## Configuration (Optional)

### Automatic configuration

Run the configuration wizard:

```bash
accelerate config
```

### Manual configuration

Use the provided `accelerate_config.yaml` file. Adjust `num_processes` according to the number of available GPUs.

## Running Training

**Important:** Always use `accelerate launch`, never run `python train.py` directly.

### Single GPU

```bash
accelerate launch --num_processes 1 train.py Configs/config.yml
```

### Multi-GPU (Recommended for Faster Training)

```bash
accelerate launch --config_file accelerate_config.yaml train.py Configs/config.yml
```

Or without configuration file:

```bash
accelerate launch --num_processes 2 train.py Configs/config.yml
```

### Using the provided script

```bash
./run_multi_gpu.sh
```

## Key Implementation Details

### train.py

- **Accelerator initialization**: Created at the beginning of `main()` (required)
- **Automatic distribution**: Model, optimizer, scheduler and dataloaders are automatically distributed
- **Logging on main process only**: TensorBoard and files are written only from the main process

### trainer.py

- **Backward pass**: Always uses `accelerator.backward()` 
- **Automatic data movement**: Data is automatically moved to correct GPU by Accelerator
- **Checkpoint saving**: Uses `accelerator.unwrap_model()` for correct model saving
- **Required parameter**: `accelerator` must be passed to Trainer constructor

## Benefits of Accelerate

1. **Clean code**: No conditional logic for single vs multi-GPU
2. **Flexibility**: Same code works on CPU, single GPU, multi GPU, TPU
3. **Automatic synchronization**: Gradients and batch normalization are automatically synchronized
4. **Mixed precision**: Easy FP16/BF16 training for faster execution
5. **Gradient accumulation**: Simple implementation for larger effective batch size

## Pokročilé nastavení

### Mixed Precision Training (FP16)

V `accelerate_config.yaml` změňte:

```yaml
mixed_precision: 'fp16'
```

### Gradient Accumulation

V `train.py` při inicializaci Accelerator:

```python
accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=4,  # efektivní batch size = batch_size * 4
)
```

### Kontrola dostupných GPU

```bash
nvidia-smi
```

### Single GPU mode

```bash
accelerate launch --num_processes 1 train.py Configs/config.yml
```

## Notes

- Effective batch size = `batch_size` × `num_processes` × `gradient_accumulation_steps`
- Learning rate may need adjustment when changing number of GPUs
- Checkpoints are compatible between single-GPU and multi-GPU training
- F0 precompute runs on all processes, but output is shared (disk cache)
- **Accelerate is always required** - do not use `python train.py` directly

## Troubleshooting

### NCCL timeout

Pokud dochází k timeoutům při inicializaci:

```bash
export NCCL_TIMEOUT=3600
accelerate launch ...
```

### Out of Memory

Snižte `batch_size` v `Configs/config.yml` nebo použijte gradient accumulation:

```python
gradient_accumulation_steps=2  # poloviční batch na GPU, stejný efekt
```

### Nerovnoměrné GPU využití

Ujistěte se, že dataset je dělitelný počtem GPU nebo použijte `drop_last=True` v dataloaderu.
