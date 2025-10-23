# Příklady konfigurace pro různé scénáře

## 1. Základní multi-GPU (2 GPU, bez mixed precision)

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: 'no'
num_processes: 2
gpu_ids: all
use_cpu: false

## 2. Multi-GPU s FP16 (rychlejší trénování)

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: 'fp16'
num_processes: 4
gpu_ids: all
use_cpu: false

## 3. Multi-GPU s BF16 (pro Ampere+ GPU - A100, RTX 3090, RTX 4090)

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: 'bf16'
num_processes: 8
gpu_ids: all
use_cpu: false

## 4. Specifické GPU (např. pouze GPU 0 a 2)

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: 'no'
num_processes: 2
gpu_ids: '0,2'
use_cpu: false

## 5. Single GPU (pro testování)

compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: 'no'
num_processes: 1
use_cpu: false

## 6. CPU only (bez GPU)

compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: 'no'
num_processes: 1
use_cpu: true

---

## Jak vybrat správnou konfiguraci:

### Podle generace GPU:

- **Pascal (GTX 1080, P100)**: mixed_precision: 'no' (nemají Tensor Cores)
- **Volta/Turing (V100, RTX 2080)**: mixed_precision: 'fp16'
- **Ampere+ (A100, RTX 3090, RTX 4090)**: mixed_precision: 'bf16' (lepší stabilita)

### Podle počtu GPU:

- Nastavte `num_processes` = počet GPU, které chcete použít
- Pokud máte 4 GPU, ale chcete použít jen 2, nastavte `num_processes: 2` a `gpu_ids: '0,1'`

### Podle velikosti modelu a dat:

- Malý model + velká data: použijte více GPU pro rychlejší zpracování
- Velký model + malá data: možná stačí 1-2 GPU
- OOM problémy: použijte mixed_precision nebo gradient_accumulation_steps

### Optimální batch size:

Efektivní batch size = batch_size * num_processes * gradient_accumulation_steps

Příklad:

- batch_size v config.yml: 32
- num_processes: 4
- gradient_accumulation_steps: 2
- Efektivní batch size: 32 * 4 * 2 = 256

## Příklady spuštění:

```bash
# Základní multi-GPU
accelerate launch --num_processes 2 train.py Configs/config.yml

# S FP16 mixed precision
accelerate launch --num_processes 4 --mixed_precision fp16 train.py Configs/config.yml

# S konfiguračním souborem
accelerate launch --config_file accelerate_config.yaml train.py Configs/config.yml

# Specifické GPU
CUDA_VISIBLE_DEVICES=0,2 accelerate launch --num_processes 2 train.py Configs/config.yml
```
