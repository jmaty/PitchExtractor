# Implementace Multi-GPU trénování s Accelerate

## Shrnutí změn

Projekt byl upraven pro použití knihovny **Hugging Face Accelerate** pro trénování. Tato implementace umožňuje:

✅ **Distribuované trénování** na 1 nebo více GPU  
✅ **Čistý kód** - žádné podmínky pro single vs multi-GPU  
✅ **Mixed precision training** - FP16/BF16 pro rychlejší běh  
✅ **Jednoduché API** - Accelerate je vždy povinné, vždy se používá

---

## Soubory upravené

### 1. `train.py`

**Změny:**

- Import `Accelerator` z accelerate
- Inicializace `Accelerator` na začátku `main()`
- Všechny komponenty (model, optimizer, dataloaders, scheduler) jsou připraveny pomocí `accelerator.prepare()`
- Logging a ukládání checkpointů pouze na main procesu
- Device se automaticky získává z `accelerator.device`

### 2. `trainer.py`

**Změny:**

- Import `Accelerator` pro type hints
- Přidán povinný parametr `accelerator` do konstruktoru
- `run()`: Vždy používá `accelerator.backward()` (žádné podmínky)
- `run()` a `eval_epoch()`: Data se nepřesouvají manuálně na device (dělá to accelerator automaticky)
- `save_checkpoint()`: Vždy používá `accelerator.unwrap_model()` a ukládá pouze na main procesu
- `load_checkpoint()`: Vždy načítá do unwrapped modelu
- **Odstraněny všechny podmínky** pro `accelerator is None`

---

## Nové soubory

**Konfigurační soubory:**

1. **`accelerate_config.yaml`** - Základní konfigurace pro 2 GPU
2. **`requirements.txt`** - Všechny závislosti včetně Accelerate (přejmenováno z requirements_multigpu.txt)
3. **`run_multi_gpu.sh`** - Bash skript pro snadné spuštění

**Dokumentace:**
4. **`MULTI_GPU_README.md`** - Kompletní návod k použití
5. **`RYCHLY_PRUVODCE_CZ.md`** - Rychlý průvodce (česky)
6. **`ACCELERATE_CONFIG_EXAMPLES.md`** - Příklady konfigurací
7. **`IMPLEMENTATION_SUMMARY.md`** - Tento soubor
8. **`CHANGELOG.md`** - Přehled změn

**Testovací skripty:**
9. **`test_setup.py`** - Testovací skript pro ověření instalace

---

## Rychlý start

### 1. Instalace

```bash
pip install accelerate
```

### 2. Konfigurace (volitelné)

```bash
# Interaktivní průvodce
accelerate config

# Nebo použijte připravený soubor
# Upravte num_processes v accelerate_config.yaml podle počtu GPU
```

### 3. Spuštění

**Důležité:** Vždy používejte `accelerate launch`, nikdy ne `python train.py` přímo!

#### Single GPU

```bash
accelerate launch --num_processes 1 train.py Configs/config.yml
```

#### Multi-GPU (doporučeno pro rychlejší trénink)

```bash
# Pomocí bash skriptu
./run_multi_gpu.sh

# Nebo přímo
accelerate launch --num_processes 2 train.py Configs/config.yml
```

---

## Výhody implementace

### 🚀 Výkon

- **Lineární škálování**: 2 GPU ≈ 2× rychleji, 4 GPU ≈ 4× rychleji
- **Mixed precision**: FP16 může urychlit trénink o 50-100%
- **Automatická optimalizace**: NCCL backend pro efektivní komunikaci mezi GPU

### 🛠️ Flexibilita

- Stejný kód funguje na: **CPU, single GPU, multi-GPU, TPU**
- Snadné přepínání mezi různými režimy změnou `--num_processes`
- Podpora gradient accumulation bez změny kódu

### 🔧 Jednoduchost

- **Čistý kód bez podmínek** - Accelerate se vždy používá
- Není potřeba ručně řešit `DistributedDataParallel`
- Automatická synchronizace gradientů a batch norm
- Žádné `if accelerator is None` v kódu

### ✅ Robustnost

- Automatické řešení edge cases (např. různé batch sizes na GPU)
- Správné uložení a načítání checkpointů
- Podpora pro resume training
- Povinná závislost = žádné runtime chyby kvůli chybějícímu acceleratoru

---

## Technické detaily

### Jak to funguje

1. **Inicializace**: `Accelerator` detekuje dostupné GPU a vytvoří procesy
2. **Příprava**: `accelerator.prepare()` zabalí model, optimizer, atd. pro distribuované počítání
3. **Forward pass**: Každé GPU zpracuje část batche
4. **Backward pass**: Gradienty se automaticky synchronizují mezi GPU
5. **Optimizer step**: Váhy se aktualizují synchronně na všech GPU

### Synchronizace

- **Gradienty**: Automaticky průměrovány přes všechna GPU (all-reduce)
- **Batch normalization**: Statistiky synchronizovány mezi GPU
- **Checkpointy**: Ukládány pouze z main procesu, ale načítány do všech

### Efektivní batch size

```
Efektivní batch size = batch_size × num_GPUs × gradient_accumulation_steps
```

**Příklad:**

- `batch_size` v config.yml: 32
- 4 GPU (`num_processes: 4`)
- Efektivní batch size: **128**

Pokud je potřeba ještě větší batch size, použijte gradient accumulation:

```python
accelerator = Accelerator(gradient_accumulation_steps=2)
# Efektivní batch size: 32 × 4 × 2 = 256
```

---

## Doporučení

### Pro různé HW konfigurace

| GPU                | Doporučená konfigurace                  |
|--------------------|----------------------------------------|
| 1× V100/A100       | Single GPU, mixed_precision: 'fp16'    |
| 2-4× V100/A100     | Multi-GPU, mixed_precision: 'fp16'     |
| 2-4× RTX 3090/4090 | Multi-GPU, mixed_precision: 'bf16'     |
| 8+ GPU             | Multi-GPU, gradient_accumulation: 2-4  |

### Learning rate scaling

Při změně počtu GPU může být nutné upravit learning rate:

```
nový_lr = původní_lr × √(počet_GPU)
```

nebo použijte linear scaling:

```
nový_lr = původní_lr × počet_GPU
```

### Optimalizace batch size

1. **Začněte s malým batch size** (např. 16) na 1 GPU
2. **Postupně zvyšujte** dokud nedojde k OOM
3. **Optimální hodnota** je obvykle ~80% VRAM capacity
4. **Pro multi-GPU**: Použijte stejný batch size per GPU

---

## Troubleshooting

### "NCCL timeout" nebo "NCCL error"

```bash
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
```

### Out of Memory (OOM)

1. Snižte `batch_size` v config.yml
2. Použijte gradient accumulation místo větších batches
3. Zapněte mixed precision ('fp16' nebo 'bf16')

### Nerovnoměrné vytížení GPU

- Dataset by měl být dělitelný počtem GPU
- Zkontrolujte, že dataloaders mají `drop_last=True` pro trénink

### Pomalý první epoch

- První epoch zahrnuje kompilaci CUDA kernelů (JIT)
- Následující epochy budou rychlejší

---

## Testování

### Verifikace funkčnosti

```bash
# 1. Test na CPU (bez GPU)
accelerate launch --cpu train.py Configs/config.yml

# 2. Test na 1 GPU
accelerate launch --num_processes 1 train.py Configs/config.yml

# 3. Test na 2 GPU
accelerate launch --num_processes 2 train.py Configs/config.yml
```

### Monitoring

```bash
# Sledování GPU využití
watch -n 0.5 nvidia-smi

# TensorBoard (stejné jako předtím)
tensorboard --logdir Checkpoint/tensorboard
```

---

## Kompatibilita

### Změny ve způsobu spuštění

⚠️ **Důležitá změna:** Accelerate je nyní povinné  
❌ Nelze používat: `python train.py Configs/config.yml`  
✅ Vždy používejte: `accelerate launch --num_processes N train.py Configs/config.yml`  
✅ Checkpointy jsou kompatibilní mezi 1 GPU a multi-GPU  
✅ Není nutné měnit existující konfigurační soubory

### Forward kompatibilita

✅ Připraveno pro budoucí rozšíření (DeepSpeed, FSDP)  
✅ Snadná migrace na větší clustery  
✅ Podpora pro Tensor Parallelism (s malými úpravami)

---

## Další kroky (volitelné rozšíření)

### 1. DeepSpeed integrace

Pro **velmi velké modely** (>1B parametrů):

```python
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
```

### 2. Gradient checkpointing

Pro **úsporu paměti**:
```python
model.gradient_checkpointing_enable()
```

### 3. Profiling

Pro **optimalizaci výkonu**:
```python
with accelerator.profile() as prof:
    train_epoch()
```

---

## Reference

- [Accelerate dokumentace](https://huggingface.co/docs/accelerate)
- [Multi-GPU best practices](https://huggingface.co/docs/accelerate/usage_guides/training)
- [Mixed precision training](https://huggingface.co/docs/accelerate/usage_guides/mixed_precision)

---

## Kontakt a podpora

Pokud narazíte na problémy:

1. Zkontrolujte `MULTI_GPU_README.md` pro běžné problémy
2. Zkuste různé konfigurace z `ACCELERATE_CONFIG_EXAMPLES.md`
3. Zapněte debug mode: `export NCCL_DEBUG=INFO`

**Hodnota 71.994** z původního dotazu může být:

- Learning rate (`lr: 0.0003` → můžete experimentovat s vyšší hodnotou při více GPU)
- F0 hodnota v Hz (typická hodnota pro mužský hlas)
- Loss hodnota během tréninku

Pro úpravu learning rate při multi-GPU:

```yaml
# V Configs/config.yml
optimizer_params:
  lr: 0.0006  # 2× původní hodnota pro 2 GPU
```
