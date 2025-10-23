# Rychlý průvodce Accelerate tréninku (Česky)

## 🎯 O projektu

Projekt používá **Hugging Face Accelerate** pro trénování na GPU. Accelerate je **povinná závislost** a používá se vždy, ať už máte 1 GPU nebo více.

## 📦 Instalace

```bash
pip install accelerate
```

## 🚀 Jak spustit

### Single GPU

```bash
accelerate launch --num_processes 1 train.py Configs/config.yml
```

### Multi-GPU (2 GPU)

```bash
accelerate launch --num_processes 2 train.py Configs/config.yml
```

### Pomocí skriptu

```bash
./run_multi_gpu.sh
```

**Důležité:** Vždy používejte `accelerate launch`, nikdy ne `python train.py` přímo!

## ⚙️ Konfigurace

V souboru `run_multi_gpu.sh` upravte:

```bash
NUM_GPUS=2  # změňte na počet GPU, které chcete použít
```

## 🧪 Test instalace

```bash
python test_setup.py
```

Tento skript zkontroluje:

- ✅ Zda máte nainstalované všechny potřebné knihovny
- ✅ Kolik GPU máte k dispozici
- ✅ Zda je konfigurace Accelerate správná
- ✅ Zda se model načte bez chyb

## 📈 Očekávané zrychlení

| Počet GPU | Zrychlení |
|-----------|-----------|
| 1 GPU     | 1× (baseline) |
| 2 GPU     | ~1.8× |
| 4 GPU     | ~3.5× |
| 8 GPU     | ~6-7× |

## 🔧 Pokročilé možnosti

### Mixed Precision (FP16) - rychlejší trénink

```bash
accelerate launch --num_processes 2 --mixed_precision fp16 train.py Configs/config.yml
```

### Použití specifických GPU

```bash
CUDA_VISIBLE_DEVICES=0,2 accelerate launch --num_processes 2 train.py Configs/config.yml
```

### Gradient accumulation - větší efektivní batch size

Upravte v `train.py`:

```python
accelerator = Accelerator(
    gradient_accumulation_steps=2  # efektivní batch = batch_size × 2 × num_GPUs
)
```

## 📊 Efektivní batch size

```
Efektivní batch size = batch_size × počet_GPU × gradient_accumulation_steps
```

**Příklad:**

- `batch_size` v `Configs/config.yml`: 32
- Počet GPU: 2
- Gradient accumulation: 1 (default)
- **Efektivní batch size: 64**

## 🛠️ Řešení problémů

### "Out of Memory"

1. Snižte `batch_size` v `Configs/config.yml`
2. Nebo použijte mixed precision: `--mixed_precision fp16`

### "NCCL timeout"

```bash
export NCCL_TIMEOUT=3600
./run_multi_gpu.sh
```

### Model není rychlejší na více GPU

- Zkontrolujte, že všechny GPU jsou stejně vytížené: `watch -n 0.5 nvidia-smi`
- Možná je dataset příliš malý nebo batch size příliš malý

## 📚 Podrobná dokumentace

- **`MULTI_GPU_README.md`** - Kompletní návod v angličtině
- **`ACCELERATE_CONFIG_EXAMPLES.md`** - Příklady různých konfigurací
- **`IMPLEMENTATION_SUMMARY.md`** - Technické detaily implementace
- **`CHANGELOG.md`** - Co se změnilo

## 💡 Tipy

### Pro začátečníky

1. Nejprve spusťte `python test_setup.py`
2. Zkuste single GPU: `accelerate launch --num_processes 1 train.py Configs/config.yml`
3. Pak zkuste 2 GPU: `accelerate launch --num_processes 2 train.py Configs/config.yml`

### Pro pokročilé

- Použijte mixed precision pro maximální rychlost
- Experimentujte s batch size pro optimální využití VRAM
- Sledujte TensorBoard pro porovnání tréninků: `tensorboard --logdir Checkpoint/tensorboard`

### Pro výzkumníky

- Checkpointy jsou kompatibilní mezi 1 GPU a multi-GPU
- Learning rate možná bude potřeba upravit při změně počtu GPU
- F0 precompute běží paralelně, ale výsledky jsou sdílené (cache na disku)

## 🎓 Příklad workflow

```bash
# 1. Test instalace
python test_setup.py

# 2. Precompute F0 (stačí jednou) - použijte single GPU
accelerate launch --num_processes 1 train.py Configs/config.yml --precompute_f0 True

# 3. Trénink na 2 GPU
accelerate launch --num_processes 2 train.py Configs/config.yml

# 4. Sledování progress
tensorboard --logdir Checkpoint/tensorboard

# 5. Monitoring GPU
watch -n 0.5 nvidia-smi
```

## ⚡ Rychlé příkazy

```bash
# Ukázat dostupné GPU
nvidia-smi

# Trénink na všech dostupných GPU
accelerate launch --num_processes $(nvidia-smi -L | wc -l) train.py Configs/config.yml

# Trénink s maximální optimalizací (FP16)
accelerate launch --num_processes 2 --mixed_precision fp16 train.py Configs/config.yml

# Debug režim (verbose output)
NCCL_DEBUG=INFO accelerate launch --num_processes 2 train.py Configs/config.yml
```

## 📞 Potřebujete pomoc?

1. ✅ Spusťte `python test_setup.py` - často odhalí problém
2. 📖 Přečtěte si `MULTI_GPU_README.md` - obsahuje řešení běžných problémů
3. 🔍 Zkontrolujte `ACCELERATE_CONFIG_EXAMPLES.md` - možná použijete špatnou konfiguraci

## ✨ Fun fact

Hodnota **71.994 Hz** z původního dotazu je typická F0 (pitch) hodnota pro mužský hlas.

- Mužský hlas: ~85-180 Hz
- Ženský hlas: ~165-255 Hz
- Hodnota 71.994 Hz je na spodní hranici, možná bas zpěvák nebo chyba v extrakci 😊

---

**Vytvořeno:** 23. října 2025  
**Verze:** 2.0.0 - Multi-GPU Support
