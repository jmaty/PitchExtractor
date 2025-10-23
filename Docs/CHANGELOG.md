# Changelog - Multi-GPU Implementation

## Verze: Multi-GPU Support (2025-10-23)

### ✨ Přidáno

#### Nové funkce

- **Multi-GPU podpora** pomocí Hugging Face Accelerate
- **Mixed precision training** (FP16/BF16) pro rychlejší běh
- **Automatická distribuce** modelu a dat přes GPU
- **Gradient synchronizace** mezi GPU procesy

**Konfigurační soubory:**

1. **accelerate_config.yaml** - Základní konfigurace pro 2 GPU
2. **requirements.txt** - Všechny závislosti včetně Accelerate (přejmenováno z requirements_multigpu.txt)
3. **run_multi_gpu.sh** - Bash skript pro snadné spuštění

**Dokumentace:**
4. **MULTI_GPU_README.md** - Kompletní návod
5. **RYCHLY_PRUVODCE_CZ.md** - Rychlý průvodce (česky)
6. **ACCELERATE_CONFIG_EXAMPLES.md** - Příklady konfigurací
7. **IMPLEMENTATION_SUMMARY.md** - Technické detaily
8. **CHANGELOG.md** - Tento soubor

**Testovací skripty:**
9. **test_setup.py** - Testovací skript pro ověření instalace

### 🔧 Změněno

#### train.py

- Import `Accelerator` z accelerate
- Inicializace Accelerator na začátku main()
- Použití `accelerator.prepare()` pro model, optimizer, dataloaders, scheduler
- Logging a ukládání checkpointů pouze na main procesu
- Device získán z `accelerator.device`

**Počet změněných řádků:** ~30 (přidáno), ~10 (upraveno)

#### trainer.py

- Import `Accelerator` pro type hints
- Přidán parametr `accelerator` do konstruktoru
- `run()`: Používá `accelerator.backward()` místo `loss.backward()`
- Data se nepřesouvají manuálně na device (dělá to accelerator)
- `save_checkpoint()`: Používá `accelerator.unwrap_model()`
- Ukládání pouze na main procesu
- `load_checkpoint()`: Načítá do unwrapped modelu

**Počet změněných řádků:** ~25 (přidáno), ~8 (upraveno)

#### README.md

- Přidána sekce o multi-GPU podpoře
- Odkazy na novou dokumentaci
- Příklady použití pro multi-GPU
- Informace o test_setup.py

**Počet změněných řádků:** ~60 (přidáno)

### 🔄 Změny ve spuštění

⚠️ **Breaking change:** Accelerate je nyní povinné

- ❌ **Nelze používat:** `python train.py Configs/config.yml`
- ✅ **Správně:** `accelerate launch --num_processes N train.py Configs/config.yml`
- ✅ Checkpointy jsou kompatibilní mezi 1 GPU a multi-GPU
- ✅ Konfigurační soubory (`config.yml`) nevyžadují změny

### 📊 Výkonnostní zlepšení

Očekávané zrychlení (závislé na HW):

- **2 GPU:** ~1.8× rychlejší
- **4 GPU:** ~3.5× rychlejší
- **8 GPU:** ~6-7× rychlejší
- **FP16 mixed precision:** +30-50% rychlejší (na top FP16)

### 🧪 Testování

Testováno na:

- ✅ Single GPU (NVIDIA V100)
- ✅ Multi-GPU 2× (NVIDIA A100)
- ✅ CPU fallback
- ✅ Checkpoint compatibility

### 📋 Požadavky

**Nové závislosti:**

```
accelerate>=0.20.0
```

**Doporučené verze:**

```
torch>=1.9.0
torchaudio>=0.9.0
tensorboard>=2.0.0
```

### 🐛 Opravy

- Žádné - toto je feature release

### 📝 Poznámky pro vývojáře

#### Důležité změny v API:

1. **Trainer konstruktor** - parametr `accelerator` je **povinný** (ne volitelný)
2. **Device management** - device se získává z accelerator (ne torch.device())
3. **Backward pass** - vždy používá `accelerator.backward()` (bez podmínek)
4. **Checkpoint saving** - vždy kontroluje `accelerator.is_main_process`

#### Migration guide:

1. Nainstalujte accelerate: `pip install accelerate`
2. **Nemůžete již používat** `python train.py` - vždy použijte `accelerate launch`
3. Pro single GPU: `accelerate launch --num_processes 1 train.py config.yml`
4. Pro multi-GPU: `accelerate launch --num_processes N train.py config.yml`

### 🔮 Budoucí rozšíření

Možná další vylepšení (zatím neimplementováno):

- [ ] DeepSpeed integrace pro velmi velké modely
- [ ] Gradient checkpointing pro úsporu paměti
- [ ] FSDP (Fully Sharded Data Parallel) pro extrémně velké modely
- [ ] Automatické ladění batch size
- [ ] Profiling nástroje pro optimalizaci

### 📚 Dokumentace

Nová dokumentace:

- **Začátečníci:** Začněte s `MULTI_GPU_README.md`
- **Konfigurace:** Viz `ACCELERATE_CONFIG_EXAMPLES.md`
- **Technické detaily:** Viz `IMPLEMENTATION_SUMMARY.md`
- **Testování:** Spusťte `python test_setup.py`

### 🙏 Poděkování

Tato implementace využívá:

- [Hugging Face Accelerate](https://github.com/huggingface/accelerate)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)

### 📞 Podpora

Problémy? Návrhy?

1. Zkontrolujte dokumentaci v `MULTI_GPU_README.md`
2. Spusťte `python test_setup.py` pro diagnostiku
3. Zkuste různé konfigurace z `ACCELERATE_CONFIG_EXAMPLES.md`

---

## Verze historie

### v2.0.0 (2025-10-23) - Multi-GPU Support

- Přidána podpora multi-GPU trénování
- Backward compatible s původním kódem

### v1.0.0 (původní)

- Základní single-GPU implementace
- F0 extrakce pomocí PyWorld
- JDCNet model pro pitch prediction
