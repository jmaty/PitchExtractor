# Souhrn zjednodušení - Accelerate je povinné

## Provedené změny

### 1. Kód byl zjednodušen

#### trainer.py
**Odstraněno:**
- ❌ Všechny podmínky `if self.accelerator is None`
- ❌ Manuální přesun dat na device: `batch = [b.to(self.device) for b in batch]`
- ❌ Podmíněný backward: `if self.accelerator is not None: ... else: loss.backward()`

**Nyní:**
- ✅ Vždy používá `self.accelerator.backward(loss)`
- ✅ Data automaticky přesouvá Accelerator
- ✅ Vždy používá `accelerator.unwrap_model()` pro checkpoint operace
- ✅ Čistší a jednodušší kód

### 2. Dokumentace aktualizována

Aktualizované soubory:
- ✅ `README.md` - zdůrazněno, že Accelerate je povinné
- ✅ `MULTI_GPU_README.md` - odstraněny zmínky o "stejně jako předtím"
- ✅ `RYCHLY_PRUVODCE_CZ.md` - aktualizovány příklady
- ✅ `IMPLEMENTATION_SUMMARY.md` - vysvětleno, že není zpětná kompatibilita
- ✅ `CHANGELOG.md` - zaznamenána breaking change

### 3. Requirements zjednodušeno

- ✅ `requirements_multigpu.txt` → přejmenováno na `requirements.txt`
- ✅ Přidány všechny závislosti (pyworld, librosa, atd.)
- ✅ Jasně uvedeno, že Accelerate je povinné

## Důvody pro tuto změnu

### ✨ Výhody

1. **Čistší kód**
   - Žádné podmínky pro `accelerator is None`
   - Jeden způsob, jak dělat věci
   - Snadnější údržba

2. **Menší prostor pro chyby**
   - Nemůže se stát, že zapomenete použít Accelerator
   - Konzistentní chování vždy

3. **Jednodušší onboarding**
   - Uživatelé se nemusí rozhodovat mezi více způsoby
   - Jeden jasný návod: "použij accelerate launch"

4. **Moderní přístup**
   - Accelerate je moderní standard
   - Připraveno na budoucí rozšíření (DeepSpeed, FSDP)

### ⚠️ Breaking Change

**DŮLEŽITÉ:** Toto je breaking change!

❌ **Nelze již používat:**
```bash
python train.py Configs/config.yml
```

✅ **Vždy používejte:**
```bash
# Single GPU
accelerate launch --num_processes 1 train.py Configs/config.yml

# Multi-GPU
accelerate launch --num_processes 2 train.py Configs/config.yml
```

## Migrace pro existující uživatele

### Před změnou (deprecated)
```bash
python train.py Configs/config.yml
```

### Po změně (required)
```bash
# Instalace
pip install accelerate

# Single GPU (stejná rychlost jako předtím)
accelerate launch --num_processes 1 train.py Configs/config.yml

# Multi-GPU (rychlejší)
accelerate launch --num_processes 2 train.py Configs/config.yml
```

### Checkpointy
✅ **Dobré zprávy:** Checkpointy jsou stále kompatibilní!
- Starý checkpoint (z `python train.py`) funguje s novým kódem
- Nový checkpoint funguje při změně počtu GPU
- Není potřeba žádná konverze

## Co se nezměnilo

✅ **Nezměněno:**
- Formát checkpointů
- Konfigurační soubory (`config.yml`)
- Model architektura
- Training loop logika
- Výstupní formáty

## Porovnání kódu

### Před (s podmínkami)
```python
def run(self, batch):
    self.optimizer.zero_grad()
    
    # Podmínka pro device
    if self.accelerator is None:
        batch = [b.to(self.device) for b in batch]
    
    x, f0, sil = batch
    f0_pred, sil_pred = self.model(x.transpose(-1, -2))
    
    # ... loss computation ...
    
    # Podmínka pro backward
    if self.accelerator is not None:
        self.accelerator.backward(loss)
    else:
        loss.backward()
```

### Po (čisté)
```python
def run(self, batch):
    self.optimizer.zero_grad()
    
    # Accelerator automaticky přesouvá data
    x, f0, sil = batch
    f0_pred, sil_pred = self.model(x.transpose(-1, -2))
    
    # ... loss computation ...
    
    # Vždy používá accelerator
    self.accelerator.backward(loss)
```

**Rozdíl:** -5 řádků, jednodušší logika, žádné podmínky

## Dokumentace

### Pro začátečníky
📖 Začněte zde: `RYCHLY_PRUVODCE_CZ.md` (česky)

### Pro pokročilé
📖 Kompletní návod: `MULTI_GPU_README.md` (anglicky)

### Pro technické detaily
📖 Implementace: `IMPLEMENTATION_SUMMARY.md`

### Pro konfiguraci
📖 Příklady: `ACCELERATE_CONFIG_EXAMPLES.md`

## FAQ

### Q: Proč je Accelerate povinné?
**A:** Zjednodušuje kód, eliminuje chyby a je moderní standard pro PyTorch trénování.

### Q: Je to pomalejší na single GPU?
**A:** Ne, overhead je minimální (~1-2%). Výhody převažují.

### Q: Můžu použít CPU?
**A:** Ano! `accelerate launch --cpu train.py config.yml`

### Q: Fungují moje staré checkpointy?
**A:** Ano, jsou plně kompatibilní.

### Q: Co když nemám GPU?
**A:** Accelerate funguje i na CPU. Použijte `--cpu` flag.

### Q: Je to složitější na nastavení?
**A:** Ne, stačí `pip install accelerate` a pak použít `accelerate launch` místo `python`.

## Závěr

Tato změna **zjednodušuje kód** a **eliminuje potenciální chyby**. Ačkoliv je to breaking change ve způsobu spuštění, migrace je velmi jednoduchá a výhody převažují.

**Hlavní výhody:**
- ✅ Čistější kód bez podmínek
- ✅ Jeden správný způsob použití
- ✅ Připraveno na budoucnost (DeepSpeed, FSDP)
- ✅ Menší prostor pro chyby

**Co musíte změnit:**
- 🔄 Instalace: `pip install accelerate`
- 🔄 Spuštění: `accelerate launch` místo `python`

Hotovo! 🎉
