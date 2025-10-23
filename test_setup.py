#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test skript pro ověření instalace a funkčnosti multi-GPU podpory
"""

import sys


def test_imports():
    """Test zda jsou všechny potřebné balíčky nainstalovány"""
    print("=" * 60)
    print("Test 1: Import balíčků")
    print("=" * 60)
    
    modules = [
        ("torch", "PyTorch"),
        ("accelerate", "Accelerate"),
        ("torchaudio", "TorchAudio"),
        ("tensorboard", "TensorBoard"),
    ]
    
    failed = []
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name:20s} ... OK")
        except ImportError as e:
            print(f"✗ {name:20s} ... CHYBÍ")
            failed.append((name, str(e)))
    
    if failed:
        print("\n❌ Chybějící balíčky:")
        for name, error in failed:
            print(f"   - {name}: {error}")
        print("\nNainstalujte pomocí:")
        print("   pip install accelerate torch torchaudio tensorboard")
        return False
    
    print("\n✅ Všechny balíčky jsou nainstalovány")
    return True


def test_cuda():
    """Test dostupnosti CUDA a GPU"""
    print("\n" + "=" * 60)
    print("Test 2: CUDA a GPU")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch verze: {torch.__version__}")
        print(f"CUDA dostupná: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA verze: {torch.version.cuda}")
            print(f"Počet GPU: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Název: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Paměť: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
            
            print("\n✅ GPU jsou dostupné pro trénování")
            return True
        else:
            print("⚠️  CUDA není dostupná - trénování bude pouze na CPU")
            print("   To je OK pro testování, ale bude velmi pomalé")
            return True
            
    except Exception as e:
        print(f"❌ Chyba při detekci CUDA: {e}")
        return False


def test_accelerate():
    """Test konfigurace Accelerate"""
    print("\n" + "=" * 60)
    print("Test 3: Accelerate konfigurace")
    print("=" * 60)
    
    try:
        from accelerate import Accelerator
        
        # Test základní inicializace
        accelerator = Accelerator()
        
        print(f"Device: {accelerator.device}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")
        print(f"Main process: {accelerator.is_main_process}")
        
        if accelerator.num_processes > 1:
            print(f"\n✅ Multi-GPU konfigurace aktivní ({accelerator.num_processes} procesy)")
        else:
            print("\n✅ Single device konfigurace (1 GPU nebo CPU)")
            
        return True
        
    except Exception as e:
        print(f"❌ Chyba při inicializaci Accelerate: {e}")
        return False


def test_model_load():
    """Test načtení modelu"""
    print("\n" + "=" * 60)
    print("Test 4: Načtení modelu")
    print("=" * 60)
    
    try:
        from model import JDCNet
        import torch
        
        model = JDCNet(num_class=1)
        print(f"Model vytvořen: {type(model).__name__}")
        
        # Spočítej parametry
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Celkem parametrů: {total_params:,}")
        print(f"Trénovatelných: {trainable_params:,}")
        print(f"Velikost modelu: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
        
        # Test forward pass
        dummy_input = torch.randn(2, 192, 80)  # batch, length, mels
        output = model(dummy_input)
        print(f"Forward pass úspěšný: {output[0].shape}")
        
        print("\n✅ Model je funkční")
        return True
        
    except Exception as e:
        print(f"❌ Chyba při načtení modelu: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test dataloaderů"""
    print("\n" + "=" * 60)
    print("Test 5: Dataloadery")
    print("=" * 60)
    
    try:
        import os.path as osp
        
        # Zkontroluj existenci dat
        data_files = [
            "Data/train_MGW500.csv",
            "Data/valid_MGW100.csv",
        ]
        
        found = []
        for f in data_files:
            if osp.exists(f):
                print(f"✓ Nalezen: {f}")
                found.append(f)
            else:
                print(f"✗ Chybí: {f}")
        
        if found:
            print(f"\n✅ Nalezeno {len(found)}/{len(data_files)} datových souborů")
            print("   Pro plné testování dataloaderů spusťte train.py")
        else:
            print("\n⚠️  Žádné datové soubory nenalezeny")
            print("   To je OK pokud testujete pouze instalaci")
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba při kontrole dat: {e}")
        return False


def main():
    """Hlavní testovací funkce"""
    print("\n" + "=" * 60)
    print("TEST MULTI-GPU PODPORY PRO PITCHEXTRACTOR")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_cuda,
        test_accelerate,
        test_model_load,
        test_dataloader,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Neočekávaná chyba v {test.__name__}: {e}")
            results.append(False)
    
    # Souhrn
    print("\n" + "=" * 60)
    print("SOUHRN TESTŮ")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Úspěšných: {passed}/{total}")
    
    if all(results):
        print("\n🎉 Všechny testy prošly!")
        print("\nMůžete spustit trénink pomocí:")
        print("   accelerate launch --num_processes 2 train.py Configs/config.yml")
        print("   nebo")
        print("   ./run_multi_gpu.sh")
        return 0
    else:
        print("\n⚠️  Některé testy selhaly")
        print("Zkontrolujte výše uvedené chyby")
        return 1


if __name__ == "__main__":
    sys.exit(main())
