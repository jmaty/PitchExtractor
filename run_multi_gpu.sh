#!/bin/bash
# Multi-GPU training script for PitchExtractor

# Nastavení proměnných prostředí
export CUDA_VISIBLE_DEVICES=0,1  # Vyberte GPU, které chcete použít
export NCCL_TIMEOUT=3600

# Počet GPU pro trénování
NUM_GPUS=2

# Cesta ke konfiguraci
CONFIG_PATH="Configs/config.yml"

# Kontrola, zda existuje accelerate
if ! command -v accelerate &> /dev/null; then
    echo "Accelerate není nainstalován. Instaluji..."
    pip install accelerate
fi

# Kontrola konfiguračního souboru
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Konfigurační soubor $CONFIG_PATH neexistuje!"
    exit 1
fi

# Spuštění tréninku
echo "Spouštím multi-GPU trénování na $NUM_GPUS GPU..."
echo "Konfigurace: $CONFIG_PATH"
echo "---"

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision no \
    train.py $CONFIG_PATH

echo "---"
echo "Trénování dokončeno!"
