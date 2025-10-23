# JDC-PitchExtractor

This repo contains the training code for deep neural pitch extractor for Voice Conversion (VC) and TTS used in [StarGANv2-VC](https://github.com/yl4579/StarGANv2-VC) and [StyleTTS](https://github.com/yl4579/StyleTTS). This is the F0 network in StarGANv2-VC and pitch extractor in StyleTTS. 

## 🚀 Multi-GPU Support with Accelerate

This repository uses [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for **distributed training** on single or multiple GPUs.

### Quick Start

```bash
# Install accelerate (REQUIRED)
pip install accelerate

# Single GPU
accelerate launch --num_processes 1 train.py Configs/config.yml

# Multi-GPU (2 GPUs)
accelerate launch --num_processes 2 train.py Configs/config.yml

# Or use the provided script
./run_multi_gpu.sh
```

**Benefits:**

- ⚡ Linear speed scaling (2 GPUs ≈ 2× faster, 4 GPUs ≈ 4× faster)
- 🎯 Automatic gradient synchronization
- 💾 Support for mixed precision training (FP16/BF16)
- 🔧 Works on CPU, single GPU, or multi-GPU with same code

**📖 For detailed documentation, see:**

- [`MULTI_GPU_README.md`](MULTI_GPU_README.md) - Complete training guide
- [`ACCELERATE_CONFIG_EXAMPLES.md`](ACCELERATE_CONFIG_EXAMPLES.md) - Configuration examples
- [`RYCHLY_PRUVODCE_CZ.md`](RYCHLY_PRUVODCE_CZ.md) - Czech quick start guide

**🧪 Test your setup:**

```bash
python test_setup.py
```

---

## Pre-requisites

1. Python >= 3.7
2. Clone this repository:

```bash
git clone https://github.com/yl4579/PitchExtractor.git
cd PitchExtractor
```

3. Install python requirements: 

```bash
pip install SoundFile torchaudio torch pyyaml click matplotlib librosa pyworld accelerate
```

4. Prepare your own dataset and put the `train_list.txt` and `val_list.txt` in the `Data` folder (see Training section for more details).

## Training

**Note:** Accelerate is required for training. Use `accelerate launch` instead of `python` to run the training script.

### Single GPU

```bash
accelerate launch --num_processes 1 train.py Configs/config.yml
```

### Multi-GPU (Recommended for Faster Training)

```bash
# Using 2 GPUs
accelerate launch --num_processes 2 train.py Configs/config.yml

# Using 4 GPUs with mixed precision
accelerate launch --num_processes 4 --mixed_precision fp16 train.py Configs/config.yml

# Using provided script (edit NUM_GPUS variable inside)
./run_multi_gpu.sh

```
Please specify the training and validation data in `config.yml` file. The data list format needs to be `filename.wav|anything`, see [train_list.txt](https://github.com/yl4579/StarGANv2-VC/blob/main/Data/train_list.txt) as an example (a subset of VCTK). Note that you can put anything after the filename because the training labels are generated ad-hoc.

Checkpoints and Tensorboard logs will be saved at `log_dir`. To speed up training, you may want to make `batch_size` as large as your GPU RAM can take. 

### IMPORTANT: DATA FOLDER NEEDS WRITE PERMISSION

Since both `harvest` and `dio` are relatively slow, we do have to save the computed F0 ground truth for later use. In [meldataset.py](https://github.com/yl4579/PitchExtractor/blob/main/meldataset.py#L77-L89), it will write the computed F0 curve `_f0.npy` for each `.wav` file. This requires write permission in your data folder. 

### F0 Computation Details

In [meldataset.py](https://github.com/yl4579/PitchExtractor/blob/main/meldataset.py#L83-L87), the F0 curves are computated using [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder), one with `harvest` and another with `dio`. Both methods are acoustic-based and are unstable under certain conditions. `harvest` is faster but fails more than `dio`, so we first try `harvest`. When `harvest` fails (determined by number of frames with non-zero values), it will compute the ground truth F0 labels with `dio`. If `dio` fails, the computed F0 will have `NaN` and will be replaced with 0. This is supposed to occur only occasionally and should not affect training because these samples are treated as noises by the neural network and deep learning models are kwown to even benefit from slightly noisy datasets. However, if a lot of your samples have this problem (say > 5%), please remove them from the training set so that the model does not learn from the failed samples. 

### Data Augmentation

Data augmentation is not included in this code. For better voice conversion results, please add your own data augmentation in [meldataset.py](https://github.com/yl4579/PitchExtractor/blob/main/meldataset.py) with [audiomentations](https://github.com/iver56/audiomentations).

## References

- [keums/melodyExtraction_JDC](https://github.com/keums/melodyExtraction_JDC)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
