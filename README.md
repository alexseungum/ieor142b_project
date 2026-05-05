# DDR Chart Generation from Audio

Automatic Dance Dance Revolution chart generation using a CNN + Transformer neural network. Given any audio file as input, the model outputs a sequence of arrow events (left, right, up, down) that form a playable, music-synchronized DDR chart.

## Overview

Dance Dance Revolution charts are currently authored manually by skilled chart artists, who design step sequences that feel natural and physically engaging while staying synchronized to the rhythm and structure of a song. Since the chart is fundamentally derived from the audio — beats, phrases, and musical accents drive where steps are placed — this mapping is a natural candidate for automation. This project learns that mapping directly from data using deep learning.

The work is inspired by [Dance Dance Convolution (Donahue et al., 2017)](https://arxiv.org/abs/1703.06891), the original paper on this problem, but differs from it in architecture, training methodology, and difficulty handling. See the Differences from DDC section below.

## Model Architecture

The model is a two-stage deep neural network pipeline.

**Stage 1: Step Placement.** Given the audio, predict whether any arrow event occurs at each beat-aligned timestep. This is a binary classification problem at every timestep.

**Stage 2: Arrow Selection.** Given the predicted step positions, predict which combination of arrows (L/D/U/R) is active at each step. This is a multi-label classification problem.

Both stages share a backbone consisting of a local CNN encoder followed by a Transformer encoder:

**Local CNN.** A 2D convolutional network processes a context window of mel spectrogram frames centered on each timestep. The convolutional filters detect local onset and rhythmic features such as beat transients and harmonic changes. Each timestep is encoded independently into a feature vector.

**Transformer Encoder.** The sequence of CNN feature vectors is passed through a stack of multi-head self-attention layers. Unlike the CNN, which has a fixed local receptive field, self-attention allows every timestep to attend to every other timestep in the full song sequence. This lets the model learn long-range musical structure — for example, that a repeating chorus section should produce similar step patterns each time it appears. This is a BERT-style encoder (full bidirectional attention over the entire sequence), not an autoregressive decoder like GPT.

**Difficulty Conditioning.** A learned embedding for the difficulty level (Beginner through Challenge) is added to every position in the sequence as a global bias. This allows a single model to handle all difficulty levels simultaneously, learning a shared musical representation across difficulties rather than training separate models per level.

The full architecture is: mel spectrogram → local CNN → positional encoding + difficulty embedding → Transformer encoder → step head + arrow head.

## Differences from DDC

The original Dance Dance Convolution paper (2017) used a pure CNN operating on local sliding windows of audio features, with separate models trained per difficulty level. This project differs in four ways:

**Transformer encoder.** DDC had no mechanism for long-range musical context. Adding a Transformer encoder on top of the CNN allows the model to reason about global song structure, which a local CNN window physically cannot capture. Transformers did not exist in their current form when DDC was published.

**Curriculum learning.** DDC trained on all difficulties simultaneously from the start. Here, training proceeds in stages: the model first trains only on Beginner charts, then Easy is added, then Medium, and so on up to Challenge. This forces the model to first learn the fundamental relationship between audio and step placement on sparse, easy charts before learning what makes a hard chart hard. The optimizer is warm-restarted at each curriculum stage.

**Shared difficulty-conditioned model.** Rather than training five separate models, a single model handles all difficulties via a learned difficulty embedding injected into every sequence position. This enables knowledge transfer across difficulties and makes the threshold knob at inference meaningful.

**Label smoothing.** DDC used standard binary cross-entropy. Here label smoothing is applied during training to prevent overconfidence on the highly imbalanced step placement task, where the vast majority of timesteps have no step.

## Threshold as a Difficulty Knob

At inference time, the `--threshold` argument controls the step placement density. The model outputs a probability at each timestep of whether a step should occur. Lowering the threshold below 0.5 causes more timesteps to cross the activation boundary, producing denser charts that feel harder. Raising it produces sparser charts. This gives a continuous difficulty tuning knob from a single trained model without retraining.

## Repository Structure

```
models/model.py          CNN + Transformer architecture, loss, and inference
utils/data_utils.py      .sm file parser and mel spectrogram extraction
utils/sm_writer.py       converts model output back to a playable .sm file
dataset.py               PyTorch Dataset with curriculum learning support
train.py                 training script with curriculum loop
generate.py              inference: any audio file → .sm chart + visualizer
visualizer.py            generates a standalone HTML chart visualizer
notebooks/train_colab.ipynb   full training and generation workflow on Colab
```

## Quickstart

**Training (Google Colab with T4 GPU recommended):**

Open `notebooks/train_colab.ipynb` in Colab, set the runtime to T4 GPU, and run cells top to bottom. The notebook handles cloning, data verification, training across curriculum stages, loss curve plotting, and chart generation.

**Generating a chart from any song:**

```bash
python generate.py \
    --audio      my_song.mp3 \
    --checkpoint checkpoints/best_model.pt \
    --difficulty 2 \
    --threshold  0.5 \
    --output     output_chart
```

This produces `output_chart/chart.sm` (load in StepMania) and `output_chart/visualizer.html` (open in any browser for an interactive scrolling chart preview).

**Threshold sweep:**

```bash
# Lower threshold = more arrows = harder feel
python generate.py --threshold 0.3 ...

# Higher threshold = fewer arrows = easier feel
python generate.py --threshold 0.7 ...
```

## Data

Training uses StepMania simfile packs where each song folder contains a `.sm` chart file and an audio file. The code searches recursively through the data directory, so any standard pack structure works. The dataset splits songs 90/10 into train and validation sets by song identity (not by chunk) to prevent leakage.

## Requirements

```
torch >= 2.0
torchaudio >= 2.0
librosa >= 0.10
soundfile >= 0.12
numpy >= 1.24
```

Install with `pip install -r requirements.txt`.

## Reference

Donahue, C., Lipton, Z., and McAuley, J. (2017). Dance Dance Convolution. *Proceedings of the 34th International Conference on Machine Learning (ICML)*.
