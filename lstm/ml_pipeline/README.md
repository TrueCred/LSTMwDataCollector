# LSTM Behavioral Pipeline

Training and inference code for the neural side of TrueCred. The main backend uses a Gaussian statistical engine by default; this pipeline produces an optional LSTM encoder that maps keystroke, scroll, and IMU sequences into a 32-dimensional "behavioral DNA" vector for similarity comparison.

Based on touch dynamics research ([Touchalytics](https://arxiv.org/pdf/1207.6231)).

## Model architecture

`BehavioralLSTM` in `model.py`:

- Keystroke branch: embedding + bidirectional LSTM on last 8 keystrokes
- Scroll branch: bidirectional LSTM on last 20 scroll events
- IMU branch: linear projection of gyro/pitch/roll statistics
- Fusion layer outputs a normalized 32-D DNA vector plus training heads for user classification and risk

## Files

| File | Purpose |
|------|---------|
| `data_loader.py` | Load enrollment data, preprocess tensors, fit/load scalers |
| `dataset.py` | PyTorch dataset and train/val split |
| `model.py` | Network definition |
| `train.py` | Training loop with early stopping and EER metric |
| `export_onnx.py` | Export encoder to ONNX for runtime inference |
| `inference.py` | `BehavioralInference` class used by the FastAPI backend |
| `reset_db.py` | Initialize SQLite templates table |
| `test_pipeline.py` | Sanity check: self-similarity vs cross-user similarity |

## Setup

```bash
pip install -r requirements.txt
```

## Training workflow

```bash
# 1. Reset templates table
python reset_db.py

# 2. Load raw data and fit scalers
python data_loader.py

# 3. Train (outputs best_model.pt)
python train.py

# 4. Export ONNX encoder
python export_onnx.py

# 5. Validate similarities
python test_pipeline.py
```

Artifacts:

- `checkpoints/best_model.pt` — trained weights
- `checkpoints/scalers.pkl` — feature scalers
- `checkpoints/key_vocab.json` — keystroke vocabulary
- `sentinel_encoder.onnx` — exported encoder (written to this folder by `export_onnx.py`)

The main backend expects the ONNX file at `checkpoints/sentinel_lstm.onnx`. After export, copy or rename:

```bash
cp sentinel_encoder.onnx checkpoints/sentinel_lstm.onnx
```

Restart `fastapiCollector` after that.

## Integration

`fastapiCollector/main.py` looks for:

```
lstm/ml_pipeline/checkpoints/sentinel_lstm.onnx
lstm/ml_pipeline/checkpoints/scalers.pkl
lstm/ml_pipeline/checkpoints/key_vocab.json
```

If they are missing, the backend still runs using the Gaussian engine and a simpler statistical fallback.

## Standalone API

`main_integration.py` exposes a minimal FastAPI server (`/enroll`, `/verify`, `/health`) that uses only the LSTM inference path. Useful for testing the pipeline in isolation:

```bash
python main_integration.py
```

## Data

Training data comes from the `raw_enrollment` table populated by the mobile app through the main backend. The Touchalytics dataset notes are in `datasets/touchalytics/readme_data.txt` for reference material.
