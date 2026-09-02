## LSTM-based Authentication System - Complete Implementation

### Files Generated (9 total)

All files have been created in `e:\projects\TrueCred\lstm\ml_pipeline\`

---

#### 1. **reset_db.py** (1,606 bytes)
- **Purpose**: Database initialization and reset
- **Function**: `reset_database(db_path)`
- **Actions**: 
  - Drops tables: `templates`, `sessions`, `live_sessions`
  - Creates fresh `templates` table with columns: `id`, `user_id` (UNIQUE), `key_dna`, `scroll_dna`, `imu_stats`, `enrolled_at`
- **Database Path**: `../backend/sentinel_lab.db` (relative to ml_pipeline)
- **Usage**: `python reset_db.py`

---

#### 2. **data_loader.py** (8,887 bytes)
- **Purpose**: Data loading, preprocessing, and scaler fitting
- **Key Functions**:
  - `load_raw_enrollment(db_path)`: Loads from `raw_enrollment` table
  - `preprocess_keystrokes(list, scaler)`: Returns [1, 8, 4] tensor (last 8 keystrokes with log1p on hold/flight)
  - `preprocess_scrolls(list, scaler)`: Returns [1, 20, 6] tensor (last 20 scrolls, direction as sin/cos)
  - `extract_imu_stats(list, scaler)`: Returns [1, 4] tensor (mean of gyro_x, gyro_y, pitch, roll)
  - `fit_scalers(unified_data)`: Fits and returns 3 StandardScalers
  - `save_scalers()`, `load_scalers()`: Pickle-based persistence
  - `save_key_vocab()`: Saves KEY_VOCAB to JSON
- **Key Vocab**: v,k,e,r,j,p,w,u,1,3,7,9,2,8,4,6 mapped to indices 1-16 (0=UNK)
- **Graceful Handling**: Empty lists return zero-padded tensors

---

#### 3. **model.py** (4,698 bytes)
- **Purpose**: LSTM neural network architecture
- **Class**: `BehavioralLSTM(nn.Module)`
- **Architecture**:
  - `key_embed`: Embedding(17, 8, padding_idx=0)
  - `key_lstm`: LSTM(11, 32, bidirectional=True) → output [batch, 64]
  - `scroll_lstm`: LSTM(6, 32, bidirectional=True) → output [batch, 64]
  - `fusion`: Linear(132, 64) → ReLU → Dropout(0.3) → Linear(64, 32) → ReLU
  - `dna_head`: Linear(32, 32) + L2 normalization → [batch, 32]
  - `classifier`: Linear(32, num_users) for training
  - `risk_head`: Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid
- **Forward Method**:
  - `forward(keystrokes, scrolls, imu_stats, return_dna=False)`
  - If `return_dna=True`: returns normalized DNA [batch, 32]
  - Else: returns (dna, logits, risk)

---

#### 4. **dataset.py** (4,811 bytes)
- **Purpose**: PyTorch Dataset implementation
- **Class**: `BehavioralDataset(Dataset)`
- **Features**:
  - Takes `unified_data` dict: {user_id: {keystrokes, scrolls, imu}}
  - Applies scalers during `__getitem__`
  - Returns dict with keys: keystrokes, scrolls, imu_stats, label
- **Split Function**: `split_dataset()` performs 80/20 train/val split by user count
- **Type Hints**: Complete with type annotations

---

#### 5. **train.py** (8,645 bytes)
- **Purpose**: Training orchestration
- **Main Components**:
  - `compute_eer(embeddings, labels)`: Equal Error Rate calculation using cosine similarity
  - `train_epoch()`: Training loop with CrossEntropyLoss + 0.5*BCELoss
  - `eval_epoch()`: Validation loop
  - `main()`: Full training pipeline
- **Training Config**:
  - Epochs: 100
  - Early Stopping: patience=15, monitors val_acc
  - Optimizer: Adam(lr=1e-3)
  - Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
  - Batch Size: 8
- **Outputs**: Saves `best_model.pt`, `scalers.pkl`, prints final EER
- **Performance**: CPU-only, targets <10 minutes training

---

#### 6. **export_onnx.py** (2,161 bytes)
- **Purpose**: Export encoder to ONNX format
- **Function**: `export_to_onnx(model_path, output_path, num_users, device)`
- **Output**: `sentinel_encoder.onnx`
- **Exported Components**: DNA output only (encoder)
- **Dynamic Axes**: Batch size is dynamic
- **ONNX Version**: 14

---

#### 7. **inference.py** (5,005 bytes)
- **Purpose**: Runtime inference engine
- **Class**: `BehavioralInference`
- **Constructor Args**:
  - `onnx_model_path`: Path to sentinel_encoder.onnx
  - `scalers_path`: Path to scalers.pkl
  - `key_vocab_path`: Path to key_vocab.json
- **Methods**:
  - `extract_dna(keystrokes_list, scrolls_list, imu_list)`: Returns [1, 32] numpy array or None
  - `compute_risk(live_dna, template_dna)`: Returns float 0-1 (1.0 - cosine_similarity)
- **Error Handling**: Graceful fallback if ONNX/scalers unavailable

---

#### 8. **test_pipeline.py** (4,284 bytes)
- **Purpose**: End-to-end pipeline validation
- **Function**: `test_pipeline()`
- **Tests**:
  - Loads model, data, and scalers
  - Extracts DNA for first 3 users
  - Computes self-similarity (should be ~1.0)
  - Computes cross-similarity with other users
  - **Assertions**:
    - `self_similarity > 0.7`
    - `cross_similarity < 0.5`
- **Output**: Prints similarities and passes/fails assertions

---

#### 9. **main_integration.py** (5,634 bytes)
- **Purpose**: FastAPI REST API endpoints
- **Request/Response Models**:
  - `EnrollRequest`: user_id, keystrokes, scrolls, imu
  - `EnrollResponse`: status, user_id, enrolled_at
  - `VerifyRequest`: user_id, keystrokes, scrolls, imu
  - `VerifyResponse`: status, risk, risk_level, user_id
- **Endpoints**:
  - **POST /enroll**: Extract DNA, save to templates table
  - **POST /verify**: Extract live DNA, compare with template, return risk with temporal smoothing
    - Thresholds: <0.35 normal, <0.60 caution, <0.80 warning, ≥0.80 critical_locked
    - Temporal smoothing: `new_risk = 0.7*prev + 0.3*current`
  - **GET /health**: Status check
- **Database**: Stores DNA vectors in templates table as JSON strings
- **Port**: 8000

---

### Workflow & Integration Points

```
1. Reset Database
   └─> reset_db.py

2. Data Preparation
   └─> data_loader.py (load, preprocess, fit scalers)
   └─> Outputs: key_vocab.json, scalers.pkl

3. Training Pipeline
   └─> train.py
   ├─> Uses: data_loader.py, dataset.py, model.py
   └─> Outputs: best_model.pt, scalers.pkl, training logs

4. Model Export
   └─> export_onnx.py
   ├─> Input: best_model.pt
   └─> Output: sentinel_encoder.onnx

5. Testing
   └─> test_pipeline.py
   ├─> Uses: model.py, inference.py, data_loader.py
   └─> Validates: DNA extraction, similarities

6. Production Deployment
   └─> main_integration.py (FastAPI)
   ├─> Uses: inference.py, scalers.pkl, sentinel_encoder.onnx
   └─> Endpoints: /enroll, /verify, /health
```

---

### Key Features

✅ **Complete Type Hints**: All functions have type annotations  
✅ **Graceful Error Handling**: Empty lists handled, ONNX/scalers optional  
✅ **CPU-Only**: No GPU requirements  
✅ **Fast Training**: <10 minutes expected  
✅ **Production Ready**: ONNX export, REST API, persistent storage  
✅ **All Paths Relative**: All paths relative to ml_pipeline/ folder  
✅ **Guard Clauses**: `if __name__ == "__main__"` in all modules  

---

### Dependencies

- torch
- numpy
- sklearn
- onnxruntime
- fastapi
- pydantic
- sqlite3 (stdlib)
- scipy (for EER computation)

---

### Database Schema

**templates table**:
```sql
CREATE TABLE templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    key_dna TEXT NOT NULL,          -- JSON array of 32 floats
    scroll_dna TEXT NOT NULL,        -- JSON array of 32 floats
    imu_stats TEXT NOT NULL,         -- JSON array of 32 floats
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

---

### Quick Start

```bash
# 1. Reset database
python reset_db.py

# 2. Load data and fit scalers
python data_loader.py

# 3. Train model
python train.py

# 4. Export to ONNX
python export_onnx.py

# 5. Test pipeline
python test_pipeline.py

# 6. Run API server
python main_integration.py
```

---

**Generation Complete**: All 9 files successfully created with full specifications implemented.
