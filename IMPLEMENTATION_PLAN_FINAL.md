# Phage-Host Interaction Prediction: Final Implementation Plan

## Implementation Status Legend
- ✅ **Completed**: Section fully implemented
- 🚧 **In Progress**: Partially implemented
- ⏳ **Pending**: Not yet implemented

---

## Project Overview
A deep learning model for predicting bacteriophage-host interactions using positive-unlabeled (PU) learning with multi-instance representations and noisy-OR aggregation.

### Key Features
- Multi-instance learning for variable numbers of proteins
- Positive-unlabeled learning (only positive interaction data available)
- Noisy-OR aggregation to handle uncertainty in protein-protein interactions
- Prevention of data leakage through careful splitting

---

## 1. Data Analysis Summary ✅ **[COMPLETED]**

### Dataset Characteristics
- **Total interactions**: 24,794 (all positive examples)
- **Unique phage IDs**: 24,794 (no duplicates)
- **Unique marker proteins**: 2,907 (72.5% appear multiple times)
- **Unique RBP proteins**: 22,310 (7% appear multiple times)

### Multi-instance Distribution
- **Host proteins (markers)**: 
  - 1 marker: 3,086 samples
  - 2 markers: 21,128 samples
- **Phage proteins (RBPs)**:
  - 1 RBP: 18,383 samples
  - 2 RBPs: 5,263 samples
  - 3 RBPs: 998 samples
  - 4+ RBPs: 150 samples (max: 18)

---

## 2. Data Splitting Strategy ✅ **[COMPLETED]**

### Implemented Approach: Simple Random Split with RBP Deduplication
- **Method**: Random 60:20:20 split followed by removal of val/test samples containing training RBPs
- **Implementation**: `scripts/simple_splitting.py`
- **Results**:
  - Train: 14,876 samples (60%)
  - Validation: 4,375 samples (17.6%)
  - Test: 4,363 samples (17.6%)
  - Data utilization: 95.2%
  - **No RBP leakage from train to val/test**

### Key Features
- Ensures model generalizes to new RBPs
- Allows marker protein overlap (biologically realistic)
- Simple and reproducible
- High data utilization

---

## 3. Model Architecture ⏳ **[PENDING]**

### 3.1 Input Processing
```python
Input Shape:
- Keys (RBPs): [B, K_max, 1280]  # ESM-2 embeddings
- Locks (Markers): [B, L_max, 1280]
- Key mask: [B, K_max]  # 1=real, 0=padding
- Lock mask: [B, L_max]
```

### 3.2 Two-Tower Encoders (Updated Design)

**Selected Architecture: Balanced Compression**
```python
Key Encoder:
  Linear(1280, 768) → LayerNorm → ReLU → Dropout(0.1) →
  Linear(768, 512) → LayerNorm → ReLU → Dropout(0.1) →
  Linear(512, 256) → LayerNorm
  
Lock Encoder:
  Linear(1280, 768) → LayerNorm → ReLU → Dropout(0.1) →
  Linear(768, 512) → LayerNorm → ReLU → Dropout(0.1) →
  Linear(512, 256) → LayerNorm
```

**Alternative Architectures to Test**:
1. **Conservative** (1280→512): Minimal information loss
2. **Moderate** (1280→384): Balance between compression and preservation
3. **Aggressive** (1280→128): Maximum compression (not recommended initially)

### 3.3 Interaction Module
```python
# Pairwise scoring with scaled dot product
S[b,k,l] = dot_product(E_k[b,k], E_l[b,l]) / sqrt(d)
P_pairs = sigmoid(S * temperature)

# Apply masking
P_pairs = P_pairs * expanded_masks

# Noisy-OR aggregation
P_bag = 1 - ∏(1 - P_pairs[k,l])
```

### 3.4 nnPU Loss Function
```python
Components:
- Positive loss: L_p = -log(P_bag_pos)
- Negative loss: L_n = -log(1 - P_bag)
- Risk correction: risk_neg = L_u - π * L_p_neg
- Final loss: L = π * L_p + max(0, risk_neg)
```

---

## 4. Evaluation Metrics ⏳ **[PENDING]**

### 4.1 Binary Classification Metrics
- **Accuracy**: Overall correctness
- **MCC**: Matthews Correlation Coefficient (balanced for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

### 4.2 Ranking Metrics
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under Precision-Recall curve

### 4.3 Top-K Metrics
- **Hit Rate@K** for k ∈ [1, 20]
- **Recall@K** for k ∈ [1, 20]

---

## 5. Implementation Components ⏳ **[PENDING]**

### 5.1 ESM-2 Embedding Generation
```python
# Planned implementation
- Model: facebook/esm2_t33_650M_UR50D
- Batch processing for efficiency
- Cache embeddings in HDF5 format
- Handle variable-length sequences
```

### 5.2 Data Pipeline
```python
# Components needed:
1. Protein sequence parser
2. ESM-2 embedding generator
3. Multi-instance bag creator
4. Negative sample generator
5. PyTorch Dataset class
```

### 5.3 Training Pipeline
```python
# Hyperparameters
learning_rate: 1e-4
batch_size: 32
epochs: 100
optimizer: AdamW
weight_decay: 1e-5
scheduler: CosineAnnealingLR
class_prior_pi: 0.3  # Estimated
embedding_dim: 256  # Or 384/512
temperature: 1.0
```

---

## 6. File Structure 🚧 **[PARTIALLY COMPLETED]**

```
phi_pos_unlabeled/
├── data/                                        ✅
│   ├── dedup.phage_marker_rbp_with_phage_entropy.tsv  ✅
│   └── processed/                               ✅
│       ├── train.tsv                           ✅
│       ├── val.tsv                             ✅
│       ├── test.tsv                            ✅
│       ├── splits.pkl                          ✅
│       ├── split_stats.txt                     ✅
│       └── embeddings.h5                       ⏳
├── scripts/                                     ✅
│   ├── simple_splitting.py                     ✅
│   ├── graph_based_splitting.py                ✅ (alternative)
│   ├── generate_embeddings.py                  ⏳
│   ├── preprocess_data.py                      ⏳
│   ├── train.py                                ⏳
│   └── evaluate.py                             ⏳
├── models/                                      ⏳
│   ├── __init__.py                            ⏳
│   ├── encoders.py                            ⏳
│   ├── mil_model.py                           ⏳
│   ├── losses.py                              ⏳
│   └── utils.py                               ⏳
├── training/                                    ⏳
│   ├── __init__.py                            ⏳
│   ├── trainer.py                             ⏳
│   ├── dataset.py                             ⏳
│   └── evaluation.py                          ⏳
├── configs/                                    ⏳
│   └── default_config.yaml                    ⏳
├── IMPLEMENTATION_PLAN_FINAL.md                ✅
└── requirements.txt                            ⏳
```

---

## 7. Implementation Phases

### Phase 1: Data Preparation 🚧 **[PARTIALLY COMPLETED]**
- [x] Analyze data structure
- [x] Create data splitting strategy
- [x] Implement splitting script
- [x] Verify no data leakage
- [ ] Generate ESM-2 embeddings
- [ ] Create data loaders

### Phase 2: Model Development ⏳ **[PENDING]**
- [ ] Implement two-tower encoders
- [ ] Create pairwise scoring mechanism
- [ ] Implement noisy-OR aggregation
- [ ] Develop nnPU loss function
- [ ] Create model configuration system

### Phase 3: Training Pipeline ⏳ **[PENDING]**
- [ ] Create PyTorch Dataset classes
- [ ] Implement training loop
- [ ] Add validation monitoring
- [ ] Implement checkpointing
- [ ] Add early stopping

### Phase 4: Evaluation ⏳ **[PENDING]**
- [ ] Implement all evaluation metrics
- [ ] Create metric tracking
- [ ] Build inference pipeline
- [ ] Generate performance reports
- [ ] Create visualization tools

---

## 8. Key Design Decisions

### Data Splitting
- **Decision**: Simple random split with RBP deduplication
- **Rationale**: Prevents memorization while maintaining high data utilization
- **Trade-off**: Lost 4.8% of data but ensures no RBP leakage

### Encoder Architecture
- **Decision**: Gradual compression (1280→768→512→256)
- **Rationale**: Preserves more information than aggressive compression
- **Trade-off**: More parameters but better representation quality

### Evaluation Focus
- **Decision**: Comprehensive metrics including Hit@K and Recall@K
- **Rationale**: Better understanding of model performance for practical use
- **Trade-off**: More computation during evaluation

---

## 9. Next Immediate Steps

1. **Generate ESM-2 Embeddings** ⏳
   - Set up ESM-2 model
   - Process all unique protein sequences
   - Save embeddings to HDF5

2. **Implement Base Model** ⏳
   - Create encoder architecture
   - Implement noisy-OR aggregation
   - Develop nnPU loss

3. **Create Data Pipeline** ⏳
   - Build PyTorch Dataset
   - Handle multi-instance bags
   - Implement negative sampling

4. **Initial Training** ⏳
   - Set up training loop
   - Run initial experiments
   - Validate on small subset

---

## 10. Risk Mitigation

### Potential Issues and Solutions

1. **Memory constraints with ESM-2**
   - Solution: Batch processing, gradient accumulation
   
2. **Class imbalance in PU learning**
   - Solution: Careful π estimation, risk correction
   
3. **Overfitting to training proteins**
   - Solution: Dropout, weight decay, early stopping
   
4. **Computational cost**
   - Solution: Mixed precision training, efficient data loading

---

## 11. Success Criteria

### Minimum Viable Model
- AUROC > 0.75
- MCC > 0.4
- F1 Score > 0.6
- Hit Rate@5 > 0.7

### Target Performance
- AUROC > 0.85
- MCC > 0.5
- F1 Score > 0.7
- Hit Rate@5 > 0.8
- Generalization to unseen RBPs

---

## 12. Documentation and Monitoring

### To Track During Development
- Training/validation loss curves
- All evaluation metrics per epoch
- Protein embedding statistics
- Model parameter counts
- Training time per epoch
- GPU memory usage

### Final Deliverables
- Trained model weights
- Performance report
- Inference pipeline
- Usage documentation
- Reproducibility instructions

---

## Notes

### Completed Work
- Data analysis and understanding ✅
- Splitting strategy design and implementation ✅
- Prevention of RBP leakage ✅
- Architecture planning with encoder options ✅
- Comprehensive evaluation metrics design ✅

### Pending Work
- ESM-2 embedding generation
- Model implementation
- Training pipeline
- Evaluation implementation
- Optimization and tuning

### Key Changes from Original Plan
1. **Splitting method**: Changed from graph-based to simple random with deduplication
2. **Encoder depth**: Reduced compression ratio (1280→256 instead of 1280→128)
3. **Evaluation metrics**: Added Hit@K and Recall@K for k=[1,20]
4. **Data split ratios**: Actual splits are 60:17.6:17.6 (due to deduplication)