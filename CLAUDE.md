# Claude Development Guidelines

## Project: Phage-Host Interaction Prediction with PU Learning

### Development Workflow
1. **Local Development**: Code is written locally on laptop in a modular fashion
2. **Remote Execution**: Code is pushed to GitHub and executed on Biowulf HPC cluster
3. **Feedback Loop**: Errors and results are reported back for debugging
4. **Iterative Refinement**: Code is updated based on cluster execution results

### Code Design Principles

#### Modularity
- Write small, focused functions with single responsibilities
- Create reusable components that can be tested independently
- Separate concerns (data processing, model definition, training, evaluation)
- Use configuration files for hyperparameters and settings

#### Code Quality Standards
- **NO SPAGHETTI CODE** - Every function should have a clear purpose
- Production-ready code only - no temporary hacks or experimental snippets
- Comprehensive docstrings for all functions and classes
- Type hints for better code clarity and IDE support
- Error handling and informative error messages
- Logging instead of print statements for production code

#### File Organization
```
- Each module should be self-contained
- Clear separation between:
  - Data processing (data/)
  - Model architecture (models/)
  - Training logic (training/)
  - Evaluation (evaluation/)
  - Utilities (utils/)
  - Configuration (configs/)
```

#### Testing Strategy
- Write code that can be tested in small chunks
- Include validation checks for inputs
- Create sample/toy data for local testing when possible
- Implement checkpoint saving for long-running processes

### Biowulf-Specific Considerations
- Memory-efficient implementations for large protein embeddings
- Batch processing capabilities for ESM-2 embedding generation
- GPU-optimized code where applicable
- Proper resource allocation in SLURM scripts
- Checkpoint/resume functionality for interrupted jobs

### Implementation Order
1. Data preprocessing utilities
2. ESM-2 embedding generation script
3. Dataset classes for PyTorch
4. Model architecture components
5. Loss functions
6. Training loop
7. Evaluation metrics
8. Inference pipeline

### Communication Protocol
- Clear error messages with full stack traces
- Log files with timestamps
- Performance metrics (time, memory usage)
- Intermediate results for validation
- Checkpoint files for resuming

### Key Technical Decisions
- **Framework**: PyTorch for deep learning
- **Embeddings**: ESM-2 (facebook/esm2_t33_650M_UR50D)
- **Data Format**: HDF5 for embedding storage
- **Config Format**: YAML for configuration files
- **Logging**: Python logging module with file handlers

### Repository Standards
- This code will be shipped as a final research repository
- All code must be:
  - Well-documented
  - Reproducible
  - Efficient
  - Maintainable
  - Professional quality

### Current Project Status
- ✅ Data analysis completed
- ✅ Data splitting implemented (simple random with RBP deduplication)
- ⏳ ESM-2 embedding generation (next step)
- ⏳ Model implementation
- ⏳ Training pipeline
- ⏳ Evaluation framework