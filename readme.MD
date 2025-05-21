# NN Playground

This repository provides mainly a trained MLP (Multi-Layer Perceptron) model for Flow123d simulations for a TSX tunnel model. 

Main file is `NN_playground.ipynb` together with the trained model `trained_model_cpu.pth` and its configuration `trained_model_cpu.yaml`.

**Important Note**: The trained model is for normalized inputs and outputs. 

## Repository Structure

- explore_save_hdf5.ipynb: Data exploration and conversion of CSV logs to HDF5 format with compression.
- train_NN.ipynb: Full pipeline for loading, normalizing, and training an MLP using PyTorch, including batching, adaptive learning rate, and loss tracking.
- reference_data.ipynb: Utilities for loading, visualizing, and saving reference or measured data (e.g., time axes) into HDF5 format.
- NN_playground.ipynb: **Main notebook for using exported and trained neural networks.**
- proposal_logs_2.h5: Main dataset in HDF5 format.
- trained_model_cpu.pth / trained_model_cpu.yaml: Example of exported trained model and its configuration.
- v1_h1_data.h5: Example reference data for evaluation.

## Main Usage: NN_playground

The NN_playground.ipynb notebook demonstrates how to load, use, and evaluate a trained neural network model. The workflow includes:

1. **Loading the Model**
   - Loads the MLP architecture and weights from a `.yaml` configuration and `.pth` state dictionary.
   - Example:
     ```python
     loaded_model = load_mlp_from_yaml('trained_model_cpu.yaml')
     ```

2. **Loading and Normalizing Data**
   - Loads input and output data from HDF5 files.
   - Normalizes inputs (per feature) and outputs (by a constant factor) to match the training regime.

3. **Loading Reference Data**
   - Loads reference vectors from HDF5 for comparison or evaluation.

4. **Model Inference**
   - Runs the trained model on normalized inputs to obtain predictions.

5. **Evaluation and Visualization**
   - Compares model predictions to reference data using log-likelihood.
