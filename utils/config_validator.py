"""
Configuration validation utilities
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


def validate_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Validate configuration file for training
    
    Args:
        config: Configuration dictionary
        logger: Optional logger for warnings
        
    Raises:
        KeyError: Missing required configuration sections
        ValueError: Invalid configuration values
        FileNotFoundError: Required paths don't exist
    """
    if logger:
        logger.info("Validating configuration...")
    
    # Check required top-level sections
    required_sections = ['model', 'training', 'data', 'evaluation', 'loss', 'dataset']
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required config section: '{section}'")
    
    # Validate model configuration
    validate_model_config(config['model'])
    
    # Validate training configuration
    validate_training_config(config['training'])
    
    # Validate data configuration
    validate_data_config(config['data'])
    
    # Validate evaluation configuration
    validate_evaluation_config(config['evaluation'])
    
    # Validate loss configuration
    validate_loss_config(config['loss'])
    
    # Validate dataset configuration
    validate_dataset_config(config['dataset'])
    
    # Validate device configuration if present
    if 'device' in config:
        validate_device_config(config['device'])
    
    # Validate logging configuration if present
    if 'logging' in config:
        validate_logging_config(config['logging'])
    
    if logger:
        logger.info("Configuration validation successful!")


def validate_model_config(model_config: Dict[str, Any]) -> None:
    """Validate model configuration"""
    
    # Check input dimension (ESM-2 embedding sizes)
    if 'input_dim' not in model_config:
        raise KeyError("Missing 'input_dim' in model config")
    
    valid_input_dims = [320, 480, 640, 1280, 2560]  # ESM-2 model sizes
    if model_config['input_dim'] not in valid_input_dims:
        raise ValueError(
            f"Invalid input_dim {model_config['input_dim']}. "
            f"Expected one of {valid_input_dims} (ESM-2 embedding dimensions)"
        )
    
    # Check encoder type
    if 'encoder_type' in model_config:
        valid_encoders = ['conservative', 'balanced', 'aggressive']
        if model_config['encoder_type'] not in valid_encoders:
            raise ValueError(
                f"Invalid encoder_type '{model_config['encoder_type']}'. "
                f"Expected one of {valid_encoders}"
            )
    
    # Check encoder dimensions
    encoder_type = model_config.get('encoder_type', 'balanced')
    encoder_dims_key = f'encoder_dims_{encoder_type}'
    if encoder_dims_key in model_config:
        encoder_dims = model_config[encoder_dims_key]
        if not isinstance(encoder_dims, list) or len(encoder_dims) == 0:
            raise ValueError(f"{encoder_dims_key} must be a non-empty list")
        
        # Check dimensions are decreasing
        prev_dim = model_config['input_dim']
        for dim in encoder_dims:
            if dim >= prev_dim:
                raise ValueError(
                    f"Encoder dimensions must be decreasing. "
                    f"Got {dim} >= {prev_dim} in {encoder_dims_key}"
                )
            prev_dim = dim
    
    # Check dropout
    if 'dropout' in model_config:
        dropout = model_config['dropout']
        if not 0 <= dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {dropout}")
    
    # Check temperature
    if 'temperature' in model_config:
        temp = model_config['temperature']
        if temp <= 0:
            raise ValueError(f"Temperature must be positive, got {temp}")
    
    # Check activation
    if 'activation' in model_config:
        valid_activations = ['relu', 'gelu', 'tanh', 'leaky_relu', 'elu']
        if model_config['activation'] not in valid_activations:
            raise ValueError(
                f"Invalid activation '{model_config['activation']}'. "
                f"Expected one of {valid_activations}"
            )


def validate_training_config(training_config: Dict[str, Any]) -> None:
    """Validate training configuration"""
    
    # Check batch size
    if 'batch_size' not in training_config:
        raise KeyError("Missing 'batch_size' in training config")
    if training_config['batch_size'] <= 0:
        raise ValueError(f"Batch size must be positive, got {training_config['batch_size']}")
    
    # Check gradient accumulation steps
    if 'gradient_accumulation_steps' in training_config:
        steps = training_config['gradient_accumulation_steps']
        if steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {steps}")
    
    # Check learning rate
    if 'learning_rate' not in training_config:
        raise KeyError("Missing 'learning_rate' in training config")
    if training_config['learning_rate'] <= 0:
        raise ValueError(f"Learning rate must be positive, got {training_config['learning_rate']}")
    
    # Check num epochs
    if 'num_epochs' not in training_config:
        raise KeyError("Missing 'num_epochs' in training config")
    if training_config['num_epochs'] <= 0:
        raise ValueError(f"Number of epochs must be positive, got {training_config['num_epochs']}")
    
    # Check optimizer
    if 'optimizer' in training_config:
        valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        if training_config['optimizer'] not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer '{training_config['optimizer']}'. "
                f"Expected one of {valid_optimizers}"
            )
    
    # Check scheduler
    if 'scheduler' in training_config:
        valid_schedulers = [
            'warmup_cosine', 'onecycle', 'cosine', 'exponential',
            'polynomial', 'step', 'plateau', 'none'
        ]
        if training_config['scheduler'] not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler '{training_config['scheduler']}'. "
                f"Expected one of {valid_schedulers}"
            )
        
        # Validate scheduler parameters
        if 'scheduler_params' in training_config:
            validate_scheduler_params(
                training_config['scheduler'],
                training_config['scheduler_params']
            )
    
    # Check gradient clipping
    if 'gradient_clip' in training_config:
        clip = training_config['gradient_clip']
        if clip <= 0:
            raise ValueError(f"Gradient clip must be positive, got {clip}")
    
    # Check weight decay
    if 'weight_decay' in training_config:
        decay = training_config['weight_decay']
        if decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {decay}")
    
    # Check early stopping
    if training_config.get('early_stopping', False):
        if 'early_stopping_patience' not in training_config:
            raise KeyError("early_stopping_patience required when early_stopping is True")
        if training_config['early_stopping_patience'] <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, "
                f"got {training_config['early_stopping_patience']}"
            )
    
    # Check calibration settings
    if training_config.get('calibrate_model', False):
        if 'calibration_method' in training_config:
            valid_methods = ['temperature', 'platt']
            if training_config['calibration_method'] not in valid_methods:
                raise ValueError(
                    f"Invalid calibration_method '{training_config['calibration_method']}'. "
                    f"Expected one of {valid_methods}"
                )


def validate_scheduler_params(scheduler_type: str, params: Dict[str, Any]) -> None:
    """Validate scheduler-specific parameters"""
    
    if scheduler_type == 'warmup_cosine':
        if 'warmup_epochs' not in params and 'warmup_steps' not in params:
            raise KeyError("warmup_cosine requires either 'warmup_epochs' or 'warmup_steps'")
    
    elif scheduler_type == 'onecycle':
        if 'pct_start' in params:
            if not 0 < params['pct_start'] < 1:
                raise ValueError(f"pct_start must be in (0, 1), got {params['pct_start']}")
        if 'anneal_strategy' in params:
            if params['anneal_strategy'] not in ['cos', 'linear']:
                raise ValueError(
                    f"anneal_strategy must be 'cos' or 'linear', "
                    f"got '{params['anneal_strategy']}'"
                )
    
    elif scheduler_type == 'exponential':
        if 'gamma' not in params:
            raise KeyError("exponential scheduler requires 'gamma' parameter")
        if not 0 < params['gamma'] <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {params['gamma']}")
    
    elif scheduler_type == 'step':
        if 'step_size' not in params:
            raise KeyError("step scheduler requires 'step_size' parameter")
        if params['step_size'] <= 0:
            raise ValueError(f"step_size must be positive, got {params['step_size']}")


def validate_data_config(data_config: Dict[str, Any]) -> None:
    """Validate data configuration"""
    
    # Check required paths
    required_paths = ['data_path', 'embeddings_path', 'splits_path']
    for path_key in required_paths:
        if path_key not in data_config:
            raise KeyError(f"Missing required path '{path_key}' in data config")
        
        path = data_config[path_key]
        if path and not Path(path).exists():
            raise FileNotFoundError(f"{path_key}: Path not found - {path}")
    
    # Check output directories (create if not exist)
    for dir_key in ['output_dir', 'checkpoint_dir']:
        if dir_key in data_config:
            dir_path = Path(data_config[dir_key])
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)


def validate_evaluation_config(eval_config: Dict[str, Any]) -> None:
    """Validate evaluation configuration"""
    
    # Check metrics
    if 'metrics' in eval_config:
        valid_metrics = [
            'accuracy', 'mcc', 'f1', 'precision', 'recall',
            'auroc', 'auprc', 'balanced_accuracy', 'cohen_kappa'
        ]
        for metric in eval_config['metrics']:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid metric '{metric}'. "
                    f"Expected one of {valid_metrics}"
                )
    
    # Check k_values
    if 'k_values' in eval_config:
        k_values = eval_config['k_values']
        if not isinstance(k_values, list):
            raise ValueError("k_values must be a list")
        for k in k_values:
            if not isinstance(k, int) or k <= 0:
                raise ValueError(f"k_values must contain positive integers, got {k}")
    
    # Check classification threshold
    if 'classification_threshold' in eval_config:
        threshold = eval_config['classification_threshold']
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"classification_threshold must be in [0, 1], got {threshold}"
            )


def validate_loss_config(loss_config: Dict[str, Any]) -> None:
    """Validate loss configuration"""
    
    # Check class prior
    if 'class_prior' not in loss_config:
        raise KeyError("Missing 'class_prior' in loss config")
    prior = loss_config['class_prior']
    if not 0 < prior < 1:
        raise ValueError(f"class_prior must be in (0, 1), got {prior}")
    
    # Check beta
    if 'beta' in loss_config:
        beta = loss_config['beta']
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
    
    # Check gamma
    if 'gamma' in loss_config:
        gamma = loss_config['gamma']
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")


def validate_dataset_config(dataset_config: Dict[str, Any]) -> None:
    """Validate dataset configuration"""
    
    # Check negative ratio
    if 'negative_ratio' in dataset_config:
        ratio = dataset_config['negative_ratio']
        if ratio <= 0:
            raise ValueError(f"negative_ratio must be positive, got {ratio}")
    
    # Check max values
    for key in ['max_markers', 'max_rbps']:
        if key in dataset_config:
            max_val = dataset_config[key]
            if max_val <= 0:
                raise ValueError(f"{key} must be positive, got {max_val}")
    
    # Check num workers
    if 'num_workers' in dataset_config:
        workers = dataset_config['num_workers']
        if workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {workers}")


def validate_device_config(device_config: Dict[str, Any]) -> None:
    """Validate device configuration"""
    
    # Check device type
    if 'device' in device_config:
        valid_devices = ['cuda', 'cpu', 'mps']
        if device_config['device'] not in valid_devices:
            raise ValueError(
                f"Invalid device '{device_config['device']}'. "
                f"Expected one of {valid_devices}"
            )
    
    # Check CUDA device ID
    if 'cuda_device_id' in device_config:
        device_id = device_config['cuda_device_id']
        if device_id < 0:
            raise ValueError(f"cuda_device_id must be non-negative, got {device_id}")


def validate_logging_config(logging_config: Dict[str, Any]) -> None:
    """Validate logging configuration"""
    
    # Check verbosity level
    if 'verbosity' in logging_config:
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if logging_config['verbosity'].lower() not in valid_levels:
            raise ValueError(
                f"Invalid verbosity '{logging_config['verbosity']}'. "
                f"Expected one of {valid_levels}"
            )
    
    # Check log frequency
    if 'log_every_n_batches' in logging_config:
        freq = logging_config['log_every_n_batches']
        if freq <= 0:
            raise ValueError(f"log_every_n_batches must be positive, got {freq}")
    
    # Check wandb settings
    if logging_config.get('use_wandb', False):
        if 'wandb_project' not in logging_config:
            raise KeyError("wandb_project required when use_wandb is True")