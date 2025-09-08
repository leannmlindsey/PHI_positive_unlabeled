"""
Logging utilities for training and evaluation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


def setup_logger(
    name: str = __name__,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Specific log file name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_dir / log_file}")
    
    return logger


class MetricLogger:
    """
    Logger for tracking training metrics
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """
        Initialize metric logger
        
        Args:
            log_dir: Directory to save metrics
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        
        self.metrics = {
            'train': [],
            'val': [],
            'test': []
        }
        
        self.current_epoch = 0
        
    def log_metrics(self, 
                   metrics: Dict[str, Any], 
                   phase: str = 'train',
                   epoch: Optional[int] = None):
        """
        Log metrics for a specific phase
        
        Args:
            metrics: Dictionary of metrics
            phase: One of 'train', 'val', 'test'
            epoch: Epoch number (uses current_epoch if None)
        """
        if epoch is None:
            epoch = self.current_epoch
        else:
            self.current_epoch = epoch
            
        # Add epoch to metrics
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Store metrics
        self.metrics[phase].append(metrics)
        
        # Save to file
        self.save()
        
    def save(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def load(self):
        """Load metrics from JSON file"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
                
    def get_best_metric(self, 
                       metric_name: str,
                       phase: str = 'val',
                       mode: str = 'max') -> Dict[str, Any]:
        """
        Get the best metric value and corresponding epoch
        
        Args:
            metric_name: Name of the metric
            phase: Phase to check ('train', 'val', 'test')
            mode: 'max' for higher is better, 'min' for lower is better
            
        Returns:
            Dictionary with best value and epoch
        """
        if not self.metrics[phase]:
            return None
            
        values = [(m.get(metric_name, None), m.get('epoch', 0)) 
                 for m in self.metrics[phase]]
        values = [(v, e) for v, e in values if v is not None]
        
        if not values:
            return None
            
        if mode == 'max':
            best_value, best_epoch = max(values, key=lambda x: x[0])
        else:
            best_value, best_epoch = min(values, key=lambda x: x[0])
            
        return {
            'value': best_value,
            'epoch': best_epoch,
            'metric': metric_name,
            'phase': phase
        }


def log_model_info(model, logger: logging.Logger):
    """
    Log model architecture information
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Log architecture
    logger.info("Model Architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            logger.info(f"  {name}: {module}")


def log_training_config(config: Dict[str, Any], logger: logging.Logger):
    """
    Log training configuration
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Training Configuration:")
    
    def log_dict(d: dict, prefix: str = "  "):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    log_dict(config)