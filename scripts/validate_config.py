#!/usr/bin/env python3
"""
Script to validate configuration files
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_validator import validate_config


def main():
    parser = argparse.ArgumentParser(description="Validate configuration file")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)
    
    # Validate configuration
    try:
        validate_config(config)
        print(f"✓ Configuration file '{args.config}' is valid!")
        
        if args.verbose:
            print("\nConfiguration summary:")
            print(f"  Model type: {config['model'].get('encoder_type', 'balanced')}")
            print(f"  Input dimension: {config['model']['input_dim']}")
            print(f"  Batch size: {config['training']['batch_size']}")
            print(f"  Learning rate: {config['training']['learning_rate']}")
            print(f"  Epochs: {config['training']['num_epochs']}")
            print(f"  Optimizer: {config['training'].get('optimizer', 'adamw')}")
            print(f"  Scheduler: {config['training'].get('scheduler', 'none')}")
            print(f"  Class prior: {config['loss']['class_prior']}")
            print(f"  Negative ratio: {config['dataset'].get('negative_ratio', 1.0)}")
            
    except KeyError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Invalid value: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()