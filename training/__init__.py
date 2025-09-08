"""
Training components for phage-host interaction prediction
"""

from .dataset import PhageHostDataset, PhageHostDataModule

__all__ = [
    'PhageHostDataset',
    'PhageHostDataModule'
]