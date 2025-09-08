"""
Model components for phage-host interaction prediction
"""

from .encoders import TwoTowerEncoder, ProteinEncoder
from .mil_model import MILModel
from .losses import nnPULoss

__all__ = [
    'TwoTowerEncoder',
    'ProteinEncoder', 
    'MILModel',
    'nnPULoss'
]