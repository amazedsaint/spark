# SPaR-K: Structure-Pseudo-Randomness with Kinetic Attention
# Author: Anoop Madhusudanan (amazedsaint@gmail.com)

from .feynman_kac_attention import FeynmanKacAttention, MultiHeadFeynmanKacAttention
from .spd_router import SPDRouter, AdaptiveSPDRouter
from .verifier_head import VerifierHead, DifferentiableStack
from .spark_transformer import SPaRKTransformerBlock, SPaRKTransformer

__version__ = "1.0.0"
__author__ = "Anoop Madhusudanan"
__email__ = "amazedsaint@gmail.com"

__all__ = [
    "FeynmanKacAttention",
    "MultiHeadFeynmanKacAttention", 
    "SPDRouter",
    "AdaptiveSPDRouter",
    "VerifierHead",
    "DifferentiableStack",
    "SPaRKTransformerBlock",
    "SPaRKTransformer"
]