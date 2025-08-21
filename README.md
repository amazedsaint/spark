# SPaR-K: Structure-Pseudo-Randomness with Kinetic Attention

A novel Transformer architecture that combines Feynman-Kac attention, structure-pseudo-randomness decomposition, and verifier heads for enhanced reasoning capabilities.

**Author:** Anoop Madhusudanan (amazedsaint@gmail.com)

## Key Features

- **Feynman-Kac Attention**: Path-integral formulation for multi-hop reasoning
- **SPD Router**: Structure vs pseudo-randomness decomposition
- **Verifier Head**: Stack-based algorithmic verification
- **Proven Results**: Perfect K-hop reachability (AUC 1.00), +11 dB SNR improvement

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/spark_config.yaml
```

## Architecture Components

1. `src/feynman_kac_attention.py` - FK attention implementation
2. `src/spd_router.py` - Structure-pseudo-randomness decomposition
3. `src/verifier_head.py` - Stack-based verification
4. `src/spark_transformer.py` - Complete SPaR-K block

## Experiments

Run validation experiments:
```bash
python experiments/k_hop_reachability.py
python experiments/snr_validation.py
python experiments/long_context_test.py
```

## Citation

```bibtex
@article{madhusudanan2025spark,
  title={SPaR-K: Structure-Pseudo-Randomness with Kinetic Attention for Enhanced Transformer Reasoning},
  author={Madhusudanan, Anoop},
  year={2025}
}
```