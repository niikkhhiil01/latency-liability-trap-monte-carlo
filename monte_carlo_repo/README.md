# Latency-Liability Trap: Monte Carlo Simulation

## The Trust-Automation Paradox: Autonomous Cryptoeconomic Governance for Regulatory Compliance in High-Risk Artificial Intelligence Systems

**Author:** Dr. Nikhil Varma  
**Affiliation:** Ramapo College of New Jersey, Anisfield School of Business 
**Journal:** *AI & Society* (Springer), submitted March 2026  

---

## Overview

This repository contains the full source code for the Monte Carlo simulation underpinning the empirical methodology of the paper. The simulation quantifies **regulatory liability exposure** accrued during a rogue AI bias event under three governance regimes:

| Regime | Name | Description |
|--------|------|-------------|
| **A** | Human-in-the-Loop (HITL) | Human auditors review flagged decisions; detection latency 2–72 hours |
| **B** | Passive Smart Contract | Blockchain logs decisions immutably; detection latency 30 min–8 hours |
| **C** | Autonomous Cryptoeconomic Governance (ACG) | Proposed framework; zk-SNARK proofs at Algorand finality (~3–7 seconds) |

### Core Liability Equation

The simulation operationalises the **Latency-Liability Trap** equation from the paper:

```
L_total = (T_d + T_m) × V × b × P          [Eq. 2]
```

Where:
- `T_d` = Time to detection (minutes, stochastic)
- `T_m` = Time to mitigation after detection (minutes, stochastic)
- `V` = Transaction volume (decisions/minute, ~Normal(100, 15))
- `b` = Bias rate (proportion of biased decisions, ~Beta(2.5, 10))
- `P` = Regulatory penalty per biased decision (USD, ~LogNormal)

---

## Repository Structure

```
monte_carlo_repo/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── src/
│   └── simulation.py                # Main simulation module (540 lines)
├── docs/
│   └── simulation_methodology.md   # Full parameter derivations & justification
├── figures/                         # Generated black-and-white figures (PNG)
│   ├── fig1_architecture.png        # ACG four-layer architecture
│   ├── fig2_sequence.png            # NIC sequence diagram
│   ├── fig3_quantization.png        # XAI-to-zkML quantization pipeline
│   ├── fig4_distributions.png       # Input distribution PDFs (A, B, C)
│   ├── fig5_cdf.png                 # Cumulative liability CDFs
│   ├── fig6_rogue_event.png         # Rogue AI event trajectory
│   ├── fig7_boxplot.png             # Box-and-whisker comparison
│   └── fig8_sensitivity.png        # Tornado sensitivity chart
└── results/
    └── sim_results.json             # Pre-computed simulation statistics (N=50,000)
```

---

## Key Results (N = 50,000 simulations)

| Metric | Regime A (HITL) | Regime B (Passive SC) | Regime C (ACG) |
|--------|----------------|-----------------------|----------------|
| **Mean Liability** | $50,735,509 | $7,972,754 | **$1,529** |
| **Median** | $32,126,361 | $5,350,346 | $1,029 |
| **95th Percentile** | $158,137,932 | $23,749,512 | $4,494 |
| **99th Percentile** | $287,434,667 | $41,794,742 | $8,080 |
| **Std Deviation** | $59,759,340 | $8,599,037 | $1,643 |
| **Detection Latency (median)** | ~36.9 hours | ~4.3 hours | **~4.5 seconds** |

> **ACG reduces mean liability by 99.997% relative to HITL and 99.98% relative to Passive Smart Contracts.**

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Simulation

```bash
# Default: 50,000 iterations, seed=42
python src/simulation.py

# Custom parameters
python src/simulation.py --n 100000 --seed 0 --output results/

# Fast test run
python src/simulation.py --n 1000
```

### Output

The script produces:
1. `results/sim_results.json` — Summary statistics for all three regimes
2. `figures/fig4_distributions.png` through `fig8_sensitivity.png` — Publication-quality B&W figures
3. Console summary table with mean, median, P95, P99 per regime

---

## Simulation Configuration

All parameters are defined in `DEFAULT_CONFIG` within `src/simulation.py`. Key settings:

```python
DEFAULT_CONFIG = {
    "n_simulations":  50_000,
    "random_seed":    42,

    # Transaction volume V ~ Normal(100, 15) decisions/min
    "V_mean": 100.0,   "V_std": 15.0,

    # Penalty P ~ LogNormal,  mean = $850/biased decision
    "P_mean": 850.0,   "P_log_std": 0.6,

    # Bias rate b ~ Beta(2.5, 10),  implied mean ~20%
    "bias_alpha": 2.5, "bias_beta": 10.0,

    # Regime A detection window: Uniform[2h, 72h]
    "Td_A_low": 120.0, "Td_A_high": 4320.0,

    # Regime B detection window: Uniform[30min, 8h]
    "Td_B_low": 30.0,  "Td_B_high": 480.0,

    # Regime C detection window: Uniform[1.8s, 7.2s] — Algorand finality
    "Td_C_low": 0.03,  "Td_C_high": 0.12,
}
```

See `docs/simulation_methodology.md` for full statistical justification.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{varma2025trust,
  title   = {The Trust-Automation Paradox: Autonomous Cryptoeconomic Governance
             for Regulatory Compliance in High-Risk Artificial Intelligence Systems},
  author  = {Varma, Nikhil; Surti, Chirag},
  journal = {AI \& Society},
  year    = {2026},
  publisher = {Springer},
  note    = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Dr. Nikhil Varma — nvarma@ramapo.edu
Ramapo College of New Jersey 

