"""
=============================================================================
Latency-Liability Trap: Monte Carlo Simulation
=============================================================================
Paper: "The Trust-Automation Paradox: Autonomous Cryptoeconomic Governance
        for Regulatory Compliance in High-Risk Artificial Intelligence Systems"
Author: Nikhil Varma
Journal: AI & SOCIETY (Springer)

Description
-----------
This module implements a Monte Carlo simulation to quantify the total
regulatory liability exposure accrued during a rogue AI bias event under
three governance regimes:

    Regime A  —  Human-in-the-Loop (HITL) auditing
    Regime B  —  Passive Smart Contract (blockchain logging)
    Regime C  —  Autonomous Cryptoeconomic Governance (ACG, proposed)

The core liability model is:

    L_total = (T_d + T_m) * V * b * P          [Eq. 2 in paper]

where:
    T_d  = time to detection (minutes)
    T_m  = time to mitigation after detection (minutes)
    V    = transaction volume (decisions per minute)
    b    = bias rate (proportion of decisions biased during drift)
    P    = regulatory penalty per biased decision (USD)

All five variables are treated as stochastic with empirically motivated
distributional assumptions. See docs/simulation_methodology.md for full
derivations and parameter justification.

Usage
-----
    python src/simulation.py                   # run with defaults
    python src/simulation.py --n 100000        # custom iteration count
    python src/simulation.py --seed 0 --n 50000 --output results/

Dependencies
------------
    numpy >= 1.22
    matplotlib >= 3.5
    scipy >= 1.8
    pandas >= 1.4

License
-------
MIT License. See LICENSE for details.
=============================================================================
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Simulation control
    "n_simulations": 50_000,
    "random_seed": 42,
    # Transaction volume: Normal(mu, sigma) decisions per minute
    "V_mean": 100.0,
    "V_std": 15.0,
    "V_min_clip": 20.0,
    "V_max_clip": 500.0,
    # Penalty per biased decision (USD): LogNormal(log_mean, log_std)
    "P_log_mean": None,       # computed from P_mean at runtime
    "P_mean": 850.0,          # mean penalty in USD
    "P_log_std": 0.6,
    # Bias rate (proportion): Beta(alpha, beta)
    "bias_alpha": 2.5,
    "bias_beta": 10.0,        # implied mean ~ 0.20
    # Regime A (HITL) — detection lag: Uniform[low, high] minutes
    "Td_A_low": 120.0,        # 2 hours
    "Td_A_high": 4320.0,      # 72 hours
    # Regime A — mitigation lag: Uniform[low, high] minutes
    "Tm_A_low": 60.0,         # 1 hour
    "Tm_A_high": 480.0,       # 8 hours
    # Regime B (Passive SC) — detection lag: Uniform[low, high] minutes
    "Td_B_low": 30.0,         # 30 minutes
    "Td_B_high": 480.0,       # 8 hours
    # Regime B — mitigation lag: Uniform[low, high] minutes
    "Tm_B_low": 30.0,
    "Tm_B_high": 240.0,       # 4 hours
    # Regime C (ACG) — detection lag: Uniform[low, high] minutes
    # Algorand block finality ~ 3.4s; model as 1-2 block confirmations
    "Td_C_low": 0.03,         # ~1.8 seconds
    "Td_C_high": 0.12,        # ~7.2 seconds
    # Regime C — mitigation lag: 0 (NIC blocks automatically)
    "Tm_C": 0.0,
}


# =============================================================================
# CORE SIMULATION
# =============================================================================

class LatencyLiabilitySimulation:
    """
    Monte Carlo simulation of the Latency-Liability Trap across three
    governance regimes.

    Parameters
    ----------
    config : dict
        Simulation configuration. Defaults to DEFAULT_CONFIG.

    Attributes
    ----------
    results : dict
        Raw simulation arrays for L_A, L_B, L_C after calling run().
    summary : pd.DataFrame
        Descriptive statistics table after calling run().
    """

    def __init__(self, config: dict = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        # Compute log_mean from P_mean so that E[P] = P_mean
        self.cfg["P_log_mean"] = (
            np.log(self.cfg["P_mean"])
            - 0.5 * self.cfg["P_log_std"] ** 2
        )
        self.results = {}
        self.summary = None
        self._rng = np.random.default_rng(self.cfg["random_seed"])

    # ------------------------------------------------------------------
    def _sample_shared(self, n: int):
        """
        Draw shared stochastic variables common to all regimes.

        Returns
        -------
        V : np.ndarray  transaction volume (decisions/min)
        b : np.ndarray  bias rate (proportion)
        P : np.ndarray  penalty per biased decision (USD)
        """
        V = self._rng.normal(
            self.cfg["V_mean"], self.cfg["V_std"], n
        ).clip(self.cfg["V_min_clip"], self.cfg["V_max_clip"])

        b = self._rng.beta(
            self.cfg["bias_alpha"], self.cfg["bias_beta"], n
        )

        P = self._rng.lognormal(
            self.cfg["P_log_mean"], self.cfg["P_log_std"], n
        )

        return V, b, P

    # ------------------------------------------------------------------
    def _liability(self, Td, Tm, V, b, P):
        """
        Compute total liability per simulation run.

        Implements Equation 2 from the paper:
            L = (T_d + T_m) * V * b * P

        Parameters
        ----------
        Td, Tm : np.ndarray  detection and mitigation lags (minutes)
        V      : np.ndarray  transaction volume (decisions/min)
        b      : np.ndarray  bias rate
        P      : np.ndarray  penalty per biased decision (USD)

        Returns
        -------
        L : np.ndarray  total liability (USD)
        """
        return (Td + Tm) * V * b * P

    # ------------------------------------------------------------------
    def _marginal_rate(self, V, b, P):
        """
        Marginal liability rate (USD per minute of governance latency).

        Implements Equation 3: dL/dt = V * b * P
        """
        return V * b * P

    # ------------------------------------------------------------------
    def run(self):
        """
        Execute the full Monte Carlo simulation for all three regimes.

        Results are stored in self.results and self.summary.
        """
        n = self.cfg["n_simulations"]
        cfg = self.cfg

        # Shared draws
        V, b, P = self._sample_shared(n)

        # Regime-specific detection and mitigation lags
        Td_A = self._rng.uniform(cfg["Td_A_low"], cfg["Td_A_high"], n)
        Tm_A = self._rng.uniform(cfg["Tm_A_low"], cfg["Tm_A_high"], n)

        Td_B = self._rng.uniform(cfg["Td_B_low"], cfg["Td_B_high"], n)
        Tm_B = self._rng.uniform(cfg["Tm_B_low"], cfg["Tm_B_high"], n)

        Td_C = self._rng.uniform(cfg["Td_C_low"], cfg["Td_C_high"], n)
        Tm_C = np.zeros(n)

        # Liability arrays
        L_A = self._liability(Td_A, Tm_A, V, b, P)
        L_B = self._liability(Td_B, Tm_B, V, b, P)
        L_C = self._liability(Td_C, Tm_C, V, b, P)

        # Marginal rate (baseline, no stochasticity needed here)
        baseline_rate = (
            cfg["V_mean"]
            * (cfg["bias_alpha"] / (cfg["bias_alpha"] + cfg["bias_beta"]))
            * cfg["P_mean"]
        )

        self.results = {
            "L_A": L_A, "L_B": L_B, "L_C": L_C,
            "Td_A": Td_A, "Td_B": Td_B, "Td_C": Td_C,
            "Tm_A": Tm_A, "Tm_B": Tm_B,
            "V": V, "b": b, "P": P,
            "marginal_rate_baseline_USD_per_min": baseline_rate,
        }

        self._build_summary()
        return self

    # ------------------------------------------------------------------
    def _build_summary(self):
        """Build descriptive statistics DataFrame."""
        rows = []
        for name, L in [
            ("Regime A: HITL", self.results["L_A"]),
            ("Regime B: Passive SC", self.results["L_B"]),
            ("Regime C: ACG (Proposed)", self.results["L_C"]),
        ]:
            rows.append({
                "Regime": name,
                "Mean (USD)": np.mean(L),
                "Median (USD)": np.median(L),
                "Std Dev (USD)": np.std(L),
                "P5 (USD)": np.percentile(L, 5),
                "P25 (USD)": np.percentile(L, 25),
                "P75 (USD)": np.percentile(L, 75),
                "P95 (USD)": np.percentile(L, 95),
                "P99 (USD)": np.percentile(L, 99),
                "Min (USD)": np.min(L),
                "Max (USD)": np.max(L),
            })
        self.summary = pd.DataFrame(rows).set_index("Regime")

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print a formatted console summary of simulation results."""
        cfg = self.cfg
        rate = self.results["marginal_rate_baseline_USD_per_min"]

        print("\n" + "=" * 70)
        print("  LATENCY-LIABILITY TRAP — MONTE CARLO SIMULATION RESULTS")
        print("=" * 70)
        print(f"  Iterations (N)       : {cfg['n_simulations']:,}")
        print(f"  Random seed          : {cfg['random_seed']}")
        print(f"  V ~ Normal({cfg['V_mean']}, {cfg['V_std']}) decisions/min")
        print(f"  b ~ Beta({cfg['bias_alpha']}, {cfg['bias_beta']})  "
              f"[mean = {cfg['bias_alpha']/(cfg['bias_alpha']+cfg['bias_beta']):.3f}]")
        print(f"  P ~ LogNormal  [mean = ${cfg['P_mean']:,.0f}/decision]")
        print(f"  Baseline marginal liability rate: ${rate:,.0f}/minute")
        print("-" * 70)

        for regime, L in [
            ("Regime A  (HITL)", self.results["L_A"]),
            ("Regime B  (Passive SC)", self.results["L_B"]),
            ("Regime C  (ACG)", self.results["L_C"]),
        ]:
            print(f"\n  {regime}")
            print(f"    Mean:    ${np.mean(L):>18,.0f}")
            print(f"    Median:  ${np.median(L):>18,.0f}")
            print(f"    P95:     ${np.percentile(L, 95):>18,.0f}")
            print(f"    P99:     ${np.percentile(L, 99):>18,.0f}")
            print(f"    Std Dev: ${np.std(L):>18,.0f}")

        print("\n" + "=" * 70)
        Td_A_med = np.median(self.results["Td_A"])
        Td_B_med = np.median(self.results["Td_B"])
        Td_C_med = np.median(self.results["Td_C"])
        print("  Median Detection Lag:")
        print(f"    Regime A: {Td_A_med/60:.1f} hours")
        print(f"    Regime B: {Td_B_med/60:.2f} hours")
        print(f"    Regime C: {Td_C_med*60:.1f} seconds")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    def save_results(self, output_dir: str = "results/"):
        """
        Save simulation outputs to CSV and JSON.

        Files written
        -------------
        results/simulation_summary.csv   — descriptive statistics table
        results/simulation_raw.csv       — per-iteration liability values
        results/simulation_config.json   — configuration used
        """
        os.makedirs(output_dir, exist_ok=True)

        # Summary CSV
        summary_path = os.path.join(output_dir, "simulation_summary.csv")
        self.summary.to_csv(summary_path)
        print(f"  Summary saved   -> {summary_path}")

        # Raw data CSV (L_A, L_B, L_C per iteration)
        raw_df = pd.DataFrame({
            "L_A_USD": self.results["L_A"],
            "L_B_USD": self.results["L_B"],
            "L_C_USD": self.results["L_C"],
            "Td_A_min": self.results["Td_A"],
            "Td_B_min": self.results["Td_B"],
            "Td_C_min": self.results["Td_C"],
            "V_dec_per_min": self.results["V"],
            "bias_rate": self.results["b"],
            "penalty_USD": self.results["P"],
        })
        raw_path = os.path.join(output_dir, "simulation_raw.csv")
        raw_df.to_csv(raw_path, index=False)
        print(f"  Raw data saved  -> {raw_path}")

        # Config JSON
        cfg_out = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in self.cfg.items()
        }
        cfg_path = os.path.join(output_dir, "simulation_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg_out, f, indent=2)
        print(f"  Config saved    -> {cfg_path}")


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(base_config: dict = None, perturbation: float = 0.20,
                          n: int = 20_000) -> pd.DataFrame:
    """
    One-at-a-time (OAT) sensitivity analysis for Regime A mean liability.

    Each parameter is perturbed by +/- `perturbation` fraction while all
    others are held at their baseline values. The resulting change in mean
    liability is recorded.

    Parameters
    ----------
    base_config  : dict    Base configuration (defaults to DEFAULT_CONFIG).
    perturbation : float   Fractional perturbation (default 0.20 = 20%).
    n            : int     Iterations per sensitivity run (default 20,000).

    Returns
    -------
    pd.DataFrame  with columns [Parameter, Baseline_Mean, Low_Mean,
                                High_Mean, Low_Delta, High_Delta]
    """
    cfg = {**DEFAULT_CONFIG, **(base_config or {}), "n_simulations": n,
           "random_seed": 0}

    params = {
        "V_mean":     "V: Transaction Volume (decisions/min)",
        "P_mean":     "P: Penalty per Decision (USD)",
        "bias_alpha": "b: Bias Rate (alpha parameter)",
        "Td_A_low":   "Td_A: Detection Lag lower bound (min)",
        "Td_A_high":  "Td_A: Detection Lag upper bound (min)",
    }

    # Baseline
    base_sim = LatencyLiabilitySimulation(cfg).run()
    base_mean = float(np.mean(base_sim.results["L_A"]))

    rows = []
    for param_key, param_label in params.items():
        # Low perturbation
        low_cfg = {**cfg, param_key: cfg[param_key] * (1 - perturbation)}
        low_mean = float(
            np.mean(LatencyLiabilitySimulation(low_cfg).run().results["L_A"])
        )
        # High perturbation
        high_cfg = {**cfg, param_key: cfg[param_key] * (1 + perturbation)}
        high_mean = float(
            np.mean(LatencyLiabilitySimulation(high_cfg).run().results["L_A"])
        )
        rows.append({
            "Parameter": param_label,
            "Baseline_Mean_USD": base_mean,
            "Low_Mean_USD": low_mean,
            "High_Mean_USD": high_mean,
            "Low_Delta_USD": low_mean - base_mean,
            "High_Delta_USD": high_mean - base_mean,
        })

    return pd.DataFrame(rows)


# =============================================================================
# ROGUE EVENT TRAJECTORY
# =============================================================================

def rogue_event_trajectory(
    V: float = 100.0,
    b: float = 0.22,
    P: float = 850.0,
    duration_hours: float = 10.0,
    resolution: int = 3000,
    regime_params: dict = None,
) -> pd.DataFrame:
    """
    Compute liability accumulation trajectories for a single deterministic
    rogue AI bias event under all three governance regimes.

    This function implements the time-integral form of Equation 1:

        L(t) = integral_{t_drift}^{min(t, T_d + T_m)} V * b * P dt

    Parameters
    ----------
    V              : float   Transaction volume (decisions/min). Default 100.
    b              : float   Bias rate. Default 0.22.
    P              : float   Penalty per biased decision (USD). Default 850.
    duration_hours : float   Simulation horizon (hours). Default 10.
    resolution     : int     Number of time steps. Default 3000.
    regime_params  : dict    Override detection/mitigation times (minutes).
                             Keys: Td_A, Tm_A, Td_B, Tm_B, Td_C.

    Returns
    -------
    pd.DataFrame with columns [t_min, t_hours, L_A, L_B, L_C]
    """
    defaults = {
        "Td_A": 400.0, "Tm_A": 150.0,
        "Td_B": 120.0, "Tm_B": 90.0,
        "Td_C": 0.08,
    }
    p = {**defaults, **(regime_params or {})}

    t = np.linspace(0, duration_hours * 60, resolution)
    rate = V * b * P  # USD per minute

    def accum(t_arr, Td, Tm):
        return np.where(
            t_arr < Td,
            0.0,
            np.where(
                t_arr < Td + Tm,
                (t_arr - Td) * rate,
                Tm * rate
            )
        )

    return pd.DataFrame({
        "t_min": t,
        "t_hours": t / 60.0,
        "L_A_USD": accum(t, p["Td_A"], p["Tm_A"]),
        "L_B_USD": accum(t, p["Td_B"], p["Tm_B"]),
        "L_C_USD": accum(t, p["Td_C"], 0.0),
        "marginal_rate_USD_per_min": rate,
    })


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latency-Liability Trap Monte Carlo Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n",      type=int,   default=50_000,
                        help="Number of Monte Carlo iterations")
    parser.add_argument("--seed",   type=int,   default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str,   default="results/",
                        help="Output directory for CSV and JSON files")
    parser.add_argument("--V",      type=float, default=100.0,
                        help="Mean transaction volume (decisions/min)")
    parser.add_argument("--P",      type=float, default=850.0,
                        help="Mean penalty per biased decision (USD)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis in addition to main simulation")
    parser.add_argument("--trajectory", action="store_true",
                        help="Export rogue event trajectory CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "n_simulations": args.n,
        "random_seed": args.seed,
        "V_mean": args.V,
        "P_mean": args.P,
    }

    print(f"\nRunning Monte Carlo simulation (N={args.n:,}, seed={args.seed})...")
    sim = LatencyLiabilitySimulation(config).run()
    sim.print_summary()
    sim.save_results(args.output)

    if args.sensitivity:
        print("\nRunning sensitivity analysis (20% OAT perturbations)...")
        sens_df = sensitivity_analysis(config, n=20_000)
        sens_path = os.path.join(args.output, "sensitivity_analysis.csv")
        sens_df.to_csv(sens_path, index=False)
        print(f"  Sensitivity saved -> {sens_path}")
        print(sens_df[["Parameter", "Low_Delta_USD", "High_Delta_USD"]].to_string(index=False))

    if args.trajectory:
        print("\nGenerating rogue event trajectory...")
        traj_df = rogue_event_trajectory(V=args.V, b=0.22, P=args.P)
        traj_path = os.path.join(args.output, "rogue_event_trajectory.csv")
        traj_df.to_csv(traj_path, index=False)
        print(f"  Trajectory saved  -> {traj_path}")
        final = traj_df.iloc[-1]
        print(f"  Final liability — A: ${final['L_A_USD']:,.0f}  "
              f"B: ${final['L_B_USD']:,.0f}  C: ${final['L_C_USD']:,.0f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
