"""Numerical results for AI adoption levies and worker transitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, linprog


@dataclass(frozen=True)
class Economy:
    name: str
    masses: np.ndarray
    productivity: np.ndarray
    labels: tuple[str, ...]


DELTA = 1.0
KAPPA = 0.30
PREPARATION = 2.0 / 75.0
LEVY_0 = 0.0
LEVY_1 = 0.18
LOSS = 0.08
SLOW_HAZARD = 0.45
FAST_HAZARD = 1.8
HORIZON = 5.0

ECONOMIES = {
    "A": Economy(
        name="A",
        masses=np.array([0.4595050199, 0.5404949801]),
        productivity=np.array([0.1145590712, 2.2766666667]),
        labels=("Low", "High"),
    ),
    "B": Economy(
        name="B",
        masses=np.array([0.1496403491, 0.3444243454, 0.5059353055]),
        productivity=np.array([0.0, 0.2316962525, 2.2766666667]),
        labels=("Inactive", "Middle", "High"),
    ),
}


def optimal_effort(productivity: np.ndarray, levy: float) -> np.ndarray:
    surplus = np.maximum(np.asarray(productivity) - levy - DELTA * PREPARATION, 0.0)
    return -DELTA + np.sqrt(DELTA**2 + 2.0 * surplus / KAPPA)


def adoption_share(effort: np.ndarray) -> np.ndarray:
    return effort / (DELTA + effort)


def flow_resources(economy: Economy, effort: np.ndarray, share: np.ndarray) -> float:
    effort_cost = KAPPA * effort**2 / 2.0 + PREPARATION * effort
    return float(np.dot(economy.masses, economy.productivity * share - effort_cost * (1.0 - share)))


def steady_welfare(economy: Economy, levy: float, loss: float = LOSS,
                   hazard: float = 1.0) -> float:
    effort = optimal_effort(economy.productivity, levy)
    share = adoption_share(effort)
    private = flow_resources(economy, effort, share)
    unemployment = np.dot(economy.masses, DELTA * share / hazard)
    return float(private - loss * unemployment)


def private_change(economy: Economy, horizon: float) -> float:
    a0 = optimal_effort(economy.productivity, LEVY_0)
    a1 = optimal_effort(economy.productivity, LEVY_1)
    s0 = adoption_share(a0)
    s1 = adoption_share(a1)
    d = s0 - s1
    baseline = flow_resources(economy, a0, s0)
    final = flow_resources(economy, a1, s1)
    decay = DELTA + a1
    effort_cost = KAPPA * a1**2 / 2.0 + PREPARATION * a1
    adjustment = np.dot(economy.masses, (economy.productivity + effort_cost) * d * (-np.expm1(-decay * horizon)) / (decay * horizon))
    return float(final - baseline + adjustment)


def displacement_kernel(economy: Economy, horizon: float, hazards: np.ndarray,
                        loss: float = LOSS) -> np.ndarray:
    a0 = optimal_effort(economy.productivity, LEVY_0)
    a1 = optimal_effort(economy.productivity, LEVY_1)
    d = adoption_share(a0) - adoption_share(a1)
    lam = (DELTA + a1)[:, None]
    mu = np.asarray(hazards)[None, :]
    h = float(horizon)
    mean_decay = -np.expm1(-mu * h) / mu
    first = DELTA * (h - mean_decay)
    second = -np.expm1(-lam * h) / lam
    difference = lam - mu
    convolution = np.empty(np.broadcast_shapes(lam.shape, mu.shape))
    np.divide(np.exp(-mu * h) - np.exp(-lam * h), difference,
              out=convolution, where=np.abs(difference) > 1e-10)
    equal = np.abs(difference) <= 1e-10
    convolution[equal] = (h * np.exp(-np.broadcast_to(mu, convolution.shape) * h))[equal]
    result = loss * d[:, None] * (first + a1[:, None] * (second - convolution)) / (h * mu)
    return result


def transport_bounds(cost: np.ndarray, row_mass: np.ndarray,
                     column_mass: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    n, m = cost.shape
    assert np.isclose(row_mass.sum(), 1.0, atol=1e-8)
    assert np.isclose(column_mass.sum(), 1.0, atol=1e-8)
    equalities = np.zeros((n + m, n * m))
    for i in range(n):
        equalities[i, i * m:(i + 1) * m] = 1.0
    for j in range(m):
        equalities[n + j, j::m] = 1.0
    marginal = np.r_[row_mass, column_mass]
    low = linprog(cost.ravel(), A_eq=equalities, b_eq=marginal,
                  bounds=(0, None), method="highs")
    high = linprog(-cost.ravel(), A_eq=equalities, b_eq=marginal,
                   bounds=(0, None), method="highs")
    if not (low.success and high.success):
        raise RuntimeError("Transport optimization failed")
    lower = low.x.reshape(cost.shape)
    upper = high.x.reshape(cost.shape)
    assert np.allclose(lower.sum(axis=1), row_mass, atol=1e-9)
    assert np.allclose(lower.sum(axis=0), column_mass, atol=1e-9)
    assert np.allclose(upper.sum(axis=1), row_mass, atol=1e-9)
    assert np.allclose(upper.sum(axis=0), column_mass, atol=1e-9)
    low_potential = low.eqlin.marginals
    high_potential = -high.eqlin.marginals
    assert np.max(low_potential[:n, None] + low_potential[None, n:] - cost) < 1e-8
    assert np.max(cost - high_potential[:n, None] - high_potential[None, n:]) < 1e-8
    assert abs(low.fun - (row_mass @ low_potential[:n] + column_mass @ low_potential[n:])) < 1e-8
    assert abs(-high.fun - (row_mass @ high_potential[:n] + column_mass @ high_potential[n:])) < 1e-8
    return float(low.fun), float(-high.fun), lower, upper


def identified_welfare(economy: Economy, horizon: float,
                       hazards: np.ndarray, proportions: np.ndarray,
                       loss: float = LOSS) -> tuple[float, float, np.ndarray, np.ndarray]:
    kernel = displacement_kernel(economy, horizon, hazards, loss)
    lower, upper, low_pi, high_pi = transport_bounds(kernel, economy.masses, proportions)
    private = private_change(economy, horizon)
    return private + lower, private + upper, low_pi, high_pi


def common_hazard_effect(economy: Economy, horizon: float, hazard: float = 1.0) -> float:
    cost = displacement_kernel(economy, horizon, np.array([hazard]))[:, 0]
    return private_change(economy, horizon) + float(economy.masses @ cost)


def marginal_stationary_gain(economy: Economy, levy: float) -> float:
    return steady_welfare(economy, levy) - steady_welfare(economy, LEVY_0)


def grid_bounds(economy: Economy, rates: np.ndarray) -> tuple[float, float]:
    adoption = np.array([float(economy.masses @ adoption_share(optimal_effort(economy.productivity, x))) for x in rates])
    widths = np.diff(rates)
    lo_integral = widths @ adoption[1:]
    hi_integral = widths @ adoption[:-1]
    private_endpoints = rates[-1] * adoption[-1] - rates[0] * adoption[0]
    unemployment_effect = LOSS * (adoption[0] - adoption[-1])
    return (float(private_endpoints - hi_integral + unemployment_effect),
            float(private_endpoints - lo_integral + unemployment_effect))


def phase_bounds(economy: Economy, horizon: float, slow_share: float,
                 loss: float) -> tuple[float, float]:
    cost = displacement_kernel(economy, horizon, np.array([SLOW_HAZARD, FAST_HAZARD]), loss)
    baseline = float(economy.masses @ cost[:, 1])
    premiums = cost[:, 0] - cost[:, 1]
    remaining = float(slow_share)
    lower = baseline
    for i in np.argsort(premiums):
        take = min(remaining, economy.masses[i])
        lower += take * premiums[i]
        remaining -= take
    remaining = float(slow_share)
    upper = baseline
    for i in np.argsort(-premiums):
        take = min(remaining, economy.masses[i])
        upper += take * premiums[i]
        remaining -= take
    private = private_change(economy, horizon)
    return private + lower, private + upper


def figure_contrast(directory: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.0))
    for j, economy in enumerate(ECONOMIES.values()):
        initial = economy.masses @ adoption_share(optimal_effort(economy.productivity, LEVY_0))
        after = economy.masses @ adoption_share(optimal_effort(economy.productivity, LEVY_1))
        ax1.plot([0, 1], [initial, after], marker='o', linewidth=1.6, label=f"Economy {economy.name}")
        private = flow_resources(economy, optimal_effort(economy.productivity, LEVY_1), adoption_share(optimal_effort(economy.productivity, LEVY_1))) - flow_resources(economy, optimal_effort(economy.productivity, LEVY_0), adoption_share(optimal_effort(economy.productivity, LEVY_0)))
        gain = marginal_stationary_gain(economy, LEVY_1)
        ax2.bar(j - 0.18, private * 1000, width=0.35, label="Private resources" if j == 0 else None)
        ax2.bar(j + 0.18, (gain - private) * 1000, width=0.35, label="Displacement losses avoided" if j == 0 else None)
    ax1.set_xticks([0, 1], ['Before', 'After'])
    ax1.set_ylim(0, 0.6)
    ax1.set_ylabel('Share of firms using AI')
    ax1.legend(frameon=False)
    ax2.axhline(0, linewidth=0.8, color='black')
    ax2.set_xticks([0, 1], ['A', 'B'])
    ax2.set_ylabel('Contribution to welfare change (× 1,000)')
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'figure_1.pdf')
    plt.close(fig)


def figure_identification(directory: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.1))
    rates = np.linspace(0.0, 0.25, 101)
    for economy in ECONOMIES.values():
        vals = np.array([marginal_stationary_gain(economy, rate) for rate in rates])
        ax1.plot(rates, vals * 1000, label=f"Economy {economy.name}")
    ax1.axhline(0, linewidth=0.8, color='black')
    ax1.axvline(LEVY_1, linestyle=':', linewidth=0.8, color='black')
    ax1.set_xlabel('Annual levy per deployment')
    ax1.set_ylabel('Stationary welfare gain (× 1,000)')
    ax1.legend(frameon=False)
    sample_counts = [2, 3, 5, 9, 17, 33]
    for economy in ECONOMIES.values():
        limits = np.array([grid_bounds(economy, np.linspace(LEVY_0, LEVY_1, n)) for n in sample_counts])
        ax2.plot(sample_counts, limits[:, 0] * 1000, marker='o', ms=3, label=f"{economy.name}: lower")
        ax2.plot(sample_counts, limits[:, 1] * 1000, linestyle='--', marker='o', ms=3, label=f"{economy.name}: upper")
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Number of levy observations')
    ax2.set_ylabel('Welfare bounds (× 1,000)')
    ax2.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / 'figure_2.pdf')
    plt.close(fig)


def figure_phase(directory: Path, economy: Economy) -> None:
    shares = np.linspace(0, 1, 150)
    losses = np.linspace(0, 0.24, 120)
    lower, upper = np.empty((len(losses), len(shares))), np.empty((len(losses), len(shares)))
    for i, loss in enumerate(losses):
        for j, share in enumerate(shares):
            lower[i, j], upper[i, j] = phase_bounds(economy, HORIZON, share, loss)
    classification = np.where(lower > 0, 2, np.where(upper < 0, 0, 1))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.contourf(shares, losses, classification, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=['#d9d9d9', '#b0c2b5', '#e5eee4'])
    ax.contour(shares, losses, lower, levels=[0], linewidths=1.2, linestyles='solid')
    ax.contour(shares, losses, upper, levels=[0], linewidths=1.2, linestyles='dashed')
    ax.scatter([economy.masses[1]], [LOSS], s=32, color='black', zorder=3)
    ax.text(.6, .21, 'Beneficial for all matchings', fontsize=9)
    ax.text(.38, .078, 'Sign not identified', fontsize=9)
    ax.text(.5, .018, 'Harmful for all matchings', fontsize=9)
    ax.set_xlabel('Share with slow job-finding hazard (0.45)')
    ax.set_ylabel('Real loss per unemployed worker-year')
    fig.tight_layout()
    fig.savefig(directory / 'figure_3.pdf')
    plt.close(fig)


def figure_horizon(directory: Path, economy: Economy, pi_lo: np.ndarray, pi_hi: np.ndarray) -> None:
    horizons = np.linspace(.25, 18, 250)
    hazards = np.array([SLOW_HAZARD, FAST_HAZARD])
    low = np.array([private_change(economy, h) + (displacement_kernel(economy, h, hazards) * pi_lo).sum() for h in horizons])
    high = np.array([private_change(economy, h) + (displacement_kernel(economy, h, hazards) * pi_hi).sum() for h in horizons])
    common = np.array([common_hazard_effect(economy, h) for h in horizons])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(horizons, low * 1000, label='Lower-bound matching')
    ax.plot(horizons, high * 1000, label='Upper-bound matching')
    ax.plot(horizons, common * 1000, linestyle='--', label='Common-hazard economy')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xlabel('Evaluation horizon (years)')
    ax.set_ylabel('Average welfare gain (× 1,000)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / 'figure_4.pdf')
    plt.close(fig)


def run(output: Path, make_figures: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    economy = ECONOMIES['B']
    proportions = np.array([economy.masses[1], 1 - economy.masses[1]])
    hazards = np.array([SLOW_HAZARD, FAST_HAZARD])
    low, high, low_pi, high_pi = identified_welfare(economy, HORIZON, hazards, proportions)
    root = brentq(lambda h: common_hazard_effect(economy, h), 1.0, 20.0, xtol=1e-13)
    results = {
        'parameters': {'delta': DELTA, 'kappa': KAPPA, 'preparation_cost': PREPARATION,
                       'levy_before': LEVY_0, 'levy_after': LEVY_1,
                       'real_loss': LOSS, 'horizon_years': HORIZON,
                       'job_finding_hazards': hazards.tolist()},
        'stationary': {k: {
            'adoption_before': float(e.masses @ adoption_share(optimal_effort(e.productivity, LEVY_0))),
            'adoption_after': float(e.masses @ adoption_share(optimal_effort(e.productivity, LEVY_1))),
            'welfare_change': marginal_stationary_gain(e, LEVY_1)
        } for k, e in ECONOMIES.items()},
        'five_year': {'private_change': private_change(economy, HORIZON),
                      'lower_bound': low, 'upper_bound': high,
                      'lower_matching': low_pi.tolist(), 'upper_matching': high_pi.tolist()},
        'common_hazard_critical_horizon_years': root,
        'grid_nine': {k: {'lower': grid_bounds(e, np.linspace(LEVY_0, LEVY_1, 9))[0],
                           'upper': grid_bounds(e, np.linspace(LEVY_0, LEVY_1, 9))[1]}
                       for k, e in ECONOMIES.items()},
    }
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    if make_figures:
        figure_contrast(output)
        figure_identification(output)
        figure_phase(output, economy)
        figure_horizon(output, economy, low_pi, high_pi)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce welfare calculations and figures.')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    results = run(args.output, not args.no_figures)
    for name, record in results['stationary'].items():
        print(f"Economy {name}: adoption {record['adoption_before']:.6f} -> "
              f"{record['adoption_after']:.6f}, welfare {record['welfare_change']:+.9f}")
    x = results['five_year']
    print(f"Five-year welfare bounds: [{x['lower_bound']:+.9f}, {x['upper_bound']:+.9f}]")
    print(f"Common-hazard critical horizon: {results['common_hazard_critical_horizon_years']:.6f} years")


if __name__ == '__main__':
    main()
