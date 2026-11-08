# AI Adoption Levies and Worker Transitions

Numerical reproduction package for *AI Adoption Levies and Worker Transitions: Welfare Rankings from Unlinked Data*.

The calculations use the analytical adoption dynamics and displacement-cost kernel described in the paper. All parameters are specified in `reproduce.py`. The examples are theoretical parameterizations, not estimates from individual employment records.

## Requirements

- Python 3.10 or newer
- NumPy, SciPy, Matplotlib

```bash
python -m pip install -r requirements.txt
```

## Reproduce the results

```bash
python reproduce.py --output results
python -m unittest discover -s tests -v
```

The first command writes `results/results.json` and four vector figures (`figure_1.pdf` through `figure_4.pdf`). The second command checks the stationary equilibria, private-welfare calculations, transport bounds, exact kernel against numerical quadrature, and the unique critical horizon in the benchmark case.

## Numerical benchmarks

| Quantity | Result |
|---|---:|
| Adoption before and after the levy, both economies | 0.50 → 0.40 |
| Stationary welfare change, A | +0.0028712404 |
| Stationary welfare change, B | −0.0025045602 |
| Five-year lower welfare bound, B | −0.0012530450 |
| Five-year upper welfare bound, B | +0.0051471106 |
| Five-year private-resource effect, B | −0.0054617336 |
| Critical horizon, common-hazard benchmark | 7.252271 years |

The two transport bounds are computed with linear programming. The program verifies the row and column marginals, primal feasibility, dual inequalities, and equality of primal and dual objectives.

## Computational methods

For a permanent levy increase, optimal adoption efforts on the active branch solve

\[
 v_i-\tau=\delta k+\delta\kappa a_i+\frac{\kappa a_i^2}{2}.
\]

With \(\lambda_i=\delta+a_i^1\) and \(d_i=s_i^0-s_i^1\), the avoided displacement flow is

\[
 e_i(t)=d_i[\delta+a_i^1\exp(-\lambda_i t)].
\]

The finite-horizon welfare kernel is

\[
 c_T(i,\mu)=\frac{\ell\beta_i}{T}\int_0^T e_i(t)
 \frac{1-\exp[-\mu(T-t)]}{\mu}\,dt.
\]

The script evaluates this kernel in closed form, including the coincident-rate case \(\lambda_i=\mu\), and verifies it against adaptive quadrature. The associated welfare bounds are the minimum and maximum of its expectation over couplings with fixed firm-type and job-finding marginals.

## Files

- `reproduce.py`: equilibrium, welfare, transport, sensitivity and figures
- `tests/test_reproduction.py`: independent numerical checks
- `requirements.txt`: Python dependencies
- `results/`: created on execution

