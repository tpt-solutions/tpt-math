"""Generate reference tables for tpt-math-prob-core's special functions.

Values come from mpmath at 30 significant digits, so the tables are exact to far
beyond the f64 tests that read them. Re-run this script to refresh the CSVs:

    python tools/golden/special_golden.py

Output goes to crates/tpt-math-prob-core/tests/golden/.
"""

from pathlib import Path

import mpmath as mp

mp.mp.dps = 30
OUT = Path(__file__).resolve().parents[2] / "crates/tpt-math-prob-core/tests/golden"


def fmt(x):
    return mp.nstr(mp.mpf(x), 17, min_fixed=-30, max_fixed=30)


def write(name, header, rows):
    path = OUT / name
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(rows)} rows")


def gamma_rows():
    rows = []
    skipped = []
    # mpmath cannot converge the hypergeometric series for much larger shapes; the
    # large-shape regime is covered by the consistency test in special.rs.
    shapes = [0.1, 0.5, 1.0, 2.5, 7.0, 30.0, 150.0, 1000.0, 1e4]
    for a in shapes:
        for f in [0.01, 0.2, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0]:
            x = mp.mpf(a) * mp.mpf(f)
            if x <= 0:
                continue
            try:
                p = mp.gammainc(a, 0, x, regularized=True)
                q = mp.gammainc(a, x, mp.inf, regularized=True)
            except mp.libmp.libhyper.NoConvergence:
                # mpmath gives up on some points near x = a for large shapes. Skip them;
                # the rest of the table still covers the regime.
                skipped.append((a, float(f)))
                continue
            rows.append([fmt(a), fmt(x), fmt(p), fmt(q)])
    print(f"gamma: skipped {len(skipped)} points mpmath could not converge: {skipped}")
    return rows


def erf_rows():
    xs = [1e-8, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, -0.7, -3.0]
    return [[fmt(x), fmt(mp.erf(x)), fmt(mp.erfc(x))] for x in xs]


def loggamma_rows():
    xs = [0.001, 0.5, 1.0, 1.5, 3.0, 10.0, 50.0, 300.0, 1e4, 1e6]
    return [[fmt(x), fmt(mp.loggamma(x))] for x in xs]


def beta_rows():
    rows = []
    for a, b in [(0.5, 0.5), (2.0, 3.0), (10.0, 0.7), (50.0, 60.0), (300.0, 2.0)]:
        for x in [0.01, 0.3, 0.5, 0.8, 0.99]:
            i = mp.betainc(a, b, 0, x, regularized=True)
            rows.append([fmt(a), fmt(b), fmt(x), fmt(i)])
    return rows


def binomial_rows():
    rows = []
    for n, k in [(5, 2), (52, 5), (100, 50), (1000, 3), (10**15, 1), (10**15, 2), (2000, 1000)]:
        rows.append([str(n), str(k), fmt(mp.log(mp.binomial(n, k)))])
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write("gamma.csv", ["a", "x", "P", "Q"], gamma_rows())
    write("erf.csv", ["x", "erf", "erfc"], erf_rows())
    write("loggamma.csv", ["x", "lgamma"], loggamma_rows())
    write("beta_reg.csv", ["a", "b", "x", "I"], beta_rows())
    write("ln_binomial.csv", ["n", "k", "ln_c"], binomial_rows())
