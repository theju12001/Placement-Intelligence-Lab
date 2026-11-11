"""Statistical evidence: descriptive stats, tests with effect sizes, CIs, odds ratios.

Test choice is rule-based and reported with each result. Nothing here
establishes causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

import config
from src.data import loader

ALPHA = 0.05


def _y(df: pd.DataFrame, target: str) -> pd.Series:
    return loader.encode_target(df[target])


def descriptive(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    num = list(schema["numeric"])
    d = df[num].describe().T
    d["missing"] = df[num].isna().sum()
    d["missing_%"] = (d["missing"] / len(df) * 100).round(2)
    d["skew"] = df[num].skew()
    d["kurtosis"] = df[num].kurt()
    return d.round(3)


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    out = pd.DataFrame({"missing": m, "missing_%": (m / len(df) * 100).round(2)})
    return out[out["missing"] > 0].sort_values("missing", ascending=False)


def correlation(df: pd.DataFrame, schema: dict, target: str) -> pd.DataFrame:
    d = df[list(schema["numeric"])].copy()
    d["placed"] = _y(df, target)
    return d.corr(method="spearman").round(3)


def outliers(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    rows = []
    for c in schema["numeric"]:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({"feature": c, "lower_fence": round(lo, 3), "upper_fence": round(hi, 3),
                     "n_outliers": n, "%": round(n / len(s) * 100, 2)})
    return pd.DataFrame(rows)


def placement_rate_by(df: pd.DataFrame, col: str, target: str, bins: int = 4) -> pd.DataFrame:
    d = df[[col]].copy()
    d["y"] = _y(df, target)
    d = d.dropna()
    if pd.api.types.is_numeric_dtype(d[col]) and d[col].nunique() > 8:
        d["group"] = pd.qcut(d[col], bins, duplicates="drop").astype(str)
    else:
        d["group"] = d[col].astype(str)
    g = d.groupby("group")["y"].agg(["sum", "count"]).reset_index()
    lo, hi = proportion_confint(g["sum"], g["count"], alpha=ALPHA, method="wilson")
    g["rate"] = g["sum"] / g["count"]
    g["ci_low"], g["ci_high"] = lo, hi
    g["small_sample"] = g["count"] < 30
    return g.rename(columns={"sum": "placed", "count": "n"})


def categorical_test(df: pd.DataFrame, col: str, target: str) -> dict:
    d = pd.DataFrame({"x": df[col].astype(str), "y": _y(df, target)}).dropna()
    ct = pd.crosstab(d["x"], d["y"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"feature": col, "test": "n/a", "note": "Not enough groups/classes."}
    small = (stats.contingency.expected_freq(ct) < 5).any()
    if ct.shape == (2, 2) and small:
        _, p = stats.fisher_exact(ct.to_numpy())
        stat, name = float("nan"), "Fisher exact (small expected counts)"
        v = np.sqrt(stats.chi2_contingency(ct, correction=False)[0] / ct.to_numpy().sum())
    else:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        stat, name = float(chi2), f"Chi-square (dof={dof})"
        n = ct.to_numpy().sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return {"feature": col, "test": name, "statistic": stat, "p_value": float(p),
            "effect_size": float(v), "effect_name": "Cramér's V",
            "significant": bool(p < ALPHA),
            "H0": f"Placement is independent of {col}.",
            "interpretation": _interp(p, v, "Cramér's V", col, small_note=bool(small))}


def numeric_test(df: pd.DataFrame, col: str, target: str) -> dict:
    d = pd.DataFrame({"x": pd.to_numeric(df[col], errors="coerce"), "y": _y(df, target)}).dropna()
    a, b = d.loc[d.y == 1, "x"], d.loc[d.y == 0, "x"]
    if len(a) < 3 or len(b) < 3:
        return {"feature": col, "test": "n/a", "note": "Too few observations in a group."}
    # Normality only informative for small/moderate n; with large n use Welch (CLT)
    normal = True
    if min(len(a), len(b)) < 40:
        normal = all(stats.shapiro(g.sample(min(len(g), 500), random_state=1))[1] > ALPHA for g in (a, b))
    if normal:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        eff = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
        out = {"test": "Welch t-test", "statistic": float(t), "effect_size": float(eff), "effect_name": "Cohen's d"}
    else:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        eff = 1 - 2 * u / (len(a) * len(b))
        out = {"test": "Mann–Whitney U", "statistic": float(u), "effect_size": float(-eff),
               "effect_name": "rank-biserial r"}
    out.update({"feature": col, "p_value": float(p), "significant": bool(p < ALPHA),
                "mean_placed": float(a.mean()), "mean_not_placed": float(b.mean()),
                "n_placed": int(len(a)), "n_not_placed": int(len(b)),
                "H0": f"{col} has the same distribution for placed and not-placed students.",
                "interpretation": _interp(p, out["effect_size"], out["effect_name"], col)})
    return out


def _interp(p, eff, name, col, small_note=False) -> str:
    mag = abs(eff)
    if name == "Cohen's d":
        size = "negligible" if mag < 0.2 else "small" if mag < 0.5 else "medium" if mag < 0.8 else "large"
    elif name == "rank-biserial r":
        size = "negligible" if mag < 0.1 else "small" if mag < 0.3 else "medium" if mag < 0.5 else "large"
    else:
        size = "negligible" if mag < 0.1 else "small" if mag < 0.3 else "medium" if mag < 0.5 else "large"
    sig = "is statistically significant" if p < ALPHA else "is not statistically significant"
    txt = (f"The association between {col} and placement {sig} at α={ALPHA} (p={p:.4g}); "
           f"the effect size is {size} ({name}={eff:.3f}). Statistical significance does not imply "
           "practical importance or causation.")
    if small_note:
        txt += " Some expected cell counts are below 5, so treat the result cautiously."
    return txt


def run_all_tests(df: pd.DataFrame, schema: dict, target: str) -> pd.DataFrame:
    rows = [numeric_test(df, c, target) for c in schema["numeric"]]
    rows += [categorical_test(df, c, target) for c in schema["categorical"]]
    out = pd.DataFrame([r for r in rows if "p_value" in r])
    if out.empty:
        return out
    # Benjamini–Hochberg FDR across the whole family of tests
    p = out["p_value"].to_numpy()
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out.loc[out.index[order], "p_adj_BH"] = np.clip(adj, 0, 1)
    out["group"] = out["feature"].map(
        lambda f: "Socioeconomic" if f in schema["socioeconomic"] else "Academic")
    return out.sort_values("p_value").reset_index(drop=True)


def logistic_odds_ratios(df: pd.DataFrame, schema: dict, target: str) -> pd.DataFrame | None:
    """Multivariable logistic regression (statsmodels): adjusted odds ratios with 95% CI."""
    import statsmodels.api as sm
    X = df[schema["features"]].copy()
    y = _y(df, target)
    keep = X.notna().all(axis=1) & y.notna()
    X, y = X[keep], y[keep]
    for c in schema["numeric"]:   # standardise → OR per 1 SD, comparable across features
        sd = X[c].std()
        X[c] = (X[c] - X[c].mean()) / sd if sd > 0 else 0.0
    X = pd.get_dummies(X, columns=list(schema["categorical"]), drop_first=True, dtype=float)
    X = sm.add_constant(X)
    try:
        res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception:
        return None
    ci = res.conf_int()
    out = pd.DataFrame({"term": res.params.index, "odds_ratio": np.exp(res.params.values),
                        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values),
                        "p_value": res.pvalues.values})
    out = out[out["term"] != "const"]
    out.attrs["n"] = int(len(y))
    out.attrs["pseudo_r2"] = float(res.prsquared)
    return out.sort_values("p_value").reset_index(drop=True)
