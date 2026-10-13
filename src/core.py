import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def attrition_by(df, col):
    g = df.groupby(col)["left"]
    return pd.DataFrame({"employees": g.size(), "left": g.sum(), "rate": g.mean()}).sort_values("rate", ascending=False)


def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def test_categorical(df, col):
    table = pd.crosstab(df[col], df["left"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"feature": col, "test": "chi2", "p_value": float(p), "effect": cramers_v(table)}


def test_numeric(df, col):
    stayed = df.loc[~df["left"], col]
    left = df.loc[df["left"], col]
    u = stats.mannwhitneyu(left, stayed, alternative="two-sided")
    # common-language effect size: probability a leaver scores higher than a stayer
    effect = u.statistic / (len(left) * len(stayed))
    return {"feature": col, "test": "mann-whitney", "p_value": float(u.pvalue), "effect": float(effect)}


def screen_features(df, numeric, categorical):
    rows = [test_numeric(df, c) for c in numeric] + [test_categorical(df, c) for c in categorical]
    out = pd.DataFrame(rows).set_index("feature")
    # Holm correction for the multiple comparisons
    order = out["p_value"].sort_values().index
    m = len(out)
    adj = {}
    running = 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * out.loc[name, "p_value"]))
        adj[name] = running
    out["p_adjusted"] = pd.Series(adj)
    out["significant"] = out["p_adjusted"] < 0.05
    return out.sort_values("p_adjusted")


def logistic_odds_ratios(df):
    d = df.copy()
    d["overtime"] = d["overtime"].astype(int)
    d["left"] = d["left"].astype(int)
    d["salary_k"] = d["salary"] / 1000
    model = smf.logit(
        "left ~ satisfaction + overtime + salary_k + years_since_promotion + tenure_years + remote_days + C(department)",
        data=d,
    ).fit(disp=False)
    ci = model.conf_int()
    out = pd.DataFrame(
        {
            "odds_ratio": np.exp(model.params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p_value": model.pvalues,
        }
    ).drop(index="Intercept")
    return out, model


def risk_groups(df, model, n_groups=5):
    d = df.copy()
    d["overtime"] = d["overtime"].astype(int)
    d["salary_k"] = d["salary"] / 1000
    d["risk"] = model.predict(d)
    d["risk_group"] = pd.qcut(d["risk"], n_groups, labels=False) + 1
    g = d.groupby("risk_group")
    return pd.DataFrame({"employees": g.size(), "predicted": g["risk"].mean(), "actual": g["left"].mean()})
