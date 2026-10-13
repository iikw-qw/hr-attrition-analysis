import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    df = data.generate_employees()
    numeric = ["age", "tenure_years", "salary", "satisfaction", "years_since_promotion", "remote_days"]
    categorical = ["department", "overtime"]
    screen = core.screen_features(df, numeric, categorical)
    odds, model = core.logistic_odds_ratios(df)
    groups = core.risk_groups(df, model)
    by_dept = core.attrition_by(df, "department")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    odds_plot = odds.drop(index=[i for i in odds.index if "department" in i]).sort_values("odds_ratio")
    y = np.arange(len(odds_plot))
    axes[0].errorbar(odds_plot["odds_ratio"], y,
                     xerr=[odds_plot["odds_ratio"] - odds_plot["ci_low"], odds_plot["ci_high"] - odds_plot["odds_ratio"]],
                     fmt="o", color="black", capsize=3)
    axes[0].axvline(1, linestyle="--", color="grey")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(odds_plot.index)
    axes[0].set_xlabel("odds ratio (95% CI)")
    axes[0].set_title("Attrition drivers")
    axes[1].plot(groups.index, groups["predicted"], marker="o", color="grey", label="predicted")
    axes[1].plot(groups.index, groups["actual"], marker="s", color="black", label="actual")
    axes[1].set_xlabel("risk quintile")
    axes[1].set_ylabel("attrition rate")
    axes[1].legend()
    axes[1].set_title("Calibration of the risk score")
    fig.tight_layout()
    fig.savefig(OUT / "drivers.png", dpi=130)

    screen.round(5).to_csv(OUT / "feature_screen.csv")
    summary = {
        "employees": int(len(df)),
        "attrition_rate": round(float(df["left"].mean()), 3),
        "highest_attrition_department": by_dept.index[0],
        "highest_department_rate": round(float(by_dept["rate"].iloc[0]), 3),
        "overtime_odds_ratio": round(float(odds.loc["overtime", "odds_ratio"]), 2),
        "satisfaction_odds_ratio_per_point": round(float(odds.loc["satisfaction", "odds_ratio"]), 2),
        "significant_features_after_holm": list(screen.index[screen["significant"]]),
        "top_risk_quintile_attrition": round(float(groups["actual"].iloc[-1]), 3),
        "bottom_risk_quintile_attrition": round(float(groups["actual"].iloc[0]), 3),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
