import numpy as np
import pandas as pd

DEPARTMENTS = ["engineering", "sales", "support", "operations", "finance"]


def generate_employees(n=3000, seed=13):
    rng = np.random.default_rng(seed)
    dept = rng.choice(DEPARTMENTS, n, p=[0.3, 0.2, 0.25, 0.15, 0.1])
    tenure = np.clip(rng.gamma(2.2, 2.0, n), 0.1, 25)
    age = np.clip(22 + tenure + rng.normal(8, 5, n), 20, 64).round()
    base = {"engineering": 4200, "sales": 3300, "support": 2600, "operations": 2900, "finance": 3800}
    salary = np.array([base[d] for d in dept]) * (1 + 0.04 * np.log1p(tenure)) * rng.lognormal(0, 0.15, n)
    satisfaction = np.clip(rng.normal(3.4, 0.9, n), 1, 5)
    overtime = rng.random(n) < np.where(dept == "support", 0.45, 0.25)
    last_promotion_years = np.clip(rng.exponential(2.0, n), 0, tenure)
    remote_days = rng.choice([0, 1, 2, 3, 5], n, p=[0.3, 0.2, 0.25, 0.15, 0.1])

    logit = (
        -1.4
        - 0.75 * (satisfaction - 3.4)
        + 0.9 * overtime
        - 0.0004 * (salary - salary.mean())
        + 0.18 * last_promotion_years
        - 0.05 * tenure
        - 0.12 * remote_days
        + np.where(dept == "support", 0.35, 0.0)
    )
    p = 1 / (1 + np.exp(-logit))
    left = rng.random(n) < p

    return pd.DataFrame(
        {
            "employee_id": np.arange(n),
            "department": dept,
            "age": age,
            "tenure_years": tenure.round(1),
            "salary": salary.round(0),
            "satisfaction": satisfaction.round(1),
            "overtime": overtime,
            "years_since_promotion": last_promotion_years.round(1),
            "remote_days": remote_days,
            "left": left,
        }
    )
