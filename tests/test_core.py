import pandas as pd
import pytest

from src import core, data


@pytest.fixture(scope="module")
def df():
    return data.generate_employees(n=2500, seed=5)


def test_attrition_table_totals(df):
    t = core.attrition_by(df, "department")
    assert t["employees"].sum() == len(df)
    assert t["left"].sum() == df["left"].sum()


def test_screen_flags_planted_effects(df):
    screen = core.screen_features(df, ["satisfaction", "salary"], ["overtime", "department"])
    assert screen.loc["satisfaction", "significant"]
    assert screen.loc["overtime", "significant"]


def test_adjusted_p_not_below_raw(df):
    screen = core.screen_features(df, ["satisfaction", "age"], ["overtime"])
    assert (screen["p_adjusted"] >= screen["p_value"] - 1e-12).all()


def test_odds_ratio_directions(df):
    odds, _ = core.logistic_odds_ratios(df)
    assert odds.loc["overtime", "odds_ratio"] > 1
    assert odds.loc["satisfaction", "odds_ratio"] < 1


def test_risk_groups_are_ordered(df):
    _, model = core.logistic_odds_ratios(df)
    groups = core.risk_groups(df, model)
    assert groups["actual"].iloc[-1] > groups["actual"].iloc[0]
    assert groups["employees"].sum() == len(df)
