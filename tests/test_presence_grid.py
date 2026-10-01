"""Tests for the Model A data preparation (no network, synthetic data).

The point is to guard against the two mistakes that would make a presence model look better than it is:
leaking the future into the features, and testing on years the model trained on.
"""

import pandas as pd
import pytest

from ml_presence_spain import build_presence_grid, pick_species, time_split


def _obs(rows):
    return pd.DataFrame(rows, columns=["year", "decimalLatitude", "decimalLongitude"])


def test_target_is_presence_in_the_next_year_for_the_same_cell():
    # One cell observed in 2010 and 2012 but not 2011.
    df = _obs([(2010, 40.2, -3.2), (2012, 40.3, -3.1)])
    grid = build_presence_grid(df, grid_deg=1.0).set_index("year")
    assert grid.loc[2010, "target_presence_next_year"] == 0  # 2011: not observed
    assert grid.loc[2011, "target_presence_next_year"] == 1  # 2012: observed
    assert grid.loc[2011, "presence"] == 0


def test_last_year_is_dropped_because_it_has_no_label():
    df = _obs([(2010, 40.2, -3.2), (2012, 40.3, -3.1)])
    grid = build_presence_grid(df, grid_deg=1.0)
    assert grid["year"].max() == 2011


def test_lag_feature_uses_the_previous_year_only():
    df = _obs([(2010, 40.2, -3.2), (2011, 40.3, -3.1), (2012, 41.5, -3.1)])
    grid = build_presence_grid(df, grid_deg=1.0)
    cell = grid[(grid["cell_lat"] == 40) & (grid["cell_lon"] == -4)].set_index("year")
    assert cell.loc[2010, "presence_lag1"] == 0  # nothing before the first year
    assert cell.loc[2011, "presence_lag1"] == 1  # observed in 2010


def test_grid_only_covers_cells_observed_at_least_once():
    df = _obs([(2010, 40.2, -3.2), (2011, 40.3, -3.1), (2012, 43.5, -8.1)])
    grid = build_presence_grid(df, grid_deg=1.0)
    assert grid[["cell_lat", "cell_lon"]].drop_duplicates().shape[0] == 2


def test_time_split_never_trains_on_test_years():
    df = _obs([(y, 40.2, -3.2) for y in range(2010, 2021)])
    grid = build_presence_grid(df, grid_deg=1.0)
    train, test = time_split(grid, test_years=3)
    assert train["year"].max() < test["year"].min()
    assert test["year"].nunique() == 3


def test_pick_species_returns_the_most_frequent_one_above_the_threshold():
    df = pd.DataFrame({"species": ["a"] * 5 + ["b"] * 3 + ["c"]})
    assert pick_species(df, min_obs=3) == "a"
    with pytest.raises(ValueError):
        pick_species(df, min_obs=10)
