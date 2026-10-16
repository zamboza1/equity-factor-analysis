"""Invalid inputs must fail before producing misleading analysis output."""
import pickle

import numpy as np
import pandas as pd
import pytest

from backend.analytics.factor_model import FactorModel
from backend.analytics.returns import compute_returns
from backend.cache import DataCache
from backend.entities import ReturnSeries
from backend.errors import CacheError, DataError


@pytest.fixture
def regression_inputs():
    dates = pd.bdate_range('2024-01-01', periods=80)
    rng = np.random.default_rng(415)
    factors = pd.DataFrame(rng.normal(0, .01, (80, 2)), index=dates,
                           columns=['Market', 'Value'])
    returns = .001 + factors['Market'] * 1.2 + rng.normal(0, .002, 80)
    return factors, returns


@pytest.mark.parametrize('kind', ['constant_factor', 'collinear_factors', 'constant_return', 'infinite_return'])
def test_invalid_regression_inputs_are_rejected(regression_inputs, kind):
    factors, returns = regression_inputs
    if kind == 'constant_factor':
        factors['Value'] = 1.
    elif kind == 'collinear_factors':
        factors['Value'] = 2 * factors['Market']
    elif kind == 'constant_return':
        returns[:] = 0.
    else:
        returns.iloc[0] = np.inf
    with pytest.raises(DataError):
        FactorModel(['Market', 'Value']).fit(ReturnSeries(returns.index, returns), factors)


def test_missing_factor_has_a_readable_data_error(regression_inputs):
    factors, returns = regression_inputs
    with pytest.raises(DataError, match='Missing factor'):
        FactorModel(['Missing']).fit(ReturnSeries(returns.index, returns), factors)


def test_duplicate_factor_dates_are_rejected(regression_inputs):
    factors, returns = regression_inputs
    factors = pd.concat([factors, factors.iloc[:1]])
    with pytest.raises(DataError, match='unique'):
        FactorModel(['Market']).fit(ReturnSeries(returns.index, returns), factors)


def test_no_residual_degrees_of_freedom_is_rejected(regression_inputs):
    factors, returns = regression_inputs
    factors = factors.iloc[:3]
    returns = returns.iloc[:3]
    with pytest.raises(DataError, match='degrees of freedom'):
        FactorModel(['Market', 'Value']).fit(ReturnSeries(returns.index, returns), factors, min_observations=2)


def test_factor_estimates_match_independent_aligned_least_squares(regression_inputs):
    factors, returns = regression_inputs
    factors['RF'] = .0001
    factors.iloc[0, 0] = np.nan
    fit = FactorModel(['Market', 'Value']).fit(ReturnSeries(returns.index, returns), factors)
    clean = factors.dropna()
    design = np.column_stack([np.ones(len(clean)), clean[['Market', 'Value']]])
    expected = np.linalg.lstsq(design, returns.loc[clean.index] - clean['RF'], rcond=None)[0]
    assert [fit.alpha, fit.betas['Market'], fit.betas['Value']] == pytest.approx(expected)
    assert fit.n_observations == 79


def test_returns_are_computed_in_date_order():
    dates = pd.date_range('2024-01-01', periods=3)
    prices = pd.Series([121., 100., 110.], index=dates[[2, 0, 1]])
    result = compute_returns(prices, method='simple')
    assert result.index.equals(dates)
    assert result.iloc[1:].tolist() == pytest.approx([.1, .1])
    assert prices.index.equals(dates[[2, 0, 1]])


@pytest.mark.parametrize('kind', ['duplicate', 'missing_date', 'infinite_price'])
def test_invalid_price_series_is_rejected(kind):
    dates = pd.date_range('2024-01-01', periods=3)
    if kind == 'duplicate': dates = pd.DatetimeIndex([dates[0], dates[0], dates[2]])
    if kind == 'missing_date': dates = pd.DatetimeIndex([dates[0], pd.NaT, dates[2]])
    prices = pd.Series([100., 110., np.inf if kind == 'infinite_price' else 121.], index=dates)
    with pytest.raises(ValueError):
        compute_returns(prices)


def test_missing_price_does_not_bridge_a_return_interval():
    prices = pd.Series([100., np.nan, 110., 121.], index=pd.date_range('2024-01-01', periods=4))
    result = compute_returns(prices, method='simple')
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(.1)


def test_labeled_returns_keep_their_dates_when_requested_order_changes():
    dates = pd.date_range('2024-01-01', periods=3)
    values = pd.Series([.01, .02, .03], index=dates)
    result = ReturnSeries(dates[::-1], values)
    pd.testing.assert_series_equal(result.returns, values, check_names=False)


def test_labeled_returns_cannot_be_reassigned_to_unrelated_dates():
    dates = pd.date_range('2024-01-01', periods=3)
    values = pd.Series([.01, .02, .03], index=dates + pd.Timedelta(days=10))
    with pytest.raises(ValueError, match='dates'):
        ReturnSeries(dates, values)


def test_positional_return_values_remain_supported():
    dates = pd.date_range('2024-01-01', periods=3)
    result = ReturnSeries(dates, pd.Series([.01, .02, .03]))
    assert result.returns.tolist() == [.01, .02, .03]


def test_corrupt_cache_is_a_miss(tmp_path):
    cache = DataCache(tmp_path)
    query = {'ticker': 'FIXTURE'}
    (tmp_path / (cache._compute_hash(query) + '.pkl')).write_bytes(b'incomplete pickle')
    assert cache.get(query) is None
    cache.set(query, {'fresh': True})
    assert cache.get(query) == {'fresh': True}


def test_failed_cache_write_preserves_the_previous_value(tmp_path):
    class Unserializable:
        def __reduce__(self):
            raise ValueError('deliberate serialization failure')
    cache = DataCache(tmp_path)
    query = {'ticker': 'FIXTURE'}
    cache.set(query, {'previous': True})
    with pytest.raises(CacheError):
        cache.set(query, Unserializable())
    assert cache.get(query) == {'previous': True}
    assert not list(tmp_path.glob('*.tmp'))
