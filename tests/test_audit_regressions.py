"""Regressions for independently reproduced audit defects; no live providers."""
from types import SimpleNamespace
import math
import numpy as np
import pandas as pd
import pytest
from backend.analytics.sabr import SABRModel, sabr_implied_vol
from backend.errors import CalibrationError, DataError


def hagan_reference(f, k, t, a, b, r, n):
    """Independent arrangement retained from the pre-change audit benchmark."""
    m=math.log(f/k);q=(f*k)**((1-b)/2);z=n*q*m/a
    zx=1-r*z/2+(2-3*r*r)*z*z/12 if abs(z)<1e-7 else z/math.log((math.sqrt(1-2*r*z+z*z)+z-r)/(1-r))
    return a*zx*(1+t*((1-b)**2*a*a/(24*q*q)+r*b*n*a/(4*q)+(2-3*r*r)*n*n/24))/(q*(1+(1-b)**2*m*m/24+(1-b)**4*m**4/1920))


@pytest.mark.parametrize('beta,nu',[(.5,.5),(1.,.5),(.5,0.)])
def test_hagan_reference_grid(beta,nu):
    alpha=2. if beta==.5 else .2
    for strike in [80.,90.,99.999,100.,100.001,110.,120.]:
        assert sabr_implied_vol(100,strike,.25,alpha,beta,-.3,nu)==pytest.approx(hagan_reference(100,strike,.25,alpha,beta,-.3,nu),rel=1e-9,abs=1e-11)


def test_calibration_of_independent_smile():
    k=np.array([80.,90.,95.,100.,105.,110.,120.])
    iv=np.array([hagan_reference(100,x,.25,2,.5,-.3,.5) for x in k])
    fit=SABRModel(.5,True).calibrate(k,iv,100,.25)
    assert fit.fit_error<1e-5
    assert fit.params.alpha==pytest.approx(2,abs=.002)
    assert fit.params.rho==pytest.approx(-.3,abs=.002)
    assert fit.params.nu==pytest.approx(.5,abs=.002)


@pytest.mark.parametrize('t',[0.,-1.,float('nan')])
def test_calibration_rejects_invalid_maturity(t):
    with pytest.raises(CalibrationError):
        SABRModel(.5,True).calibrate(np.array([90,100,110]),np.array([.2,.2,.2]),100,t)


def test_factor_loader_normalizes_daily_period_index(monkeypatch):
    import pandas_datareader.data as reader
    from backend.data_sources.factors import get_factor_data
    idx=pd.period_range('2024-01-01',periods=3,freq='D')
    ff=pd.DataFrame({'Mkt-RF':[1,2,3],'SMB':[.1,.2,.3],'HML':[.3,.4,.5],'RF':[.02,.02,.02]},index=idx)
    mom=pd.DataFrame({'Mom   ':[.4,.5,.6]},index=idx.to_timestamp())
    monkeypatch.setattr(reader,'DataReader',lambda name,*a:{0:mom if 'Momentum' in name else ff})
    result=get_factor_data('2024-01-01','2024-01-03')
    assert isinstance(result.index,pd.DatetimeIndex)
    assert list(result.index)==list(pd.date_range('2024-01-01',periods=2))
    assert result.loc['2024-01-02','Market']==pytest.approx(.02)
    assert result.loc['2024-01-02','RF']==pytest.approx(.0002)


def run_cli_fixture(monkeypatch,tmp_path):
    import backend.cli as cli
    from backend.data_sources import factors,macro_events
    import yaml
    dates=pd.bdate_range('2024-01-01',periods=253,tz='America/New_York')
    f=pd.DataFrame(np.random.default_rng(741).normal(0,.02,(253,4)),index=dates,columns=['Market','Value','Size','Momentum']);f['RF']=.0002
    simple=f['RF']+1.2*f['Market']
    prices=pd.Series(np.r_[100,100*np.cumprod(1+simple.iloc[1:])],index=dates)
    captured={};calls=[]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli,'get_close_prices',lambda *a,**kw:prices)
    monkeypatch.setattr(factors,'get_factor_data',lambda *a,**kw:f)
    monkeypatch.setattr(yaml,'safe_load',lambda *a:{})
    monkeypatch.setattr(macro_events,'get_recent_events_for_analysis',lambda **kw:[])
    if hasattr(macro_events,'get_events_for_analysis'):monkeypatch.setattr(macro_events,'get_events_for_analysis',lambda *a,**kw:[])
    def option(*a,**kw):
        calls.append(a)
        raise DataError('Fixture should not fetch options when disabled')
    monkeypatch.setattr(cli,'get_option_chain',option)
    def report(self,**kw):captured.update(kw);return str(tmp_path/'fixture.md')
    monkeypatch.setattr(cli.Report,'generate_report',report)
    cli.analyze_command(SimpleNamespace(ticker='AAPL',start='2024-01-01',end='2024-12-31',fit_sabr=False,run_id=None))
    return captured,calls


def test_cli_factor_model_uses_arithmetic_excess_returns(monkeypatch,tmp_path):
    captured,_=run_cli_fixture(monkeypatch,tmp_path)
    fit=captured['factor_result']
    assert fit.alpha==pytest.approx(0,abs=1e-12)
    assert fit.betas['Market']==pytest.approx(1.2,abs=1e-12)


def test_major_ticker_respects_disabled_sabr(monkeypatch,tmp_path):
    _,calls=run_cli_fixture(monkeypatch,tmp_path)
    assert calls==[]


def test_web_keeps_validation_errors_as_client_errors():
    from fastapi.testclient import TestClient
    import backend.web as web
    with TestClient(web.app) as client:
        response=client.post('/api/run-analysis',json={'tickers':['AAPL'],'start_date':'2010-01-01','end_date':'2024-01-01','fit_sabr':False})
    assert response.status_code==400
    assert response.json()['detail']=='Date range cannot exceed 5 years'


def test_report_image_urls_resolve_from_nested_route():
    from backend.web import markdown_to_html
    assert 'src="/assets/run/event_impact.png"' in markdown_to_html('![Event Impact](assets/run/event_impact.png)')


def test_ticker_lookup_does_not_match_other_symbol(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    import backend.web as web
    monkeypatch.setattr(web,'reports_dir',tmp_path)
    (tmp_path/'AAPL_old.md').write_text('**R²:** 0.123')
    with TestClient(web.app) as client:
        assert client.get('/api/analysis-results/A').status_code==404
