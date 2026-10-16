from pathlib import Path
import json, math, re
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from backend.errors import DataError
from backend.data_sources import macro_events, prices
from backend.data_sources.options import option_time_to_expiry
from backend.analytics.event_impact import EventImpactStudy, EventImpact
from backend.reporting.report import Report


def test_verified_calendar_and_exclusive_end():
    events=macro_events.get_events_for_analysis('2024-10-01','2025-01-01')
    actual={(e.name,str(e.timestamp.date())) for e in events}
    assert actual=={('CPI','2024-10-10'),('CPI','2024-11-13'),('CPI','2024-12-11'),('NFP','2024-10-04'),('NFP','2024-11-01'),('NFP','2024-12-06'),('FOMC','2024-11-07'),('FOMC','2024-12-18')}
    assert macro_events.get_events_for_analysis('2026-01-01','2026-12-31')==[]
    assert macro_events.get_events_for_analysis('2024-11-13','2024-11-14')[0].name=='CPI'
    assert macro_events.get_events_for_analysis('2024-11-12','2024-11-13')==[]


def test_intraday_provider_receives_timestamps_and_extended_hours(monkeypatch):
    seen={}
    def history(**kw):
        seen.update(kw)
        return pd.DataFrame({'Close':[100,101]},index=pd.date_range('2024-01-31 13:30',periods=2,freq='min',tz='America/New_York'))
    monkeypatch.setattr(prices.yf,'Ticker',lambda _:SimpleNamespace(history=history))
    start=pd.Timestamp('2024-01-31 13:30',tz='America/New_York');end=start+pd.Timedelta(hours=3)
    prices.get_prices('AAPL',start,end,interval='1m',prepost=True)
    assert pd.Timestamp(seen['start'])==start and pd.Timestamp(seen['end'])==end
    assert seen['prepost'] is True


def test_intraday_slices_broad_data_and_honors_cache_flag(monkeypatch):
    import backend.analytics.event_impact as module
    event=macro_events.MacroEvent('FOMC',pd.Timestamp('2024-01-31 14:00',tz='US/Eastern'),'US/Eastern','Fixture')
    index=pd.date_range('2024-01-31 09:30',periods=480,freq='min',tz='US/Eastern')
    data=pd.Series(np.where(index.hour>=17,150.,100.),index=index)
    seen={}
    def fetch(*a,**kw):seen.update(kw);return data
    monkeypatch.setattr(module,'get_close_prices',fetch)
    impact=EventImpactStudy()._analyze_event_intraday('AAPL',event,None,False)
    assert impact.post_event_return==0
    assert impact.window_start==pd.Timestamp('2024-01-31 13:30',tz='US/Eastern')
    assert impact.window_end==pd.Timestamp('2024-01-31 16:00',tz='US/Eastern')
    assert seen['use_cache'] is False and seen['prepost'] is True


def test_daily_event_cannot_use_prices_after_requested_end(monkeypatch):
    import backend.analytics.event_impact as module
    event=macro_events.MacroEvent('CPI',pd.Timestamp('2024-11-13 08:30',tz='US/Eastern'),'US/Eastern','Fixture')
    data=pd.Series([100.,101.,99.],index=pd.date_range('2024-11-12',periods=3,tz='US/Eastern'))
    monkeypatch.setattr(module,'get_close_prices',lambda *a,**kw:data)
    impact=EventImpactStudy().analyze_event('AAPL',event,analysis_end='2024-11-14')
    assert impact.cumulative_return==pytest.approx(math.log(1.01))
    assert impact.n_observations==1
    assert impact.window_end.date().isoformat()=='2024-11-13'


def test_same_day_expiry_retains_fractional_time_and_rejects_expired():
    noon=pd.Timestamp('2024-11-15 12:00',tz='America/New_York')
    assert option_time_to_expiry('2024-11-15',now=noon)==pytest.approx(4/(24*365))
    assert option_time_to_expiry(noon+pd.Timedelta(hours=3),now=noon)==pytest.approx(3/(24*365))
    with pytest.raises(DataError,match='after'):
        option_time_to_expiry('2024-11-15',now=noon+pd.Timedelta(hours=5))


def test_friday_event_includes_the_next_available_session(monkeypatch):
    import backend.analytics.event_impact as module
    event = macro_events.MacroEvent('NFP', pd.Timestamp('2024-11-01 08:30', tz='US/Eastern'), 'US/Eastern', 'Fixture')
    data = pd.Series([100., 101., 104.], index=pd.DatetimeIndex(['2024-10-31', '2024-11-01', '2024-11-04'], tz='US/Eastern'))
    def fetch(ticker, start, end, **kwargs):
        return data.loc[(data.index >= pd.Timestamp(start, tz='US/Eastern')) & (data.index < pd.Timestamp(end, tz='US/Eastern'))]
    monkeypatch.setattr(module, 'get_close_prices', fetch)
    impact = EventImpactStudy().analyze_event('AAPL', event, analysis_end='2025-01-01')
    assert impact.post_event_return == pytest.approx(math.log(1.01))
    assert impact.cumulative_return == pytest.approx(math.log(1.04))
    assert impact.window_end.date().isoformat() == '2024-11-04'
    assert impact.n_observations == 2


def test_empty_call_side_preserves_available_puts(monkeypatch):
    from backend.data_sources import options
    ticker=SimpleNamespace(info={'regularMarketPrice':100},options=['2026-12-18'],
        option_chain=lambda:SimpleNamespace(calls=pd.DataFrame(),puts=pd.DataFrame({'strike':[100.],'impliedVolatility':[.2]})))
    monkeypatch.setattr(options.yf,'Ticker',lambda _:ticker)
    chain=options.get_option_chain('FIXTURE')
    assert chain.strikes.tolist()==[100.]
    assert chain.put_ivs.tolist()==[.2]
    assert chain.call_ivs.tolist()==[0.]


def impact(day,post):
    event=macro_events.MacroEvent('CPI',pd.Timestamp(day+' 08:30',tz='US/Eastern'),'US/Eastern','Fixture')
    return EventImpact(event,'FIXTURE',0.,post,post,event.timestamp-pd.Timedelta(days=1),event.timestamp,1,'daily')


def test_report_assets_are_immutable_and_numerical_companion_is_precise(monkeypatch,tmp_path):
    import backend.reporting.report as module
    def plot(items,path):Path(path).write_text(str(items[0].post_event_return))
    monkeypatch.setattr(module,'plot_event_impact',plot)
    first=Path(Report(str(tmp_path)).generate_report('FIXTURE',None,None,[],[impact('2024-10-10',.0123456789)]))
    first_data=json.loads(first.with_suffix('.json').read_text())
    first_asset=tmp_path/first_data['event_chart_url'].lstrip('/')
    snapshot=first_asset.read_bytes()
    second=Path(Report(str(tmp_path)).generate_report('FIXTURE',None,None,[],[impact('2024-11-13',.02)]))
    assert first!=second and first_asset.read_bytes()==snapshot
    assert first_data['events_data'][0]['Post-Return']==.0123456789
    assert 'assets/'+first_asset.parent.name+'/event_impact.png' in first.read_text()
    assert 'normal historical ranges' not in first.read_text()


def test_repeated_event_types_have_dated_bars(monkeypatch,tmp_path):
    import backend.reporting.charts as charts
    seen={}
    def save(*a,**kw):
        ax=charts.plt.gca()
        seen['heights']=[p.get_height() for p in ax.patches]
        seen['labels']=[t.get_text() for t in ax.get_xticklabels()]
    monkeypatch.setattr(charts.plt,'savefig',save)
    charts.plot_event_impact([impact('2024-11-13',.02),impact('2024-10-10',.01)],str(tmp_path/'chart.png'))
    assert seen['heights'][-2:]==[.01,.02]
    assert '2024-10-10' in seen['labels'][0] and '2024-11-13' in seen['labels'][1]


def test_report_uses_p_value_not_fixed_t_cutoff(tmp_path):
    from backend.analytics.factor_model import FactorFitResult
    fit=FactorFitResult({'Market':1.},0,.2,{'Market':2.01,'alpha':0},{'Market':.055339,'alpha':1},30,['Market'])
    report=Path(Report(str(tmp_path)).generate_report('FIXTURE',None,fit,[],[]))
    assert '| 2.01 | 0.0553 | No |' in report.read_text()
    assert '|t-stat| > 2' not in report.read_text()
    from backend.web import report_results
    result=report_results(report,'FIXTURE')
    assert result['factors'][0]['p-value']==.055339
