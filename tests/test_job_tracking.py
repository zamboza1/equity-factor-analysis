from types import SimpleNamespace
import json
from fastapi.testclient import TestClient
import backend.web as web


def test_request_only_completes_with_its_own_successful_worker(monkeypatch,tmp_path):
    monkeypatch.setattr(web,'reports_dir',tmp_path)
    monkeypatch.setattr(web,'analysis_jobs',{})
    processes=[]
    def launch(command,**kw):
        p=SimpleNamespace(code=None,command=command)
        p.poll=lambda:p.code
        processes.append(p)
        return p
    monkeypatch.setattr(web.subprocess,'Popen',launch)
    (tmp_path/'AAPL_old.md').write_text('**R²:** 0.123')
    with TestClient(web.app) as client:
        body={'tickers':['AAPL'],'start_date':'2024-01-01','end_date':'2024-12-31','fit_sabr':False}
        first=client.post('/api/run-analysis',json=body).json()['jobs'][0]['job_id']
        second=client.post('/api/run-analysis',json=body).json()['jobs'][0]['job_id']
        assert first!=second
        assert client.get('/api/jobs/'+first).json()['status']=='running'
        assert '--fit-sabr' not in processes[0].command
        assert processes[0].command[-2:]==['--run-id',first]
        (tmp_path/f'AAPL_{first}.md').write_text('**R²:** 0.456')
        assert client.get('/api/jobs/'+first).json()['status']=='running'
        processes[0].code=0
        complete=client.get('/api/jobs/'+first).json()
        assert complete['status']=='succeeded'
        assert complete['result']['r_squared']==.456
        assert complete['result']['report_path']==f'AAPL_{first}.md'
        assert client.get('/api/jobs/'+second).json()['status']=='running'
        (tmp_path/'jobs'/f'{second}.log').write_text('✗ Error: Insufficient overlapping data: 9 observations\n')
        processes[1].code=1
        failed=client.get('/api/jobs/'+second).json()
        assert failed['status']=='failed'
        assert failed['error']=='Insufficient overlapping data: 9 observations'
        assert 'result' not in failed
        assert json.loads((tmp_path/'jobs'/f'{second}.json').read_text())['exit_code']==1


def test_success_without_report_is_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(web,'reports_dir',tmp_path)
    monkeypatch.setattr(web,'analysis_jobs',{})
    monkeypatch.setattr(web.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(poll=lambda:0))
    with TestClient(web.app) as client:
        j=client.post('/api/run-analysis',json={'tickers':['A'],'start_date':'2024-01-01','end_date':'2024-12-31'}).json()['jobs'][0]['job_id']
        (tmp_path/'AAPL_unrelated.md').write_text('**R²:** 0.999')
        assert client.get('/api/jobs/'+j).json()['status']=='failed'
        assert client.get('/api/jobs/missing').status_code==404


def test_empty_ticker_list_and_equal_dates_rejected():
    with TestClient(web.app) as client:
        body={'tickers':[],'start_date':'2024-01-01','end_date':'2024-12-31'}
        assert client.post('/api/run-analysis',json=body).status_code==422
        body.update(tickers=['AAPL'],end_date='2024-01-01')
        assert client.post('/api/run-analysis',json=body).status_code==400
