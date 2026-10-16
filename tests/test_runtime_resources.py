"""Application resources must not depend on the current directory."""
from fastapi.testclient import TestClient

from backend.data_sources.macro_events import load_events_from_csv
from backend.web import app


def test_react_page_and_calendar_load_outside_the_source_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        response = client.get('/')
    assert response.status_code == 200
    assert 'Include current options (SABR)' in response.text
    assert 'id="start-date"' in response.text
    assert len(load_events_from_csv()) == 8
