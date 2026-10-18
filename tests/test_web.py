"""Upload endpoint: folder uploads, mixed file types, ordering — pipeline stubbed (no API calls)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    started = {}

    async def fake_run(input_path, opts, job_id):
        started.update(input=input_path, opts=opts, job_id=job_id)

    monkeypatch.setattr(main, "run_pipeline", fake_run)
    return TestClient(main.app), started


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_folder_upload_keeps_supported_files_in_path_order(client):
    c, started = client
    files = [
        ("files", ("catalog/page10.png", _png(), "image/png")),
        ("files", ("catalog/notes.txt", b"hello", "text/plain")),
        ("files", ("catalog/page2.pdf", b"%PDF-1.7 minimal", "application/pdf")),
    ]
    r = c.post("/jobs", files=files, data={"mode": "cascade", "folder_name": "catalog"}, follow_redirects=False)
    assert r.status_code == 303
    saved = sorted(p.name for p in started["input"].iterdir())
    assert saved == ["0001_page2.pdf", "0002_page10.png"]          # natural order, .txt skipped
    assert main.job_store[started["job_id"]].filename == "catalog/ (2 files)"


def test_upload_with_no_supported_files_is_rejected(client):
    c, _ = client
    r = c.post("/jobs", files=[("files", ("notes.txt", b"hello", "text/plain"))], data={"mode": "cascade"})
    assert r.status_code == 400


def test_customer_text_hides_model_names():
    msg = ("warning: title text not readable by OCR — name confirmed by agreement of "
           "gemini:gemini-3.8-flash and claude:claude-opus-5")
    assert main.customer_text(msg).endswith("agreement of two independent AI models")
    assert "gemini" not in main.customer_text("error from gemini:gemini-3.8-flash")
    assert main.is_internal_note("not escalated: stronger model unavailable")


def test_demo_api_omits_cost(client, monkeypatch):
    c, _ = client
    main.job_store["demo-job"] = main.JobStatus(job_id="demo-job", status="done", filename="x.pdf",
                                                cost_usd=0.5, cost_inr=48.0, tiers="gemini:x")
    internal = c.get("/api/jobs/demo-job?view=internal").json()
    demo = c.get("/api/jobs/demo-job?view=demo").json()
    assert internal["cost_usd"] == 0.5
    assert "cost_usd" not in demo and "tiers" not in demo and demo["cost_inr"] == 0.0


def test_demo_home_lists_only_this_sessions_jobs(client):
    c, started = client
    main.job_store["other-company-job"] = main.JobStatus(job_id="other-company-job", status="done",
                                                         filename="OtherCo_secret.pdf", created_at="2026-01-01")
    r = c.post("/jobs?view=demo", files=[("files", ("mine.pdf", b"%PDF-1.7", "application/pdf"))],
               data={"mode": "cascade"}, follow_redirects=False)
    assert started["opts"].update_catalog is False                 # prospect data stays out of the master
    page = c.get("/?view=demo").text
    assert "mine.pdf" in page and "OtherCo_secret.pdf" not in page
    assert "mine.pdf" not in c.get("/?view=demo&fresh=1").text      # next company starts clean
    assert "OtherCo_secret.pdf" in c.get("/?view=internal").text    # your own view still lists everything


def test_type_is_detected_from_content_not_name(client):
    c, started = client
    r = c.post("/jobs", files=[("files", ("scan.pdf", _png(), "application/pdf"))],
               data={"mode": "cheapest"}, follow_redirects=False)
    assert r.status_code == 303
    assert [p.name for p in started["input"].iterdir()] == ["0001_scan.png"]
    assert started["opts"].tiers == "gemini:gemini-3.5-flash-lite"
