import os, json, sys
import requests

BASE = os.environ.get("CVAT_URL", "http://localhost:8080").rstrip("/")
USER = os.environ["CVAT_USER"]
PASS = os.environ["CVAT_PASS"]
TASK_ID = int(os.environ.get("CVAT_TASK_ID", "2"))
JOB_ID = int(os.environ.get("CVAT_JOB_ID", "2"))
OUT = os.environ.get("CVAT_BACKUP_OUT")

s = requests.Session()
# try login
r = s.post(f"{BASE}/api/auth/login", json={"username": USER, "password": PASS})
if r.status_code >= 400:
    # try with email field variations? CVAT accepts username or email
    r = s.post(f"{BASE}/api/auth/login", json={"email": USER, "password": PASS})
if r.status_code >= 400:
    print("LOGIN_FAIL", r.status_code, r.text[:300])
    sys.exit(2)
print("LOGIN_OK")
def get(path):
    rr = s.get(f"{BASE}{path}")
    if not rr.ok:
        return {"error": rr.status_code, "text": rr.text[:500]}
    return rr.json()
backup = {
    "task": get(f"/api/tasks/{TASK_ID}"),
    "task_annotations": get(f"/api/tasks/{TASK_ID}/annotations"),
    "job_annotations": get(f"/api/jobs/{JOB_ID}/annotations"),
    "labels": get(f"/api/labels?task_id={TASK_ID}"),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(backup, f, ensure_ascii=False, indent=2)
ta = backup["task_annotations"]; ja = backup["job_annotations"]
print("SAVED", OUT)
print("TASK_SHAPES", len(ta.get("shapes", [])) if isinstance(ta, dict) else ta)
print("JOB_SHAPES", len(ja.get("shapes", [])) if isinstance(ja, dict) else ja)
t = backup["task"]
if isinstance(t, dict):
    print("TASK", t.get("id"), t.get("name"))
