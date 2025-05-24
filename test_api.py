import requests

# Test /api/overview
resp = requests.get("http://127.0.0.1:8000/api/overview?dimension=year")
print("Overview (year):", resp.json()[:3])

# Test /api/trend (no filters)
resp = requests.post("http://127.0.0.1:8000/api/trend", json={})
print("Trend (all):", resp.json()[:3])

# Test /api/forecast (next 7 days)
resp = requests.post("http://127.0.0.1:8000/api/forecast", json={"periods":7})
print("Forecast (7 days):", resp.json()[-3:]) 