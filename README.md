# Hotel Check-in & Housekeeping DES (Streamlit App)

This app visualizes a discrete-event simulation (DES) of a hotel front desk + housekeeping system.

## Features
- Time-varying **front desk staffing** and **housekeeping shift**
- Check-in/checkout policy, arrivals by hour-of-day profile
- Room states: **VC** (vacant-clean), **VD** (vacant-dirty), **O** (occupied)
- KPIs: wait times, utilization, occupancy, early-checkin success
- Time series plots (queues, occupancy), distributions, and downloads

## Quick start
```bash
# 1) Create a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

## Files
- `app.py` — Streamlit UI that runs the simulation and builds charts
- `hotel_des2.py` — DES engine (no external dependencies)
- `requirements.txt` — Python dependencies

## Notes
- All times inside the engine are in **hours**; UI shows waits in **minutes**
- You can change staffing and shift shapes directly from the sidebar
- Random seed is configurable for reproducibility

## Roadmap (optional)
- Room types & VIP priority
- Rush-clean priority queue
- Maintenance / out-of-order states
- Scenario compare and batch runs
