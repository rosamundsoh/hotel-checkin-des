# app.py
import streamlit as st
import pandas as pd
import numpy as np
from hotel_des2 import HotelDES2

st.set_page_config(page_title="Hotel Check-in & Housekeeping DES", layout="wide")
st.title("🏨 Hotel Check-in & Housekeeping Discrete-Event Simulation")
st.caption("Interactive simulation of front desk flow, room readiness, and housekeeping.")

with st.sidebar:
    st.header("Simulation Inputs")
    col_a, col_b = st.columns(2)
    with col_a:
        n_rooms = st.number_input("Rooms", 10, 2000, 200, 10)
        sim_days = st.number_input("Sim days (measured)", 1, 60, 7, 1)
        warmup_days = st.number_input("Warm-up days", 0, 60, 7, 1)
        seed = st.number_input("Random seed", 0, 10_000, 42, 1)
    with col_b:
        mean_daily_arrivals = st.number_input("Mean daily arrivals", 1, 2000, 80, 1)
        avg_los_nights = st.number_input("Avg LOS (nights)", 1.0, 14.0, 2.0, 0.1)
        checkin_hour = st.number_input("Check-in hour", 0.0, 23.5, 15.0, 0.5)
        checkout_hour = st.number_input("Checkout hour", 0.0, 23.5, 12.0, 0.5)

    st.subheader("Front Desk")
    fd_min = st.slider("FD service time - min (min)", 1, 15, 3)
    fd_mode = st.slider("FD service time - mode (min)", 2, 20, 6)
    fd_max = st.slider("FD service time - max (min)", 3, 30, 10)

    st.subheader("Housekeeping")
    hk_mean_clean = st.slider("Mean clean time (min)", 10, 120, 35)
    hk_sigma = st.slider("Clean time variability (lognormal sigma)", 0.1, 1.5, 0.5, 0.1)
    hk_start = st.slider("HK shift start (hour)", 0.0, 12.0, 9.0, 0.5)
    hk_end = st.slider("HK shift end (hour)", 12.0, 24.0, 17.0, 0.5)
    hk_cleaners = st.number_input("Cleaners on shift", 0, 200, 12, 1)

    st.subheader("Front Desk Staffing (simple profile)")
    fd_agents_night = st.number_input("Agents 00–08", 0, 50, 2, 1)
    fd_agents_morn = st.number_input("Agents 08–12", 0, 50, 3, 1)
    fd_agents_peak = st.number_input("Agents 12–20", 0, 50, 6, 1)
    fd_agents_even = st.number_input("Agents 20–24", 0, 50, 3, 1)

    run_btn = st.button("▶️ Run Simulation", use_container_width=True)

# Construct schedules from inputs

def fd_schedule(t):
    hod = t % 24
    if 0 <= hod < 8:
        return fd_agents_night
    elif 8 <= hod < 12:
        return fd_agents_morn
    elif 12 <= hod < 20:
        return fd_agents_peak
    else:
        return fd_agents_even


def hk_schedule(t):
    hod = t % 24
    if hk_start <= hod < hk_end:
        return hk_cleaners
    return 0

# Run simulation when clicked
if run_btn:
    with st.spinner('Running simulation...'):
        model = HotelDES2(
            n_rooms=int(n_rooms),
            sim_days=int(sim_days),
            warmup_days=int(warmup_days),
            mean_daily_arrivals=float(mean_daily_arrivals),
            avg_los_nights=float(avg_los_nights),
            checkin_hour=float(checkin_hour),
            checkout_hour=float(checkout_hour),
            fd_service_tri_mins=(int(fd_min), int(fd_mode), int(fd_max)),
            hk_mean_clean_mins=int(hk_mean_clean),
            hk_lognorm_sigma=float(hk_sigma),
            hk_shift_start=float(hk_start),
            hk_shift_end=float(hk_end),
            fd_schedule=fd_schedule,
            hk_cleaners_schedule=hk_schedule,
            random_seed=int(seed),
        )
        summary = model.run()
        metrics = model.metrics

    st.success('Simulation complete!')

    # KPIs
    res = summary['results (averages over measured window)']
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Avg arrival→room (min)", f"{res['avg_total_arrival_to_room_minutes']:.1f}")
    kpi_cols[1].metric("Wait after FD (min)", f"{res['avg_wait_for_room_after_fd_minutes']:.1f}")
    kpi_cols[2].metric("FD Utilization", f"{res['front_desk_utilization']*100:.1f}%")
    kpi_cols[3].metric("HK Utilization", f"{res['housekeeping_utilization']*100:.1f}%")

    kpi_cols2 = st.columns(4)
    kpi_cols2[0].metric("Avg Occ Rate", f"{res['avg_occupancy_rate']*100:.1f}%")
    kpi_cols2[1].metric("FD queue len", f"{res['avg_front_desk_queue_len']:.2f}")
    kpi_cols2[2].metric("VD queue len", f"{res['avg_housekeeping_queue_len']:.2f}")
    kpi_cols2[3].metric("Early-checkin success", f"{res['early_checkin_success_rate_given_eligible']*100:.1f}%")

    st.markdown("---")
    st.subheader("Time Series")
    # Build time series dataframe (hours → days:hours label)
    def fmt_time(h):
        day = int(h//24)
        hh = int(h%24)
        mm = int((h*60)%60)
        return f"D{day} {hh:02d}:{mm:02d}"

    ts_hours = [t for t,_ in metrics['occ_obs']]
    df_ts = pd.DataFrame({
        'time_h': ts_hours,
        'time': [fmt_time(t) for t in ts_hours],
        'occupancy': [o for _,o in metrics['occ_obs']],
    })
    # Align FD and VD queues by nearest index length
    fd_q = [q for _,q in metrics['fd_queue_obs']]
    vd_q = [q for _,q in metrics['cleaning_queue_obs']]
    n = len(df_ts)
    df_ts['fd_queue'] = fd_q[:n] if len(fd_q)>=n else fd_q + [fd_q[-1] if fd_q else 0]*(n-len(fd_q))
    df_ts['vd_queue'] = vd_q[:n] if len(vd_q)>=n else vd_q + [vd_q[-1] if vd_q else 0]*(n-len(vd_q))
    df_ts['occ_rate'] = df_ts['occupancy'] / float(n_rooms)

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df_ts.set_index('time_h')[['occ_rate']], height=280)
        st.caption("Occupancy rate over time (0–1)")
    with c2:
        st.line_chart(df_ts.set_index('time_h')[['fd_queue','vd_queue']], height=280)
        st.caption("Queue lengths: Front Desk and Vacant-Dirty (awaiting clean)")

    st.markdown("---")
    st.subheader("Wait Time Distributions (minutes)")
    fd_wait_min = np.array(metrics['fd_wait_times'])*60.0
    room_wait_min = np.array(metrics['room_wait_times'])*60.0
    total_wait_min = np.array(metrics['total_to_room_times'])*60.0

    def percentile_block(arr):
        if len(arr)==0:
            return {"p50":0,"p90":0,"p95":0}
        return {
            'p50': float(np.round(np.percentile(arr,50),1)),
            'p90': float(np.round(np.percentile(arr,90),1)),
            'p95': float(np.round(np.percentile(arr,95),1)),
        }

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Front Desk wait**")
        st.bar_chart(pd.DataFrame(fd_wait_min, columns=['FD Wait (min)']).value_counts().reset_index(drop=True))
        p = percentile_block(fd_wait_min)
        st.caption(f"P50 {p['p50']} | P90 {p['p90']} | P95 {p['p95']}")
    with col2:
        st.write("**Wait for room after FD**")
        st.bar_chart(pd.DataFrame(room_wait_min, columns=['Room Wait (min)']).value_counts().reset_index(drop=True))
        p = percentile_block(room_wait_min)
        st.caption(f"P50 {p['p50']} | P90 {p['p90']} | P95 {p['p95']}")
    with col3:
        st.write("**Total time: Arrival → Room**")
        st.bar_chart(pd.DataFrame(total_wait_min, columns=['Total Wait (min)']).value_counts().reset_index(drop=True))
        p = percentile_block(total_wait_min)
        st.caption(f"P50 {p['p50']} | P90 {p['p90']} | P95 {p['p95']}")

    st.markdown("---")
    st.subheader("Download Outputs")
    # Summary JSON
    st.download_button("Download summary (JSON)", data=str(summary).encode('utf-8'), file_name='summary.txt')

    # Time series CSV
    ts_csv = df_ts.to_csv(index=False).encode('utf-8')
    st.download_button("Download time series (CSV)", data=ts_csv, file_name='time_series.csv')

else:
    st.info("Set your inputs on the left, then click **Run Simulation** to generate KPIs and charts.")
