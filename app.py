import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.cyberguard_engine import CyberGuardEngine
from src.traffic_simulator import generate_live_flow


st.set_page_config(
    page_title="CyberGuard | Next-Gen AI IDS",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def init_engine():
    engine = CyberGuardEngine()
    metrics = engine.train()
    return engine, metrics


engine, train_metrics = init_engine()

if "event_log" not in st.session_state:
    st.session_state.event_log = []


def create_event(flow: dict, analysis: dict) -> dict:
    return {
        "Timestamp": time.strftime("%H:%M:%S"),
        "Source IP": flow["src_ip"],
        "Dst Port": flow["dst_port"],
        "Verdict": analysis["verdict"],
        "Risk Score": analysis["risk_score"],
        "Detection Engine": analysis["detection_source"],
        "Primary XAI Reason": analysis["top_reasons"][0]["summary"],
        "Mitigation Action": analysis["mitigation"]["action"],
        "Firewall Rule": analysis["mitigation"]["command"],
    }


# Header
st.title("🛡️ CyberGuard: Next-Gen Hybrid AI Intrusion Detection System")
st.caption(
    "Dual-Engine Defense: Supervised Behavioral Classification (Random Forest) + "
    "Unsupervised Zero-Day Anomaly Detection (Isolation Forest) + Explainable AI (XAI)"
)

# Sidebar controls
with st.sidebar:
    st.header("SOC Control Center")
    st.success(
        f"AI Model Validation Accuracy: {train_metrics['accuracy'] * 100:.2f}%"
    )

    attack_probability = st.slider(
        "Simulated Threat Intensity (%)",
        min_value=5,
        max_value=90,
        value=35,
        step=5,
    ) / 100.0
    batch_size = st.number_input(
        "Flows per Simulation Step",
        min_value=1,
        max_value=50,
        value=10,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        capture_btn = st.button(
            "Capture Traffic",
            use_container_width=True,
            type="primary",
        )
    with col_btn2:
        clear_btn = st.button("Clear Logs", use_container_width=True)

    if clear_btn:
        st.session_state.event_log = []
        st.rerun()

    st.divider()
    st.subheader("Manual Threat Sandbox")
    st.caption("Test how CyberGuard responds to custom network flow metrics:")

    custom_pps = st.slider("Packet Rate (pps)", 10.0, 15000.0, 140.0)
    custom_syn = st.slider("SYN Flag Ratio", 0.0, 1.0, 0.08)
    custom_port_entropy = st.slider(
        "Destination Port Entropy",
        0.0,
        6.0,
        0.45,
    )
    custom_payload_entropy = st.slider(
        "Payload Entropy",
        0.1,
        8.0,
        4.8,
    )
    custom_failed_auth = st.slider("Failed Auth Attempts", 0, 60, 0)

    if st.button("Inspect Custom Flow", use_container_width=True):
        manual_flow = {
            "src_ip": "203.0.113.99",
            "dst_port": 443,
            "packet_rate_pps": custom_pps,
            "bytes_per_packet": 600.0,
            "syn_flag_ratio": custom_syn,
            "dst_port_entropy": custom_port_entropy,
            "payload_entropy": custom_payload_entropy,
            "flow_duration_ms": 900.0,
            "failed_auth_count": float(custom_failed_auth),
            "ground_truth": "Manual_Test",
        }
        result = engine.analyze_flow(manual_flow)
        st.session_state.event_log.insert(0, create_event(manual_flow, result))


# Run a traffic-capture batch.
if capture_btn:
    for _ in range(batch_size):
        flow = generate_live_flow(attack_probability=attack_probability)
        analysis = engine.analyze_flow(flow)
        st.session_state.event_log.insert(0, create_event(flow, analysis))


# Populate initial demo flows if the log is empty.
if not st.session_state.event_log:
    for _ in range(12):
        flow = generate_live_flow(attack_probability=0.40)
        analysis = engine.analyze_flow(flow)
        st.session_state.event_log.append(create_event(flow, analysis))


log_df = pd.DataFrame(st.session_state.event_log)

# Top KPI metrics
total_flows = len(log_df)
threats_df = log_df[log_df["Verdict"] != "Normal"]
zero_days = len(log_df[log_df["Verdict"] == "Zero_Day_Anomaly"])
average_risk = log_df["Risk Score"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Inspected Flows", total_flows)
kpi2.metric(
    "Detected Intrusions",
    len(threats_df),
    delta=f"{(len(threats_df) / total_flows) * 100:.1f}% threat ratio",
    delta_color="inverse",
)
kpi3.metric("Zero-Day Anomalies Caught", zero_days)
kpi4.metric("Average Network Risk Score", f"{average_risk:.1f} / 100")

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    pie_chart = px.pie(
        log_df,
        names="Verdict",
        title="Traffic Classification Breakdown",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    st.plotly_chart(pie_chart, use_container_width=True)

with chart_col2:
    risk_chart = px.histogram(
        log_df,
        x="Risk Score",
        color="Verdict",
        nbins=15,
        title="Threat Risk Score Distribution (Hybrid Engine)",
    )
    st.plotly_chart(risk_chart, use_container_width=True)

# Live Explainable AI alert feed
st.subheader("Real-Time Telemetry & Explainable AI (XAI) Alert Feed")
st.dataframe(
    log_df[
        [
            "Timestamp",
            "Source IP",
            "Dst Port",
            "Verdict",
            "Risk Score",
            "Detection Engine",
            "Primary XAI Reason",
            "Mitigation Action",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

# Automated active-defense firewall rules
st.subheader("Automated SOAR Mitigation Commands (Active Prevention)")

if not threats_df.empty:
    latest_threat = threats_df.iloc[0]
    st.warning(
        f"**Latest Alert:** `{latest_threat['Verdict']}` from "
        f"`{latest_threat['Source IP']}` "
        f"(Risk Score: {latest_threat['Risk Score']}) — "
        f"**Root Cause:** {latest_threat['Primary XAI Reason']}"
    )
    st.code(latest_threat["Firewall Rule"], language="bash")
else:
    st.info("No active threats detected in current buffer.")