import time
import pandas as pd
import plotly.express as px
import streamlit as st
from cyberguard_engine import CyberGuardEngine
from traffic_simulator import FEATURE_COLUMNS, generate_live_flow

st.set_page_config(page_title="CyberGuard | Next-Gen AI IDS", page_icon="🛡️", layout="wide")

@st.cache_resource
def init_engine():
    engine = CyberGuardEngine()
    metrics = engine.train()
    return engine, metrics

engine, train_metrics = init_engine()

initialize_demo_log = "event_log" not in st.session_state
if initialize_demo_log:
    st.session_state.event_log = []

st.title("🛡️ CyberGuard: Next-Gen Hybrid AI IDS")
st.caption("Dual-Engine Defense: Random Forest + Isolation Forest + XAI")

with st.sidebar:
    st.success(f"Accuracy: {train_metrics['accuracy'] * 100:.2f}%")
    attack_prob = st.slider("Threat Intensity (%)", 5, 90, 35) / 100.0
    batch_size = st.number_input("Flows per Step", 1, 50, 10)
    capture_btn = st.button("⚡ Capture Traffic", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Logs", use_container_width=True)
    if clear_btn:
        st.session_state.event_log = []
        st.rerun()

if capture_btn or initialize_demo_log:
    num = batch_size if capture_btn else 12
    for _ in range(num):
        flow = generate_live_flow(attack_probability=attack_prob if capture_btn else 0.40)
        analysis = engine.analyze_flow(flow)
        st.session_state.event_log.insert(0, {
            "Timestamp": time.strftime("%H:%M:%S"),
            "Source IP": flow["src_ip"],
            "Verdict": analysis["verdict"],
            "Risk Score": analysis["risk_score"],
            "Detection Engine": analysis["detection_source"],
            "Primary XAI Reason": analysis["top_reasons"][0]["summary"],
            "Mitigation Action": analysis["mitigation"]["action"],
            "Firewall Rule": analysis["mitigation"]["command"],
        })

log_df = pd.DataFrame(st.session_state.event_log)
k1,k2,k3 = st.columns(3)
k1.metric("Total Flows", len(log_df))
k2.metric("Intrusions", len(log_df[log_df["Verdict"]!="Normal"]))
k3.metric("Avg Risk", f"{log_df['Risk Score'].mean():.1f}")

st.dataframe(log_df, use_container_width=True, hide_index=True)

if not log_df[log_df["Verdict"]!="Normal"].empty:
    st.code(log_df[log_df["Verdict"]!="Normal"].iloc[0]["Firewall Rule"], language="bash")
