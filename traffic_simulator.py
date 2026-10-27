import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "packet_rate_pps",
    "bytes_per_packet",
    "syn_flag_ratio",
    "dst_port_entropy",
    "payload_entropy",
    "flow_duration_ms",
    "failed_auth_count",
]


def generate_training_dataset(
    n_samples: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a labeled synthetic network-flow dataset for training the Hybrid IDS.

    Zero-day exploits are intentionally excluded from supervised training so
    the Isolation Forest anomaly engine can be tested on unseen threats.
    """
    rng = np.random.default_rng(seed)
    records = []

    # Distribution: 65% Normal, 35% known attacks.
    labels = rng.choice(
        [
            "Normal",
            "DDoS_SYN_Flood",
            "Port_Scan",
            "Brute_Force",
            "Data_Exfiltration",
        ],
        size=n_samples,
        p=[0.65, 0.12, 0.10, 0.08, 0.05],
    )

    for label in labels:
        records.append(_sample_flow_by_profile(label, rng))

    return pd.DataFrame(records)


def generate_live_flow(attack_probability: float = 0.30) -> dict:
    """
    Simulate a single real-time network flow, including occasional unseen
    Zero_Day_Exploit anomalies.
    """
    rng = np.random.default_rng()

    if rng.random() > attack_probability:
        profile = "Normal"
    else:
        profile = rng.choice(
            [
                "DDoS_SYN_Flood",
                "Port_Scan",
                "Brute_Force",
                "Data_Exfiltration",
                "Zero_Day_Exploit",
            ],
            p=[0.30, 0.25, 0.20, 0.15, 0.10],
        )

    flow = _sample_flow_by_profile(profile, rng)
    flow["src_ip"] = (
        f"192.168.1.{rng.integers(10, 250)}"
        if profile == "Normal"
        else (
            f"{rng.choice([45, 103, 185, 198])}."
            f"{rng.integers(1, 255)}."
            f"{rng.integers(1, 255)}."
            f"{rng.integers(2, 254)}"
        )
    )
    flow["dst_port"] = int(rng.choice([80, 443, 22, 3389, 53, 8080, 3306]))

    return flow


def _sample_flow_by_profile(
    profile: str,
    rng: np.random.Generator,
) -> dict:
    """Generate statistical features for a traffic class."""
    if profile == "Normal":
        return {
            "packet_rate_pps": float(np.clip(rng.normal(120, 45), 10, 350)),
            "bytes_per_packet": float(np.clip(rng.normal(650, 180), 120, 1460)),
            "syn_flag_ratio": float(np.clip(rng.normal(0.08, 0.04), 0.0, 0.25)),
            "dst_port_entropy": float(
                np.clip(rng.normal(0.45, 0.20), 0.05, 1.5)
            ),
            "payload_entropy": float(
                np.clip(rng.normal(4.8, 0.6), 2.5, 6.2)
            ),
            "flow_duration_ms": float(
                np.clip(rng.normal(850, 300), 50, 2500)
            ),
            "failed_auth_count": float(
                rng.choice([0, 0, 0, 1], p=[0.85, 0.05, 0.05, 0.05])
            ),
            "ground_truth": "Normal",
        }

    elif profile == "DDoS_SYN_Flood":
        return {
            "packet_rate_pps": float(
                np.clip(rng.normal(8500, 2200), 3500, 18000)
            ),
            "bytes_per_packet": float(
                np.clip(rng.normal(64, 12), 40, 120)
            ),
            "syn_flag_ratio": float(
                np.clip(rng.normal(0.94, 0.04), 0.80, 1.0)
            ),
            "dst_port_entropy": float(
                np.clip(rng.normal(0.20, 0.10), 0.0, 0.6)
            ),
            "payload_entropy": float(
                np.clip(rng.normal(1.2, 0.4), 0.2, 2.5)
            ),
            "flow_duration_ms": float(
                np.clip(rng.normal(120, 40), 10, 300)
            ),
            "failed_auth_count": 0.0,
            "ground_truth": "DDoS_SYN_Flood",
        }

    elif profile == "Port_Scan":
        return {
            "packet_rate_pps": float(
                np.clip(rng.normal(1400, 400), 500, 3000)
            ),
            "bytes_per_packet": float(
                np.clip(rng.normal(60, 10), 40, 100)
            ),
            "syn_flag_ratio": float(
                np.clip(rng.normal(0.88, 0.06), 0.70, 1.0)
            ),
            "dst_port_entropy": float(
                np.clip(rng.normal(4.6, 0.5), 3.2, 6.0)
            ),
            "payload_entropy": float(
                np.clip(rng.normal(1.0, 0.3), 0.1, 2.0)
            ),
            "flow_duration_ms": float(
                np.clip(rng.normal(45, 20), 5, 150)
            ),
            "failed_auth_count": 0.0,
            "ground_truth": "Port_Scan",
        }

    elif profile == "Brute_Force":
        return {
            "packet_rate_pps": float(
                np.clip(rng.normal(320, 80), 150, 650)
            ),
            "bytes_per_packet": float(
                np.clip(rng.normal(280, 50), 150, 450)
            ),
            "syn_flag_ratio": float(
                np.clip(rng.normal(0.22, 0.06), 0.10, 0.40)
            ),
            "dst_port_entropy": float(
                np.clip(rng.normal(0.15, 0.05), 0.01, 0.35)
            ),
            "payload_entropy": float(
                np.clip(rng.normal(5.4, 0.4), 4.2, 6.5)
            ),
            "flow_duration_ms": float(
                np.clip(rng.normal(2100, 500), 1000, 4000)
            ),
            "failed_auth_count": float(rng.integers(12, 65)),
            "ground_truth": "Brute_Force",
        }

    elif profile == "Data_Exfiltration":
        return {
            "packet_rate_pps": float(
                np.clip(rng.normal(950, 250), 450, 1800)
            ),
            "bytes_per_packet": float(
                np.clip(rng.normal(1420, 40), 1300, 1500)
            ),
            "syn_flag_ratio": float(
                np.clip(rng.normal(0.03, 0.01), 0.01, 0.08)
            ),
            "dst_port_entropy": float(
                np.clip(rng.normal(0.30, 0.10), 0.05, 0.60)
            ),
            "payload_entropy": float(
                np.clip(rng.normal(7.75, 0.15), 7.3, 7.99)
            ),
            "flow_duration_ms": float(
                np.clip(rng.normal(6500, 1200), 3500, 10000)
            ),
            "failed_auth_count": 0.0,
            "ground_truth": "Data_Exfiltration",
        }

    else:  # Zero_Day_Exploit: stealthy encrypted covert channel / abnormal timing
        return {
            "packet_rate_pps": float(
                np.clip(rng.normal(480, 90), 300, 750)
            ),
            "bytes_per_packet": float(
                np.clip(rng.normal(1150, 120), 900, 1450)
            ),
            "syn_flag_ratio": float(
                np.clip(rng.normal(0.55, 0.08), 0.40, 0.75)
            ),
            "dst_port_entropy": float(
                np.clip(rng.normal(2.8, 0.4), 2.0, 3.8)
            ),
            "payload_entropy": float(
                np.clip(rng.normal(7.85, 0.10), 7.5, 7.99)
            ),
            "flow_duration_ms": float(
                np.clip(rng.normal(9200, 1500), 6000, 14000)
            ),
            "failed_auth_count": float(rng.integers(2, 6)),
            "ground_truth": "Zero_Day_Exploit",
        }