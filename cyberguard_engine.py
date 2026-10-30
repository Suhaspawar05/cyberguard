import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from traffic_simulator import FEATURE_COLUMNS, generate_training_dataset


class CyberGuardEngine:
    """
    Hybrid AI intrusion detection engine:

    1. Supervised Random Forest for known multi-class threat classification.
    2. Unsupervised Isolation Forest trained on normal baseline traffic to
       catch unseen zero-day anomalies.
    3. Explainable AI (XAI) Z-score deviation analyzer for human-readable alerts.
    4. Automated SOAR firewall command generator.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        self.anomaly_detector = IsolationForest(
            n_estimators=150,
            contamination=0.04,
            random_state=42,
            n_jobs=-1,
        )

        self.normal_means = {}
        self.normal_stds = {}
        self.is_trained = False

    def train(self, df: pd.DataFrame = None) -> dict:
        """Train the supervised classifier and unsupervised zero-day detector."""
        if df is None:
            df = generate_training_dataset(n_samples=3500)

        X = df[FEATURE_COLUMNS]
        y = df["ground_truth"]

        # Store normal baseline statistics for Explainable AI (XAI).
        normal_df = df[df["ground_truth"] == "Normal"][FEATURE_COLUMNS]
        self.normal_means = normal_df.mean().to_dict()
        self.normal_stds = normal_df.std().replace(0, 1e-6).to_dict()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the supervised threat classifier on all known classes.
        self.classifier.fit(X_train_scaled, y_train)

        # Train the anomaly detector only on normal baseline flows.
        normal_mask = y_train == "Normal"
        self.anomaly_detector.fit(X_train_scaled[normal_mask])

        self.is_trained = True
        self.save_models()

        y_pred = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "accuracy": accuracy,
            "report": report,
            "samples_trained": len(X_train),
        }

    def analyze_flow(self, flow: dict) -> dict:
        """
        Evaluate a network flow using both AI engines, compute a unified
        Threat Risk Score (0-100), explain top contributing features, and
        provide an actionable mitigation command.
        """
        if not self.is_trained:
            self.load_or_train()

        feature_vector = pd.DataFrame(
            [{column: flow[column] for column in FEATURE_COLUMNS}]
        )
        scaled_vector = self.scaler.transform(feature_vector)

        # Engine 1: supervised classification.
        probabilities = self.classifier.predict_proba(scaled_vector)[0]
        classes = self.classifier.classes_
        predicted_index = int(np.argmax(probabilities))
        predicted_class = classes[predicted_index]
        confidence = float(probabilities[predicted_index])
        normal_index = list(classes).index("Normal")
        supervised_attack_probability = 1.0 - float(
            probabilities[normal_index]
        )

        # Engine 2: unsupervised Isolation Forest anomaly score.
        # decision_function returns positive for normal and negative for anomalies.
        raw_anomaly_score = float(
            self.anomaly_detector.decision_function(scaled_vector)[0]
        )
        is_outlier = int(self.anomaly_detector.predict(scaled_vector)[0]) == -1

        # Normalize the anomaly score into a 0-100 risk metric.
        unsupervised_risk = float(
            np.clip((0.15 - raw_anomaly_score) * 220, 0, 100)
        )

        # Hybrid decision fusion logic.
        if predicted_class != "Normal" and confidence >= 0.55:
            final_verdict = predicted_class
            detection_source = "Supervised Behavioral Classifier"
            risk_score = round(
                max(supervised_attack_probability * 100, unsupervised_risk),
                1,
            )
        elif is_outlier or unsupervised_risk >= 62.0:
            final_verdict = "Zero_Day_Anomaly"
            detection_source = "Unsupervised Anomaly Engine (Zero-Day)"
            risk_score = round(max(unsupervised_risk, 75.0), 1)
        else:
            final_verdict = "Normal"
            detection_source = "Baseline Verified"
            risk_score = round(min(unsupervised_risk * 0.4, 25.0), 1)

        explanations = self.explain_prediction(flow)
        mitigation = self.generate_mitigation(
            final_verdict,
            flow.get("src_ip", "0.0.0.0"),
            flow.get("dst_port", 80),
            risk_score,
        )

        return {
            "verdict": final_verdict,
            "risk_score": risk_score,
            "confidence": round(confidence * 100, 1),
            "detection_source": detection_source,
            "top_reasons": explanations,
            "mitigation": mitigation,
        }

    def explain_prediction(self, flow: dict) -> list[dict]:
        """
        Explain predictions by measuring each feature's absolute Z-score
        relative to learned normal traffic.
        """
        contributions = []

        for column in FEATURE_COLUMNS:
            value = float(flow[column])
            mean = self.normal_means.get(column, 0.0)
            std = self.normal_stds.get(column, 1.0)
            z_score = abs(value - mean) / std
            direction = "above" if value >= mean else "below"

            contributions.append(
                {
                    "feature": column,
                    "value": round(value, 2),
                    "normal_avg": round(mean, 2),
                    "z_score": round(z_score, 2),
                    "summary": (
                        f"{column} = {value:.2f} "
                        f"({z_score:.1f}σ {direction} normal avg {mean:.2f})"
                    ),
                }
            )

        contributions.sort(
            key=lambda item: item["z_score"],
            reverse=True,
        )
        return contributions[:3]

    @staticmethod
    def generate_mitigation(
        verdict: str,
        src_ip: str,
        dst_port: int,
        risk_score: float,
    ) -> dict:
        """Generate OS firewall rules based on the threat type."""
        if verdict == "Normal":
            return {"action": "ALLOW", "command": "No action required."}

        if verdict == "DDoS_SYN_Flood":
            return {
                "action": "RATE_LIMIT & SYN_COOKIE",
                "command": (
                    f"sudo iptables -A INPUT -s {src_ip} -p tcp --syn "
                    "-m limit --limit 1/s -j ACCEPT && "
                    f"sudo iptables -A INPUT -s {src_ip} -j DROP"
                ),
            }
        elif verdict == "Port_Scan":
            return {
                "action": "QUARANTINE_SCANNER_IP",
                "command": (
                    f"sudo iptables -A INPUT -s {src_ip} "
                    "-m recent --name portscan --set -j DROP"
                ),
            }
        elif verdict == "Brute_Force":
            return {
                "action": "BLOCK_AUTH_PORT",
                "command": (
                    f"sudo iptables -A INPUT -s {src_ip} -p tcp "
                    f"--dport {dst_port} -j REJECT --reject-with tcp-reset"
                ),
            }
        else:  # Data exfiltration or zero-day anomaly.
            return {
                "action": "IMMEDIATE_HOST_ISOLATION",
                "command": (
                    f"sudo iptables -I INPUT 1 -s {src_ip} -j DROP && "
                    f"sudo iptables -I OUTPUT 1 -d {src_ip} -j DROP"
                ),
            }

    def save_models(self):
        """Save the trained model bundle to disk."""
        joblib.dump(
            {
                "scaler": self.scaler,
                "classifier": self.classifier,
                "anomaly_detector": self.anomaly_detector,
                "normal_means": self.normal_means,
                "normal_stds": self.normal_stds,
            },
            os.path.join(self.model_dir, "cyberguard_bundle.pkl"),
        )

    def load_or_train(self) -> dict:
        """Load a saved model bundle, or train and save a new one."""
        bundle_path = os.path.join(
            self.model_dir,
            "cyberguard_bundle.pkl",
        )

        if os.path.exists(bundle_path):
            data = joblib.load(bundle_path)
            self.scaler = data["scaler"]
            self.classifier = data["classifier"]
            self.anomaly_detector = data["anomaly_detector"]
            self.normal_means = data["normal_means"]
            self.normal_stds = data["normal_stds"]
            self.is_trained = True
            return {"status": "loaded_from_disk"}

        return self.train()
