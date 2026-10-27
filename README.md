# CyberGuard

## Next-Gen Hybrid AI for Intrusion Detection and Cybersecurity

CyberGuard is an AI-powered Network Intrusion Detection and Prevention System
(NIDS/IPS). It combines supervised behavioral classification, unsupervised
zero-day anomaly detection, and Explainable AI (XAI) in an interactive security
operations dashboard.

## Why CyberGuard Is Different

### Dual-engine hybrid detection

- **Random Forest classifier:** identifies known attack patterns:
  `DDoS_SYN_Flood`, `Port_Scan`, `Brute_Force`, and `Data_Exfiltration`.
- **Isolation Forest anomaly engine:** learns from normal baseline traffic to
  flag anomalous flows, including simulated zero-day threats not included in
  supervised training.

### Explainable AI root-cause attribution

CyberGuard calculates Z-score deviations from normal traffic for each flow
feature. The dashboard displays the top deviations to help analysts understand
why a flow was flagged.

### Automated active mitigation (SOAR Lite)

For detected threats, CyberGuard generates Linux `iptables` commands for
actions such as blocking or rate-limiting traffic. Review generated commands
before running them; the application displays commands but does not execute
them automatically.

### Interactive SOC dashboard

The Streamlit and Plotly dashboard includes a simulated traffic feed, detection
metrics and charts, and a manual threat sandbox for testing custom flow
features.

## Quickstart

### Requirements

- Python
- Git

### Setup and run

```bash
# Clone the repository
git clone https://github.com/Suhaspawar05/CyberGuard-IDS.git
cd CyberGuard-IDS

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the CyberGuard SOC dashboard
streamlit run app.py
```

The first launch trains the models and saves the model bundle to
`models/cyberguard_bundle.pkl`. The `.gitignore` excludes this generated file
and your virtual environment from Git.

## Publish Your Local Project to GitHub

If you are starting from a local project folder that is not already a Git
repository, run these commands from the `CyberGuard-IDS` folder. Git must be
installed on your machine.

```bash
git init
git add .
git commit -m "Initial commit: CyberGuard Hybrid AI IDS dashboard"
git branch -M main
```

Create a new empty repository named `CyberGuard-IDS` on
[GitHub](https://github.com/new). Leave the options to add a README, license,
or `.gitignore` unchecked, then connect and push your local project:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/CyberGuard-IDS.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username. If you cloned this
project from GitHub using the Quickstart above, it is already a Git repository;
use `git add`, `git commit`, and `git push` without running `git init` or adding
another `origin` remote.

## Possible Next Steps

- Add optional live packet capture with Scapy. Capturing traffic may require
  elevated permissions; use only on networks you own or are authorized to
  monitor.
- Expose threat scoring through a FastAPI endpoint.
- Add preprocessing support for a real intrusion-detection dataset, such as
  CIC-IDS2017 or NSL-KDD, and document the feature mapping.