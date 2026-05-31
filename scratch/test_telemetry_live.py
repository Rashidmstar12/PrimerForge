"""Live integration script to test uvicorn FastAPI Telemetry API in real-time."""

import time
import requests
import numpy as np

# 1. Simulate a classic sigmoidal qPCR curve (positive sample)
Cq_true = 23.0
baseline = 12.0
max_val = 110.0
k = 0.55

cycles_pos = []
np.random.seed(101)
for t in range(40):
    val = baseline + max_val / (1.0 + np.exp(-k * (t - Cq_true)))
    cycles_pos.append(float(val + np.random.normal(0, 0.05)))

# 2. Simulate flat noise (negative control sample)
cycles_neg = [float(12.0 + np.random.normal(0, 0.08)) for _ in range(40)]

payload = {
    "experiments": [
        {
            "forward_seq": "ATGCATGCATGCATGC",
            "reverse_seq": "GCATGCATGCATGCAT",
            "fluorescence_cycles": cycles_pos,
            "temperature_dissociation": [60.0, 70.0, 80.0, 90.0],
            "fluorescence_dissociation": [10.0, 8.0, 2.5, 0.1]
        },
        {
            "forward_seq": "AAAATTTTGCGCATGC",
            "reverse_seq": "GCGCGCGCATATATAT",
            "fluorescence_cycles": cycles_neg
        }
    ]
}

url = "http://127.0.0.1:8000/api/v1/telemetry/ingest"

print("==========================================================")
print("     PRIMERFORGE LIVE REAL-TIME TELEMETRY TEST SCRIPT     ")
print("==========================================================")
print(f"Sending POST request to live API endpoint: {url}...")

try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nIngest Status:", data["status"].upper())
        print(f"Processed Experiments Count: {data['processed']}\n")
        
        for res in data["results"]:
            success_str = "SUCCESS" if res["empirical_success"] else "FAILURE"
            print(f"[Experiment {res['experiment_index']}]")
            print(f"  Forward: {res['forward']}")
            print(f"  Reverse: {res['reverse']}")
            print(f"  Classification: {success_str} (Cq={res['Cq']}, Melt Peaks={res['melt_peaks']})")
            
        print("\n[Online Calibration Sigmoid (Platt)]")
        print(f"  platt_a: {data['calibration_metrics']['platt_a']}")
        print(f"  platt_b: {data['calibration_metrics']['platt_b']}")
        
        print("\n[EWC Fine-Tuning Losses]")
        print(f"  New loss: {data['calibration_metrics']['new_losses'][-1]:.4f}")
        print(f"  EWC penalty: {data['calibration_metrics']['ewc_penalties'][-1]:.4f}")
        print(f"  Replay loss: {data['calibration_metrics']['replay_losses'][-1]:.4f}")
    else:
        print(f"Server Error Response: {response.text}")
except Exception as e:
    print(f"Error connecting to uvicorn server: {e}")
print("==========================================================\n")
