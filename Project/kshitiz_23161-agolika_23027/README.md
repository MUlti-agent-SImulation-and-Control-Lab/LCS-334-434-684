# Risk-Aware Hybrid LQR-MPC Navigation for Autonomous Systems

**Authors:** Kshitiz Kumar Sinha (23161), Agolika BM (23027)  
**Course:** LCS-334/434/684 — Linear Control Systems  
**GitHub (Full Project):** [https://github.com/Erebuzzz/Risk-Aware-Hybrid-LQR-MPC-Navigation-for-Autonomous-Systems](https://github.com/Erebuzzz/Risk-Aware-Hybrid-LQR-MPC-Navigation-for-Autonomous-Systems)

---

## 📁 Repository Structure

```
kshitiz_23161-agolika_23027/
├── README.md               ← You are here
├── requirements.txt        ← All Python dependencies
├── src/                    ← Source code
│   ├── hybrid_controller/  ← Core control library (LQR, MPC, Adaptive MPC, Hybrid Blender)
│   ├── ros2_nodes/         ← ROS2 controller nodes for Gazebo deployment
│   └── run_simulation.py   ← Main simulation entry point
├── logs/                   ← Simulation log files (CSV + JSON telemetry)
├── results/                ← All generated plots (tracking, error, control inputs)
└── report/
    ├── report.tex          ← LaTeX source
    ├── report.pdf          ← Compiled report
    └── plots/              ← Plots referenced by the report
```

---

## ⚙️ Step-by-Step Instructions to Run

### Prerequisites
- Python 3.10 or later
- pip

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install the Core Controller Library

```bash
cd src/hybrid_controller
pip install -e .
cd ../..
```

### Step 3: Run the Simulations

All simulations are run from the `src/` directory. Each mode generates plots into `results/`.

```bash
cd src

# 1. LQR Controller (obstacle-free trajectory tracking)
python run_simulation.py --mode lqr

# 2. Tube MPC Controller (obstacle avoidance with CVXPY/OSQP)
python run_simulation.py --mode mpc

# 3. Adaptive Nonlinear MPC (CasADi/IPOPT with LMS adaptation)
python run_simulation.py --mode adaptive

# 4. Hybrid LQR-MPC Blended Controller
python run_simulation.py --mode hybrid
```

### Step 4: View the Results

After running, plots will be saved to `results/`:
- `results/LQR/figure8/` — LQR tracking, error, and control plots
- `results/MPC/figure8/default/` — MPC obstacle avoidance, error, control plots
- `results/AdaptiveMPC/figure8/default/` — Adaptive MPC plots
- `results/Hybrid/figure8/default/` — Hybrid trajectory, error, blending, control plots

### Step 5: Read the Report

The detailed evaluation report is at `report/report.pdf`.

---

## 🧪 What Each Mode Does

| Mode | Controller | Obstacles | Solver | Key Feature |
|------|-----------|-----------|--------|-------------|
| `lqr` | Pure LQR | None | SciPy DARE | Baseline tracking accuracy |
| `mpc` | Tube MPC | 3 circular | CVXPY + OSQP | Linearized obstacle avoidance |
| `adaptive` | Adaptive NMPC | 3 circular | CasADi + IPOPT | Online LMS parameter estimation |
| `hybrid` | Hybrid LQR-MPC | 3 circular | Both | Sigmoid-blended risk-aware switching |

---

## 📊 Summary of Results

| Metric | LQR | Tube MPC | Adaptive NMPC | Hybrid |
|--------|-----|----------|---------------|--------|
| Mean Error (m) | 0.005 | 1.638 | 1.282 | **0.218** |
| Final Error (m) | 0.003 | 2.269 | 1.978 | **0.056** |
| Solve Time (ms) | <1 | 163 | 145 | 102 |

---

## 📝 Notes
- The ROS2 Gazebo deployment (Docker-based) is documented in the full GitHub repository linked above.
- Simulation logs (CSV telemetry) are in the `logs/` directory for post-run debugging.
