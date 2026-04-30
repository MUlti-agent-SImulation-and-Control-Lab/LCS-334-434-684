# Event Trigger Polynomial Control Project

## Team Members

* Akshit (23034)
* Priyanshu (23246)

---

## Repository Structure

```
Akshit_23034-Priyanshu_23246/
├── README.md
├── requirements.txt
├── src/                # MATLAB source code files
├── logs/               # (Not used / placeholder)
└── report/
    ├── Final_Simulation_report.pdf
    └── Final_theory_report.pdf
```

---

## Requirements

* MATLAB R2020 or above

---

## Files Description (src/)

* `OneDcontrol.m` → Simulation of one-dimensional trajectory tracking
* `TwoDControl_Circle.m` → Simulation of two-dimensional circular trajectory tracking
* `Fig8.m` → Simulation of figure-8 trajectory tracking
* `First3D.m` → Simulation of three-dimensional trajectory tracking
* `TwoDBoundness.m` → Analysis of trajectory boundedness
* `TwoDCirclewithLYAPUNOV.m` → Lyapunov-based control for circular trajectory

---

## Report

The project reports are available in the `report/` folder:

* `Final_Simulation_report.pdf` → Contains simulation results and analysis
* `Final_theory_report.pdf` → Contains theoretical background and derivations given in the paper

---

## How to Run

1. Open MATLAB
2. Navigate to the `src/` folder
3. Run any of the following scripts:

   * `OneDcontrol.m`
   * `TwoDControl_Circle.m`
   * `Fig8.m`
   * `First3D.m`
   * `TwoDBoundness.m`
   * `TwoDCirclewithLYAPUNOV.m`

---

## Output

* The scripts generate plots showing trajectory tracking performance
* Results correspond to those presented in the reports inside `report/`

---

## Verification

The outputs generated from the MATLAB scripts match the results and analysis provided in:

```
report/Final_Simulation_report.pdf  

```

---

## Notes

* Ensure all files remain in their respective folders
* No modification is required to run the code
* The project executes successfully using the specified MATLAB version
