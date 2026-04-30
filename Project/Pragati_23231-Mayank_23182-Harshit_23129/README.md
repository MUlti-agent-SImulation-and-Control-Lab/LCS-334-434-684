Work Done:
1. Created Dynamic environments(pedestrians, multiple vehicles, obstacles, etc.)
2. Test algorithms
3. In model architecture, replaced LSTM with LTCs
4. SAC perform better than DQN
5. Results(averge over 5 experiments)

Future Work:
1. Test with high-speed vehicles
2. Sensor-fusion

LCS_project_Webots: contains all the files to run the environments
LCS_project_results: contains all the results

Hardware requirements and runtime:
-RTX 5060 8GB VRAM
-run for ~10-12 hours

To run it:
1. Install Webots: https://cyberbotics.com/
2. Open LCS_project_Webots/city.wbt
3. Install all the dependencies:
   pip install torch, numpy, matplotlib
4. Recommended: Run on GPU
