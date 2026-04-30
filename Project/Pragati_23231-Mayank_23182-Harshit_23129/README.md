Installation Guide
1. Install Webots Simulator
Download and install Webots 2025a (or latest) from the official Cyberbotics website:
https://cyberbotics.com/

2. Set Up Python Dependencies
Ensure you have Python 3.9+ installed. Install the required libraries via pip:

Bash
pip install torch numpy matplotlib

3. Configure Webots Python Path
Open Webots.

Go to Preferences > Python Command.

Set the path to your local Python executable (e.g., python3 or the path to your virtual environment).

How to Run the Simulator
Step 1: Loading the Environment
Launch Webots.

Go to File > Open World....

Navigate to LCS_project_Webots/city.wbt and click Open.

Step 2: Selecting the Controller
In the Webots Scene Tree (left sidebar), locate the Robot/UGV node.

Expand the node and find the controller field.

Select the desired strategy:

LCS_code_updated.py (Proposed LTC-SAC)

Step 3: Execution
Click the Real-time run (single arrow) or Fast run (double arrow) button at the top of the interface.

Training logs and performance plots will be automatically generated in the LCS_project_results folder.