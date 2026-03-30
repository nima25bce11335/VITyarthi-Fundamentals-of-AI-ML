# VITyarthi-Fundamentals-of-AI-ML
This Predictive Maintenance script implements Linear Regression to predict a machine's Remaining Useful Life. It uses Common Distributions (Normal/Uniform) to simulate sensor data and employs Validation Sets to ensure model accuracy. It satisfies your syllabus goals for Supervised Learning and real-world Case Studies.

Project Overview:
This project demonstrates a Supervised Learning approach to solving a real-world industrial problem: predicting the Remaining Useful Life (RUL) of machinery. By analyzing sensor data like temperature and vibration, the system acts as a Problem Solving Agent to prevent equipment failure through statistical forecasting.

Technical Logic:
The model follows the standard AI/ML pipeline:


1.Environment Simulation: Generates synthetic data where RUL is a function of operational hours and stress factors (vibration/temp).


2.Statistical Decision Theory: Uses an Estimator (Linear Regression) to find the line of best fit between sensor inputs and machine longevity.


3.Predictive Diagnostics: An interactive interface allows users to input real-time telemetry to receive an instant health report.

How to Run:
1.Prerequisites: Ensure you have numpy, pandas, and sklearn installed.

2.Execution: Run the script in a Python environment.

python predictive_maintenance.py
3.Interaction: Enter the requested values for Vibration, Temperature, and Hours Run when prompted to see the AI's prediction.

Evaluation Metrics:
The system evaluates the "Rationality" of the agent using:


1.Mean Squared Error (MSE): Measures the average squared difference between estimated and actual values.


2.R-squared Score: Indicates how much of the variance in machine life is explained by our sensor features.
