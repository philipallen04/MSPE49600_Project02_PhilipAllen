# MSPE 49600 - Project 2: Predictive Modeling and Engineering Optimization

**Student:** Philip Allen  
**Course:** MSPE 49600 - Data Analytics for Motorsports  
**Semester:** Fall 2026

## Project overview

This project uses repeated Indianapolis Motor Speedway road-course setup data to build and validate quadratic response-surface models for mean Sector 3 time and mean top speed. The validated surrogate models are then used for bounds-only optimization, constrained optimization with a 284.0 km/h minimum predicted top-speed requirement, grid verification, and a quadratic penalty-function study.

The analysis is intentionally restricted to the tested design region:
- Rear-wing angle: 5 to 13 deg
- Rear ride height: 35 to 45 mm

## Data

Source dataset: `ims_setup_project02.csv`

The data contain 135 lap observations covering 27 unique setup configurations. Repeated laps are grouped by rear-wing angle and rear ride height before train/test splitting to avoid leakage between repeated observations of the same setup.

## Validation summary

| Response | Test RMSE | Test R2 |
|---|---:|---:|
| Mean Sector 3 time | 0.02585 s | 0.99799 |
| Mean top speed | 0.16435 km/h | 0.98583 |

These results support using the response surfaces as local surrogate models inside the tested design region, while not establishing them as physical laws outside the observed setup space.

## Optimization summary

- Bounds-only optimum: approximately 10.54 deg wing, 38.87 mm ride height
  - Predicted Sector 3 time: 31.211 s
  - Predicted top speed: 283.273 km/h
  - Infeasible under the 284.0 km/h requirement

- Final constrained recommendation: approximately 9.54 deg wing, 38.33 mm ride height
  - Predicted Sector 3 time: 31.269 s
  - Predicted top speed: 284.000 km/h
  - The top-speed constraint is active

A 101 x 101 feasible grid search gives a nearby best feasible design, and five substantially different SLSQP starting points converge to the same constrained candidate. The penalty study also approaches the same design as the penalty parameter increases.

The recommendation remains model-based and requires an additional track test or higher-fidelity simulation before being treated as a confirmed physical optimum.

## Main files

- `philipallen04_pr02_notebook.ipynb` - executed analysis notebook
- `philipallen04_pr02_notebook.pdf` - PDF export with visible outputs
- `philipallen04_pr02_notebook.py` - Python export of the notebook analysis
- `philipallen04_pr02_grouped.csv` - setup-level summary data
- `philipallen04_pr02_validation.csv` - surrogate validation metrics
- `philipallen04_pr02_optimization_runs.csv` - constrained optimization sensitivity runs
- `philipallen04_pr02_penalty.csv` - penalty-function study
- `philipallen04_pr02_recommendation.csv` - final candidate comparison
- `philipallen04_pr02_validation.png` - validation plot
- `philipallen04_pr02_unconstrained.png` - bounds-only optimization plot
- `philipallen04_pr02_constrained.png` - constrained optimization plot

## Environment

The project uses Python with:
- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- Jupyter / nbconvert

See `requirements.txt` for the versions used to execute this repository-ready copy.

## Running the project

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Keep `ims_setup_project02.csv` in the same directory as the notebook and Python export.
4. Run the notebook from top to bottom, or execute:
   ```bash
   python philipallen04_pr02_notebook.py
   ```
5. Confirm that the required CSV and PNG outputs are regenerated.
