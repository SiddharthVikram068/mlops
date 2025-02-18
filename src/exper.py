import mlflow

# Get all experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}, Location: {exp.artifact_location}")

# Get all runs for a specific experiment (replace with your experiment ID)
experiment_id = "0"  # Change this if needed
runs = mlflow.search_runs(experiment_ids=[experiment_id])
print(runs)

