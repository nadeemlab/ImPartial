import mlflow 

def init_mlflow(experiment_name, run_name):
    # mlflow.set_tracking_uri("file:///nadeem_lab/Gunjan/mlruns")
    # mlflow.set_tracking_uri("http://10.0.3.10:8000") # old
    mlflow.set_tracking_uri("http://10.0.3.12:8000")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")

    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.start_run(run_name=run_name)
