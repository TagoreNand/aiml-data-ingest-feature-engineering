import mlflow
mlflow.set_tracking_uri('sqlite:///mlruns.db')
client = mlflow.MlflowClient()
versions = client.search_model_versions("name='aiml_platform'")
for v in versions:
    print(f'Version: {v.version}, Run ID: {v.run_id}')
latest = max(versions, key=lambda v: int(v.version))
client.set_registered_model_alias('aiml_platform', 'champion', latest.version)
print(f'Champion alias set on version {latest.version}!')
