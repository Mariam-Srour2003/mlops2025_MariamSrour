import mlflow.sklearn

# After training, test loading:
model = mlflow.sklearn.load_model("models:/NYC_Taxi_gb/None")
print("Model loaded successfully by name!")
