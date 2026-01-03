from src.ml_project.pipelines.pipeline import TaxiPipeline



config_file = 'config/config.yaml'  # Path to your config file
pipeline = TaxiPipeline(config_file)
model, best_model_name, output_df = pipeline.run()

print("Best Model Name:", best_model_name)
print("Model Output:", output_df.head())