from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
import sagemaker

def create_inference_pipeline(role, bucket):
    input_data = f"s3://{bucket}/data/test.csv"
    model_path = f"s3://{bucket}/models/your_trained_model.tar.gz"

    processor = SKLearnProcessor(
        framework_version="0.23-1", 
        role=role,
        instance_type="ml.m5.large",
        instance_count=1
    )

    inference_step = ProcessingStep(
        name="BatchInference",
        processor=processor,
        code="scripts/batch_inference.py",
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            ),
            sagemaker.processing.ProcessingInput(
                source=model_path,
                destination="/opt/ml/processing/model"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/predictions"
            )
        ],
        job_arguments=[
            "--input", "/opt/ml/processing/input",
            "--model", "/opt/ml/processing/model",
            "--output", "/opt/ml/processing/output"
        ]
    )
    pipeline = Pipeline(
        name="mlops-inference-pipeline",
        steps=[inference_step]
    )

    return pipeline

if __name__ == "__main__":
    session = sagemaker.Session()
    role = "arn:aws:iam::029937870282:role/SageMakerExecutionRole"
    bucket = session.default_bucket()
    pipeline = create_inference_pipeline(role, bucket)
    pipeline.upsert(role_arn=role)
    pipeline.start()
