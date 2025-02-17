import logging
import os
from pathlib import Path
import tarfile
from aws_setup import create_iam_role
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Deploying model to SageMaker to region: {}".format(os.getenv('REGION', 'us-east-1')))

def create_model_tarfile(model_dir: Path, code_dir: Path) -> Path:
    """Create a tar.gz file containing the model and code."""
    # Ensure directories exist
    model_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Create tar.gz file
    tar_path = Path('model.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(model_dir, arcname=model_dir.name)
    
    return tar_path

def deploy_to_sagemaker():
    try:
        # Get IAM role
        role_arn = create_iam_role()
        
        # Setup AWS session
        region = os.getenv('REGION', 'us-east-1')
        logger.info(f"Creating SageMaker session in region: {region}")
        boto_session = boto3.Session(region_name=region)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        
        # Setup paths using pathlib
        base_dir = Path.cwd()
        model_dir = base_dir / 'model'
        code_dir = model_dir / 'code'
        
        # Create and upload model.tar.gz
        tar_path = create_model_tarfile(model_dir, code_dir)
        
        # Upload to S3
        model_data = sagemaker_session.upload_data(
            str(tar_path), 
            bucket=sagemaker_session.default_bucket(),
            key_prefix='distilgpt2'
        )
        
        # Create HuggingFace model
        model = HuggingFaceModel(
            model_data=model_data,
            transformers_version="4.26.0",
            pytorch_version="1.13.1",
            py_version="py39",
            role=role_arn,
            entry_point='inference.py',
            source_dir=str(code_dir), 
            sagemaker_session=sagemaker_session,
            env={
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                'SAGEMAKER_PROGRAM': 'inference.py',
                'HF_MODEL_ID': str(os.getenv('MODEL_ID'))
            }
        )
        
        logger.info("Deploying quantized model to SageMaker...")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name='distilgpt2',
            wait=True,
            timeout=900,
            tags=[]
        )
        
        logger.info(f"Model deployed to SageMaker endpoint: {predictor.endpoint_name}")
        return predictor
        
    except Exception as e:
        logger.error(f"Error deploying to SageMaker: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        if tar_path.exists():
            tar_path.unlink()

if __name__ == "__main__":
    deploy_to_sagemaker()