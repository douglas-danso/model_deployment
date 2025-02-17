import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("REGION", "us-west-2")
    SAGEMAKER_ENDPOINT_NAME: str = os.getenv("SAGEMAKER_ENDPOINT_NAME")
    PAYLOAD_TEMPLATE: dict = {
        "prompt":"",
        "max_length": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }

settings = Settings()