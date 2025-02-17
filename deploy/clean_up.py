import boto3

def cleanup():
    sage = boto3.client('sagemaker', region_name='us-east-1')
    sage.delete_endpoint(EndpointName='opt-125m-quantized')

if __name__ == "__main__":
    cleanup()