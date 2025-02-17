import boto3
import json
from config import settings

def invoke_sagemaker_endpoint(endpoint_name, prompt):
    """
    Invoke the SageMaker endpoint with a given prompt and return the generated text.
    """
    # Create a SageMaker runtime client
    client = boto3.client('sagemaker-runtime',region_name=settings.AWS_REGION)
    
    # Define the payload to send to the endpoint.
    payload_dict = dict(settings.PAYLOAD_TEMPLATE)
    payload_dict["prompt"] = prompt
    
    # Call the endpoint
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload_dict)
    )
    
    # Read and decode the response
    response_body = response['Body'].read().decode("utf-8")
    result = json.loads(response_body)
    return result

if __name__ == "__main__":
    # Set the name of your deployed endpoint; adjust if necessary.
    endpoint_name = "distilgpt2"
    
    # Define your test prompttest_
    test_prompt = "hello Clara, how are you today?"
    
    # Invoke the endpoint
    result = invoke_sagemaker_endpoint(endpoint_name, test_prompt)
    
    # Print the generated text
    print("Generated text:")
    print(result)
