import os
import boto3
import json

def create_iam_role():
    iam = boto3.client('iam', region_name=os.getenv('REGION', 'us-east-1'))
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        response = iam.create_role(
            RoleName='OllamaSageMakerRole',
            AssumeRolePolicyDocument=json.dumps(trust_policy)
        )
        
        iam.attach_role_policy(
            RoleName='OllamaSageMakerRole',
            PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
        )
        
        return response['Role']['Arn']
    except iam.exceptions.EntityAlreadyExistsException:
        response = iam.get_role(RoleName='OllamaSageMakerRole')
        return response['Role']['Arn']