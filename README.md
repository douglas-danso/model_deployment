**Deploy Hugging Face Model to AWS SageMaker**

**Overview**
This project demonstrates how to deploy a Hugging Face model (*distilgpt2*) to AWS SageMaker. It includes scripts for:
- Deploying the model to SageMaker
- Invoking the deployed endpoint with a prompt
- Setting up AWS resources
- Cleaning up the deployment

**Project Structure**
```
.
├── .env
├── app
│   ├── __init__.py
│   ├── config.py         # Contains configuration settings including AWS credentials and endpoint details
│   └── test_script.py    # Invokes the SageMaker endpoint and prints the generated text
├── deploy
│   ├── __init__.py
│   ├── aws_setup.py      # Sets up necessary AWS resources
│   ├── clean_up.py       # Deletes the SageMaker endpoint
│   └── sagemaker_deploy.py  # Deploys the Hugging Face model to SageMaker
├── model
│   └── code
│       ├── inference.py  # Contains inference logic: loading the model, making predictions, handling I/O
│       └── requirements.txt
└── requirements.txt      # Contains all project dependencies
```

**Prerequisites**
- Python 3.8 or above
- An AWS account with the necessary permissions

**Setup**

1. **Create and Activate a Virtual Environment**
   It is recommended to use a virtual environment to manage dependencies.

   Using Python's built-in `venv`:
   ```sh
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

   Alternatively, using `virtualenv`:
   ```sh
   pip install virtualenv
   virtualenv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   With your virtual environment activated, install the project dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory with the following content:
   ```
   AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
   AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>
   REGION=us-west-2
   SAGEMAKER_ENDPOINT_NAME=distilgpt2
   MODEL_ID=distilbert/distilgpt2
   ```

**Usage**

**Deploying the Model**
Deploy the Hugging Face model to AWS SageMaker by running:
```sh
python deploy/sagemaker_deploy.py
```

**Cleaning Up Resources**
After you are done, clean up by deleting the SageMaker endpoint:
```sh
python deploy/clean_up.py
```

**Additional Information**
- **Configuration:** You can adjust AWS credentials and endpoint settings in `app/config.py`.
- **Model Inference:** The `inference.py` file (located in `model/code/`) handles model loading, prediction, and input/output processes.
