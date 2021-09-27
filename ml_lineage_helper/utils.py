import sys
import os
import subprocess
import numpy as np
import sagemaker
import boto3
from botocore.config import Config


class StatusIndicator:
    def __init__(self):
        self.previous_status = None
        self.need_newline = False

    def update(self, status):
        if self.previous_status != status:
            if self.need_newline:
                sys.stdout.write("\n")
            sys.stdout.write(status + " ")
            self.need_newline = True
            self.previous_status = status
        else:
            sys.stdout.write(".")
            self.need_newline = True
        sys.stdout.flush()

    def end(self):
        if self.need_newline:
            sys.stdout.write("\n")


class SageMakerSession:
    """Custom SageMakerSession class with sensible default properties"""

    # Default constructor
    def __init__(
        self,
        bucket_name=None,
        region="us-east-1",
        role_name=None,
        aws_profile_name="default",
    ):
        self.bucket_name = bucket_name
        self.region = region
        self.role_name = role_name
        self.aws_profile_name = aws_profile_name

        self.get_sagemaker_session()

    def get_sagemaker_session(self):
        try:
            # You're using a SageMaker notebook
            self.role_arn = sagemaker.get_execution_role()
            self.session = sagemaker.Session()
            self.session.config = Config(
                connect_timeout=5, read_timeout=60, retries={"max_attempts": 20}
            )
            self.bucket_name = self.session.default_bucket()
            self.bucket_s3_uri = f"s3://{self.bucket_name}"
            self.region = self.session.boto_region_name
        except ValueError:
            # You're using a notebook somewhere else
            print("Setting role and SageMaker session manually...")

            iam = boto3.client("iam", region_name=self.region)
            sagemaker_client = boto3.client(
                "sagemaker",
                region_name=self.region,
                config=Config(
                    connect_timeout=5, read_timeout=60, retries={"max_attempts": 20}
                ),
            )

            self.role_arn = iam.get_role(RoleName=self.role_name)["Role"]["Arn"]
            boto3.setup_default_session(
                region_name=self.region, profile_name=self.aws_profile_name
            )
            self.session = sagemaker.Session(
                sagemaker_client=sagemaker_client, default_bucket=self.bucket_name
            )
            self.bucket_s3_uri = f"s3://{self.bucket_name}"


def upload_df_to_s3(df, s3_uri, sagemaker_session, csv=True, header=True):
    """Save a Pandas DataFrame as CSV and upload to S3

    Args:
        df (pandas.DataFrame): Pandas DataFrame
        s3_uri (str): S3 URI of where you want the CSV to be stored
        sagemaker_session (SageMakerSession): Custom SageMakerSession
        csv (bool): If false, DataFrame will be written as numpy file
        header (bool): If false, the header of the dataframe will not be written
    """
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    s3_client = boto3.client("s3", region_name=sagemaker_session.region)
    s3_uri_split = s3_uri.split("/")
    file_name = s3_uri_split[-1]
    bucket = s3_uri_split[2]
    prefix = ("/").join(s3_uri_split[3:-1])
    if csv:
        df.to_csv(f"./data/{file_name}", index=False)
    else:
        np.save(f"./data/{file_name}", df.to_numpy())
    s3_client.upload_file(
        Filename=f"data/{file_name}", Bucket=bucket, Key=f"{prefix}/{file_name}"
    )
    print(f"Uploaded {file_name} to {s3_uri}.")


def get_repo_link(cwd: str, entry_point_script_path: str, processing_code=True):
    """Construct git url of the processing or training code

    Args:
        cwd (str): Current working directory (e.g. os.cwd())
        entry_point_script_path (str): This is relative to your cwd (e.g. code/processing.py)
        processing_code (bool): (If True, repo link will be added to processing code artifact propert, else will be added to training code artifact property)

    Returns:
        repo_link (str): The git url of the processing or training code
    """

    result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    output = result.stdout

    if "git@ssh" in output:
        git_https = (
            output.split("\n")[0]
            .split("\t")[1][:-8]
            .replace(":", "/")
            .replace("git@ssh.", "https://")
            .split(".git")[0]
        )

    elif "git@" in output:
        git_https = (
            output.split("\n")[0]
            .split("\t")[1][:-8]
            .replace(":", "/")
            .replace("git@", "https://")
            .split(".git")[0]
        )

    else:
        git_https = output.split("\n")[0].split("\t")[1][:-8].split(".git")[0]

    repo_name = git_https.split("/")[-1]

    result = subprocess.run(["git", "branch"], capture_output=True, text=True)
    output = result.stdout
    branch = output.strip()[2:]

    cwd_list = cwd.split("/")
    repo_name_index = cwd_list.index(repo_name)
    relative_path = "/".join(cwd_list[repo_name_index + 1 :])
    repo_link = f"{git_https}/blob/{branch}/{relative_path}/{entry_point_script_path}"
    if processing_code:
        return ("processing_code", repo_link)
    return ("training_code", repo_link)
