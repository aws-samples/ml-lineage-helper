from setuptools import setup

setup(
    name="ml-lineage-helper",
    version="0.1",
    description="A wrapper around SageMaker ML Lineage Tracking extending ML Lineage to end-to-end ML lifecycles, including additional capabilities around Feature Store groups, queries, and other relevant artifacts.",
    url="https://github.com/aws-samples/ml-lineage-helper",
    author="Bobby Lindsey",
    author_email="bwlind@amazon.com",
    license="Apache-2.0",
    packages=["ml_lineage_helper"],
    install_requires=[
        "numpy",
        "boto3>=1.17.74",
        "sagemaker>2.49.1",
        "pandas",
        "networkx",
        "matplotlib",
        "numpy",
    ],
)
