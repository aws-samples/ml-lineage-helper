# https://docs.aws.amazon.com/sagemaker/latest/dg/lineage-tracking-entities.html
# https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-lineage/sagemaker-lineage.ipynb

import base64
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.context import Context
from sagemaker.lineage.action import Action
from sagemaker.lineage.association import Association
from typing import Optional, Iterator
from sagemaker.lineage._api_types import AssociationSummary
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import boto3
from .utils import SageMakerSession


class MLLineageHelper:
    def __init__(
        self, sagemaker_session=None, sagemaker_model_name_or_model_s3_uri=None
    ):
        """Default constructor

        Args:
            sagemaker_model_name_or_model_s3_uri (str): SageMaker model name or S3 URI of model package
            sagemaker_session (SageMakerSession): SageMakerSession object
        """
        if sagemaker_session:
            self.sagemaker_session = sagemaker_session
        else:
            self.sagemaker_session = SageMakerSession()
        self.sagemaker_model_name_or_model_s3_uri = sagemaker_model_name_or_model_s3_uri
        self.df = None
        if self.sagemaker_model_name_or_model_s3_uri:
            self.df = self.get_ml_lineage()

    def get_associations(
        self, src_arn: Optional[str] = None, dest_arn: Optional[str] = None
    ) -> Iterator[AssociationSummary]:
        """Given an arn retrieve all associated lineage entities.
        The arn must be one of: experiment, trial, trial component, artifact, action, or context.
        Args:
            src_arn (str, optional): The arn of the source. Defaults to None.
            dest_arn (str, optional): The arn of the destination. Defaults to None.
        Returns:
            array: An array of associations that are either incoming or outgoing from the lineage
            entity of interest.
        """
        if src_arn:
            associations: Iterator[AssociationSummary] = Association.list(
                source_arn=src_arn, sagemaker_session=self.sagemaker_session.session
            )
        else:
            associations: Iterator[AssociationSummary] = Association.list(
                destination_arn=dest_arn,
                sagemaker_session=self.sagemaker_session.session,
            )
        return associations

    def get_ml_lineage(self):
        """Crawl up the lineage graph"""
        # If input is model name, find the model package uri in S3
        if not self.sagemaker_model_name_or_model_s3_uri.endswith(".tar.gz"):
            sagemaker_client = boto3.client(
                "sagemaker", region_name=self.sagemaker_session.region
            )
            s3_uri_of_model = sagemaker_client.describe_model(
                ModelName=self.sagemaker_model_name_or_model_s3_uri
            )["PrimaryContainer"]["ModelDataUrl"]
        # Otherwise the input was already the S3 uri of the model package
        else:
            s3_uri_of_model = self.sagemaker_model_name_or_model_s3_uri

        # Start with the model artifact (the end of the DAG)
        model_summary = list(
            Artifact.list(
                source_uri=s3_uri_of_model,
                max_results=1,
                sagemaker_session=self.sagemaker_session.session,
            )
        )[0]
        model_artifact_arn = model_summary.artifact_arn

        # Start working upstream in the DAG
        upstream_associations = list(
            self.get_associations(dest_arn=model_artifact_arn)
        )  # Training job
        second_order_upstream_associations = list(
            self.get_associations(dest_arn=upstream_associations[0].source_arn)
        )
        upstream_associations.extend(second_order_upstream_associations)

        # Find the TrainingData artifact so we can fetch the SageMaker Processing artifact if it exists
        for association in second_order_upstream_associations:
            source_name = association.source_name
            if not source_name:
                source_name = Artifact.load(association.source_arn).artifact_name
            if source_name == "TrainingData" or source_name == "TestingData":
                training_data_artifact_summary = association
                third_order_upstream_associations = list(
                    self.get_associations(
                        dest_arn=training_data_artifact_summary.source_arn
                    )
                )
                upstream_associations.extend(third_order_upstream_associations)
                fourth_order_upstream_associations = []
                for each in third_order_upstream_associations:
                    fourth_order_upstream_associations.extend(
                        list(
                            self.get_associations(
                                dest_arn=third_order_upstream_associations[0].source_arn
                            )
                        )
                    )
                upstream_associations.extend(fourth_order_upstream_associations)

                try:
                    # Get artifacts that fed into the SM processing job
                    fifth_order_upstream_associations = list(
                        self.get_associations(
                            dest_arn=fourth_order_upstream_associations[0].source_arn
                        )
                    )
                    # Get feature groups produced by SM processing job
                    fifth_order_upstream_associations.extend(
                        list(
                            self.get_associations(
                                src_arn=fourth_order_upstream_associations[0].source_arn
                            )
                        )
                    )
                    upstream_associations.extend(fifth_order_upstream_associations)
                except:
                    pass
        data = []
        for association in upstream_associations:
            name_source = association.source_name
            association_type = association.association_type
            name_destination = association.destination_name
            artifact_source_arn = association.source_arn
            if not name_source:
                name_source = Artifact.load(artifact_source_arn).artifact_name
            artifact_destination_arn = association.destination_arn
            if not name_destination:
                name_destination = Artifact.load(artifact_destination_arn).artifact_name
            try:
                source_uri = Artifact.load(association.source_arn).source.source_uri
            except:
                source_uri = None
            if name_source == "TrainingData":
                # Set properties
                artifact = Artifact.load(training_data_artifact_summary.source_arn)
                try:
                    base64_feature_store_query_string = artifact.properties[
                        "Base64FeatureStoreQueryString"
                    ]
                except:
                    pass
            else:
                base64_feature_store_query_string = None

            git_url = None
            if name_source == "TrainingCode" or name_source == "ProcessingCode":
                # Set properties
                artifact = Artifact.load(artifact_source_arn)
                try:
                    git_url = artifact.properties["GitURL"]
                except:
                    pass

            if name_source:
                data.append(
                    [
                        name_source,
                        association_type,
                        name_destination,
                        artifact_source_arn,
                        artifact_destination_arn,
                        source_uri,
                        base64_feature_store_query_string,
                        git_url,
                    ]
                )
        df = pd.DataFrame(
            data=data,
            columns=[
                "Name/Source",
                "Association",
                "Name/Destination",
                "Artifact Source ARN",
                "Artifact Destination ARN",
                "Source URI",
                "Base64 Feature Store Query String",
                "Git URL",
            ],
        )
        df.drop_duplicates(ignore_index=True, inplace=True)
        self.df = df
        return df

    def graph(self):
        """Visually represent the lineage DAG"""
        plt.figure(3, figsize=(16, 14))
        graph = nx.DiGraph()
        graph.add_edges_from([(each[0], each[2]) for each in self.df.values])
        nx.draw_networkx(
            graph,
            node_size=800,
            node_color="orange",
            alpha=0.9,
            font_size=14,
            pos=nx.spring_layout(graph),
        )
        plt.show()

    def get_artifact_if_exists(self, artifact_source_uri, artifact_name, properties):
        """Check if artifact already exists

        Args:
            artifact_source_uri (str): Source URI of the artifact
            artifact_name (str): Name of the artifact

        Returns:
            sagemaker.lineage.Artifact or None

        """
        # Check if artifact_source_uri already exists
        try:
            artifact_list = list(
                Artifact.list(
                    source_uri=artifact_source_uri,
                    max_results=1,
                    sagemaker_session=self.sagemaker_session.session,
                )
            )
        except Exception as e:
            artifact_list = []
        if artifact_list:
            artifact_summary = artifact_list[0]
            print(
                f"Using existing artifact, {artifact_name}: {artifact_summary.artifact_arn}\n"
            )
            try:
                artifact_properties = artifact_summary.properties
            except:
                artifact_properties = properties
            if properties or artifact_properties:
                artifact = Artifact(
                    artifact_arn=artifact_summary.artifact_arn,
                    artifact_name=artifact_name,
                    source=artifact_summary.source,
                    artifact_type=artifact_summary.artifact_type,
                    properties=properties,
                    sagemaker_session=self.sagemaker_session.session,
                )
                artifact.save()
            else:
                artifact = Artifact(
                    artifact_arn=artifact_summary.artifact_arn,
                    artifact_name=artifact_name,
                    source=artifact_summary.source,
                    artifact_type=artifact_summary.artifact_type,
                    sagemaker_session=self.sagemaker_session.session,
                )
                artifact.save()
            return artifact
        else:
            return None

    def create_artifact(
        self,
        artifact_name,
        artifact_source_uri,
        artifact_type,
        properties=None,
    ):
        """Check if artifact already exists

        Args:
            artifact_name (str): Name of the artifact
            artifact_source_uri (str): Source URI of the artifact
            artifact_type (str): Type of the artiface ('Code', 'DataSet', etc...)
            properties (dict): Optional custom properties for the artifact

        Returns:
            sagemaker.lineage.Artifact

        """
        artifact = self.get_artifact_if_exists(
            artifact_source_uri, artifact_name, properties
        )
        if not artifact:
            artifact = Artifact.create(
                artifact_name=artifact_name,
                source_uri=artifact_source_uri,
                artifact_type=artifact_type,
                properties=properties,
                sagemaker_session=self.sagemaker_session.session,
            )
            # artifact.save()
            print(f"Created {artifact_name} artifact: {artifact.artifact_arn}\n")
        return artifact

    def associate_artifacts(
        self, artifacts, association_type, artifact_destination_arn
    ):
        """Asssociate two artifacts together

        Args:
            artifacts (list): List of Artifacts
            association_type (str): Type of association ('Produced', 'ContributedTo', etc...)
            artifact_destination_arn (str): Destination Artifact ARN
            sagemaker_session (SageMakerSession): SageMakerSession object

        Returns:
            None

        """
        if type(artifacts) != list:
            artifacts = [artifacts]
        for artifact in artifacts:
            if artifact == None:
                continue
            if type(artifact) == Artifact:
                source_arn = artifact.artifact_arn
            else:
                source_arn = artifact
            try:
                Association.create(
                    source_arn=source_arn,
                    destination_arn=artifact_destination_arn,
                    association_type=association_type,
                    sagemaker_session=self.sagemaker_session.session,
                )
                print(
                    f"Associated {artifact.artifact_arn} and {artifact_destination_arn} with association {association_type}\n"
                )
            except:
                print(
                    f"Association already exists between {source_arn} and {artifact_destination_arn}\n"
                )

    def create_training_job_artifacts(self, training_job_info, repo_links):
        """Create training job artifacts

        Args:
            training_job_info (dict): Training job information
            repo_links (list(tuples)): Repo links for processing and training code

        Returns:
            tuple of SageMaker Artifacts: code, training_data, and testing_data artifacts
        """
        # Create training job artifacts
        try:
            code_s3_uri = training_job_info["HyperParameters"][
                "sagemaker_submit_directory"
            ][1:-1]
        except:
            code_s3_uri = None
        training_data_s3_uri = training_job_info["InputDataConfig"][0]["DataSource"][
            "S3DataSource"
        ]["S3Uri"]
        testing_data_s3_uri = training_job_info["InputDataConfig"][1]["DataSource"][
            "S3DataSource"
        ]["S3Uri"]

        training_data_artifact = self.create_artifact(
            "TrainingData", training_data_s3_uri, "DataSet"
        )
        testing_data_artifact = self.create_artifact(
            "TestingData", testing_data_s3_uri, "DataSet"
        )
        if code_s3_uri:
            code_artifact = self.create_artifact("TrainingCode", code_s3_uri, "Code")
        else:
            code_artifact = None
        # Add git URL to code artifact
        if code_s3_uri and repo_links:
            for repo_link in repo_links:
                if repo_link[0] == "training_code":
                    code_artifact.properties = {"GitURL": repo_link[1]}
                    code_artifact.save()
        return code_artifact, training_data_artifact, testing_data_artifact

    def create_model_artifact(self, model_name, training_job_info):
        """Create a model artifact

        Args:
            model_name (str): Name of the SageMaker model
            training_job_info (dict): Training job information

        Returns:
            sagemaker.lineage.artifact.Artifact: Model artifact
        """
        if model_name:
            sagemaker_client = boto3.client(
                "sagemaker", region_name=self.sagemaker_session.region
            )
            trained_model_s3_uri = sagemaker_client.describe_model(
                ModelName=model_name
            )["PrimaryContainer"]["ModelDataUrl"]
        else:
            trained_model_s3_uri = training_job_info["ModelArtifacts"][
                "S3ModelArtifacts"
            ]
        # SageMaker automatically creates a model artifact so
        # we'll get an existing artifact back but it won't have
        # our custom attributes
        model_artifact = self.create_artifact(
            "Model",
            trained_model_s3_uri,
            "Model",
            properties={"SageMakerModelName": model_name},
        )
        # So set the custom attributes
        if model_artifact:
            model_artifact.artifact_name = "Model"
            model_artifact.source_uri = trained_model_s3_uri
            model_artifact.artifact_type = "Model"
            model_artifact.properties = {"SageMakerModelName": model_name}
            model_artifact.save()
        return model_artifact

    def create_sm_processing_job_artifacts(
        self, sagemaker_processing_job_description, repo_links
    ):
        """Create SageMaker Processing job artifacts

        Args:
            sagemaker_processing_job_description (dict): SageMaker Processing job information
            repo_links (list(tuples)): Repo links for processing and training code

        Returns:
            tuple of sagemaker.lineage.artifact.Artfiact: processing code, raw data, and processing job artifacts
        """
        try:
            processing_script_uri = sagemaker_processing_job_description[
                "ProcessingInputs"
            ][1]["S3Input"]["S3Uri"]
            processing_input_data_uri = sagemaker_processing_job_description[
                "ProcessingInputs"
            ][0]["S3Input"]["S3Uri"]
        except:
            processing_script_uri = sagemaker_processing_job_description[
                "ProcessingInputs"
            ][0]["S3Input"]["S3Uri"]
            processing_input_data_uri = sagemaker_processing_job_description[
                "AppSpecification"
            ]["ContainerArguments"][3]
        processing_job_arn = sagemaker_processing_job_description["ProcessingJobArn"]

        sagemaker_processing_code_artifact = self.create_artifact(
            "ProcessingCode", processing_script_uri, "Code"
        )
        # Add git URL to code artifact
        if repo_links:
            for repo_link in repo_links:
                if repo_link[0] == "processing_code":
                    sagemaker_processing_code_artifact.properties = {
                        "GitURL": repo_link[1]
                    }
                    sagemaker_processing_code_artifact.save()

        sagemaker_processing_input_data_artifact = self.create_artifact(
            "ProcessingInputData", processing_input_data_uri, "DataSet"
        )

        sagemaker_processing_job_artifact = self.create_artifact(
            "ProcessingJob", processing_job_arn, "ProcessingJob"
        )

        return (
            sagemaker_processing_code_artifact,
            sagemaker_processing_input_data_artifact,
            sagemaker_processing_job_artifact,
        )

    def create_ml_lineage(
        self,
        estimator_or_training_job_name,
        model_name=None,
        query=None,
        feature_group_names: list = None,
        sagemaker_processing_job_description=None,
        repo_links: list = None,
    ):
        """Do ML Lineage Tracking

        Args:
            estimator_or_training_job_name (sagemaker.Estimator or str): An estimator object (object you get after training a model) or the training job name
            model_name (str): Name of the model in SageMaker (SageMaker console --> Inference --> Models)
            query (str): The query you used to query the Feature Store and get your training and test data
            feature_group_names (list): An optional list of Feature Group names
            sagemaker_processing_job_description (dict): The job description of your SageMaker Processing job
            repo_links ([Tuples]): List of tuples containing repo links of processing and training code

        Returns:
            MLLineage
        """

        sagemaker_client = boto3.client(
            "sagemaker", region_name=self.sagemaker_session.region
        )
        if type(estimator_or_training_job_name) != str:
            training_job_info = (
                estimator_or_training_job_name.latest_training_job.describe()
            )
        else:
            training_job_info = sagemaker_client.describe_training_job(
                TrainingJobName=estimator_or_training_job_name
            )
        (
            code_artifact,
            training_data_artifact,
            testing_data_artifact,
        ) = self.create_training_job_artifacts(training_job_info, repo_links)

        model_artifact = self.create_model_artifact(model_name, training_job_info)

        input_artifacts = [training_data_artifact, testing_data_artifact, code_artifact]

        # Create Feature Store artifacts and query
        if query:
            # Base64 encode query string
            base64_encoded_query_bytes = base64.b64encode(query.encode("utf-8"))
            base64_encoded_query_string = str(base64_encoded_query_bytes, "utf-8")
            # Cut off
            if len(base64_encoded_query_string) > 256:
                base64_encoded_query_string = base64_encoded_query_string[0:256]

            training_data_artifact.properties = {
                "Base64FeatureStoreQueryString": base64_encoded_query_string
            }
            training_data_artifact.save()
            testing_data_artifact.properties = {
                "Base64FeatureStoreQueryString": base64_encoded_query_string
            }
            testing_data_artifact.save()

        if feature_group_names:
            feature_group_artifacts = []
            for feature_group_name in feature_group_names:
                feature_group_arn = sagemaker_client.describe_feature_group(
                    FeatureGroupName=feature_group_name
                )["FeatureGroupArn"]
                temp_artifact = self.create_artifact(
                    f"fg-{feature_group_name}", feature_group_arn, "FeatureGroup"
                )
                feature_group_artifacts.append(temp_artifact)

        # Create SageMaker Processing job artifacts
        if sagemaker_processing_job_description:
            sm_processing_job_artifacts = self.create_sm_processing_job_artifacts(
                sagemaker_processing_job_description, repo_links
            )
            sagemaker_processing_code_artifact = sm_processing_job_artifacts[0]
            sagemaker_processing_input_data_artifact = sm_processing_job_artifacts[1]
            sagemaker_processing_job_artifact = sm_processing_job_artifacts[2]

        # Get trial component which will contain metrics from the training job
        training_job_name = training_job_info["TrainingJobName"]
        trial_component = sagemaker_client.describe_trial_component(
            TrialComponentName=f"{training_job_name}-aws-training-job"
        )
        trial_component_arn = trial_component["TrialComponentArn"]

        # Associate input artifacts
        self.associate_artifacts(input_artifacts, "ContributedTo", trial_component_arn)
        self.associate_artifacts(
            trial_component_arn, "Produced", model_artifact.artifact_arn
        )

        # Associate Feature Group artifacts to training data artifact
        if feature_group_names and feature_group_artifacts:
            self.associate_artifacts(
                feature_group_artifacts,
                "ContributedTo",
                training_data_artifact.artifact_arn,
            )
            self.associate_artifacts(
                feature_group_artifacts,
                "ContributedTo",
                testing_data_artifact.artifact_arn,
            )

        # Associate SageMaker Processing job artifacts
        if sagemaker_processing_job_description:
            self.associate_artifacts(
                [
                    sagemaker_processing_code_artifact,
                    sagemaker_processing_input_data_artifact,
                ],
                "ContributedTo",
                sagemaker_processing_job_artifact.artifact_arn,
            )
            if feature_group_names and feature_group_artifacts:
                for feature_group_artifact in feature_group_artifacts:
                    self.associate_artifacts(
                        sagemaker_processing_job_artifact,
                        "ContributedTo",
                        feature_group_artifact.artifact_arn,
                    )
            else:  # Training data generated by processing job, not Feature Store data
                self.associate_artifacts(
                    sagemaker_processing_job_artifact,
                    "Produced",
                    training_data_artifact.artifact_arn,
                )

        self.sagemaker_model_name_or_model_s3_uri = model_name
        return self.get_ml_lineage()

    def delete_associations(self, arn):
        """Delete an artifact's associations.

        Args:
            arn (str): Artifact ARN
        """
        # Delete incoming associations
        incoming_associations = Association.list(destination_arn=arn)
        for summary in incoming_associations:
            assct = Association(
                source_arn=summary.source_arn,
                destination_arn=summary.destination_arn,
                sagemaker_session=self.sagemaker_session.session,
            )
            assct.delete()

        # Delete outgoing associations
        outgoing_associations = Association.list(source_arn=arn)
        for summary in outgoing_associations:
            assct = Association(
                source_arn=summary.source_arn,
                destination_arn=summary.destination_arn,
                sagemaker_session=self.sagemaker_session.session,
            )
            assct.delete()

    def delete_lineage_data(self):
        """Delete ALL lineage data in your account.
        """
        for summary in Context.list():
            print(f"Deleting context {summary.context_name}")
            self.delete_associations(summary.context_arn)
            ctx = Context(context_name=summary.context_name, sagemaker_session=self.sagemaker_session.session)
            ctx.delete()

        for summary in Action.list():
            print(f"Deleting action {summary.action_name}")
            self.delete_associations(summary.action_arn)
            actn = Action(action_name=summary.action_name, sagemaker_session=self.sagemaker_session.session)
            actn.delete()

        for summary in Artifact.list():
            print(f"Deleting artifact {summary.artifact_arn} {summary.artifact_name}")
            self.delete_associations(summary.artifact_arn)
            artfct = Artifact(
                artifact_arn=summary.artifact_arn,
                sagemaker_session=self.sagemaker_session.session,
            )
            artfct.delete(disassociate=True)
