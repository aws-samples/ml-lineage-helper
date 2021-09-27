"""
Query SageMaker ML Lineage
"""

from collections import deque
from sagemaker.lineage.artifact import Artifact
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
import boto3
from ..ml_lineage import MLLineageHelper
from ..utils import SageMakerSession


class QueryLineage:
    def __init__(self, sagemaker_session=None):
        """Default constructor

        Args:
            sagemaker_session (SageMakerSession): SageMakerSession object
        """
        if sagemaker_session:
            self.sagemaker_session = sagemaker_session
        else:
            self.sagemaker_session = SageMakerSession()
        self.ml_lineage_tracking = MLLineageHelper(self.sagemaker_session)

    def get_data_sources_from_feature_group(
        self, artifact_or_fg_arn: str, max_depth: int = 5
    ):
        """
        Given a Feature Group, find all associated data sources.

        Args:
            artifact_or_fg_arn (str): Artifact or Feature Group ARN

        Returns:
            data_source_df (pd.DataFrame): DataFrame containing data sources
        """
        if "artifact" in artifact_or_fg_arn:
            artifact_arn = artifact_or_fg_arn
        else:
            artifact_list = list(
                Artifact.list(
                    source_uri=artifact_or_fg_arn,
                    max_results=1,
                    sagemaker_session=self.sagemaker_session.session,
                )
            )
            artifact_arn = artifact_list[0].artifact_arn

        queue = deque()
        discovered = []
        queue.append(artifact_arn)
        discovered.append(artifact_arn)
        dataset_artifact_arns = []
        current_depth = 0
        # While queue isn't empty
        try:
            while queue:
                if current_depth > max_depth:
                    break
                node_name = queue.popleft()  # Dequeue
                neighbors = list(
                    self.ml_lineage_tracking.get_associations(src_arn=node_name)
                )
                neighbors.extend(
                    list(self.ml_lineage_tracking.get_associations(dest_arn=node_name))
                )
                for neighbor in neighbors:
                    if neighbor.source_arn not in discovered:
                        if neighbor.source_type == "DataSet":
                            dataset_artifact_arns.append(neighbor.source_arn)
                        queue.append(neighbor.source_arn)
                        discovered.append(neighbor.source_arn)

                    if neighbor.destination_arn not in discovered:
                        if neighbor.destination_type == "DataSet":
                            dataset_artifact_arns.append(neighbor.destination_arn)
                        queue.append(neighbor.destination_arn)
                        discovered.append(neighbor.destination_arn)
                current_depth += 1
        except Exception as e:  # throttling error
            pass

        dataset_names = []
        dataset_s3_uris = []
        for dataset_artifact_arn in dataset_artifact_arns:
            dataset_artifact = Artifact.load(dataset_artifact_arn)
            dataset_s3_uris.append(dataset_artifact.source.source_uri)
            dataset_names.append(dataset_artifact.artifact_name)
        data_source_df = pd.DataFrame(
            {
                "Datset Name": dataset_names,
                "Dataset Artifact ARN": dataset_artifact_arns,
                "Dataset S3 URI": dataset_s3_uris,
            }
        )
        return data_source_df

    def get_feature_groups_from_model(
        self, artifact_arn_or_model_name: str, max_depth: int = 5
    ):
        """
        Given a Feature Group, find all associated models.

        Args:
            artifact_arn_or_model_name (str): Artifact ARN or SageMaker model name

        Returns:
            feature_groups_df (pd.DataFrame): DataFrame containing models
        """

        if "artifact" in artifact_arn_or_model_name:
            artifact_arn = artifact_arn_or_model_name
        else:
            sagemaker_client = boto3.client(
                "sagemaker", region_name=self.sagemaker_session.region
            )
            model_data_s3_uri = sagemaker_client.describe_model(
                ModelName=artifact_arn_or_model_name
            )["PrimaryContainer"]["ModelDataUrl"]
            artifact_list = list(
                Artifact.list(
                    source_uri=model_data_s3_uri,
                    max_results=1,
                    sagemaker_session=self.sagemaker_session.session,
                )
            )
            artifact_arn = artifact_list[0].artifact_arn

        queue = deque()
        discovered = []
        queue.append(artifact_arn)
        discovered.append(artifact_arn)
        feature_group_artifact_arns = []
        current_depth = 0
        # While queue isn't empty
        while queue:
            if current_depth > max_depth:
                break
            node_name = queue.popleft()  # Dequeue
            neighbors = list(
                self.ml_lineage_tracking.get_associations(dest_arn=node_name)
            )
            for neighbor in neighbors:
                if neighbor.source_arn not in discovered:
                    if neighbor.source_type == "FeatureGroup":
                        feature_group_artifact_arns.append(neighbor.source_arn)
                    queue.append(neighbor.source_arn)
                    discovered.append(neighbor.source_arn)
            current_depth += 1

        feature_group_names = []
        feature_group_arns = []
        feature_group_s3_uris = []
        feature_group_table_names = []
        for feature_group_artifact_arn in feature_group_artifact_arns:
            feature_group_artifact = Artifact.load(feature_group_artifact_arn)
            feature_group_name = feature_group_artifact.artifact_name
            feature_group_names.append(feature_group_name)
            feature_group_arn = Artifact.load(
                feature_group_artifact_arn
            ).source.source_uri
            feature_group_arns.append(feature_group_arn)

            # Get Feature Group info
            fg = FeatureGroup(
                feature_group_name.replace("fg-", ""),
                sagemaker_session=self.sagemaker_session.session,
            )
            fg_info = fg.describe()

            feature_group_s3_uri = fg_info["OfflineStoreConfig"]["S3StorageConfig"][
                "S3Uri"
            ]
            feature_group_s3_uris.append(feature_group_s3_uri)

            feature_group_table_name = fg_info["OfflineStoreConfig"][
                "DataCatalogConfig"
            ]["TableName"]
            feature_group_table_names.append(feature_group_table_name)
        feature_group_df = pd.DataFrame(
            {
                "Feature Group Name": feature_group_names,
                "Feature Group Artifact ARN": feature_group_artifact_arns,
                "Feature Group ARN": feature_group_arns,
                "Feature Group S3 URI": feature_group_s3_uris,
                "Feature Group Table Name": feature_group_table_names,
            }
        )
        return feature_group_df

    def get_models_from_feature_group(
        self, artifact_or_fg_arn: str, max_depth: int = 5
    ):
        """
        Given a Feature Group, find all associated models.

        Args:
            artifact_or_fg_arn (str): Artifact or Feature Group ARN

        Returns:
            models_df (pd.DataFrame): DataFrame containing models
        """

        if "artifact" in artifact_or_fg_arn:
            artifact_arn = artifact_or_fg_arn
        else:
            artifact_list = list(
                Artifact.list(
                    source_uri=artifact_or_fg_arn,
                    max_results=1,
                    sagemaker_session=self.sagemaker_session.session,
                )
            )
            artifact_arn = artifact_list[0].artifact_arn

        queue = deque()
        discovered = []
        queue.append(artifact_arn)
        discovered.append(artifact_arn)
        model_artifact_arns = []
        current_depth = 0
        # While queue isn't empty
        while queue:
            if current_depth > max_depth:
                break
            node_name = queue.popleft()  # Dequeue
            neighbors = list(
                self.ml_lineage_tracking.get_associations(src_arn=node_name)
            )
            for neighbor in neighbors:
                if neighbor.destination_arn not in discovered:
                    if neighbor.destination_type == "Model":
                        model_artifact_arns.append(neighbor.destination_arn)
                    queue.append(neighbor.destination_arn)
                    discovered.append(neighbor.destination_arn)
            current_depth += 1

        model_names = []
        model_s3_uris = []
        created_bys = []
        for model_artifact_arn in model_artifact_arns:
            artifact = Artifact.load(model_artifact_arn)
            try:
                model_names.append(artifact.properties["SageMakerModelName"])
            except:
                model_names.append(None)
            model_s3_uris.append(artifact.source.source_uri)
            try:
                created_bys.append(artifact.created_by["UserProfileName"])
            except:
                created_bys.append(None)
        models_df = pd.DataFrame(
            {
                "SageMaker Model Name": model_names,
                "Model S3 URI": model_s3_uris,
                "Created By": created_bys,
            }
        )
        return models_df

    def get_feature_groups_from_data_source(
        self, artifact_arn_or_s3_uri: str, max_depth: int = 5
    ):
        """
        Given a data source, find all associated Feature Groups.

        Args:
            artifact_arn_or_s3_uri (str): Artifact ARN or S3 URI of data source

        Returns:
            feature_group_df (pd.DataFrame): DataFrame containing Feature Groups
        """
        if "artifact" in artifact_arn_or_s3_uri:
            artifact_arn = artifact_arn_or_s3_uri
        else:
            artifact_list = list(
                Artifact.list(
                    source_uri=artifact_arn_or_s3_uri,
                    max_results=1,
                    sagemaker_session=self.sagemaker_session.session,
                )
            )
            artifact_arn = artifact_list[0].artifact_arn

        queue = deque()
        discovered = []
        queue.append(artifact_arn)
        discovered.append(artifact_arn)
        feature_group_artifact_arns = []
        current_depth = 0
        # While queue isn't empty
        try:
            while queue:
                if current_depth > max_depth:
                    break
                node_name = queue.popleft()  # Dequeue
                neighbors = list(
                    self.ml_lineage_tracking.get_associations(src_arn=node_name)
                )
                neighbors.extend(
                    list(self.ml_lineage_tracking.get_associations(dest_arn=node_name))
                )
                for neighbor in neighbors:
                    if neighbor.source_arn not in discovered:
                        if neighbor.source_type == "FeatureGroup":
                            feature_group_artifact_arns.append(neighbor.source_arn)
                        queue.append(neighbor.source_arn)
                        discovered.append(neighbor.source_arn)

                    if neighbor.destination_arn not in discovered:
                        if neighbor.destination_type == "FeatureGroup":
                            feature_group_artifact_arns.append(neighbor.destination_arn)
                        queue.append(neighbor.destination_arn)
                        discovered.append(neighbor.destination_arn)
                current_depth += 1
        except Exception as e:  # throttling error
            pass

        feature_group_names = []
        feature_group_arns = []
        feature_group_s3_uris = []
        feature_group_table_names = []
        for feature_group_artifact_arn in feature_group_artifact_arns:
            feature_group_artifact = Artifact.load(feature_group_artifact_arn)
            feature_group_name = feature_group_artifact.artifact_name
            feature_group_names.append(feature_group_name)
            feature_group_arn = Artifact.load(
                feature_group_artifact_arn
            ).source.source_uri
            feature_group_arns.append(feature_group_arn)

            # Get Feature Group info
            fg = FeatureGroup(
                feature_group_name.replace("fg-", ""),
                sagemaker_session=self.sagemaker_session.session,
            )
            fg_info = fg.describe()

            feature_group_s3_uri = fg_info["OfflineStoreConfig"]["S3StorageConfig"][
                "S3Uri"
            ]
            feature_group_s3_uris.append(feature_group_s3_uri)

            feature_group_table_name = fg_info["OfflineStoreConfig"][
                "DataCatalogConfig"
            ]["TableName"]
            feature_group_table_names.append(feature_group_table_name)
        feature_group_df = pd.DataFrame(
            {
                "Feature Group Name": feature_group_names,
                "Feature Group Artifact ARN": feature_group_artifact_arns,
                "Feature Group ARN": feature_group_arns,
                "Feature Group S3 URI": feature_group_s3_uris,
                "Feature Group Table Name": feature_group_table_names,
            }
        )
        return feature_group_df
