import time
import boto3
import pandas as pd
from sagemaker.feature_store.feature_definition import FeatureDefinition
from sagemaker.feature_store.feature_group import FeatureGroup

from ml_lineage_helper import StatusIndicator
from ml_lineage_helper import SageMakerSession


class FeatureStore:
    def __init__(self, feature_group_name: str, sagemaker_session: SageMakerSession):
        self.feature_group_name = feature_group_name
        self.sagemaker_session = sagemaker_session
        self.feature_store_client = boto3.client(
            "sagemaker-featurestore-runtime", region_name=self.sagemaker_session.region
        )
        try:
            self.feature_group = FeatureGroup(
                self.feature_group_name, self.sagemaker_session.session
            )
            self.table_name = self.feature_group.athena_query().table_name
        except:
            self.feature_group = None
            self.table_name = None

    def get_feature_definitions(self, df, feature_group):
        # Dtype int_, int8, int16, int32, int64, uint8, uint16, uint32
        # and uint64 are mapped to Integral feature type.

        # Dtype float_, float16, float32 and float64
        # are mapped to Fractional feature type.

        # string dtype is mapped to String feature type.

        # Our schema of our data that we expect
        # _after_ SageMaker Processing
        feature_definitions = []
        for column in df.columns:
            feature_type = feature_group._DTYPE_TO_FEATURE_DEFINITION_CLS_MAP.get(
                str(df[column].dtype), None
            )
            feature_definitions.append(
                FeatureDefinition(column, feature_type)
            )  # you can alternatively define your own schema
        return feature_definitions

    def wait_for_feature_group_creation_complete(self, feature_group):
        status_indicator = StatusIndicator()
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            status_indicator.update(status)
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        status_indicator.end()
        if status != "Created":
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")

    def ingest_df_into_feature_group(self, df):
        success, fail = 0, 0
        for _, row_series in df.astype(str).iterrows():
            record = []
            for key, value in row_series.to_dict().items():
                record.append({"FeatureName": key, "ValueAsString": str(value)})
            response = self.feature_store_client.put_record(
                FeatureGroupName=self.feature_group_name, Record=record
            )
            if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                success += 1
            else:
                fail += 1
        print(f"Success = {success}")
        print(f"Fail = {fail}")

    def create_feature_group(
        self,
        df,
        feature_group_description,
        offline_feature_group_s3_uri,
        enable_online=True,
        id_name=None,
        event_time_name=None,
        ingest=True,
    ):
        feature_group = FeatureGroup(
            name=self.feature_group_name,
            sagemaker_session=self.sagemaker_session.session,
        )
        # Add id and event time column
        id_name = "event_id"
        event_time_name = "event_time"
        df[id_name] = df.index
        current_time_sec = int(round(time.time()))
        df[event_time_name] = pd.Series([current_time_sec] * len(df), dtype="float64")
        feature_definitions = self.get_feature_definitions(df, feature_group)
        feature_group.feature_definitions = feature_definitions
        try:
            print(f'Trying to create feature group "{self.feature_group_name}" \n')
            feature_group.create(
                description=feature_group_description,
                record_identifier_name=id_name,
                event_time_feature_name=event_time_name,
                role_arn=self.sagemaker_session.role_arn,
                s3_uri=offline_feature_group_s3_uri,
                enable_online_store=enable_online,
            )
            self.wait_for_feature_group_creation_complete(feature_group)
        except Exception as e:
            code = e.response["Error"]["Code"]
            if code == "ResourceInUse":
                print(f"Using existing feature group: {self.feature_group_name}")
            else:
                raise (e)
        if ingest:
            print(f"Ingesting dataframe into {self.feature_group_name}...")
            self.ingest_df_into_feature_group(df)
        self.feature_group = feature_group
        return feature_group

    def query_feature_group(self, query, query_output_s3_uri=None, wait=True):
        """
        :param: query: str
        :param: query_output_s3_uri: str

        :return: pandas.DataFrame
        """
        feature_group_athena_query = self.feature_group.athena_query()
        if not query_output_s3_uri:
            query_output_s3_uri = (
                f"{self.sagemaker_session.bucket_s3_uri}/query_results"
            )
        try:
            feature_group_athena_query.run(
                query_string=query, output_location=query_output_s3_uri
            )
            if wait:
                feature_group_athena_query.wait()
                return (
                    feature_group_athena_query.as_dataframe(),
                    feature_group_athena_query,
                )
            else:
                return None, None
        except Exception as e:
            print(e)
            print(
                f'\nNote that the "{self.feature_group.name}" Feature Group is a table called "{feature_group_athena_query.table_name}" in Athena.'
            )
