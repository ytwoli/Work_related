from datetime import timedelta
import pandas as pd
from pandas import Timestamp

from feast import (
    Entity,
    FeatureService,
    ValueType,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource
)

from feast.types import Float64, Int32, String
inverter = Entity(name="inverter", 
              join_keys=["_id"], 
              value_type=ValueType.STRING,
              description="id of the inverter"
              )

source = FileSource(
    name="inverter_feature_view",
    path="data/inventers.parquet",
    event_timestamp_column="ts_day",
)

inverter_fv = FeatureView(
    name="inverters_information",
    entities=[inverter],
    ttl=timedelta(days=1),
    schema=[
        Field(name="did", dtype=String),
        Field(name="m", dtype=String),
        Field(name="noId", dtype=String),
        Field(name="pId", dtype=String),
        Field(name="serial", dtype=String)
    ],
    source=source
)
