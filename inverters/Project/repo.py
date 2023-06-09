from datetime import timedelta
import pandas as pd
from pandas import Timestamp
import os
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

from feast.types import Float64, Int32, String, Bool 
from dotenv import load_dotenv

load_dotenv()
source_path = os.getenv("SOURCE_PATH")

inverter = Entity(name="inverter", 
              join_keys=["_id"], 
              value_type=ValueType.STRING,
              description="id of the inverter"
              )

source = FileSource(
    name="inverter_feature_view",
    path=source_path,
    timestamp_field="ts_day",
    created_timestamp_column="m.ts",
)

inverter_fv = FeatureView(
    name="inverters_information",
    entities=[],
    ttl=timedelta(days=1),
    schema=[
        Field(name="_id", dtype=String, description="inverter id"),
        Field(name="did", dtype=String, description="device id"),
        Field(name="m.v", dtype=Float64, description="meassage of voltage"),
        Field(name="m.p0", dtype=Int32, description="message of power"),
        Field(name="m.pac", dtype=Float64,description="message of pac"),
        Field(name="m.tw", dtype=Int32),
        Field(name="m.ar",dtype=Float64),
        Field(name="m.rr",dtype=Float64),
        Field(name="m.t",dtype=Float64),
        Field(name="noId", dtype=Bool, description="If it does not have Id"),
        Field(name="pid", dtype=String, description="panel Id"),
        Field(name="serial", dtype=String, description="serial number")
    ],
    source=source
)
