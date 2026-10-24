
import json

import numpy as np
import pandas as pd
import requests

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from river import stats

from config import (
    PHYSICAL_BOUNDS,
    PROCESSED_DIR,
    RAW_DIR,
    TIMEZONE,
    WEATHER_COLUMNS,
    HOT_PERCENTILE,
    WARMUP_PERCENTAGE,
    MIN_CONSECUTIVE_HOT_DAYS,
    WARMUP_START,
)





