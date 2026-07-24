import json
import logging
import random
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
DEFAULT_MODEL = "best_match"
OVERLAP_START = pd.Timestamp("2017-01-01T00:00:00Z")


def _chunk_date_ranges(start_date: str, end_date: str) -> Iterable[tuple[str, str]]:
    current = pd.Timestamp(start_date).normalize()
    final = pd.Timestamp(end_date).normalize()
    while current <= final:
        chunk_end = min(current + pd.offsets.YearEnd(0), final)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = (chunk_end + pd.Timedelta(days=1)).normalize()


def _cache_file_path(cache_dir: Path, lat: float, lon: float, start: str, end: str, model: str) -> Path:
    safe_model = model.replace("/", "_")
    return cache_dir / f"nwp_{lat:.4f}_{lon:.4f}_{safe_model}_{start}_{end}.json"


def _read_cached_payload(cache_file: Path) -> dict:
    with cache_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fetch_payload(url: str, retries: int = 5, base_sleep_seconds: float = 1.5) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == retries - 1:
                raise
        except URLError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise

        sleep_seconds = base_sleep_seconds * (2**attempt) + random.uniform(0.0, 0.5)
        logger.warning("NWP fetch retry %d/%d after error: %s", attempt + 1, retries, last_error)
        time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to fetch NWP payload: {last_error}")


def _load_or_fetch_chunk(
    lat: float,
    lon: float,
    start: str,
    end: str,
    cache_dir: Path,
    model: str,
) -> dict:
    cache_file = _cache_file_path(cache_dir, lat, lon, start, end, model)
    if cache_file.exists():
        return _read_cached_payload(cache_file)

    query = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,surface_pressure",
            "models": model,
            "timezone": "UTC",
            "start_date": start,
            "end_date": end,
        }
    )
    payload = _fetch_payload(f"{HISTORICAL_FORECAST_URL}?{query}")
    with cache_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return payload


def _hourly_payload_to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly:
        return pd.DataFrame()

    frame = pd.DataFrame(hourly)
    if frame.empty or "time" not in frame.columns:
        return pd.DataFrame()

    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time"]).set_index("time").sort_index()
    frame = frame.rename(
        columns={
            "wind_speed_10m": "nwp_wind_speed",
            "wind_direction_10m": "nwp_wind_dir",
            "wind_gusts_10m": "nwp_wind_gust",
            "surface_pressure": "nwp_pressure",
        }
    )
    numeric_cols = ["nwp_wind_speed", "nwp_wind_dir", "nwp_wind_gust", "nwp_pressure"]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[numeric_cols]


def _resample_to_half_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    half_hour_index = pd.date_range(frame.index.min(), frame.index.max(), freq="30min", tz="UTC")
    reindexed = frame.reindex(half_hour_index)

    radians = np.radians(reindexed["nwp_wind_dir"])
    reindexed["nwp_wind_dir_sin"] = np.sin(radians)
    reindexed["nwp_wind_dir_cos"] = np.cos(radians)

    linear_cols = ["nwp_wind_speed", "nwp_wind_gust", "nwp_pressure", "nwp_wind_dir_sin", "nwp_wind_dir_cos"]
    reindexed[linear_cols] = reindexed[linear_cols].interpolate(
        method="time",
        limit_direction="both",
    )
    reindexed["nwp_wind_dir"] = (
        np.degrees(np.arctan2(reindexed["nwp_wind_dir_sin"], reindexed["nwp_wind_dir_cos"])) + 360.0
    ) % 360.0

    return reindexed[
        [
            "nwp_wind_speed",
            "nwp_wind_dir",
            "nwp_wind_dir_sin",
            "nwp_wind_dir_cos",
            "nwp_wind_gust",
            "nwp_pressure",
        ]
    ]


def fetch_nwp_history(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cache_path: str | Path = "data/nwp_cache",
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """
    Fetch Open-Meteo Historical Forecast data for MOS-style post-processing.

    This uses Open-Meteo's historical forecast archive, not the reanalysis archive:
    the endpoint returns stitched forecast-model output derived from historical runs.
    It is suitable for forecast-bias correction features, but this helper does not
    preserve individual issue times or exact lead-time provenance.
    """
    cache_dir = Path(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _chunk_date_ranges(start_date, end_date):
        payload = _load_or_fetch_chunk(lat, lon, chunk_start, chunk_end, cache_dir, model)
        frame = _hourly_payload_to_frame(payload)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("Open-Meteo historical forecast fetch returned no hourly payloads.")

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.loc[combined.notna().any(axis=1)]
    combined = combined.loc[combined.index >= OVERLAP_START]
    if combined.empty:
        raise ValueError("Open-Meteo overlap window produced no non-null NWP rows for this location.")

    return _resample_to_half_hourly(combined)
