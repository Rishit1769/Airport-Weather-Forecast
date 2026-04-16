import pandas as pd
import re

df = pd.read_csv("mumbai_metar_progress.csv")

parsed_data = []

for row in df["raw_metar"]:
    
    # ❌ Skip bad rows
    if "NIL" in row:
        continue

    try:
        # Extract FULL timestamp (YYYYMMDDHHMM)
        full_time_match = re.search(r'(\d{12})', row)
        full_time = full_time_match.group(1) if full_time_match else None

        # Convert to datetime
        if full_time:
            dt = pd.to_datetime(full_time, format="%Y%m%d%H%M")
        else:
            continue

        # Wind
        wind_match = re.search(r'(\d{3})(\d{2})KT', row)
        wind_dir = int(wind_match.group(1)) if wind_match else None
        wind_speed = int(wind_match.group(2)) if wind_match else None

        # Visibility
        vis_match = re.search(r'\s(\d{4})\s', row)
        visibility = int(vis_match.group(1)) if vis_match else None

        # Temperature
        temp_match = re.search(r'(\d{2})/(\d{2})', row)
        temp = int(temp_match.group(1)) if temp_match else None

        # Pressure
        pressure_match = re.search(r'Q(\d{4})', row)
        pressure = int(pressure_match.group(1)) if pressure_match else None

        parsed_data.append({
            "datetime": dt,
            "wind_dir": wind_dir,
            "wind_speed": wind_speed,
            "visibility": visibility,
            "temp": temp,
            "pressure": pressure
        })

    except:
        continue

clean_df = pd.DataFrame(parsed_data)

# Sort properly (VERY important)
clean_df = clean_df.sort_values("datetime")

# Drop missing
clean_df = clean_df.dropna()

clean_df.to_csv("clean_weather_data.csv", index=False)

print("✅ Clean parsing done")
print(clean_df.head())