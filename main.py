import logging

import data_pipeline
import eval_engine
import mod_pressure
import mod_temperature
import mod_visibility
import mod_wind
import mod_wind_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main():
    print("Initializing Data Pipeline...")
    df = data_pipeline.get_engineered_data()

    results = {}

    print("Training Temperature Specialist...")
    results["temp"] = mod_temperature.train_and_predict(df)

    print("Training Pressure Specialist...")
    results["pressure"] = mod_pressure.train_and_predict(df)

    print("Training Visibility Specialist...")
    results["visibility"] = mod_visibility.train_and_predict(df)

    print("Training Wind Specialist...")
    results["wind"] = mod_wind.train_and_predict(df)

    print("Training Wind MOS Specialist...")
    results["wind_v2"] = mod_wind_v2.train_and_predict(df)

    print("Generating Master Dashboard...")
    eval_engine.generate_combined_dashboard(results)
    print("Pipeline Complete. Master dashboard saved.")


if __name__ == "__main__":
    main()
