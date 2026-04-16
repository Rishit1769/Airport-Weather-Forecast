from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import pandas as pd
from datetime import datetime, timedelta

STATION = "VABB"

start_date = datetime(2005, 1, 1)
end_date = datetime(2026, 4, 1)

delta = timedelta(days=30)

all_data = []

def start_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    return webdriver.Chrome(options=options)

driver = start_driver()

current_date = start_date
iteration = 0

while current_date < end_date:
    next_date = min(current_date + delta, end_date)

    print(f"\nFetching: {current_date.date()} → {next_date.date()}")

    driver.get("https://www.ogimet.com/metars.phtml.en")
    time.sleep(3)

    # ICAO
    driver.find_element(By.NAME, "lugar").clear()
    driver.find_element(By.NAME, "lugar").send_keys(STATION)

    # FORMAT
    Select(driver.find_element(By.NAME, "fmt")).select_by_visible_text("TXT")

    # BEGIN DATE
    Select(driver.find_element(By.NAME, "ano")).select_by_visible_text(str(current_date.year))
    Select(driver.find_element(By.NAME, "mes")).select_by_visible_text(current_date.strftime("%B"))
    Select(driver.find_element(By.NAME, "day")).select_by_visible_text(f"{current_date.day:02d}")
    Select(driver.find_element(By.NAME, "hora")).select_by_visible_text("00")

    # END DATE
    Select(driver.find_element(By.NAME, "anof")).select_by_visible_text(str(next_date.year))
    Select(driver.find_element(By.NAME, "mesf")).select_by_visible_text(next_date.strftime("%B"))
    Select(driver.find_element(By.NAME, "dayf")).select_by_visible_text(f"{next_date.day:02d}")
    Select(driver.find_element(By.NAME, "horaf")).select_by_visible_text("23")

    # SUBMIT
    driver.find_element(By.XPATH, "//input[@value='send']").click()
    time.sleep(5)

    body_text = driver.find_element(By.TAG_NAME, "body").text
    lines = body_text.split("\n")

    count_before = len(all_data)

    for line in lines:
        if "METAR" in line and STATION in line:
            all_data.append({
                "date_range": str(current_date.date()),
                "raw_metar": line
            })

    print(f"Collected: {len(all_data) - count_before}")

    # SAVE PROGRESS
    pd.DataFrame(all_data).to_csv("mumbai_metar_progress.csv", index=False)

    current_date = next_date
    iteration += 1

    if iteration % 5 == 0:
        driver.quit()
        time.sleep(2)
        driver = start_driver()

driver.quit()

df = pd.DataFrame(all_data)
df.to_csv("mumbai_metar_3years.csv", index=False)

print(f"\n✅ DONE. Total rows: {len(df)}")