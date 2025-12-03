import datetime

import matplotlib.pyplot as plt
import numpy as np

from environmentaltools.common import read
from environmentaltools.spectral import analysis


locations = {
    "playa_Granada": (36.0, -3.54),
    "Happisburgh": (52.85, 1.75),
    "Morioka": (40, 145),
    "Madagascar": (-20, 50),
    "LaSerena": (-30, -72.0)}

start_date = datetime.date(2023, 12, 1)
end_date = datetime.date(2023, 12, 28)
time_resolution_minutes = 1

database_path = "./src/environmentaltools/data/spectral/tidal/"

analysis.check_and_download_tidal_model(model_name="EOT20", database_path=database_path)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
axs[5].axis('off')  # Hide the unused subplot
for index, (location_name, (LAT, LON)) in enumerate(locations.items()):
    print(f"Processing location: {location_name} (Lat: {LAT}, Lon: {LON})")
    results = analysis.tidal_reconstruction_from_models(
        LAT,
        LON,
        start_date,
        end_date,
        'EOT20',
        database_path,
        time_resolution_minutes,
        database_path,
        location_name)
    
    predictions, datetime_index = results['predictions'], results['datetime_index']
    
    data = read.csv(f"./src/environmentaltools/data/spectral/tidal/serie_{location_name}.csv", ts=True)

    print("Max. dif: ", np.max(predictions - np.ravel(data.eta)), " cm")
    print(
        "RMSE: ",
        1 / len(predictions) * np.sqrt(np.sum((predictions**2 - np.ravel(data.eta) ** 2))),
        " cm",
    )

    axs[index].plot(datetime_index, predictions, label="EOT20")
    axs[index].plot(data, label="TPXO9-atlas")
    axs[index].set_xlabel("Time (min) [28 days]")
    axs[index].set_ylabel("elevation (cm)")
    axs[index].legend()
plt.tight_layout()
plt.show()