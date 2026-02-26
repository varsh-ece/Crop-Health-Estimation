from flask import Flask, render_template, request
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# SentinelHub credentials
config = SHConfig()
config.sh_client_id = "92859096-4143-4586-a925-5a5d6feb33a2"
config.sh_client_secret = "Gv5vP3w7ZyHZ2EIspT8OFEqwyPiQcPc8"

# Calculate NDVI and return array
def calculate_ndvi(bbox):
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B08"],
            output: { bands: 2 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B08];
    }
    """
    request_sh = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2024-01-01", "2024-01-31")
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(256, 256),
        config=config
    )

    data = request_sh.get_data()[0]
    red = data[:, :, 0].astype(float)
    nir = data[:, :, 1].astype(float)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.nanmean(ndvi), ndvi

# Generate color-coded NDVI map
def generate_ndvi_map(ndvi_array, location_name):
    colormap = np.zeros((ndvi_array.shape[0], ndvi_array.shape[1], 3))
    colormap[ndvi_array < 0.3] = [1, 0, 0]                     # Red
    colormap[(ndvi_array >= 0.3) & (ndvi_array < 0.5)] = [1, 0.65, 0]  # Orange
    colormap[ndvi_array >= 0.5] = [0, 0.8, 0]                 # Green

    plt.imshow(colormap)
    plt.axis('off')
    if not os.path.exists("static"):
        os.makedirs("static")
    map_filename = f"static/ndvi_map_{location_name.replace(' ', '_')}.png"
    plt.savefig(map_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return map_filename

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    location = request.form.get("location", "Selected Location")
    lat = request.form.get("latitude")
    lon = request.form.get("longitude")
    language = request.form.get("language")

    if not lat or not lon:
        return "Location not found."

    lat, lon = float(lat), float(lon)
    bbox = BBox([lon-0.01, lat-0.01, lon+0.01, lat+0.01], crs=CRS.WGS84)

    ndvi_value, ndvi_array = calculate_ndvi(bbox)
    ndvi_map_file = generate_ndvi_map(ndvi_array, location)

    if ndvi_value < 0.3:
        status_key = "serious"
        color = "#ff4d4d"
    elif ndvi_value < 0.5:
        status_key = "moderate"
        color = "#ffa500"
    else:
        status_key = "healthy"
        color = "#4CAF50"

    messages = {
        "english": {
            "healthy": "Crop Health is Excellent 🌱",
            "moderate": "Crop is Moderately Stressed ⚠️",
            "serious": "Critical Condition! Immediate Attention Required 🚨"
        },
        "tamil": {
            "healthy": "பயிர் நல்ல நிலையில் உள்ளது 🌱",
            "moderate": "பயிர் சற்றே பாதிக்கப்பட்டுள்ளது ⚠️",
            "serious": "மிகவும் ஆபத்தான நிலை! உடனடி கவனம் தேவை 🚨"
        }
    }

    result_message = messages[language][status_key]

    return render_template(
        "result.html",
        location=location,
        ndvi=round(ndvi_value, 2),
        message=result_message,
        color=color,
        ndvi_map=ndvi_map_file
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    