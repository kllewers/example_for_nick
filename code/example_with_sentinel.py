import ee
import folium

ee.Authenticate()

ee.Initialize(project='ee-krle4401')

# Define your area of interest
area_of_interest = ee.Geometry.Rectangle([73.687, 18.524, 74.213, 18.985])  # Example coordinates

# Load Sentinel-2 data
sentinel_data = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
    .filterDate('2022-01-01', '2022-01-10') \
    .filterBounds(area_of_interest) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Filtering out cloudy images

# Function to mask clouds based on the Sentinel-2 QA band
def maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
             .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

# Apply the cloud mask
sentinel_data = sentinel_data.map(maskS2clouds).median()

# Example of creating training regions manually
urban = ee.Geometry.Point([74.004, 18.562])
vegetation = ee.Geometry.Point([73.807, 18.957])
water = ee.Geometry.Point([73.789, 18.560])

# Use these points to sample the input imagery to get training data.
training = sentinel_data.sampleRegions(
  collection=ee.FeatureCollection([
    ee.Feature(urban, {'class': 0}),
    ee.Feature(vegetation, {'class': 1}),
    ee.Feature(water, {'class': 2})
  ]),
  properties=['class'],
  scale=20
)

# Train a classifier.
classifier = ee.Classifier.smileRandomForest(50).train(training, 'class')

# Classify the image.
classified_image = sentinel_data.classify(classifier)

'''def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map data © Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Add EE drawing method to folium.Map.
folium.Map.add_ee_layer = add_ee_layer

# Assume 'classified_image' is an EE Image from previous steps
# Create a folium map object.
my_map = folium.Map(location=[18.516726, 73.856255], zoom_start=10)

# Set visualization parameters.
vis_params = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}

# Add the classified image layer to the map using the new method.
my_map.add_ee_layer(classified_image, vis_params, 'LULC')

# Add a layer control panel to the map.
my_map.add_child(folium.LayerControl())

# Display the map.
my_map'''

import folium

# Function to add Earth Engine layers
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map data © Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Patch the method to folium
folium.Map.add_ee_layer = add_ee_layer

# Create map
my_map = folium.Map(location=[-34.737709, 19.831708], zoom_start=10)

# Visualization parameters for classified image
vis_params = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}
my_map.add_ee_layer(classified_image, vis_params, 'LULC')

# Dictionary of data
data_dict = {1: {'bands': {'AOT': 0.0145, 'B1': 0.1105, 'B11': 0.2370, 'B12': 0.1676, 'B2': 0.1090, 'B3': 0.1256, 'B4': 0.1204, 'B5': 0.1674, 'B6': 0.3424, 'B7': 0.4085, 'B8': 0.4202, 'B8A': 0.4371, 'B9': 0.4339, 'MSK_CLDPRB': 0, 'MSK_SNWPRB': 0, 'QA10': 0, 'QA20': 0, 'QA60': 0, 'SCL': 0.0004, 'TCI_B': 0.0009, 'TCI_G': 0.0027, 'TCI_R': 0.0022, 'WVP': 0.1861}, 'Latitude': -33.714651479, 'Longitude': 19.241065829, 'RandomID': 1, 'class_value': 'Gum', 'class_code': 1}}

# Add points from the dictionary to the map
for point_id, point_info in data_dict.items():
    folium.Marker(
        location=[point_info['Latitude'], point_info['Longitude']],
        popup=f"ID: {point_id}<br>Class: {point_info['class_value']} ({point_info['class_code']})",
        icon=folium.Icon(color='green' if point_info['class_code'] == 1 else 'red')
    ).add_to(my_map)

# Add layer control and display map
my_map.add_child(folium.LayerControl())
my_map
