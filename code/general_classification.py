import ee
import folium

ee.Authenticate()
ee.Initialize(project='ee-krle4401')

# Making an ROI/AOI based on a circle, should be able to upload shapefile path, too
center_point = ee.Geometry.Point([19.84332462338792, -34.737038455808126])
radius = 5000  # meters
area_of_interest = center_point.buffer(radius)

# Load Sentinel-2 data
sentinel_data = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
    .filterDate('2020-01-01', '2022-01-10') \
    .filterBounds(area_of_interest) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  #the 20 param, only allows for images with less than 20% cloud cover

# Function to mask clouds based on the Sentinel-2 QA band
def maskS2clouds(image):
    qa = image.select('QA60') 
    cloudBitMask = 1 << 10 #bit 10 on Sentinel is clouds
    cloudshadowBitMask = 1 << 11 #bit 11 on Sentinel is cloud shadow
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
             .And(qa.bitwiseAnd(cloudshadowBitMask).eq(0))
    return image.updateMask(mask).divide(10000) #dividing by DNs

# Apply the cloud mask
sentinel_data = sentinel_data.map(maskS2clouds).median()

# Update coordinates for training points within the new AOI
urban = ee.Geometry.Point([19.85332462338792, -34.737038455808126])  # Example change
vegetation = ee.Geometry.Point([19.84332462338792, -34.727038455808126])
water = ee.Geometry.Point([19.83332462338792, -34.747038455808126])

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

def add_ee_layer(self, ee_image_object, vis_params, name):
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

# Create a folium map object centered on the new AOI.
my_map = folium.Map(location=[-34.737038455808126, 19.84332462338792], zoom_start=10)

# Set visualization parameters.
vis_params = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}

# Add the classified image layer to the map using the new method.
my_map.add_ee_layer(classified_image, vis_params, 'LULC')

# Add a layer control panel to the map.
my_map.add_child(folium.LayerControl())

# Display the map.
my_map
