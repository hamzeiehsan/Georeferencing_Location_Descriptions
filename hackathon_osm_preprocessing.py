import json
import logging

import geojson
from shapely.geometry import shape

logging.basicConfig(level=logging.INFO)

# key:vals
key_vals = {'name': ['name'], 'type': [], 'district': [], 'source': 'OSM'}

# loading point datasets
logging.info('loading point datasets...')
with open('hackathon_data/osm/points.json', encoding='utf-8') as fp:
    points = geojson.load(fp)

docs = []
# name, type, source, coordinates, district
for feature in points['features']:
    if 'name' in feature['properties'].keys():
        point = feature['geometry']['coordinates']
        oid = feature['properties']['osm_id']
        name = feature['properties']['name']
        coordinates = '{lat},{lon}'.format(lat=point[1], lon=point[0])
        docs.append({'id': oid, 'name': name, 'coordinates': coordinates, 'source': 'OSM'})

with open('hackathon_data/osm/out.points.json', 'w', encoding='utf-8') as fp:
    json.dump(docs, fp)

# lines, multilines, and polygons: geometry to points
for file_name in ['lines', 'multilinestrings', 'multipolygons']:
    docs = []
    logging.info('loading {} dataset'.format(file_name))
    with open('hackathon_data/osm/{}.json'.format(file_name), encoding='utf-8') as fp:
        data = geojson.load(fp)
    for feature in data['features']:
        if 'name' in feature['properties'].keys() and 'osm_id' in feature['properties'].keys():
            name = feature['properties']['name']
            oid = feature['properties']['osm_id']
            geom = shape(feature['geometry'])
            coordinates = '{lat},{lon}'.format(lat=geom.centroid.y, lon=geom.centroid.x)
            docs.append({'id': oid, 'name': name, 'coordinates': coordinates, 'source': 'OSM'})
    with open('hackathon_data/osm/out.{}.json'.format(file_name), 'w', encoding='utf-8') as fp:
        json.dump(docs, fp)
