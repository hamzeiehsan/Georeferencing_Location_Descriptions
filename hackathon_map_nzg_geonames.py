import logging

import pandas as pd
from geopy import distance

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info('reading datasets')
    geonames = pd.read_csv('hackathon_data/NZ_data.csv')
    geonames = geonames[geonames.scale.notnull()]  # unique: 44652
    nzg = pd.read_csv('hackathon_data/gaz_names.csv')  # unique: 45051
    nzg_names = nzg['name'].astype(str).to_list()  # exact matches: 38724
    geonames.name = geonames['name'].astype(pd.StringDtype())
    matches = []
    for index, row in nzg.iterrows():
        matched = {}
        name = str(row['name'])
        pid = row['name_id']
        nzg_lon = float(row['crd_longitude'])
        nzg_lat = float(row['crd_latitude'])
        record = geonames[geonames['name'] == name]
        if len(record) == 1:  # 21630
            matched['toponym'] = name
            matched['pid'] = pid
            matched['type'] = record['type'].values[0]
            matched['scale'] = record['scale'].values[0]
            matches.append(matched)
        elif len(record) > 1:  # 17094
            min_distance = 1000000000000000
            selected = False
            info = {}
            for index2, rec in record.iterrows():
                g_lat = float(rec['latitude'])
                g_lon = float(rec['longitude'])
                distance_val = distance.distance((g_lat, g_lon), (nzg_lat, nzg_lon)).km
                if distance_val < min_distance or not selected:
                    selected = True
                    info['type'] = rec['type']
                    info['scale'] = rec['scale']
                    min_distance = distance_val
            matched['toponym'] = name
            matched['type'] = info['type']
            matched['scale'] = info['scale']
            matched['pid'] = pid
            matched['name'] = name
            matches.append(matched)
    logging.info('simple matching process is achieved ...')
