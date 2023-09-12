import json
import logging

import jenkspy
import pandas as pd
from geopy import distance

logging.basicConfig(level=logging.INFO)

with open('hackathon_data/osm_matched.json', encoding='utf-8') as fp:
    matched = json.load(fp)


def calculate_area(boundingbox):
    x = distance.distance((float(boundingbox[0]), float(boundingbox[2])),
                          (float(boundingbox[1]), float(boundingbox[2]))).km

    y = distance.distance((float(boundingbox[0]), float(boundingbox[2])),
                          (float(boundingbox[1]), float(boundingbox[2]))).km
    return x * y


areas = []
nz_gaz = pd.read_csv('hackathon_data/gaz_names.csv')
nz_gaz = nz_gaz[['name_id', 'name']]
nz_gaz
for pid, values in matched.items():
    if 'boundingbox' in values.keys():
        bbox = values['boundingbox']
        if bbox is not None and len(bbox) == 4:
            area = calculate_area(bbox)
            matched[pid]['area'] = area
    if 'area' not in matched[pid].keys():
        matched[pid]['area'] = 0
    areas.append(matched[pid]['area'])

areas = sorted(areas)

df = pd.DataFrame(matched).transpose()
df.index = df.index.astype(int)
breaks = jenkspy.jenks_breaks(areas, nb_class=30)
df['scale'] = pd.cut(df['area'], bins=breaks, labels=list(range(1, 31)))
df = df.merge(nz_gaz, how='inner', left_index=True, right_on='name_id')
df.to_csv('hackathon_data/osm_matched_scale.csv', index_label='id')
