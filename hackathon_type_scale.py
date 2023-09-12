import json
import logging
import time

import pandas as pd
from OSMPythonTools.nominatim import Nominatim
from geopy import distance
from pyproj.transformer import TransformerGroup

logging.basicConfig(level=logging.INFO)

datums = {'NZGD2000': 4167, 'RSRGD2000': 4762, 'WGS-84': 4326}
nominatim = Nominatim()


def in_bounding_box(lat, lon, bounding_box):
    if float(bounding_box[0]) <= lat <= float(bounding_box[1]):
        if float(bounding_box[2]) <= lon <= float(bounding_box[3]):
            return True
    return False


def is_near(lat1, lon1, lat2, lon2):
    if distance.distance((lon1, lat1), (lon2, lat2)).km < 2:
        return True
    return False


if __name__ == '__main__':
    dataset = pd.read_csv('hackathon_data/gaz_names.csv')

    osm_additional_info = {}
    unmatched = {}
    errors = {}
    for index, row in dataset.iterrows():
        name = str(row['name']).strip()
        pid = int(str(row['name_id']))
        datum = str(row['crd_datum'])
        tg = TransformerGroup(datums[datum], datums['WGS-84'])
        raw_lat = float(row['crd_latitude'])
        raw_lon = float(row['crd_longitude'])
        lat, lon = tg.transformers[0].transform(raw_lat, raw_lon)

        # fetch osm data
        try:
            result = nominatim.query(name + ', NZ')
            results_list = result.toJSON()
        except Exception:
            logging.error('error in fetching {}'.format(name))
            errors[pid] = name
            results_list = []

        for record in results_list:
            if 'boundingbox' in record.keys() and record['boundingbox'] is not None and len(record['boundingbox']) == 4:
                if in_bounding_box(lat, lon, record['boundingbox']):
                    osm_additional_info[pid] = record
                    break
            elif 'lat' in record.keys() and lon in record.keys():
                if is_near(lat1=lat, lon1=lon, lat2=float(record['lat']), lon2=float(['lon'])):
                    osm_additional_info[pid] = record
                    break

        if pid not in osm_additional_info.keys():
            unmatched[pid] = name
        if len(osm_additional_info) + len(unmatched) > 50:
            logging.info('50 records processed, '
                         'total matched: {0} and unmatched: {1}'.format(len(osm_additional_info), len(unmatched)))
            time.sleep(0.5)

    logging.info('all processed, time to write it in JSON files...')

    with open('hackathon_data/osm_matched.json', 'w', encoding='utf-8') as fp:
        json.dump(osm_additional_info, fp)

    with open('hackathon_data/unmatched.json', 'w', encoding='utf-8') as fp:
        json.dump(unmatched, fp)

    with open('hackathon_data/matching_errors.json', 'w', encoding='utf-8') as fp:
        json.dump(errors, fp)

    logging.info('writing the output files is finished...')
