import logging
import uuid

import pandas as pd
import pysolr
from pyproj.transformer import TransformerGroup

logging.basicConfig(level=logging.INFO)

solr = pysolr.Solr('http://45.113.235.161:8983/solr/hackbio/', always_commit=True)

solr.ping()

datums = {'NZGD2000': 4167, 'RSRGD2000': 4762, 'WGS-84': 4326}


def dummy_insert():
    solr.add([
        {
            "id": "doc_1",
            "toponym": "just a Test",
            "coordinates": "45.034,-124.3234",
            "place_type": "hill"
        },
        {
            "id": "doc_2",
            "toponym": ["Just test 2", "Test 2", "TEST2"],
            "coordinates": "-43.3242,124.343455",
            "place_type": "bush"
        },
    ])


def dummy_delete():
    solr.delete(id=['doc_1', 'doc_2'])


def read_osm():
    points = pd.read_json('hackathon_data/osm/out.points.json')
    lines = pd.read_json('hackathon_data/osm/out.lines.json')
    mlines = pd.read_json('hackathon_data/osm/out.multilinestrings.json')
    polies = pd.read_json('hackathon_data/osm/out.multipolygons.json')
    df = pd.concat([points, lines, mlines, polies])
    df[['lat', 'lon']] = df.coordinates.str.split(',', expand=True)
    df['lat'] = df.lat.astype(float)
    df['lon'] = df.lon.astype(float)
    return df.groupby('name').agg({'lat': ['mean'], 'lon': ['mean'], 'source': ['min']})


if __name__ == '__main__':
    logging.info('start loading gazetteers to Apache Solr.')
    logging.info('working on NZ Gazetteers')
    dataset = pd.read_csv('hackathon_data/gaz_names.csv')
    maori_names = pd.read_csv('hackathon_data/maori_names.csv')
    nz_names = dataset.name.astype(str).to_list()

    buffer = 500
    docs = []
    for index, row in dataset.iterrows():
        name = str(row['name']).strip()
        datum = str(row['crd_datum'])
        tg = TransformerGroup(datums[datum], datums['WGS-84'])
        raw_lat = float(row['crd_latitude'])
        raw_lon = float(row['crd_longitude'])
        lat, lon = tg.transformers[0].transform(raw_lat, raw_lon)
        district = str(row['land_district'])
        pid = str(row['name_id'])
        place_type = str(row['feat_type'])
        document = {'id': pid, 'toponym': [name], 'type': [place_type],
                    'coordinates': '{lat},{lon}'.format(lat=lat, lon=lon), 'source': ['NZG'],
                    'district': [district]}
        docs.append(document)
        if len(docs) > buffer:
            solr.add(docs)
            docs = []
            logging.info('{} added to SOLR'.format(buffer))

    docs = []
    osm_data = read_osm()
    for name, row in osm_data.iterrows():
        if str(name).strip() not in nz_names:
            lat = row.lat
            lon = row.lon
            source = 'OSM'
            pid = str(uuid.uuid4())
            docs.append({'id': pid, 'toponym': [str(name).strip()], 'type': [],
                         'coordinates': '{lat},{lon}'.format(lat=lat.mean(), lon=lon.mean()), 'source': ['source'],
                         'district': []})
            if len(docs) > buffer:
                solr.add(docs)
                docs = []
                logging.info('{} from OSM added to SOLR'.format(buffer))
