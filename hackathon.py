import Levenshtein
import difflib
import itertools
import json
import logging
import warnings

import distance as distance_string
import geopandas as gpd
import pandas as pd
import pysolr
import spacy
from allennlp.predictors.predictor import Predictor
from geopy import distance, Point
from numpy import arctan2, sin, cos, degrees

import hackathon_parser as parser

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.ERROR)

ner = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
fg_ner = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-02-11.tar.gz")
nlp = spacy.load("en_core_web_sm")
valid_roles = ['B-LOC', 'U-LOC', 'U-GPE', 'B-GPE', 'U-FAC', 'B-FAC', 'U-ORG', 'B-ORG']
scale_regions = pd.read_csv('hackathon_data/NZ_data_regions.csv')
unique_type_dict = {'hill': 'hill', 'hills': 'hill', 'slope': 'cliff(s)', 'slopes': 'cliff(s)'}  # todo
logging.info('pretrained models are loaded...')

types = []
types_abbreviations = {'Mt. ': 'Mount ', 'Mt ': 'Mount '}
places_abbreviations = {'NZ': 'New Zealand', 'AUK': 'Auckland', 'BOP': 'Bay of Plenty', 'CAN': 'Canterbury',
                        'GIS': 'Gisborne', 'HKB': 'Hawke\'s Bay', 'MBH': 'Marlborough', 'MWT': 'Manawatu Wanganui',
                        'NSN': 'Nelson', 'NTL': 'Northland', 'OTA': 'Otago', 'STL': 'Southland', 'TAS': 'Tasman',
                        'TKI': 'Taranaki', 'WKO': 'Waikato', 'WGN': 'Wellington', 'WTC': 'West Coast',
                        'CIT': 'Chatham Islands Territory'}
prep_starts = {'In ': 'in ', 'Near ': 'near ', 'At ': 'at ', 'Of ': 'of ', 'Within ': 'within ', 'From ': 'from ',
               'To ': 'to ', 'Between ': 'between ', 'Beside ': 'beside ', 'Along ': 'along ',
               'Above ': 'above ', 'Around ': 'around ', 'Below ': 'below ', 'About ': 'about ', 'South': 'south ',
               'East ': 'east ', 'West ': 'west ', 'North ': 'north', 'South-': 'south', 'North-': 'north',
               'Northwest ': 'northwest ', 'Northeast ': 'northeast ', 'SouthWest ': 'southwest ',
               'Southeast': 'southeast '}


def calculate_distance(lat1, lat2, lon1, lon2):
    return distance.distance((lon1, lat1), (lon2, lat2)).km


def calculate_bearing(lat1, lat2, lon1, lon2):
    dL = lon2 - lon1
    X = cos(lat2) * sin(dL)
    Y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)
    return (degrees(arctan2(X, Y)) + 360) % 360


def preprocessing(sentence):
    for key, val in types_abbreviations.items():
        if key in sentence:
            sentence = sentence.replace(key, val)
    for key, val in places_abbreviations.items():
        if key in sentence:
            sentence = sentence.replace(key, val)
    for key, val in prep_starts.items():
        if sentence.startswith(key):
            sentence = sentence.replace(key, val)
            break
    return sentence.replace('/', ' ').replace('-', ' ')


class NER:  # find generic place types
    def __init__(self, sentence):
        self.sentence = sentence

    @staticmethod
    def place_names_tags(tags, words):
        places = []
        place = ''
        for i, (tag, word) in enumerate(zip(tags, words)):
            if tag.startswith('B-') and tag in valid_roles:
                place += word + ' '
            if len(place) > 0:
                if tag.startswith('I-'):
                    place += word + ' '
                elif tag.startswith('L-'):
                    place += word
                    places.append(place)
                    place = ''
            if tag.startswith('U-') and tag in valid_roles:
                if word not in parser.DIRECTIONS.keys():
                    places.append(word)
        return places

    def extract_place_names(self):
        ner_res = ner.predict(self.sentence)
        ner_places = NER.place_names_tags(ner_res['tags'], ner_res['words'])
        fg_ner_res = fg_ner.predict(self.sentence)
        fg_ner_places = NER.place_names_tags(fg_ner_res['tags'], fg_ner_res['words'])
        return NER.merge_place_lists(ner_places, fg_ner_places)

    @staticmethod
    def merge_place_lists(l1, l2):  # not working properly -- duplicates:
        # changed test wih 'Marlborough, hills about Queen Charlotte Sound'
        lm = []
        already_l1 = []
        already_l2 = []
        for p1 in l1:
            if ', ' in p1:
                already_l1.append(p1)
                continue
            for p2 in l2:
                if ', ' in p2:
                    already_l2.append(p2)
                    continue
                if p2 not in already_l2:
                    if p1 == p2:
                        lm.append(p1)
                        already_l2.append(p2)
                        already_l1.append(p1)
                        break
                    elif p1 in p2:
                        lm = []
                        already_l1 = []
                        already_l2 = []
                    elif p2 in p1:
                        already_l2.append(p2)
                        already_l1.append(p1)
                        lm.append(p1)
                        break
        for p1 in l1:
            if p1 not in already_l1:
                lm.append(p1)
        for p2 in l2:
            if p2 not in already_l2:
                lm.append(p2)
        return lm


class Gazetteer:
    def __init__(self):
        self.solr = pysolr.Solr('http://localhost:8983/solr/hackbio/', always_commit=True)
        self.level1 = gpd.read_file('hackathon_data/shape_files/level1.shp')
        self.level2 = gpd.read_file('hackathon_data/shape_files/level2.shp')
        self.name_l11 = self.level1['REGC2020_1'].astype(str).to_list()
        self.name_l12 = self.level1['REGC2020_2'].astype(str).to_list()
        self.name_l2 = self.level2['NAME'].astype(str).to_list()

    def find_administrative_matches(self, toponym):
        res = {'name': toponym, 'toponym': [], 'type': ['region_adm'], 'coordinates': '', 'source': 'SHAPE', 'bbox': []}
        record = None
        level = None
        matched_name = None
        for name in self.name_l11:
            if toponym in name:
                record = self.level1[self.level1['REGC2020_1'] == name]
                matched_name = name
                level = 'Level_1'
                break
        if record is None:
            for name in self.name_l12:
                if toponym in name:
                    record = self.level1[self.level1['REGC2020_2'] == name]
                    matched_name = name
                    level = 'Level_1'
                    break
        if record is None:
            for name in self.name_l2:
                if toponym in name:
                    record = self.level2[self.level2['NAME'] == name]
                    matched_name = name
                    level = 'Level_2'
                    break

        if record is not None:
            res['coordinates'] = '{lat},{lon}'. \
                format(lat=float(record.geometry.centroid.y), lon=float(record.geometry.centroid.x))
            res['bbox'] = record.geometry.bounds
            res['toponym'] = [matched_name]
            res['level'] = level
            return [res]
        return None

    def find_possible_locations(self, toponym, best=False):
        record = self.find_administrative_matches(toponym)
        if record is not None:
            return record
        if self.solr is not None:
            try:
                logging.debug('finding the match for {}'.format(toponym))
                matches = []
                # exact match
                search_results = self.solr.search('toponym:"' + toponym + '"'
                                                  , **{'hl': 'true', 'hl.fragsize': 10,
                                                       'rows': 20, 'fl': 'toponym, coordinates, type, source'})
                if len(search_results.docs) > 0:
                    matches.extend(search_results.docs)
                else:  # fuzzy match -- find similarities
                    search_results = self.solr.search('toponym:' + toponym
                                                      , **{'hl': 'true', 'hl.fragsize': 10,
                                                           'rows': 20, 'fl': 'toponym, coordinates, type, source'})
                    if len(search_results) > 0:
                        matches.extend(search_results.docs)

                logging.debug('{0} has {1} ambiguity'.format(toponym, len(matches)))
                return Gazetteer.solr_similarity(toponym, matches, best=best)
            except Exception:
                return []

    def find_by_districts(self, district, toponym):
        if self.solr is not None:
            try:
                logging.debug('finding the match for {} by district'.format(toponym))
                matches = []
                # exact match
                search_results = self.solr.search('district: "{}" and toponym:"'.format(district) + toponym + '"'
                                                  , **{'hl': 'true', 'hl.fragsize': 10,
                                                       'rows': 10, 'fl': 'toponym, coordinates, type, source'})
                if len(search_results.docs) > 0:
                    matches.extend(search_results.docs)
                else:  # fuzzy match -- find similarities
                    search_results = self.solr.search('district: "{}" and toponym:'.format(district) + toponym
                                                      , **{'hl': 'true', 'hl.fragsize': 10,
                                                           'rows': 20, 'fl': 'toponym, coordinates, type, source'})
                    if len(search_results) > 0:
                        matches.extend(search_results.docs)

                logging.debug('{0} has {1} ambiguity'.format(toponym, len(matches)))
                return Gazetteer.solr_similarity(toponym, matches)
            except Exception:
                return []

    @staticmethod
    def solr_similarity(toponym, matched_list, best=False):
        matched_nzg = []  # focus on NZG dataset
        for matched_record in matched_list:
            if 'NZG' in matched_record['source']:
                matched_nzg.append(matched_record)
        if len(matched_nzg) > 0:
            matched_list = matched_nzg
        filtered = []
        filtered_85 = []
        max_sem = -1
        best_match = None
        for m in matched_list:
            m['name'] = toponym
            sems = []
            for pname in m['toponym']:
                sems.append(Gazetteer.similarity(toponym, pname))
            sem = max(sems)
            if sem > max_sem:
                max_sem = sem
                best_match = m
            if sem > 0.7:
                filtered.append(m)
            if sem > 0.85:
                filtered_85.append(m)
        if len(filtered) == 0 or best:
            return [best_match]
        if len(filtered_85) > 0:
            filtered = filtered_85
        return filtered

    @staticmethod
    def similarity(s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        if s1 == s2:
            return 1
        try:
            diffl = difflib.SequenceMatcher(None, s1, s2).ratio()
            lev = Levenshtein.ratio(s1, s2)
            sor = 1 - distance_string.sorensen(s1, s2)
            jac = 1 - distance_string.jaccard(s1, s2)
            return (diffl + lev + sor + jac) / 4
        except Exception:
            return 0


class Disambiguate:
    def __init__(self, context, nz_gaz):
        self.context = context
        self.nz_gaz = nz_gaz
        self.locations = None
        self.resolved = None
        self.find_locations()

    def find_locations(self):
        self.locations = {}
        best = False
        if len(self.context) == 1:
            best = True
        for place in self.context:
            ambigous_place = self.nz_gaz.find_possible_locations(place, best=best)
            if ambigous_place is not None:
                self.locations[place] = ambigous_place

    def disambiguate(self):
        if self.resolved is None:
            self.resolved = self.disambiguate_area()
        return self.resolved

    def disambiguate_area(self):
        list_vals = []
        list_toponyms = []
        for toponym, values in self.locations.items():
            list_toponyms.append(toponym)
            list_vals.append(values)

        combs = list(itertools.product(*list_vals))
        comb_value = 1000000000000000000
        comb_min = None
        for comb in combs:
            measure = Disambiguate.measure_bbox(comb)
            if comb_min is None or comb_value > measure:
                comb_min = comb
                comb_value = measure
        return comb_min

    @staticmethod
    def measure_bbox(combination):
        min_lat = 90
        max_lat = -90
        min_lon = 180
        max_lon = -180
        for vals in combination:
            coordinates = vals['coordinates'].split(',')
            lat = float(coordinates[0])
            lon = float(coordinates[1])
            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat
            if lon < min_lon:
                min_lon = lon
            if lon > max_lon:
                max_lon = lon
        return distance.distance((min_lat, min_lon), (max_lat, max_lon)).km


class EstimateLocation:
    def __init__(self, unambigous_context, locality_description, lat, lon, generic=None, locations=None, anchors=None):
        self.context = unambigous_context
        self.description = locality_description
        self.generic = generic
        self.anchors = anchors
        self.locations = locations
        self.mil_lat = None
        self.mil_lon = None
        self.locality = None
        self.level = None
        self.lat = lat
        self.lon = lon
        self.predicated_coordinates = []
        self.error = {}

    def calculate_error(self, predicted_lat, predicted_lon, analysis_type):
        self.error[analysis_type] = distance.distance((self.lat, self.lon), (predicted_lat, predicted_lon)).km
        logging.debug('prediction error using {0} heuristics: {1}'.
                      format(analysis_type, self.error[analysis_type]))

    def calculate_errors(self):
        for predictions in self.predicated_coordinates:
            self.calculate_error(predictions['lat'], predictions['lon'], predictions['type'])
        if 'SL' not in self.error.keys():
            self.error['SL'] = self.error['LL']
        if 'MIL' not in self.error.keys():
            self.error['MIL'] = self.error['SL']
        if 'DAS1' not in self.error.keys():
            self.error['DAS1'] = self.error['MIL']
        if 'DAS2' not in self.error.keys():
            self.error['DAS2'] = self.error['DAS1']
        if 'GR' not in self.error.keys():
            self.error['GR'] = self.error['DAS2']

    def estimate(self):
        self.estimate_by_zoom_out()
        self.estimate_by_mil_scale()
        self.estimate_by_mil_locations()
        self.estimate_by_mil_anchors()
        self.estimate_by_generics()

    def estimate_by_mil_scale(self):
        sl_lat = None
        sl_lon = None
        best_scale_coordinates = None
        best_scale_value = 1
        for place in self.context:
            if len(place['toponym']) > 0 and 'NZG' in place['source']:
                toponym = place['toponym'][0]
                scale_estimate = EstimateScale.get_scale(toponym)
                if scale_estimate is not None:
                    if scale_estimate >= best_scale_value:
                        best_scale_value = scale_estimate
                        best_scale_coordinates = place['coordinates']
        if best_scale_coordinates is not None:
            coords = best_scale_coordinates.split(',')
            sl_lat = float(coords[0])
            sl_lon = float(coords[1])
            self.predicated_coordinates.append({'type': 'SL', 'lat': sl_lat, 'lon': sl_lon})
            self.mil_lat = sl_lat
            self.mil_lon = sl_lon

    def estimate_by_mil_locations(self):
        mil_lat = None
        mil_lon = None
        localities = []
        for place in self.context:
            if 'type' in place.keys() and 'region_adm' in place['type']:
                localities.append(place['name'])
        if len(localities) == len(self.context):
            localities = []
        if self.locations is not None and len(self.locations) > 0:
            if len(self.locations) == 1:
                mil = list(self.locations[0].values())[0]
                for place in self.context:
                    if place['name'] == mil:
                        coords = place['coordinates'].split(',')
                        mil_lat = float(coords[0])
                        mil_lon = float(coords[1])
                        break
            else:
                toponyms = []
                for location in self.locations:
                    if list(location.values())[0] not in localities:
                        toponyms.append(list(location.values())[0])
                selected = None
                mil_importance = {}
                if self.anchors is not None and len(self.anchors) > 0:
                    for anchor in anchors:
                        if anchor['anchor'] in toponyms:
                            if anchor['anchor'] not in mil_importance.keys():
                                mil_importance[anchor['anchor']] = 0
                            mil_importance[anchor['anchor']] += 1
                else:
                    for location in self.locations:
                        for prep, anchor in location.items():
                            if anchor not in mil_importance.keys():
                                mil_importance[anchor] = 0
                            if prep in parser.prepositions_type.keys():
                                mil_importance[anchor] += parser.prepositions_type[prep]
                            else:
                                mil_importance[anchor] += 2
                max_importance = -1
                for toponym in toponyms:
                    if toponym in mil_importance.keys() and mil_importance[toponym] > max_importance:
                        max_importance = mil_importance[toponym]
                        selected = toponym
                for place in self.context:
                    if selected == place['name'] and place['name'] in toponyms:
                        coords = place['coordinates'].split(',')
                        mil_lat = float(coords[0])
                        mil_lon = float(coords[1])
                        break
        else:
            if self.context is not None:
                if len(self.context) == 1:
                    coords = self.context[0]['coordinates'].split(',')
                    mil_lat = float(coords[0])
                    mil_lon = float(coords[1])
                else:
                    self.estimate_by_mil_scale()

        if mil_lat is not None:
            self.predicated_coordinates.append({'type': 'MIL', 'lat': mil_lat,
                                                'lon': mil_lon})
            self.mil_lat = mil_lat
            self.mil_lon = mil_lon

    def estimate_by_zoom_out(self):  # choose the last location
        last_index = -1
        last_coords = None
        for place in self.context:
            if place['name'] in self.description and self.description.index(place['name']) > last_index:
                last_index = self.description.index(place['name'])
                last_coords = place['coordinates'].split(',')
        ll_lat = float(last_coords[0])
        ll_lon = float(last_coords[1])
        self.predicated_coordinates.append({'type': 'LL', 'lat': ll_lat, 'lon': ll_lon})
        self.mil_lat = ll_lat
        self.mil_lon = ll_lon

    def estimate_by_generics(self):
        if self.generic is not None:
            locality = None
            level = None
            for place in self.context:
                if 'type' in place.keys() and 'region_adm' in place['type']:
                    locality = place['toponym'][0]
                    level = place['level']
                    break
            if level is not None and locality is not None:
                for g in self.generic:
                    estimated_res = \
                        EstimateLocation.generic_in_locality(g, locality, level, self.mil_lat, self.mil_lon)
                    if estimated_res is not None and len(estimated_res) == 2:
                        self.predicated_coordinates.append({'type': 'GR',
                                                            'lat': estimated_res[0], 'lon': estimated_res[1]})
                        break

    def estimate_by_mil_anchors(self):  # between + distance angle
        if self.anchors is None:
            return
        localities = []
        for place in self.context:
            if 'type' in place.keys() and 'region_adm' in place['type']:
                localities.append(place['name'])
        if len(localities) == len(self.context):
            localities = []
        for anchor in self.anchors:
            # s1: distance -> to a place
            if anchor['type'] == 'S1':
                start = None
                end_dir = None
                if toponym in localities:
                    continue
                toponym = anchor['anchor']
                dir_toponym = anchor['dir_anchor']
                d = distance.GeodesicDistance(kilometers=anchor['dis'])
                for place in self.context:
                    if place['name'] == toponym:
                        coords = place['coordinates'].split(',')
                        start = Point(float(coords[0]), float(s_lon))
                    elif place['name'] == dir_toponym:
                        coords = place['coordinates'].split(',')
                        end_dir = Point(float(coords[0]), float(s_lon))
                if start is not None and end_dir is not None:
                    bearing = calculate_bearing(start.latitude, end_dir.latitude, start.longitude, end_dir.longitude)
                    prediction = d.destination(point=start, bearing=bearing)
                    self.predicated_coordinates.append({'type': 'DAS1', 'lat': prediction.latitude,
                                                        'lon': prediction.longitude})
            # s2: distance in cardinal direction
            if anchor['type'] == 'S2':
                start = None
                bearing = (anchor['dir'] + 180) % 360
                d = distance.GeodesicDistance(kilometers=anchor['dis'])
                toponym = anchor['anchor']
                for place in self.context:
                    if place['name'] == toponym:
                        coords = place['coordinates'].split(',')
                        s_lat = float(coords[0])
                        s_lon = float(coords[1])
                        start = Point(s_lat, s_lon)
                        break
                if start is not None:
                    prediction = d.destination(point=start, bearing=bearing)
                    self.predicated_coordinates.append({'type': 'DAS2', 'lat': prediction.latitude,
                                                        'lon': prediction.longitude})

    @staticmethod
    def generic_in_locality(g, locality, level, mil_lat, mil_lon):
        val = None
        if g in unique_type_dict.keys():
            val = unique_type_dict[g]
        else:
            return None
        type_filtered = scale_regions[scale_regions['type'] == val]
        locality_filtered = type_filtered[type_filtered[level] == locality]
        selected = None
        min_distance = 10000000000
        for index, row in locality_filtered.iterrows():
            distance_value = distance.distance((mil_lat, mil_lon), (float(row['latitude']), float(row['longitude'])))
            if selected is None or min_distance >= distance_value:
                selected = [float(row['latitude']), float(row['longitude'])]
                min_distance = distance_value
        return selected


def generate_anchors(anchor_raw):
    anchors = []
    formatted_anchors = {}
    for a in anchor_raw:
        toponym = a['anchor']
        if toponym not in formatted_anchors.keys():
            formatted_anchors[toponym] = []
        formatted_anchors[toponym].append(a)
    for toponym, info in formatted_anchors.items():
        if len(info) == 2 and info[0]['type'] != info[1]['type']:  # S2
            if info[0]['type'] == 'distance':
                distance_info = info[0]
                angle = info[1]
            else:
                distance_info = info[1]
                angle = info[1]
            anchors.append({'type': 'S2', 'anchor': toponym, 'dir': angle['value'],
                            'dis': distance_info['value'] * parser.UNITS[distance_info['unit']]})
    return anchors


class EstimateScale:
    names = scale_regions['name'].astype(str).to_list()

    @staticmethod
    def get_scale(toponym):
        scale = None
        if toponym in EstimateScale.names:
            record = scale_regions[scale_regions['name'] == toponym]
            scale = record['scale']
        if scale is not None and len(scale) == 1:
            return scale.values[0]
        return None


def summarize_errors(coordinate_errors):
    summary = {}
    for e in coordinate_errors:
        c_error = e['error']
        for c_t_error, v_error in c_error.items():
            if c_t_error not in summary.keys():
                summary[c_t_error] = []
            summary[c_t_error].append(v_error)
    sorted_summary = {}
    for key, val in summary.items():
        if len(val) > 10:
            sorted_summary[key] = sorted(val)[:int(len(val) * 0.75)]
        else:
            sorted_summary[key] = sorted(val)
    average_summary = {}
    for key, val in sorted_summary.items():
        average_summary[key] = {'rate': sum(val) / len(val), 'len': len(val)}
    return average_summary


if __name__ == '__main__':
    logging.info('script started...')
    nz_gaz = Gazetteer()
    dataset = pd.read_csv('hackathon_data/locality_descriptions.csv')
    dataset.name = dataset.Locality.astype(str)

    # osm_types = pd.read_csv('hackathon_data/osm_types.csv')
    # type_names = osm_types['name'].astype(str).to_list()

    parse_model = parser.CPARSER()

    resolved_locations = []

    counter = 0

    coordinate_errors = []
    description_analysis_errors = []

    # testing
    # dataset_list = []
    # for i in range(2728):
    #     dataset_list.append(dataset)
    # dataset = pd.concat(dataset_list)

    for index, row in dataset.iterrows():
        counter += 1
        # try:
        if True:
            l_description = preprocessing(row['Locality'])
            ner_model = NER(l_description)
            places = ner_model.extract_place_names()

            # disambiguate
            disambiguation = Disambiguate(context=places, nz_gaz=nz_gaz)
            ambiguous = disambiguation.locations
            resolved = disambiguation.disambiguate()
            resolved_locations.append(resolved)

            # parse
            tree = parse_model.construct_tree(l_description)
            generics = tree.get_generic_nouns()
            place_types = [g for g in generics if g in unique_type_dict.keys()]
            labels = {'toponyms': places, 'place_types': place_types}
            tree.analyze(labels, l_description)
            logging.debug('Tree:\n{}'.format(tree))
            logging.debug(tree.anchors)

            locations = tree.locations
            anchors = generate_anchors(tree.anchors)
            logging.debug(anchors)

            # estimate
            estimate = EstimateLocation(unambigous_context=resolved, locality_description=l_description,
                                        lat=row['Lat'], lon=row['Long'], generic=place_types, locations=locations,
                                        anchors=anchors)
            estimate.estimate()
            estimate.calculate_errors()
            logging.debug(estimate.predicated_coordinates)
            logging.debug(estimate.error)
            coordinate_errors.append({'index': index, 'description': l_description, 'error': estimate.error})
        # except Exception:
        #    logging.error('exception in analyzing {0} counter {1}'.format(row, counter))
        #    description_analysis_errors.append(index)
        if counter % 100 == 0:
            print('counter: {}'.format(counter))
        if counter % 200 == 0:
            with open('hackathon_data/tmp-processing.json', 'w', encoding='utf-8') as fp:
                json.dump(coordinate_errors, fp)
            with open('hackathon_data/tmp-errors.json', 'w', encoding='utf-8') as fp:
                json.dump(description_analysis_errors, fp)
            print('count: {0}, errors {1}'.format(counter, len(description_analysis_errors)))
            print('coordinate prediciton error:\n\t {}'.format(summarize_errors(coordinate_errors)))
