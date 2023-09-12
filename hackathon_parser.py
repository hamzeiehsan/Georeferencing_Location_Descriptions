import logging
import re

import anytree.cachedsearch as search
from allennlp.predictors.predictor import Predictor
from anytree import AnyNode, RenderTree, PostOrderIter

logging.basicConfig(level=logging.ERROR)

UNITS = {'meters': 0.001, 'kilometers': 1, 'miles': 1.609, 'mile': 1.609, 'meter': 0.001, 'kilometer': 1,
         'km': 1, 'm': 0.001}

DIRECTIONS = {'above': 0, 'below': 180, 'north': 0, 'south': 180, 'east': 270, 'west': 90, 'southeast': 225,
              'northeast': 315, 'southwest': 135, 'northwest': 45, 'SE': 225, 'SW': 135, 'NE': 315, 'NW': 45,
              'West': 90, 'North': 0, 'East': 270, 'South': 180, 'NorthEast': 315, 'North-east': 315, 'North-East': 315,
              'NorthWest': 45, 'North-West': 45, 'North-west': 45, 'SouthEast': 225, 'South-East': 225,
              'South-east': 225, 'SouthWest': 135, 'South-West': 135, 'South-west': 135}

parsemodel = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

spatiotemporal_propositions = ['in', 'of', 'on', 'at', 'within', 'from', 'to', 'near', 'close', 'between', 'beside',
                               'by', 'since', 'until', 'before', 'after', 'close to', 'near to', 'closest to',
                               'nearest to', 'above', 'around', 'below', 'about', 'along']

prepositions_type = {'in': 1, 'inside': 1, 'of': 4, 'on': 3, 'at': 3, 'within': 4, 'from': 4, 'to': 3,
                     'near': 4, 'close': 4, 'between': 2, 'beside': 4, 'close to': 4, 'near to': 4,
                     'closest to': 4, 'nearest to': 4, 'above': 3, 'around': 3, 'below': 3, 'about': 2,
                     'along': 3}

ENCODINGS = dict(
    {'toponyms': 'P', 'place_types': 'p', 'spatial_relationship': 'r'})


def is_overlap(position, indices):
    for index in indices:
        if position[0] >= index[0] and position[1] <= index[1]:
            return True
    return False


class CPARSER:
    @staticmethod
    def parse(sentence):
        res = parsemodel.predict(sentence)
        return res['hierplane_tree']['root']

    @staticmethod
    def construct_tree(sentence):
        parse_results = CPARSER.parse(sentence)
        return ConstituencyParser(parse_results)


class ConstituencyParser:
    def __init__(self, parse_dict):
        self.parse_dict = parse_dict
        self.tree = None
        self.root = None
        self.construct_tree()
        self.locations = []
        self.anchors = []

    def construct_cleaning_labels(self, results, question):
        orders = ['toponyms', 'place_types']
        indices = []
        labelled = {}
        for order in orders:
            values = results[order]
            role = ENCODINGS[order]
            for v in values:
                temp = v
                if temp not in question:
                    temp = temp.replace(" 's", "'s")
                matches = re.finditer(temp, question)
                matches_positions = [[match.start(), match.end()] for match in matches]
                for position in matches_positions:
                    if not is_overlap(position, indices):
                        labelled[v + '--' + str(position[0])] = {'start': position[0],
                                                                 'end': position[1],
                                                                 'role': role,
                                                                 'pos': 'NOUN'}
                indices.extend(matches_positions)
        return labelled

    def construct_tree(self):
        root = AnyNode(name=self.parse_dict['word'], nodeType=self.parse_dict['nodeType'], role='',
                       spans={'start': 0, 'end': len(self.parse_dict['word'])})
        if 'children' in self.parse_dict.keys():
            for child in self.parse_dict['children']:
                self.add_to_tree(child, root)
        self.root = root
        self.tree = RenderTree(root)

    def add_to_tree(self, node, parent):
        local_start = parent.name.find(node['word'])
        n = AnyNode(name=node['word'], nodeType=node['nodeType'], parent=parent, role='',
                    spans={'start': parent.spans['start'] + local_start,
                           'end': parent.spans['start'] + local_start + len(node['word'])})
        if 'children' in node.keys():
            for child in node['children']:
                self.add_to_tree(child, n)

    def render(self):
        self.tree = RenderTree(self.root)

    def __repr__(self):
        if self.tree is None:
            return "Empty Tree"
        res = ""
        for pre, fill, node in self.tree:
            res += "%s%s (%s) {%s}" % (pre, node.name, node.nodeType, node.role) + "\n"
        return res

    def label_tree(self):
        self.clean_tree()
        res = self.label_conjunctions()
        res = {**res, **self.label_non_platial_objects()}
        res = {**res, **self.label_numbers()}
        self.update()
        return res

    def find_node_by_exact_name(self, string):
        return search.findall_by_attr(self.root, string)

    def find_node_by_name(self, string):
        res = self.find_node_by_exact_name(string)
        if len(res) > 0:
            return res
        return search.findall(self.root, filter_=lambda node: node.name in string.split())

    def label_role(self, name, role, clean=False):
        nodes = self.find_node_by_name(name)
        if len(nodes) == 1:
            nodes[0].role = role
            if clean:
                nodes[0].children = []
        else:
            min_depth = 1000
            selected = None
            for node in nodes:
                if node.depth < min_depth and node.name not in ['of']:
                    min_depth = node.depth
                    selected = node
                else:
                    node.parent = None
                selected.name = name
                selected.spans = {'start': self.root.name.index(name), 'end': self.root.name.index(name) + len(name)}
                selected.nodeType = 'NP'
                selected.role = role

    def clean_tree(self):
        named_objects = search.findall(self.root, filter_=lambda node: node.role in ("P", "p"))
        for named_object in named_objects:
            if len(named_object.siblings) == 1 and (named_object.siblings[0].nodeType == 'DT'):
                named_object.parent.role = named_object.role
                named_object.parent.name = named_object.name
                named_object.parent.nodeType = named_object.nodeType
                named_object.parent.children = []
            elif len(named_object.siblings) == 1 and named_object.siblings[0].role == named_object.role:
                named_object.parent.role = named_object.role
            elif len(named_object.siblings) == 2 and ',' in named_object.parent.name:
                for sibling in named_object.siblings:
                    if sibling.role == named_object.role:
                        named_object.parent.role = named_object.role
                        break

    def label_spatiotemporal_relationships(self):
        named_objects = search.findall(self.root, filter_=lambda node: node.role in ("P", "p"))
        res_relationships = {}
        for named_object in named_objects:
            for sibling in named_object.siblings:
                if sibling.nodeType == 'IN' and \
                        sibling.name in spatiotemporal_propositions:
                    sibling.role = 'R'
                    if sibling.name + '--' + str(sibling.spans['start']) not in res_relationships.keys():
                        res_relationships[sibling.name + '--' + str(sibling.spans['start'])
                                          ] = {'start': sibling.spans['start'],
                                               'end': sibling.spans['end'], 'role': 'R', 'pos': 'ADP'}
                    named_object.parent.role = 'LOCATION'
        self.resolve_relations()
        return res_relationships

    def resolve_relations(self):
        locations = search.findall(self.root, filter_=lambda node: node.role == 'LOCATION')

        for location in locations:
            found = False
            rel = None
            anchor = None
            for child in location.children:
                if child.role == 'R':
                    found = True
                    rel = child.name
                if found and child.role == 'P':
                    temp = child
                    while len(temp.children) != 0:
                        for c_child in temp.children:
                            if c_child.role == 'P':
                                temp = c_child
                                if len(temp.children) == 0:
                                    break
                    if temp.role == 'P' and len(temp.children) == 0:
                        anchor = temp
                        break
            if anchor is not None:
                self.locations.append({rel: anchor.name})

        if len(self.locations) > 0:
            self.distance_and_angles()

    def distance_and_angles(self):
        string = self.root.name
        anchors = []
        distances = []
        angles = []
        for key, value in UNITS.items():
            if ' ' + key + ' ' in string:
                distances.append(key)

        if len(distances) > 0:
            for key, value in DIRECTIONS.items():
                if ' ' + key + ' ' in string:
                    angles.append(key)

            distance_nodes = search.findall(self.root, filter_=lambda node: node.name in distances and
                                                                            len(node.children) == 0)
            angle_nodes = search.findall(self.root, filter_=lambda node: node.name in angles and
                                                                         len(node.children) == 0)

            places = search.findall(self.root, filter_=lambda node: node.role == 'P' and len(node.children) == 0)
            span_places = {}
            for place in places:
                span_places[place.spans['start']] = place.name
            spans = sorted(list(span_places.keys()))
            for distance_node in distance_nodes:
                for span in spans:
                    if distance_node.spans['start'] < span:
                        numbers = search.findall(distance_node.parent, filter_=lambda node: node.role == 'n')
                        number = 1
                        if len(numbers) > 0:
                            try:
                                number = numbers[0].name
                            except:
                                logging.info('error in converting {} to float'.format(numbers[0].name))
                        anchors.append({'type': 'distance', 'unit': distance_node.name, 'value': number,
                                        'anchor': span_places[span]})
                        break
            for angle_node in angle_nodes:
                for span in spans:
                    if angle_node.spans['start'] < span:
                        anchors.append({'type': 'angle', 'value': DIRECTIONS[angle_node.name],
                                        'anchor': span_places[span]})
                        break
        logging.debug('distance: {0} and angles:{1}'.format(distances, angles))
        self.anchors = anchors
        logging.debug(anchors)

    def all_encodings(self):
        res = {}
        roles = search.findall(self.root, filter_=lambda node: node.role != '')
        for role in roles:
            key = role.role
            val = role.name
            if key not in res.keys():
                res[key] = []
            res[key].append(val)
        return res

    def label_complex_spatial_relationships(self, prep, pattern):
        matched = False
        context = prep.parent
        text = ''
        while not matched:
            regex_search = re.search(pattern.strip(), context.name)
            if regex_search is not None:
                matched = True
                text = context.name[regex_search.regs[0][0]: regex_search.regs[0][1]]
                break
            if context.parent is None:
                break
            context = context.parent
        if matched:
            if context.name == text:
                context.role = 'R'
            else:
                nodes = ConstituencyParser.iterate_and_find(context, text)
                new_node = AnyNode(name=text, nodeType='IN', role='R', spans={'start': nodes[0].spans['start'],
                                                                              'end': nodes[len(nodes) - 1].spans[
                                                                                  'end']})
                before = []
                after = []

                firstparent = nodes[0].parent
                if firstparent != context:
                    for child in context.children:
                        if self.root.name.index(child.name) + len(child.name) <= self.root.name.index(text):
                            before.append(child)

                for child in firstparent.children:
                    if child in nodes:
                        break
                    before.append(child)

                lastparent = prep.parent
                for child in lastparent.children:
                    if child not in nodes:
                        after.append(child)
                while lastparent != context:
                    lastparent = lastparent.parent
                    for child in lastparent.children:
                        if self.root.name.index(text) + len(text) <= self.root.name.index(child.name):
                            after.append(child)
                context.children = []
                for b in before:
                    b.parent = context

                for node in nodes:
                    node.parent = new_node

                new_node.parent = context

                for a in after:
                    a.parent = context

    @staticmethod
    def iterate_and_find(node, text):
        res = []
        for child in node.children:
            if child.name in text:
                res.append(child)
                text = text.replace(child.name, '', 1)
            elif text.strip() != '':
                res.extend(ConstituencyParser.iterate_and_find(child, text))
        return res

    def clean_locations(self):
        named_objects = search.findall(self.root, filter_=lambda node: node.role == 'LOCATION')
        if len(named_objects) == 2:
            if named_objects[0].depth < named_objects[1].depth:
                if self.root.name.index(named_objects[0].name) < self.root.name.index(named_objects[1].name):
                    ConstituencyParser.merge(node1=named_objects[0], node2=named_objects[1])
                else:
                    ConstituencyParser.merge(node1=named_objects[0], node2=named_objects[1], order=False)
            else:
                if self.root.name.index(named_objects[0].name) < self.root.name.index(named_objects[1].name):
                    ConstituencyParser.merge(node1=named_objects[1], node2=named_objects[0], order=False)
                else:
                    ConstituencyParser.merge(node1=named_objects[1], node2=named_objects[0])

    def clean_phrases(self):
        single_child_nodes = search.findall(self.root, filter_=lambda node: len(node.children) == 1)
        for node in single_child_nodes:
            try:
                if node.role == '':
                    node.role = node.children[0].role
                node.nodeType = node.children[0].nodeType
                children = node.children[0].children
                node.children[0].parent = None
                node.children = children
            except:
                logging.error('error in cleaning...')

        incorrect_types = search.findall(self.root, filter_=lambda node: len(node.children) > 0 and
                                                                         node.role in ['p', 'P'])
        for it in incorrect_types:
            if len(search.findall(it, filter_=lambda node: node != it and node.role in ['p', 'P'])) == 0:
                it.role = ''

    @staticmethod
    def merge(node1, node2, order=True):
        node = None
        start = min(node1.spans['start'], node2.spans['start'])
        end = max(node1.spans['end'], node2.spans['end'])
        if order:
            node = AnyNode(name=node1.name + ' ' + node2.name, nodeType=node1.nodeType, role=node1.role,
                           spans={'start': start, 'end': end})
        else:
            node = AnyNode(name=node2.name + ' ' + node1.name, nodeType=node1.nodeType, role=node1.role,
                           spans={'start': start, 'end': end})
        node.parent = node1.parent
        if order:
            node1.parent = node
            node2.parent = node
        else:
            node2.parent = node
            node1.parent = node

    def update(self):
        for node in PostOrderIter(self.root):
            if len(node.children) > 0:
                name = ''
                for child in node.children:
                    name += child.name + ' '
                if node.name != name:
                    node.name = name.strip()
                if len(node.children) == 1 and (node.role == '' or node.role == node.children[0].role) and \
                        node.nodeType == node.children[0].nodeType:
                    node.role = node.children[0].role
                    node.children = node.children[0].children

    def label_numeric_values(self):
        nodes = search.findall(self.root, filter_=lambda node: node.nodeType == 'CD' and node.role == '' and
                                                               len(node.children) == 0)
        if len(nodes) > 0:
            for node in nodes:
                node.role = 'n'
            self.label_numbers()

    def label_numbers(self):
        numbers = search.findall(self.root, filter_=lambda node: node.role == '' and node.nodeType == 'CD')
        units = {}
        for num in numbers:
            num.role = 'n'
            check = False
            added = False
            for sibling in num.parent.children:
                if sibling == num:
                    check = True
                elif check and sibling.name in UNITS.keys():
                    if num.parent.role == '':
                        num.parent.role = 'MEASURE'
                    if num.name + ' ' + sibling.name in self.root.name:
                        units[num.name + ' ' + sibling.name + '--' + str(num.spans['start'])] = {
                            'start': num.spans['start'],
                            'end': sibling.spans['end'] + 1,
                            'role': 'n',
                            'pos': 'NUM'}
                        added = True
            if not added and num.parent.nodeType == 'QP' and num.parent.parent is not None:
                found = False
                for child in num.parent.parent.children:
                    if child == num.parent:
                        found = True
                    elif found and child.name in UNITS.keys():
                        new_node = AnyNode(child.parent, role='MEASURE', name=num.name + ' ' + child.name,
                                           nodeType='NP', spans={
                                'start': self.root.name.index(num.name + ' ' + child.name),
                                'end': self.root.name.index(num.name + ' ' + child.name) +
                                       len(num.name + ' ' + child.name)
                            })
                        num.parent = new_node
                        child.parent = new_node
                        units[new_node.name + '--' + str(new_node.spans['start'])] = {'start': new_node.spans['start'],
                                                                                      'end': new_node.spans['end'],
                                                                                      'role': 'n',
                                                                                      'pos': 'NUM'
                                                                                      }
            else:
                units[num.name + '--' + str(num.spans['start'])] = {'start': num.spans['start'],
                                                                    'end': num.spans['end'],
                                                                    'role': 'n',
                                                                    'pos': 'NUM'
                                                                    }
        return units

    def analyze(self, place_and_types, sentence):
        pnts = self.construct_cleaning_labels(place_and_types, sentence)
        for k, v in pnts.items():
            self.label_role(re.split('--', k.strip())[0], v['role'], clean=True)
        self.clean_tree()
        self.label_spatiotemporal_relationships()
        self.label_numeric_values()
        self.label_numbers()

    def get_generic_nouns(self):
        nodes = search.findall(self.root, filter_=lambda node: (node.nodeType == 'NN' or node.nodeType == 'NNS' or
                                                                node.nodeType == 'NNP') and len(node.children) == 0)
        generics = []
        for node in nodes:
            if node.name.islower():
                generics.append(node.name)
        return generics

    @staticmethod
    def context_builder(list_str, node):
        boolean_var = True
        for string in list_str:
            boolean_var = boolean_var and string in node.name  # multi-word?
        return boolean_var

    def search_context(self, list_str):
        nodes = search.findall(self.root, filter_=lambda node: ConstituencyParser.context_builder(list_str, node))
        max_depth = -1
        selected = None
        for node in nodes:
            if node.depth > max_depth:
                max_depth = node.depth
                selected = node
        return selected

    @staticmethod
    def valid_node_selection(nodes, valid_pos_tags, invalid_tags):
        if len(nodes) == 1:
            return nodes[0]
        max_depth = -1
        selected = None
        for node in nodes:
            invalid_child = search.findall(node, filter_=lambda child: child != node and child.nodeType in invalid_tags)
            if len(invalid_child) == 0 and node.nodeType in valid_pos_tags and max_depth < node.depth:
                max_depth = node.depth
                selected = node
        return selected

    @staticmethod
    def find_exact_match(context, name):
        matches = search.findall(context, filter_=lambda node: node.name == name)
        max_depth = 1000
        selected = None
        for match in matches:
            if max_depth > match.depth:
                max_depth = match.depth
                selected = match
        selected.children = []
        return selected


if __name__ == '__main__':
    c_parser = CPARSER()
    sentences = ['Buller, Paparoa Mountains, north flank of Mt Euclid, c. 1-1.5km east of Morgan Tarn.',
                 'Auckland Island, lower slopes about Musgrave Inlet',
                 'Nelson, about l km SE of Lake Peel, in the track to Balloon Hut',
                 'Marlborough, hills about Queen Charlotte Sound',
                 'Lake Ellesmere Spit  = Kaitorete Spit  – About Midway along length.',
                 'Mokaikai Scenic Reserve, above Whareana Bay,  North Cape',
                 'Heaphy Track, slopes above Perry Saddle.',
                 'Canterbury, Loburn area, White Rock, above Karetu River',
                 'Lookout above Huia, Waitakere Ranges',
                 'Canterbury, Banks Peninsula: above Breeze Bay, Lyttleton Harbour.',
                 'Hunua, bush above Hunua Falls']
    for sentence in sentences:
        constituencies = c_parser.construct_tree(sentence)
        print(sentence)
        print(constituencies)
        print(constituencies.get_generic_nouns())
        print('\n\n')
