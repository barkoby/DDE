"""
Get a path to XSD files in ORE convention and the name of the candidate schema file.
Run data description similarity between each of the XSD schemas to the target, return the result of a first line matcher
in ORE's CSV format
"""


import xml.etree.ElementTree as ET
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from flair.data import Sentence, Label
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, FlairEmbeddings
from typing import List
import torch
from flair.models import SequenceTagger
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk import download
from gensim.models import Word2Vec, KeyedVectors
import math
from sklearn import preprocessing
import csv
import os
import sys
from lxml import etree


# get schema from pangea XML file
def get_schema(xml_file):
    # get schema from xml file
    current_schema = []
    xml_tree = ET.parse(xml_file)
    xml_root = xml_tree.getroot()
    for parameter_name in xml_root.findall(prefix+'matrixColumn/'+prefix+'parameter/'+prefix+'name'):
        #print(parameter.text)
        current_schema.append(parameter_name.text)
    return current_schema


# get data description from pangea XML file
def get_data_description(xml_file):
    # get data description from XML file.
    # The description in pangea API is given under the tag 'matrixColumn' for each parameter
    current_description = []
    xml_tree = ET.parse(xml_file)
    xml_root = xml_tree.getroot()
    # for matrix_column in xml_root.findall(prefix + 'matrixColumn/'+prefix+'parameter'):
    for matrix_column in xml_root.findall(prefix + 'matrixColumn'):
        parameter_description = ''
        num_of_params = 0
        ignore_flag = False
        for parameter_tag in matrix_column.findall('.//'):
            #print(parameter_tag)
            if num_of_params > 0 and ignore_flag is False:
                # parameter_description += ', '
                parameter_description += ' '
            if parameter_tag.text != '\n' and parameter_tag.text != '':
                parameter_description += parameter_tag.text
                num_of_params += 1
                ignore_flag = False
            else:
                ignore_flag = True
        current_description.append(parameter_description)
    return current_description


# Runs Oceanic NER on a given text and return the entities
def get_sentence_ner(text):
    # run oceanic NER for the a given input text
    sentence = Sentence(text)
    model.predict(sentence)
    entities = []
    for entity in sentence.get_spans('ner'):
        m_string = str(entity)
        if any([m_string.startswith(onto_class) for onto_class in selected_ontology_classes]):
        # if any([m_string.startswith(onto_class) for onto_class in all_ontology_classes]):
            # get the specific entity for the selected ontology classes
            s = re.split('\\: \"\\b', m_string)[-1]
            s = s[:-1]
            entities.append(s)
    return entities


# gets DDE list from entities
def get_dde(description_list):
    dde_list = []
    for line in description_list:
        dde_list.append(get_sentence_ner(line))
    # dde_list = unique(dde_list)
    return dde_list


def get_similarity_matrix(input_list, target_list):
    # compute similarity matrix for 2 data description lists.
    # return numpy array of similarities (1st line matcher)
    sim_matrix = []
    for input_attribute in input_list:
        row_value = []
        for output_attribute in target_list:
            if input_attribute == '' or output_attribute == '' or input_attribute is None or output_attribute is None:
                row_value.append(0)
            else:
                dist = compute_wmd(input_attribute, output_attribute)
                row_value.append(dist)
        sim_matrix.append(row_value)
    arr = np.array(sim_matrix)
    # multiply the matrix by -1
    arr = -1 * arr
    # clean Nan values and normalize the matrix to a [0,1] range
    clean_matrix(arr)
    return arr


def prepare_list(t_list):
    if t_list is None or t_list == '':
        return ''
    else:
        current_str = list_to_string(t_list)
        current_str = remove_stop_words(current_str)
        current_str = list_to_string(current_str)
        return current_str


# remove stop words as given in NLTK English
def remove_stop_words(text):
    # Remove stopwords from an input string
    new_text = WhitespaceTokenizer().tokenize(text)
    # new_text = word_tokenize(text)
    new_text = [w for w in new_text if w not in stop_words]
    return new_text


def compute_wmd(text1, text2):
    # compute word movers distance according to an oceanic word2vec model.
    # note that the distance could be infinity for words not in the vocabulary
    distance = wmv_model.wmdistance(unique(text1), unique(text2))  # Compute WMD
    return distance


def list_to_string(lis):
    str1 = ''
    for ele in lis:
        str1 += str(ele)
        str1 += ' '
    return str1


def clean_matrix(a):
    # assign infinite values to 0 (no similarity) and normalize the matrix to a [0,1] scale
    try:
        max_a = np.nanmax(a[a != np.inf])
        print(max_a)
        min_a = np.nanmin(a[a != -np.inf])
        print(min_a)
    except:
        print('An exception occurred')
        print(a)
    # iterate the matrix and normalize the values of finite numbers, assign 0 to non-finite numbers
    for x in np.nditer(a, op_flags=['readwrite']):
        if math.isfinite(x):
            x[...] = (x-min_a)/(max_a-min_a)
        else:
            x[...] = 0


def get_ore_schema_xsd_element(xsd_file_path):
    # get schema from ORE XSD file
    with open(sys.argv[1] + '/' + xsd_file_path, 'r', encoding='utf-8') as f:
        xsd_ele = etree.parse(f)
    return xsd_ele


def schema_names(xsd_ele):
    names = xsd_ele.xpath("//xs:element/@name", namespaces=namespaces)
    return names


def schema_descriptions(xsd_ele):
    descriptions = xsd_ele.xpath("//xs:element/@description", namespaces=namespaces)
    return descriptions


def schema_column_indexes(xsd_ele):
    column_indexes = xsd_ele.xpath("//xs:element/@column_index", namespaces=namespaces)
    return column_indexes


def write_sim_matrix(cand_file_name, cand, target, sim):
    cand_onto_name = cand_file_name.strip('.xsd')
    target_onto_name = tar_file_name.strip('.xsd')
    with open(cand_onto_name + '2' + target_onto_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(cand)):
            cand_element_name = cand_onto_name + '.' + cand[i]
            for j in range(len(target)):
                target_element_name = target_onto_name + '.' + target[j]
                writer.writerow([cand_element_name, target_element_name, sim[i, j]])



def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def unique_words(my_string):
    words = my_string.split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    text = ''
    for unique_word in unique_words:
        text = text + ' ' + unique_word
    return text


torch.device('cuda')

stop_words = stopwords.words('english')
prefix = 'http://www.w3.org/2001/XMLSchema'
# prefix = '{http://www.pangaea.de/MetaData}'
namespaces = {"xs": "http://www.w3.org/2001/XMLSchema"}


# all_ontology_classes = ['B-DateTime','B-LatLon','B-Depth','B-Investigator','B-GeoRegion','B-Organization','B-PlatformType','B-PlatformName','B-MeasuredVariable','B-Unit','B-FundingAgency','B-Method','B-ProcessingType','B-Device','B-Program','B-DatasetID','B-Campaign']
all_ontology_classes = ['DateTime-span','LatLon-span','Depth-span','Investigator-span','GeoRegion-span','Organization-span','PlatformType-span','PlatformName-span','MeasuredVariable-span','Unit-span','FundingAgency-span','Method-span','ProcessingType-span','Device-span','Program-span','DatasetID-span','Campaign-span']

# only classes from the following list will be selected
selected_ontology_classes = ['DateTime-span', 'LatLon-span', 'Depth-span', 'GeoRegion-span', 'MeasuredVariable-span', 'Unit-span', 'Method-span', 'ProcessingType-span', 'Device-span']

wmv_model = KeyedVectors.load_word2vec_format("oceanic_300_word2vec.bin", binary=True)
wmv_model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
model = SequenceTagger.load_from_file('resources/final-model.pt')

if len(sys.argv) != 3:
    # invalid command line arguments were given, throw exception and exit
    raise Exception('invalid arguments were given')
directory = os.fsencode(sys.argv[1])
candidate_files = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(sys.argv[2]):
        tar_file_name = filename
    elif filename.endswith(".xsd"):
        candidate_files.append(filename)
        print(filename)

target_xsd = get_ore_schema_xsd_element(tar_file_name)
target_schema = schema_names(target_xsd)
target_description = schema_descriptions(target_xsd)
# target_description = [['Depth'], ['Date'], ['Latitude'], ['Longitude'], ['Temperature'], ['Practical salinity'], ['Pressure half-tide pressure sensor'], ['oxygen'], ['phytoplankton flow cytometry'], ['Wind speed'], [['Wind direction']]]

candidate_schemas = []
candidate_descriptions = []
for cand_file_name in candidate_files:
    cand_xsd = get_ore_schema_xsd_element(cand_file_name)
    cand_schema = schema_names(cand_xsd)
    candidate_schemas.append(cand_schema)
    cand_description = schema_descriptions(cand_xsd)
    candidate_descriptions.append(cand_description)


# read XML files
# target_xml_file_name = '897335_metadata.xml'
# target_xml_file_name = '904040_metadata.xml'
# input_xml_file_name = '897335_metadata.xml'

# DEBUG
with open('debug_dde_110120.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)

    sim_matrix_list = []
    target_dde = get_dde(target_description)
    target_dde[1] = ['Date']
    target_dde[10] = ['Wind Direction']

    csv_writer.writerow(['target', target_description, target_dde, unique(target_dde)])
    # print(target_dde)
    i = 0
    for candidate in candidate_descriptions:
        candidate_dde = get_dde(candidate)
        # print(candidate)
        # print(target_dde)
        sim = get_similarity_matrix(candidate_dde, target_dde)
        sim_matrix_list.append(sim)
        # print(candidate_dde)
        csv_writer.writerow([candidate_files[i], candidate, candidate_dde, unique(candidate_dde)])
        i = i+1


for k in range(len(candidate_files)):
    current_sim = sim_matrix_list[k]
    current_cand_name = candidate_files[k]
    current_candidate_schema = candidate_schemas[k]
    write_sim_matrix(current_cand_name, current_candidate_schema, target_schema, current_sim)

"""
# fixed_input_dde = []
# fixed_target_dde = []
for entry in input_dde:
    fixed_input_dde.append(prepare_list(entry))
for entry in target_dde:
    fixed_target_dde.append(prepare_list(entry))
"""

"""
sim = get_similarity_matrix(input_dde, target_dde)
print(sim)

with open('similarity_matrix.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(input_schema)):
        for j in range(len(target_schema)):
            writer.writerow([input_schema[i], target_schema[j], sim[i,j]])

"""
