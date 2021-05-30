"""
Get a path to XSD files in ORE convention and the name of the candidate schema file.
Run data description similarity between each of the XSD schemas to the target, return the result of a first line matcher
in ORE's CSV format Using the radius measure
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
from csv import reader

import fasttext
import fasttext.util
model = fasttext.load_model('/home/kobyb/fastText/cc.en.300.bin')
model.get_dimension()
import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm

fast_vocab = model.words

wmv_model = KeyedVectors.load_word2vec_format("oceanic_300_word2vec.bin", binary=True)
wmv_model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
print(wmv_model['phytoplankton'])

word_vectors = wmv_model.wv
vocab = word_vectors.vocab

# get schema from pangea XML file
def get_schema(xml_file):
    # get schema from xml file
    current_schema = []
    xml_tree = ET.parse(xml_file)
    xml_root = xml_tree.getroot()
    for parameter_name in xml_root.findall(prefix+'matrixColumn/'+prefix+'parameter/'+prefix+'name'):
        current_schema.append(parameter_name.text)
    return current_schema

# get data description from pangea XML file
def get_data_description(xml_file):
    # get data description from XML file.
    # The description in pangea API is given under the tag 'matrixColumn' for each parameter
    current_description = []
    xml_tree = ET.parse(xml_file)
    xml_root = xml_tree.getroot()
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

def get_ore_schema_xsd_element(xsd_file_path):
    # get schema from ORE XSD file
    with open(xsd_file_path, 'r', encoding='utf-8') as f:
        xsd_ele = etree.parse(f)
    return xsd_ele

def schema_names(xsd_ele):
    names = xsd_ele.xpath("//xs:element/@name", namespaces=namespaces)
    return names

def schema_descriptions(xsd_ele):
    descriptions = xsd_ele.xpath("//xs:element/@description", namespaces=namespaces)
    return descriptions

torch.device('cuda')

stop_words = stopwords.words('english')
prefix = 'http://www.w3.org/2001/XMLSchema'
namespaces = {"xs": "http://www.w3.org/2001/XMLSchema"}

candidate_files = []
# tar_file_name = 'my_target.xsd'
tar_file_name = 'user_target.xsd'
# ore_path = 'ore_xsd'
# ore_path = 'pangea/ore_format/'
ore_path = 'pangea/ore_format/240421/'
directory = os.fsencode(ore_path)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(tar_file_name):
        tar_file_name = filename
    elif filename.endswith(".xsd"):
        candidate_files.append(filename)

candidate_schemas = []
candidate_descriptions = []
for cand_file_name in candidate_files:
    cand_xsd = get_ore_schema_xsd_element(ore_path + '/' + cand_file_name)
    cand_schema = schema_names(cand_xsd)
    candidate_schemas.append(cand_schema)
    cand_description = schema_descriptions(cand_xsd)
    candidate_descriptions.append(cand_description)

target_xsd = get_ore_schema_xsd_element(ore_path + '/' + tar_file_name)
target_schema = schema_names(target_xsd)
target_description = schema_descriptions(target_xsd)

# print(candidate_descriptions[0])
def clean_desc(desc_list):
    # remove words that aren't in the vocabulary
    # desc = ' '.join(desc_list)
    # tok = WhitespaceTokenizer().tokenize(desc)
    word_list = []
    for word in desc_list:
        if word in word_vectors.vocab:
            word_list.append(word)
    # remove duplicates
    a = list(set(word_list))
    # return word_list
    return a

def centroid(sentence):
    words_vec = []
    words = WhitespaceTokenizer().tokenize(sentence)
    n = len(words)
    # for word in words:
        # words_vec.append(model.get_word_vector(word))
        # if word in wmv_model.vocab:
        #    words_vec.append(wmv_model[word])
    for word in words:
        if word in fast_vocab:
            words_vec.append(model.get_word_vector(word))
        ## words_vec.append(model.get_word_vector(word))
        # if word in wmv_model.vocab:
        #    words_vec.append(wmv_model[word])
    words_sum = np.sum(words_vec, axis=0)
    centroid = words_sum / n
    return centroid

def cluster_radius(w_vec, centroid):
    # print('cluster_radius')
    # print('words_vec: ',words_vec)
    # print('centroid: ', centroid)
    n = len(w_vec)
    words_vec = getVectorListFromWords(w_vec)
    try:
        radii_sum = np.sqrt((1 / n) * np.sum(np.square(1 - np.dot(words_vec, centroid) / (LA.norm(words_vec) * LA.norm(centroid))),axis=0))
    except ZeroDivisionError:
        return 1
    return radii_sum

def cosine_sim(A,B):
    return np.dot(A,B)/(norm(A)*norm(B))

def clusters_radius_measure(radii_sum1, radii_sum2):
    r = np.abs(radii_sum1 - radii_sum2)
    if np.isnan(r).any():
        r = 1.0
    return r

def elements_radius(centroid1, centroid2):
    r = 1 - cosine_sim(centroid1,centroid2)
    if np.isnan(r).any():
        r = 1.0
    return r

def getVectorListFromWords(words):
    words_vec = []
    for word in words:
        words_vec.append(model.get_word_vector(word))
    #    if word in wmv_model.vocab:
    #        words_vec.append(wmv_model[word])
    for word in words:
        if word in model.words:
            words_vec.append(model.get_word_vector(word))
        words_vec.append(model.get_word_vector(word))
    return words_vec

def compute_radius(text1, text2):
    # compute the radius measure according to a fasttext word embeddings.
    # t_text1 = WhitespaceTokenizer().tokenize(text1)
    # t_text2 = WhitespaceTokenizer.tokenize(text2)
    centroid1 = centroid(text1)
    centroid2 = centroid(text2)
    return elements_radius(centroid1, centroid2)

def compute_radius_clusters(text1, text2):
    # compute the radius measure for cousters according to a fasttext word embeddings.
    # t_text1 = WhitespaceTokenizer().tokenize(text1)
    # t_text2 = WhitespaceTokenizer.tokenize(text2)
    centroid1 = centroid(text1)
    cr1 = cluster_radius(getVectorListFromWords(text1), centroid1)
    centroid2 = centroid(text2)
    cr2 = cluster_radius(getVectorListFromWords(text2), centroid2)
    return clusters_radius_measure(cr1, cr2)

def compute_r_clusters_from_vectors(w_vector1, w_vector2, c1, c2):
    cr1 = cluster_radius(w_vector1, c1)
    cr2 = cluster_radius(w_vector2, c2)
    return clusters_radius_measure(cr1, cr2)

def compute_distance(text1, text2):
    # compute the distance between the text according to fasttext word embeddings
    centroid1 = centroid(text1)
    centroid2 = centroid(text2)
    r = cosine_sim(centroid1,centroid2)
    if np.isnan(r).any():
        r = 0.0
    return r

def centroid_radius(c1, c2):
    return 1 - cosine_sim(c1,c2)

def compute_sim_matrix(input_list, target_list):
    # compute similarity matrix for 2 data description lists.
    # return numpy array of similarities (1st line matcher)
    sim_matrix = []
    for input_attribute in input_list:
        row_value = []
        for output_attribute in target_list:
            if input_attribute == '' or output_attribute == '' or input_attribute is None or output_attribute is None:
                row_value.append(1.0)
            else:
                # dist = compute_radius(input_attribute, output_attribute)
                # dist = compute_radius_clusters(input_attribute, output_attribute)
                dist = compute_distance(input_attribute, output_attribute)
                row_value.append(dist)
        sim_matrix.append(row_value)
    arr = np.array(sim_matrix)
    return arr

def write_sim_matrix(cand_file_name, cand, target, sim):
    cand_onto_name = cand_file_name.strip('.xsd')
    target_onto_name = tar_file_name.strip('.xsd')
    with open(ore_path + '/' + cand_onto_name + '2' + target_onto_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(cand)):
            cand_element_name = cand_onto_name + '.' + cand[i]
            for j in range(len(target)):
                target_element_name = target_onto_name + '.' + target[j]
                writer.writerow([cand_element_name, target_element_name, sim[i, j]])

