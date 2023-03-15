from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import os
import xml.etree.ElementTree as ET
import re

folder_path = 'documents'

# List all files in the folder
files = os.listdir(folder_path)
data = list()
# Loop through each file in the folder and read its contents
for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r') as f:
        html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        pre_tag = soup.find('pre')
        text = pre_tag.get_text().replace('\n', ' ').replace('\t', '').split(' PM ', 1)[0]
        text = text.split(' AM ', 1)[0]
        data.append(text)

tfidf_1 = TfidfVectorizer(norm=None,use_idf=True,smooth_idf=False) # use_idf=True

sparse_doc_term_matrix_1 = tfidf_1.fit_transform(data)
dense_doc_term_matrix_1 = sparse_doc_term_matrix_1.toarray()
index = tfidf_1.get_feature_names_out()

print(index)
print(dense_doc_term_matrix_1)

with open('query_devel.xml', 'r') as f:
    data = f.read()

# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")

# Finding all instances of tag
# `unique`
b_unique = Bs_data.find_all()

print(str(b_unique[1]).replace('<DOCNO> ' + str(1) + ' </DOCNO>', '').split('\n'))
print(type(b_unique))

# Using find() to extract attributes
# of the first instance of the tag
# b_name = Bs_data.find('DOC')
#
# print(b_name)

# Extracting the data stored in a
# specific attribute of the
# `child` tag
