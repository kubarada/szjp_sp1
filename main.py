from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import os
import numpy as np

folder_path = 'szjp_sp1-main/documents'

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


with open('szjp_sp1-main/query_devel.xml', 'r') as f:
    html_doc = f.read()
    soup = BeautifulSoup(html_doc, 'html.parser')

    # Get the text from the <ALL> tag
    doc_text = soup.find('all').get_text()

# Print the plain text
doc_lines = doc_text.splitlines()
doc_lines = [x for x in doc_lines if x != '']
doc_lines = [x for x in doc_lines if x != ' ']

result = []
current_text = ''
for item in doc_lines:
    if item.strip().isdigit():
        if current_text != '':
            result.append(current_text.strip())
            current_text = ''
        result.append(item.strip())
    else:
        current_text += ' ' + item

if current_text != '':
    result.append(current_text.strip())

print(result)

query_dict = {}
for i in range(0, len(result), 2):
    key = result[i]
    value = result[i+1]
    query_dict[key] = value

print(query_dict)
query_list = query_dict.values()
print(query_list)


q=tfidf_1.transform(query_list)       ### !!

print(index)
dense_q=q.toarray()
print(dense_q)


