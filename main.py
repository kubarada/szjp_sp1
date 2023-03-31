from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import os
import numpy as np

folder_path = './documents'

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

with open('./query_devel.xml', 'r') as f:
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

query_dict = {}
for i in range(0, len(result), 2):
    key = result[i]
    value = result[i+1]
    query_dict[key] = value


query_list = query_dict.values()

tfidf = TfidfVectorizer(norm=None,use_idf=True,smooth_idf=True,sublinear_tf=True,max_df=0.65)
sparse_doc_term_matrix = tfidf.fit_transform(data)

dense_doc_term_matrix = sparse_doc_term_matrix.toarray()
index = tfidf.get_feature_names_out()


f_output = open("vystup_vyhledavaciho_programu.txt", "w")

q = tfidf.transform(query_list) ### !!
dense_q=q.toarray()
sim = cosine_similarity(sparse_doc_term_matrix, q)
sim_T = sim.transpose()
for i in range(len(sim_T)):
    indexes = np.ndarray.argsort(sim_T[i])[-100:][::-1]

    for j in indexes:
        doc_name = "CACM-" + str(j + 1).zfill(4)
        doc_eval = str(sim_T[i][j])
        f_output.write(str(i+1) + "\t" + doc_name + "\t" + doc_eval + "\n")

