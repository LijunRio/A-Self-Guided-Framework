import json
import random
import numpy as np
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from TFIDF import ClassTFIDF
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from KMVE_RG.config import config as args


def _preprocess_text(documents):
    """ Basic preprocessing of text
    Steps:
        * Lower text
        * Replace \n and \t with whitespace
        * Only keep alpha-numerical characters
    """
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
    cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
    return cleaned_documents


def get_item(cls, ann, sentence_all):
    examples = ann[cls]  # split-> train valid test
    print(cls, '_len:', len(examples))
    for i in range(len(examples)):
        sentence = examples[i]['report']
        sentence_all.append({cls + '_' + str(i): sentence})


def get_all_data(split, ann_path):
    ann = json.loads(open(ann_path, 'r').read())  # load json格式的注释
    sentence_all = []
    for cls in split:
        get_item(cls, ann, sentence_all)
    print('sentence_alllen:', len(sentence_all))
    data = []

    for item in sentence_all:
        sentence = list(item.values())[0]
        data.append(sentence)
    print('data:', len(data))
    return data, sentence_all, ann


def _check_class_nums(topics, topic_model):
    # check the class nums
    cls_num = {}
    for item in topics:
        if item not in cls_num:
            cls_num.update({item: str(item)})

    result = len(cls_num) == topic_model.get_topic_info().shape[0]
    assert result is True, 'cls_nums need to equal to topic_model.get_topic_info().shape'


def shuffle_result(topics, topic_model, ann, data, all_sentence, shuffle=False):
    _check_class_nums(topics, topic_model)
    all_data = []
    for i in range(len(data)):
        label = topics[i] + 1  # 不要-1的标签，全部从0开始
        key_list = list(all_sentence[i].keys())[0].split('_')
        origin = ann[key_list[0]][int(key_list[1])]
        origin.update({'label': label})
        all_data.append(origin)
    if shuffle is True:
        random.shuffle(all_data)
        print('shuffle data complieted !')
    return all_data


ann_path = args.ann_path
split = ['train', 'val', 'test']
data, all_sentence, origin_ann = get_all_data(split=split, ann_path=ann_path)

embedding_method = SentenceTransformer("paraphrase-MiniLM-L6-v2")
documents = pd.DataFrame({"Document": data,
                          "ID": range(len(data)),
                          "Topic": None})
embeddings = embedding_method.encode(data)

# UMAP algorithm settings
umap_model = umap.UMAP(n_neighbors=15,
                       n_components=2,
                       min_dist=0.0,
                       metric='cosine',
                       low_memory=False)

hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15,
                                metric='euclidean',
                                cluster_selection_method='eom',
                                prediction_data=True)

# Reduction
umap_model.fit(embeddings, y=None)
umap_embeddings = umap_model.transform(embeddings)
new_embeddings = np.nan_to_num(umap_embeddings)

# Clustering
hdbscan_model.fit(umap_embeddings)
documents['Topic'] = hdbscan_model.labels_
probabilities = hdbscan_model.probabilities_
sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
topic_size = dict(zip(sizes.Topic, sizes.Document))

# Calculate Similarity
n_gram_range = (1, 1)
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
vectorizer_model = CountVectorizer(ngram_range=n_gram_range)
documents = _preprocess_text(documents_per_topic.Document.values)
vectorizer_model.fit(documents)
words = vectorizer_model.get_feature_names()
X = vectorizer_model.transform(documents)
m = len(documents)
transformer = ClassTFIDF().fit(X, n_samples=m, multiplier=None)
c_tf_idf = transformer.transform(X)
topic_sim_matrix = cosine_similarity(c_tf_idf)
# topic_sim_matrix = cosine_similarity(X)

sns.set_theme()
ax = sns.heatmap(topic_sim_matrix, cmap="GnBu")
sns.set(font_scale=10)
print('complete')
plt.show()
