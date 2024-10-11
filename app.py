from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
data = newsgroups.data
vectorizer = TfidfVectorizer(stop_words='english') 
term_doc_matrix = vectorizer.fit_transform(data)

'''         machine     learning       bicycle
    doc1    0           1               2
    doc2    1           0               2
    doc3    0           5               0
'''

def lsa(tdm, num_components):
    svd_model = TruncatedSVD(n_components=num_components)
    svd_matrix = svd_model.fit_transform(tdm)

    #'returns a vector of variance explained by each dimension'
    explained_variance = svd_model.explained_variance_ratio_
    return svd_matrix, explained_variance, svd_model

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    #Perform lsa on term-document-matrix ()
    svd_matrix, explained_variance, svd_model = lsa(term_doc_matrix, 110)
    
    query_matrix = vectorizer.transform([query])
    query_lsa = svd_model.transform(query_matrix)

    #Have the lsa sparse matricies of all documents plus queries. Use cosine similarity to find the similarities to return top n documents
    similarities = cosine_similarity(query_lsa, svd_matrix)
    #Flatten similarities into 1D array so it can be read and sorted properly later
    similarities = similarities.flatten()


    #Take the result from LSA and return a list of the top 10 documents
    #array of integers of top 10 document indecies
    top_doc_indicies = np.argsort(similarities)[-10:][::-1]
    top_documents = []

    for i in top_doc_indicies.tolist():
        top_documents.append(data[i])
    similarities = similarities[top_doc_indicies]

    return top_documents, similarities.tolist(), top_doc_indicies.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
