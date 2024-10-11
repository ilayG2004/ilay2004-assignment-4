from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
def create_term_document_matrix(query, documents):
    #Think of query as a list ["machine","learning", "bicycle"]
    '''  machine        learning        bicycle
    doc1     0           1           2
    doc2     1           0           2
    doc3     0           0           0
    '''
    frequency_array = []
    for doc in documents:
        words = doc.lower().split()
        #subarray for the document i, of how many times each word appears
        #Assuming query is list. Within the entire document count the number of times a word in query appears
        frequency_sublist = [words.count(word) for word in query]
        frequency_array.append(frequency_sublist)
    term_document_matrix = np.array(frequency_array)
    return term_document_matrix

def svd(tdm, rank):
    T,S,D=np.linalg.svd(tdm,full_matrices=False)
    '''
    T: Term-to-concept matrix (left singular vectors).
    Σ: A diagonal matrix of singular values, which represent the importance of the latent concepts.
    D: Document-to-concept matrix (right singular vectors).
    '''
    T_reduced = T[:, :rank] #doc-to-concept similarity
    S_reduced = np.diag(S[:rank]) #'strength' of each concept
    D_reduced = D[:rank, :] #term-to-concept similarity

    compressed_tdm = np.dot(T_reduced, np.dot(S_reduced,D_reduced))
    
    return compressed_tdm, T_reduced, S_reduced, D_reduced

def project_query_to_latent_space(query, T_reduced):
    #Frequency of each word across all documents
    query_frequency = np.array([query.count(word) for word in query])
    query_frequency = query_frequency.reshape(1, -1)  # Convert to 2D (1, n)
    # Project the query frequency into the latent space
    query_projected = np.dot(query_frequency, T_reduced)
    
    return query_projected

def find_relevant_documents(query, T_reduced, D_reduced):
    #Project the query into the latent space
    query_projected = project_query_to_latent_space(query, T_reduced)
    
    #Calculate cosine similarity between the projected query (number of times a vocab word appears) 
    #in the document to measure relevance of document
    similarities = cosine_similarity([query_projected], D_reduced).flatten()

    return similarities

def get_top_relevant_documents(similarities, documents, top_k=5):
    #Indices of the top 5 similar documents
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    #Retrieve the documents based on the top indices
    relevant_docs = [(index, documents[index], similarities[index]) for index in top_indices]
    
    return top_indices

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    newsgroups = fetch_20newsgroups(subset='all')
    documents = newsgroups.data

    words = query.split() #Parse the query string into a list to be read later

    tdm = create_term_document_matrix(words, documents)
    compressed_tdm, T_reduced, S_reduced, D_reduced = svd(tdm, 110)
    similarities = find_relevant_documents(words, T_reduced, D_reduced)
    relevant_docs = get_top_relevant_documents(similarities,documents, 5)

    return documents, similarities, relevant_docs

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
