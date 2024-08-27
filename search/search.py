import nltk
import json
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

def find_most_relevant_page(query, json_file='../output/results.json'):
    stop_words = set(stopwords.words('english'))
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = [item['content'] for item in data]

    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    most_relevant_idx = cosine_similarities.argmax()

    if cosine_similarities[most_relevant_idx] > 0:
        data[most_relevant_idx]['score'] = cosine_similarities[most_relevant_idx]
        return data[most_relevant_idx]
    else:
        return None
