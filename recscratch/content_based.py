# Content-based is the method of Recommendation

# library
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

# local library
from recscratch.utils.processing import content_processing 


class ContentBased(): 
    # Support function
    # indexing the titles
    def indexing_item(self, col, title_col):
        item2item_encoded = pd.Series(col.index, index=col[title_col]).drop_duplicates()
        return item2item_encoded
    def get_name_of_item(self, item2item_encoded, index, title_col):
        name_item = item2item_encoded.iloc[index][title_col]
        return name_item

    # language processing function
    def processing_on_1_sent(self, data):
        # Input: string
        # lowercase
        data = data.lower()
        # remove punctuation and special characters
        data = re.sub('\W+',' ', data)
        # remove excess whitespace
        data = data.strip()
        # remove StopWord
        data = ' '.join([word for word in data.split() if word not in stopwords.words("english")])
        # word tokenize
        data = word_tokenize(data)
        data = ' '.join(data)
        # Output: string
        return data
    

    # preprocessing on list of natural language data
    def processing_on_list(self, col):
        list_processed = []
        for i in tqdm(range(len(col))):
            list_processed.append(self.processing_on_1_sent(col[i]))
        return list_processed
    
    # feature extraction with tf_idf
    def fit_tfidf(self, col):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(col)

        # Output: tfidf of content
        return tfidf_matrix
    
    # calculate similarity on all of data
    def fit_similarity_all_data(self, overview_matrix): 
        # Input: shape: (num_of_sentence, embedding_dim)
        distance_similarity = linear_kernel(overview_matrix, overview_matrix)
        return distance_similarity

    # calculate similarity on one vec of data
    def fit_similarity_one_data(self, one_vec, list_vec): 
        # Input: one vector, list of vector
        sim_synthesis = cosine_similarity([one_vec], list_vec)
        # sim_synthesis = np.argsort(sim_synthesis)[::-1]
        # Output: shape (one_vec, list_vec)
        return sim_synthesis

    # get recommendation for 1 item
    def get_recommendations(self, title_name, cosine_sim, item2item_encoded, topK=10):
        item_index = item2item_encoded[title_name]
        
        sim_scores = list(enumerate(cosine_sim[item_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:topK]

        item_encoded = [i[0] for i in sim_scores]
        item_names = [title for title, index in item2item_encoded.items() if index in item_encoded]
        # Output: list of item names
        return item_names

if __name__ == "__main__":
    
    ## Movielens dataset ##
    # Use 'overview' column for content-based recommendation
    movies = content_processing("../data_example/Movielens/ml-latest-small", "movies_metadata_test.csv", ['title', 'overview'])

    ## Test function ##
    content_based = ContentBased()
    list_data_processed = content_based.processing_on_list(movies['overview'])
    tfidf_feature = content_based.fit_tfidf(list_data_processed)

    cosine_sim_matrix = content_based.fit_similarity_all_data(tfidf_feature)
    # one_vec = cosine_sim_matrix[0]
    # list_vec = cosine_sim_matrix[0:10]
    # print(content_based.fit_similarity_one_data(one_vec, list_vec))

    item2item_encoded = content_based.indexing_item(movies,'title')
    recommendation = content_based.get_recommendations("Father of the Bride Part II", cosine_sim_matrix, item2item_encoded)
    print(recommendation)