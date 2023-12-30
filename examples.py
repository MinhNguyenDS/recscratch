# local library
from recscratch.utils.processing import content_processing, rating_processing
from recscratch.utils.evaluation import mean_reciprocal_rank, ndcg_at_k

from recscratch.non_personalized import MeanRating, MostPop
from recscratch.content_based import ContentBased
from recscratch.memory_based import UserKNN, ItemKNN

if __name__ == "__main__":
    
    ## Movielens dataset ##
    # Use 'overview' column for content-based recommendation
    movies = content_processing("data_example/Movielens/ml-latest-small", "movies_metadata_test.csv", ['title', 'overview'])
    X_train, X_test = rating_processing("data_example/Movielens/ml-latest-small", "ratings.csv", "userId", "movieId", "rating", 0.998, 22)
    ratings = rating_processing("data_example/Movielens/ml-latest-small", "ratings.csv" , "userId", "movieId", "rating")

    ## Book dataset ##
    X_train, X_test = rating_processing("data_example/Book", "Ratings.csv", "User-ID", "ISBN", "Book-Rating", 0.998, 22)
    ratings = rating_processing("data_example/Book", "Ratings.csv" , "User-ID", "ISBN", "Book-Rating")

    ## MeanRating ## 
    RatPred = MeanRating()
    users_ratings = RatPred.avg_calculate(X_train)
    predictions = RatPred.get_recommendations(X_test)
    print(predictions)

    ## MostPop ##
    mostpop = MostPop()
    item_count = mostpop.item_count(X_train[:1000])
    predictions = mostpop.get_recommendations(X_test)
    print(predictions)

    ## Content based ##
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

    # Collaborative Filtering UserKNN ##
    userKNN = UserKNN(X_train)
    recommendation = userKNN.get_recommendations(1, 100, sim_name='cosine')
    print(recommendation)
    userKNN.get_recommendation_on_dataframe(X_test, K=40, sim_name='pearson')
    userKNN.get_recommendation_on_dataframe(X_test, K=100, sim_name='cosine')
    
    # Collaborative Filtering UserKNN ##
    itemKNN = ItemKNN(X_test)
    recommendation = itemKNN.get_recommendations(1, 100, sim_name='cosine')
    print(recommendation)
    itemKNN.get_recommendation_on_dataframe(X_test, K=100, sim_name='pearson')
    itemKNN.get_recommendation_on_dataframe(X_test, K=100, sim_name='cosine')