# Source code is flowed by the code of recommender (https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb)

# library 
import pandas as pd
import itertools

# local library
from utils.processing import rating_processing


def filter_by(df, filter_by_df, filter_by_cols):
    return df.loc[
        ~df.set_index(filter_by_cols).index.isin(
            filter_by_df.set_index(filter_by_cols).index
        )
    ]

class MeanRating : 
    # Calculate avg ratings from the training set
    def avg_calculate(self, df):
        users_ratings = df.groupby(["userid"])["rating"].mean()
        users_ratings = users_ratings.to_frame().reset_index()
        users_ratings.rename(columns={"rating": "AvgRating"}, inplace=True)
        self.users_ratings = users_ratings
        return users_ratings
    
     # Generate rating prediction for the test set
    def get_recommendations(self, df, userid = None):
        predictions = pd.merge(df, self.users_ratings, on=["userid"], how="inner")
        if userid != None:
            return predictions.loc[predictions['userid'] == userid]
        # Output: Data frame
        return predictions
    
   
class MostPop:
    def item_count(self, df):
        item_counts = df["itemid"].value_counts().to_frame().reset_index()
        item_counts.columns = ["itemid", "Count"]
        self.item_counts = item_counts
        return item_counts
    
    def get_recommendations(self, df, userid = None):
        user_item_col = ["userid", "itemid"]

        # Cross join users and items
        test_users = df['userid'].unique()
        user_item_list = list(itertools.product(test_users, self.item_counts['itemid']))
        users_items = pd.DataFrame(user_item_list, columns=user_item_col)

        # Remove seen items (items in the train set) as we will not recommend those again to the users
        users_items_remove_seen = filter_by(users_items, df, user_item_col)

        # Generate recommendations
        recommendations = pd.merge(self.item_counts, users_items_remove_seen, on=['itemid'], how='inner')
        
        if userid != None:
            return recommendations.loc[recommendations['userid'] == userid]
        return recommendations

        
if __name__ == "__main__":
    # ## Movielens dataset ##
    # X_train, X_test = rating_processing("../data_example/Movielens/ml-latest-small", "ratings.csv", "userId", "movieId", "rating", 0.998, 22)

    ## Book dataset ##
    X_train, X_test = rating_processing("../data_example/Book", "Ratings.csv", "User-ID", "ISBN", "Book-Rating", 0.998, 22)

    ## test function ## 
    # RatPred = MeanRating()
    # users_ratings = RatPred.avg_calculate(X_train)
    # predictions = RatPred.get_recommendations(X_test)

    mostpop = MostPop()
    item_count = mostpop.item_count(X_train[:1000])
    predictions = mostpop.get_recommendations(X_test)
    print(predictions)
