# Memory-based is in the Collaborative Filtering method of Recommendation 

# library
from math import sqrt, isnan
from numpy.linalg import norm
from numpy import dot
from tqdm import tqdm

# local library
from recscratch.utils.processing import rating_processing 

class UserKNN():
    # ratings: ratings dataframe
    def __init__(self, ratings):
        self.ratings = ratings
    
    # Support function
    # get rating from the user-item pair
    def get_rating(self, userid, itemid):
        rating_data = self.ratings.loc[(self.ratings.userid == userid) & (self.ratings.itemid == itemid), 'rating']
        if rating_data.empty:
            return None  # Return None or handle the case when the rating data is not found
        return (rating_data.iloc[0])
    # get items of user's history
    def get_itemids_his(self,userid):
        return (self.ratings.loc[(self.ratings.userid==userid), 'itemid'].tolist())

    # Pearson calculation
    def pearson_correlation_score(self, userid1, userid2):
        both_item_count = []
        list_item_user1 = self.ratings.loc[self.ratings.userid == userid1, 'itemid'].to_list()
        list_item_user2 = self.ratings.loc[self.ratings.userid == userid2, 'itemid'].to_list()

        for element in list_item_user1:
            if element in list_item_user2:
                both_item_count.append(element)

        if (len(both_item_count) == 0):
            return 0

        rating_sum_1 = sum([self.get_rating(userid1, i) for i in both_item_count])
        avg_rating_sum_1 = rating_sum_1 / len(both_item_count)  # mean rating of user1
        rating_sum_2 = sum([self.get_rating(userid2, i) for i in both_item_count])
        avg_rating_sum_2 = rating_sum_2 / len(both_item_count)  # mean rating of user2

        numerator = sum([(self.get_rating(userid1, i) - avg_rating_sum_1) * (self.get_rating(userid2, i) - avg_rating_sum_2) for i in both_item_count])
        denominator = sqrt(sum([pow((self.get_rating(userid1, i) - avg_rating_sum_1), 2) for i in both_item_count])) * sqrt(sum([pow((self.get_rating(userid2, i) - avg_rating_sum_2), 2) for i in both_item_count]))

        # output: int
        if (denominator == 0):
            return 0
        return numerator/denominator
    
    # Cosine calculation
    def distance_similarity_score(self,userid1,userid2):
        both_watch_count = 0
        for element in self.ratings.loc[self.ratings.userid==userid1,'itemid'].tolist():
            if element in self.ratings.loc[self.ratings.userid==userid2,'itemid'].tolist():
                both_watch_count += 1
        if both_watch_count == 0 :
            return 0

        rating1 = []
        rating2 = []
        for element in self.ratings.loc[self.ratings.userid==userid1,'itemid'].tolist():
            if element in self.ratings.loc[self.ratings.userid==userid2,'itemid'].tolist():
                rating1.append(self.get_rating(userid1,element))
                rating2.append(self.get_rating(userid2,element))

        denominator = norm(rating1)*norm(rating2)

        # output: int
        if denominator == 0:
            return 0
        return dot(rating1, rating2)/denominator

    # get K user nearest
    def get_most_similar_user(self, userid, K_number_user, similarity_name):
        userid_in_rat = self.ratings.userid.unique().tolist()

        if similarity_name == 'pearson':
            print("===similarity is being calculated by pearson...===")
            similarity_score =  [(self.pearson_correlation_score(userid, user_i), user_i) for user_i in tqdm(userid_in_rat) if user_i != userid]

        if similarity_name == 'cosine':
            print("===similarity is being calculated by cosine...===")
            similarity_score =  [(self.distance_similarity_score(userid, user_i), user_i) for user_i in tqdm(userid_in_rat) if user_i != userid]

        similarity_score.sort() # ascending
        similarity_score.reverse() # descending

        # output: [(score, userid)]
        return similarity_score[:K_number_user]

    # get recommendation for 1 user
    ## args: userid, K_number_user=10, sim_name=['cosine', 'pearson'], topK=10
    def get_recommendations(self, userid, K_number_user, sim_name, topK = None):
        total = {}
        sum_similarity = {}
        list_user_nearest = self.get_most_similar_user(userid, K_number_user, sim_name)
        
        print("===recommending...===")
        for similarity_score, user_id in tqdm(list_user_nearest): # loop each user in list nearest user 
            score = similarity_score
            itemids = self.get_itemids_his(user_id)
            
            for itemid in itemids:
                # item id of other user's history
                if itemid not in self.get_itemids_his(userid):
                    if itemid not in total:
                        total[itemid] = 0
                        sum_similarity[itemid] = 0

                    total[itemid] += self.get_rating(user_id, itemid)*score
                    sum_similarity[itemid] += score

            # calculate rating according to Aggregate weighted ratings
            list_ranking = []
            for userid, tot in total.items():
                if sum_similarity[userid] == 0:
                    list_ranking.append((0,userid))
                else:
                    rating = tot/(sum_similarity[userid])
                    list_ranking.append((rating,userid))

        # calculate rating according to Aggregate weighted ratings
        list_ranking = [(tot/sum_similarity[itemid], itemid) for itemid, tot in total.items()]
        list_ranking = [tup for tup in list_ranking if not isnan(tup[0])]

        list_ranking.sort()
        list_ranking.reverse()

        # output: [(ratings, itemid)]
        if topK != None:
            return list_ranking[:topK]
        return list_ranking
    
    def get_recommendation_on_dataframe(self, df, K=40, sim_name='cosine'):
        # Input: dataframe
            # example:
            #
            # userid,itemid
            # 1     ,1     
            # 5     ,1  
            # 7     ,1    
            # 15    ,1
            # ...
        y_pred = []
        userid_list = df['userid'].tolist()
        itemid_list = df['itemid'].tolist()
        
        print("------ Predicting on dataframe with {} records ------".format(len(userid_list)))

        for i in range(len(userid_list)):        # each user
            list_R = self.get_recommendations(userid_list[i], K, sim_name)  

            # print("list_R", len(list_R))

            check = 0
            for j in list_R:
                if(itemid_list[i] == j[1]):     #j[1] itemID
                    y_pred.append(j[0]) #j[0] score
                    check = 1
            if(check == 0):
                y_pred.append(0)
            print('----- Predicting on {}th'.format(i), 'is {}'.format(y_pred[i]))
        # Output: list
        return y_pred

class ItemKNN():
    # ratings: ratings dataframe
    def __init__(self, ratings):
        self.ratings = ratings
    
    # Support function
    # get rating from the user-item pair
    def get_rating(self, userid, itemid):
        rating_data = self.ratings.loc[(self.ratings.userid == userid) & (self.ratings.itemid == itemid), 'rating']
        if rating_data.empty:
            return None  # Return None or handle the case when the rating data is not found
        return rating_data.iloc[0]
    # get users of item's history
    def get_userids_his(self, itemid):
        return (self.ratings.loc[(self.ratings.itemid==itemid), 'userid'].tolist())

    # Pearson calculation
    def pearson_correlation_score(self, itemid1, itemid2):
        both_user_count = []
        list_user_item1 = self.ratings.loc[self.ratings.itemid == itemid1, 'userid'].to_list()
        list_user_item2 = self.ratings.loc[self.ratings.itemid == itemid2, 'userid'].to_list()

        for userid in list_user_item1:
            if userid in list_user_item2:
                both_user_count.append(userid)
        if len(both_user_count) == 0:
            return 0

        rating_sum_1 = sum([self.get_rating(i, itemid1) for i in both_user_count])
        avg_rating_sum_1 = rating_sum_1 / len(both_user_count)  # mean rating of item1
        rating_sum_2 = sum([self.get_rating(i, itemid2) for i in both_user_count])
        avg_rating_sum_2 = rating_sum_2 / len(both_user_count)  # mean rating of item2

        numerator = sum([(self.get_rating(i, itemid1) - avg_rating_sum_1) * (self.get_rating(i, itemid2) - avg_rating_sum_2) for i in both_user_count])
        denominator = sqrt(sum([pow((self.get_rating(i, itemid1) - avg_rating_sum_1), 2) for i in both_user_count])) * sqrt(sum([pow((self.get_rating(i, itemid2) - avg_rating_sum_2), 2) for i in both_user_count]))

        # output: int
        if denominator == 0:
            return 0
        return numerator / denominator
    
    # Cosine calculation
    def distance_similarity_score(self,itemid1,itemid2):
        both_item_count = 0
        for element in self.ratings.loc[self.ratings.itemid==itemid1,'userid'].tolist():
            if element in self.ratings.loc[self.ratings.itemid==itemid2,'userid'].tolist():
                both_item_count += 1
        if both_item_count == 0 :
            return 0

        rating1 = []
        rating2 = []
        for element in self.ratings.loc[self.ratings.itemid==itemid1,'userid'].tolist():
            if element in self.ratings.loc[self.ratings.itemid==itemid2,'userid'].tolist():
                rating1.append(self.get_rating(element,itemid1))
                rating2.append(self.get_rating(element,itemid2))

        denominator = norm(rating1)*norm(rating2)

        # # output: int
        if denominator == 0:
            return 0
        return dot(rating1, rating2)/denominator

    # get K item nearest
    def get_most_similar_item(self, itemid, K_number_item, similarity_name):
        itemid_in_rat = self.ratings.itemid.unique().tolist()

        if similarity_name == 'pearson':
            print("===similarity is being calculated by pearson...===")
            similarity_score =  [(self.pearson_correlation_score(itemid, user_i), user_i) for user_i in tqdm(itemid_in_rat) if user_i != itemid]

        if similarity_name == 'cosine':
            print("===similarity is being calculated by cosine...===")
            similarity_score =  [(self.distance_similarity_score(itemid, user_i), user_i) for user_i in tqdm(itemid_in_rat) if user_i != itemid]

        similarity_score.sort() # ascending
        similarity_score.reverse() # descending

        # output: [(score, itemid)]
        return similarity_score[:K_number_item]

    # get recommendation for 1 user
    ## args: itemid, K_number_item=10, sim_name=['cosine', 'pearson'], topK=10
    def get_recommendations(self, itemid, K_number_item, sim_name, topK = None):

        total = {}
        sum_similarity = {}
        list_item_nearest = self.get_most_similar_item(itemid, K_number_item, sim_name)

        print("===recommending...===")
        for similarity_score, item_id in tqdm(list_item_nearest): # loop each item in list nearest item
            score = similarity_score
            userids = self.get_userids_his(item_id)

            for userid in userids:
                # userid of other item's history
                if userid not in self.get_userids_his(itemid):
                    if userid not in total:
                        total[userid] = 0
                        sum_similarity[userid] = 0

                    total[userid] += self.get_rating(userid, item_id)*score
                    sum_similarity[userid] += score

            # calculate rating according to Aggregate weighted ratings
            list_ranking = []
            for itemid, tot in total.items():
                if sum_similarity[itemid] == 0:
                    list_ranking.append((0,itemid))
                else:
                    rating = tot/(sum_similarity[itemid])
                    list_ranking.append((rating,itemid))

        for userid, tot in total.items():
            if sum_similarity[userid] != 0:  # Check if denominator is non-zero
                ranking_value = tot / sum_similarity[userid]
                list_ranking.append((ranking_value, userid))
            else:
            # Handle the case when the denominator is zero
                list_ranking.append((0, userid))  # Assign a default value, such as 0

        list_ranking = [tup for tup in list_ranking if not isnan(tup[0])]

        list_ranking.sort()
        list_ranking.reverse()

        # output: [(ratings, userid)]
        if topK != None:
            return list_ranking[:topK]
        return list_ranking
    
    def get_recommendation_on_dataframe(self, df, K=40, sim_name='cosine'):
        # Input: dataframe
            # example:
            #
            # userid,itemid
            # 1     ,1     
            # 5     ,1  
            # 7     ,1    
            # 15    ,1
            # ...
        y_pred = []
        userid_list = df['userid'].tolist()
        itemid_list = df['itemid'].tolist()
        
        print("------ Predicting on dataframe with {} records ------".format(len(itemid_list)))

        for i in range(len(itemid_list)):        # each user
            list_R = self.get_recommendations(itemid_list[i], K, sim_name)  

            # print("list_R", len(list_R))

            check = 0
            for j in list_R:
                if(userid_list[i] == j[1]):     #j[1] itemID
                    y_pred.append(j[0]) #j[0] score
                    check = 1
            if(check == 0):
                y_pred.append(0)
            print('----- Predicting on {}th'.format(i), 'is {}'.format(y_pred[i]))
        # Output: list
        return y_pred

if __name__ == "__main__":
    
    ## Movielens dataset ##
    X_train, X_test = rating_processing("../data_example/Movielens/ml-latest-small", "ratings.csv", "userId", "movieId", "rating", 0.998, 22)
    ratings = rating_processing("../data_example/Movielens/ml-latest-small", "ratings.csv" , "userId", "movieId", "rating")

    ## Book dataset ##
    # X_train, X_test = rating_processing("../data_example/Book", "Ratings.csv", "User-ID", "ISBN", "Book-Rating", 0.002, 22)
    # ratings = rating_processing("../data_example/Book", "Ratings.csv" , "User-ID", "ISBN", "Book-Rating")

    ## Test function ##
    userKNN = UserKNN(ratings)
    userKNN.get_recommendations(1, K=40, sim_name='pearson')
    userKNN.get_recommendation_on_dataframe(X_test, K=40, sim_name='pearson')
    # userKNN.get_recommendation_on_dataframe(X_test, K=100, sim_name='cosine')
    
    # itemKNN = ItemKNN(ratings)
    # itemKNN.get_recommendations(1, K=100, sim_name='pearson')
    # itemKNN.get_recommendation_on_dataframe(X_test, K=100, sim_name='pearson')
    # itemKNN.get_recommendation_on_dataframe(X_test, K=100, sim_name='cosine')