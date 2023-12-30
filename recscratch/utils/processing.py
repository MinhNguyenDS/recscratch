# library
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def content_processing(link_folder, content_file, list_content_col = []):
    content_items = pd.read_csv(link_folder+'/'+content_file)
    content_items = content_items.fillna('')
    if list_content_col != []:
        content_items = content_items[list_content_col]
    return content_items


def rating_processing(link_folder, rating_file, user_col, item_col, rating_col, split_size = None, seed = None):
    ratings = pd.read_csv(link_folder+"/"+rating_file)
    ratings = ratings[[user_col, item_col, rating_col]]
    ratings = ratings.rename(columns={user_col: "userid", item_col: "itemid", rating_col: "rating"})
    
    if split_size:
        return train_test_split(ratings, train_size=split_size, random_state=seed)
    return ratings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("link_folder", type=str, help="link of dataset folder")
    parser.add_argument("rating_file", type=str, help="name of rating file")
    parser.add_argument("content_file", type=str, help="name of content file")
    args = parser.parse_args()

    ## test func on Movielens dataset ## 
    # ratings = rating_processing(args.link_folder, args.rating_file , "userId", "movieId", "rating")
    # X_train, X_test = rating_processing(args.link_folder, args.rating_file , "userId", "movieId", "rating", 0.8, 42)
    # movies = content_processing(args.link_folder, args.content_file)
    # print(ratings)

    # test func on Book dataset ##
    ratings = rating_processing(args.link_folder, args.rating_file , "User-ID", "ISBN", "Book-Rating")
    X_train, X_test = rating_processing(args.link_folder, args.rating_file , "User-ID", "ISBN", "Book-Rating", 0.8, 42)
    books = content_processing(args.link_folder, args.content_file)
    print(books)
