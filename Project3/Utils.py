movies_w_imgurl_path = "./data/movies_w_imgurl.csv"
ratings_train_path = "./data/ratings_train.csv"
ratings_val_path = "./data/ratings_val.csv"
tags_path = "./data/tags.csv"

embedding_dim = 8
num_ratings = 9

# model_name = "param.data"
# model_name = "compare_rating_val"
model_name = "replaced_simmv"
epochs = 500
lr_rate = 0.001

sim_k = 0.5
second_sim_k = 0.6