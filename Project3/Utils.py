movies_w_imgurl_path = "./data/movies_w_imgurl.csv"
ratings_train_path = "./data/ratings_train.csv"
ratings_val_path = "./data/ratings_val.csv"
tags_path = "./data/tags.csv"

embedding_dim = 8
num_ratings = 9

model_name = "param.data"
# model_name = "replaced_simmv"
# model_name = "replace_user_avg"
# model_name = "replace_zero"
epochs = 100
lr_rate = 0.001

sim_k = 0.75
second_sim_k = 0.5