# recommend the items with the most popular one.
import numpy as np
import pandas as pd

data = pd.read_csv('ml-latest-small/ratings.csv')

user_idx = data['userId'].unique()  # id for all the user
np.random.shuffle(user_idx)  # shuffle the indices
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# for each movie, calculate the average ratings and rank them.
movie_count = []
movie_id = []
for idx1 in train_id:  # 针对train_id中的每个用户
    user_record = data[data['userId'] == idx1]
    for idx2, row in user_record.iterrows():
        # 检查list中是否有该电影的id
        if row['movieId'] in movie_id:
            idx = movie_id.index(row['movieId'])  # 找到该位置
            movie_count[idx] += 1  # 计数加一
        else:
            # 否则新加入movie_id
            movie_id.append(row['movieId'])
            movie_count.append(1)

index = np.argsort(movie_count)
movie_id=np.array(movie_id)
sorted_movie_id = movie_id[np.flipud(index)]#倒序排列
