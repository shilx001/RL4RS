import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

np.random.seed(1)
data = pd.read_csv('ml-latest-small/ratings.csv')

# 统计一下用户的分布
# hist = data.hist(column='userId',bins=610)
print('Max rating count of a user: ', np.max(data['userId'].value_counts()))
print('Min rating count of a user: ', np.min(data['userId'].value_counts()))
print('Average rating count:', np.mean(data['userId'].value_counts()))

user_idx = data['userId'].unique()  # id for all the user
np.random.shuffle(user_idx)
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# 对训练集中每个电影进行统计，统计其平均得分
movie_total_rating = []
movie_count = []
movie_id = []
for idx1 in train_id:  # 针对train_id中的每个用户
    user_record = data[data['userId'] == idx1]
    for idx2, row in user_record.iterrows():
        # 检查list中是否有该电影的id
        if row['movieId'] in movie_id:
            idx = movie_id.index(row['movieId'])  # 找到该位置
            movie_count[idx] += 1  # 计数加一
            movie_total_rating[idx] += row['rating']
        else:
            # 否则新加入movie_id
            movie_id.append(row['movieId'])
            movie_count.append(1)
            movie_total_rating.append(row['rating'])
print('Total movie count:', len(movie_id))

# 计算average rating并排序
movie_total_rating = np.array(movie_total_rating)
movie_count = np.array(movie_count)
movie_average_rating = movie_total_rating / movie_count
recommend_index = np.argsort(-movie_average_rating)
print('Average rating for movies:', np.mean(movie_average_rating))
print('Average rating count for movies:', np.mean(movie_count))

# 按照popularity进行推荐，并对测试集的reward进行评测。
result = []  # 评测结果以list进行存储，存储内容为
k = 30  # 用于评估的top-k参数
alpha = 0


# 评分是正则化到[-1,1]间
def normalize(rating):
    max_rating = 5
    min_rating = 0.5
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)


for idx1 in test_id:  # 针对test_id中的每个用户
    user_record = data[data['userId'] == idx1]  # 找出每个用户的index
    user_watched_list = []  # 用户已观看过的电影list
    user_rating_list = []
    cp = []  # consecutive positive count
    cn = []  # consecutive negative count
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    r = 0  # user rating
    for idx2, row in user_record.iterrows():  # 针对每个测试集的用户
        # 推荐电影为其没观看过的且排行最早的电影
        user_rating_list.append(row['rating'])  # 记录user_rating_list
        if idx2 is not 1:
            cp.append(np.sum(np.array(user_rating_list)[:-1] > 3))
            cn.append(np.sum(np.array(user_rating_list)[:-1] <= 3))
        else:
            cp.append(0)
            cn.append(0)
        rec_count = 0
        current_recommend = []
        # 针对recommend_index里面所有的电影,找出用户没有看过的k个电影推荐给用户
        for movie_idx in recommend_index:
            if movie_idx in user_watched_list:  # 如果看过，则continue
                continue
            else:
                # 如果没看过则加入推荐名单,每次推荐k个给用户
                rec_count += 1
                current_recommend.append(movie_idx)
                if rec_count > k:
                    break
        # 对推荐电影进行评估
        if row['movieId'] in current_recommend:  # 如果当前用户看的电影在推荐的列表中
            if row['rating'] > 3:  # 如果评分大于3，则证明推荐的是用户喜欢的
                tp += 1
            else:  # 否则不是用户喜欢的
                fp += 1
            r = normalize(row['rating'])
        else:  # 如果当前用户选择列表不在推荐的列表中,则要看用户喜欢不
            if row['rating'] > 3:  # 证明推荐系统不推荐的用户喜欢
                fn += 1
            else:
                tn += 1
                # r = -1
        result.append([r + alpha * (cp[-1] - cn[-1]), tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)])

print('Result: Reward, Precision, Recall')
print(np.mean(np.array(result).reshape([-1, 3]), axis=0))
