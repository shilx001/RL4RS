import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from ddpg_v2 import *
import math
import pickle
# DDPG-KNN
np.random.seed(1)
data = pd.read_csv('ml-latest-small/ratings.csv')
#data = pd.read_table('ratings.dat',sep='::',names=['userId','movieId','rating','timestep'])

user_idx = data['userId'].unique()  # id for all the user
np.random.shuffle(user_idx)
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# 找出这种集合中总共有多少电影
# 找出集合中有多少个电影。
movie_id = []
for idx1 in user_idx:  # 针对train_id中的每个用户
    user_record = data[data['userId'] == idx1]
    for idx2, row in user_record.iterrows():
        # 检查list中是否有该电影的id
        if row['movieId'] in movie_id:
            idx = movie_id.index(row['movieId'])  # 找到该位置
        else:
            # 否则新加入movie_id
            movie_id.append(row['movieId'])

# 针对训练集建立user-rating matrix,构建item的feature matrix
rating_mat = np.zeros([len(train_id), len(movie_id)])
movie_id = np.array(movie_id)
for idx in train_id:  # 针对每个train数据
    record = data[data['userId'] == idx]  # record有多个数据，所以row_index也有多个
    for _, row in record.iterrows():  # 针对每个用户的每条评分
        r = np.where(train_id == idx)
        c = np.where(row['movieId'] == movie_id)
        rating_mat[r, c] = row['rating']


def get_feature(input_id):
    # 根据输入的movie_id得出相应的feature
    movie_index = np.where(movie_id == input_id)
    return rating_mat[:, movie_index]


# 根据item_id找出相应的动作
BASE = 2  # 输出动作的进制
output_action_dim = np.ceil(math.log(len(movie_id), BASE))  # DDPG输出动作的维度
output_action_bound = 1.0 / BASE


def action_mapping(item_id):
    # 根据movie的id返回其转换的连续型变量
    output_action = []
    item_id = np.where(movie_id == item_id)
    item_id = item_id[0]
    while item_id / BASE > 0:
        output_action.append(item_id % BASE)
        item_id = item_id // BASE
    return np.hstack(
        (np.array(output_action).flatten(), np.zeros([int(output_action_dim) - len(output_action)])))  # 针对不满的要补0


def get_movie(movie_mask):
    # 根据电影编码得到电影的index
    return np.sum(BASE ** np.cumsum(movie_mask))


action_mask_set = []
# 针对每个movie构建action mask集合
for idx in movie_id:
    action_mask_set.append(action_mapping(idx))

MAX_SEQ_LENGTH = 32
agent = DDPG(state_dim=len(train_id) + 1, action_dim=int(output_action_dim), action_bound=output_action_bound,
             max_seq_length=MAX_SEQ_LENGTH)

print('Start training.')
start_time = datetime.datetime.now()
# 根据训练数据对DDPG进行训练。
global_step = 0
for id1 in train_id:
    user_record = data[data['userId'] == id1]  # 找到该用户的所有
    state = []
    reward = []
    action = []
    for _, row in record.iterrows():  # 针对每个用户的评分数据，对state进行录入
        movie_feature = get_feature(row['movieId'])  # 用户的movie feature
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        state.append(current_state)
        reward.append(row['rating'])
        action.append(action_mapping(row['movieId']))
    # 针对每个state,把reward
    for i in range(2, len(state)):
        current_state = state[:i - 1]  # 到目前为止所有的state
        current_state_length = i - 1
        next_state = state[:i]
        next_state_length = i
        current_reward = reward[i]
        current_action = action[i]
        if current_state_length > MAX_SEQ_LENGTH:
            current_state = current_state[-MAX_SEQ_LENGTH:]
            current_state_length = MAX_SEQ_LENGTH
        if next_state_length > MAX_SEQ_LENGTH:
            next_state = next_state[-MAX_SEQ_LENGTH:]
            next_state_length = MAX_SEQ_LENGTH
        done = 0
        if i is len(state) - 1:
            done = 1
        agent.store(current_state, current_state_length, current_action, current_reward, next_state,
                    next_state_length, done)
    memory_length = agent.replay_buffer.get_size()
    a_loss, c_loss = agent.train(int(memory_length / 32))
    print('Step ', global_step)
    print('Actor loss: ', a_loss)
    print('Critic loss: ', c_loss)
    global_step += 1
    agent.replay_buffer.clear()

print('Training finished.')
end_time = datetime.datetime.now()
print('Training time(seconds):', (end_time - start_time).seconds)

print('Begin test.')


def normalize(rating):
    max_rating = 5
    min_rating = 0.5
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)


start_time = datetime.datetime.now()
# TEST阶段
result = []
K = 10
N = 10  # top-N evaluation
for idx1 in test_id:  # 针对test_id中的每个用户
    user_record = data[data['userId'] == idx1]
    user_watched_list = []
    user_rating_list = []
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    r = 0
    all_state = []
    for idx2, row in user_record.iterrows():  # 针对每个电影记录
        user_rating_list.append(row['rating'])
        current_movie = row['movieId']
        current_state = np.hstack((get_feature(current_movie).flatten(), row['rating']))
        all_state.append(current_state)
        if len(all_state) > 1:  # 针对第二个电影开始推荐
            temp_state = all_state[:-1]
            if len(temp_state) > MAX_SEQ_LENGTH:
                temp_state = temp_state[-MAX_SEQ_LENGTH:]
            proto_action = agent.get_action(temp_state, len(temp_state))  # DDPG-knn输出的Proto action
            # 根据proto_action找K个最近的动作
            dist = np.sqrt(np.sum(
                (np.array(action_mask_set).reshape([-1, int(output_action_dim)]) - proto_action.flatten()) ** 2,
                axis=1))
            sorted_index = np.argsort(dist)
            nearest_index = sorted_index[:K]
            # 评估nearest_index的value
            eval_state = []
            eval_length = []
            eval_action = []
            # 对temp_state进行补0
            temp_length = len(temp_state)
            if len(temp_state) < MAX_SEQ_LENGTH:
                padding_mat = np.zeros([MAX_SEQ_LENGTH - len(temp_state), len(train_id) + 1])
                temp_state = np.vstack((temp_state, padding_mat))
            for idx3 in nearest_index:
                eval_state.append(temp_state)
                eval_action.append(np.array(action_mask_set[idx3]))
                eval_length.append(temp_length)
            critic_value = agent.eval_critic(eval_state, eval_length, eval_action)
            # 推荐Q值最高的N个
            recommend_index = nearest_index[np.argsort(critic_value.flatten())[:N]]
            recommend_movie = list(movie_id[recommend_index])  # 转为list
            # 针对每个推荐item评估下
            if row['movieId'] in recommend_movie:
                if row['rating'] > 3:
                    tp += 1
                else:
                    fp += 1
                r = normalize(row['rating'])
            else:
                if row['rating'] > 3:
                    fn += 1
                else:
                    tn += 1
        result.append([r, tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)])

pickle.dump(result, open('ddpg_knn', mode='wb'))
print('Result:')
print(np.mean(np.array(result).reshape([-1, 3]), axis=0))
end_time = datetime.datetime.now()
print('Testing time(seconds):', (end_time - start_time).seconds)
