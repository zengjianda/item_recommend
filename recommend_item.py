# coding:utf8
import math
import operator

import numpy as np
import pandas as pd
import redis
from apscheduler.schedulers.background import BackgroundScheduler
import time
import pymysql

#初始化数据
def init_data():
    con = pymysql.connect(host="host",port=port,user="user", passwd="pwd", db="db",charset="utf8")
    sql = "select uuid,user_id from uuid_userid"
    #初始化uuid和userid的对应关系
    uuid = pd.read_sql(sql=sql,con=con)
    pool = redis.ConnectionPool(host="host", port=6379, password="pwd")
    r = redis.Redis(connection_pool=pool)
    for i in range(len(uuid)):
        r.set("uuid_"+str(uuid.iloc[i].uuid),str(uuid.iloc[i].user_id))
    #初始化热销商品
    sql = "select items from hot_item"
    hot_item = pd.read_sql(sql=sql,con=con)
    r.set("hot_item",str(hot_item.iloc[0]["items"]))
    con.close()

#读取数据并做一些处理
def readData():
    con = pymysql.connect(host="host", port=port, user="user",
                          passwd="passwd", db="db", charset="utf8")
    sql = "select user_id,product_id,orders_num from recommend_detail_sale"
    data = pd.read_sql(sql,con)
    con.close()
    # 将空由0替换
    data = data.fillna(0)
    # 将product_id 转化为整数
    data["product_id"] = data["product_id"].apply(lambda x: int(x))
    # 过滤掉产品为空的数据
    data = data[(data.product_id > 0)]
    # 过滤一个用户购买过多个商品的记录，只购买过一个商品的用户记录对共现矩阵没有影响，这样的数据没有意义
    user_data = data.groupby("user_id").size()
    user_data = user_data[user_data > 1]
    data = data[data.user_id.isin(user_data.keys())]
    return data

#根据用户分组
def get_all_user_data(data):
    user_data = data.groupby("user_id").size()
    user_data = user_data[user_data > 1]
    return user_data

#创建商品的数据字典
def product_index_dic(data):
    all_product_id = list(set(data["product_id"].values.tolist()))
    product_to_index = {}
    index_to_product = {}
    for index, value in enumerate(all_product_id):
        product_to_index[value] = index
        index_to_product[index] = value
    return product_to_index, index_to_product


# 第一步创建用户-物品的倒排索引
def create_user_item_index(user_data, data, product_to_index):
    user_item_index = {}
    for user_id in user_data.keys():
        product_ids = data[data.user_id == user_id]["product_id"].values.tolist()
        for index, value in enumerate(product_ids):
            product_ids[index] = product_to_index[value]
        user_item_index[user_id] = product_ids
    return user_item_index


# 第二步创建共现矩阵
def create_matrix_c(product_length, user_item_index):
    matrix_c = np.zeros((product_length, product_length))
    # 循环用户-商品倒排索引 对于同一个用户购买的任意的两个商品 在共现矩阵中都要加1
    for user_id in user_item_index:
        product_ids = user_item_index[user_id]
        for i, value in enumerate(product_ids):
            if (i < len(product_ids) - 1):
                list_other = product_ids[(i + 1):len(product_ids)]
                for second_product_index in list_other:
                    matrix_c[value][second_product_index] += 1
                    matrix_c[second_product_index][value] += 1
    return matrix_c


# 第三步根据算法得到商品的相似矩阵 算法：cij/sqrt(|N(i)|*|N(j)|)
def create_matrix_w(data, product_to_index, product_length, matrix_c):
    product_index_count_dic = {}
    product_group = data.groupby("product_id").size()
    for product_id in product_group.keys():
        product_index_count_dic[product_to_index[product_id]] = product_group[product_id]
    matrix_w = np.zeros((product_length, product_length))
    # 共现矩阵大于0的下标list
    index_i_list, index_j_list = np.where(matrix_c > 0)
    for index, value in enumerate(index_i_list):
        i = value
        j = index_j_list[index]
        score = matrix_c[i][j] / math.sqrt(product_index_count_dic[i] * product_index_count_dic[j])
        matrix_w[i][j] = score
        matrix_w[j][i] = score
    return matrix_w


# 归一化函数
def normalize(value):
    value = (value - np.min(value)) / (np.max(value) - np.min(value))
    return value


# 第四步创建用户的喜好商品矩阵：并进行归一化
def create_user_like_item_dic(user_data, data, product_length, product_to_index):
    user_like_item_dic = {}
    for user_id in user_data.keys():
        user_like_item = data[data.user_id == user_id]
        user_item_like_matrix = np.zeros(product_length)
        for i in range(len(user_like_item)):
            index = product_to_index[user_like_item.iloc[i].product_id]
            user_item_like_matrix[index] = user_like_item.iloc[i].orders_num
        user_like_item_dic[user_id] = normalize(user_item_like_matrix)
    return user_like_item_dic


# 获得某个商品的最相似的k个商品
def getMostSimilar(matrix_w, index, k):
    c_list = matrix_w[index]
    similar_item = pd.DataFrame({"value": c_list})
    similar_item = similar_item.sort_values(by="value", ascending=False).iloc[0:k]
    similar_item_dic = {}
    for i in range(len(similar_item)):
        similar_item_dic[similar_item.iloc[i].name] = similar_item.iloc[i].value
    return similar_item_dic


# 为用户推荐的商品
def reommendItem(user_id, matrix_w, user_like_item_dic, k):
    recommend_dic = {}
    user_like_list = user_like_item_dic[user_id]
    user_like_item_index_list = np.where(user_like_list > 0)
    user_like_item_index_list = user_like_item_index_list[0]
    for product_index in user_like_item_index_list:
        like_score = user_like_list[product_index]
        most_similar_item = getMostSimilar(matrix_w, product_index, k)
        for key in most_similar_item.keys():
            if key in user_like_item_index_list:
                continue
            # 最终得分是用户对商品的喜欢程度 * 商品的相似程度
            score = like_score * most_similar_item[key]
            if key in recommend_dic.keys():
                score += recommend_dic[key]
            recommend_dic[key] = score
    # 返回得分最高的k个商品
    sorted_x = sorted(recommend_dic.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    return sorted_x[0:k]


# 第五步给用户推荐商品

def recommend_for_user(user_like_item_dic, matrix_w, index_to_product):
    user_recommend = {}
    for user_id in user_like_item_dic.keys():
        recommend_dic = reommendItem(user_id, matrix_w, user_like_item_dic, 10)
        value = ""
        for key in recommend_dic:
            index = key[0]
            if value == "":
                value += str(index_to_product[index])
            else:
                value += "," + str(index_to_product[index])
        user_recommend[user_id] = value
    return user_recommend


def save_value_to_redis(user_recommend):
    pool = redis.ConnectionPool(host="host", port=port, password="pwd")
    r = redis.Redis(connection_pool=pool)
    for key in user_recommend.keys():
        r.set("user_recommend_" + str(key),user_recommend[key])

def train_data():
    print("job start")
    init_data()
    print("init data finish")
    data = readData()
    print("read data finish")
    product_to_index, index_to_product = product_index_dic(data)
    user_data = get_all_user_data(data)
    user_item_index = create_user_item_index(user_data, data, product_to_index)
    product_length = len(product_to_index)
    print(product_length)
    matrix_c = create_matrix_c(product_length, user_item_index)
    print("matrix_c create finish")
    matrix_w = create_matrix_w(data, product_to_index, product_length, matrix_c)
    print("matrix_w create finish")
    user_like_item_dic = create_user_like_item_dic(user_data, data, product_length, product_to_index)
    print("user_like_item_dic create finish")
    user_recommend = recommend_for_user(user_like_item_dic, matrix_w, index_to_product)
    print("user_recommend create finish")
    save_value_to_redis(user_recommend)
    print("save value finish")

def test():
    print("job start---------")

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_data, 'cron', hour=5,minute=0,second=0)
    scheduler.start()
    while(True):
        time.sleep(2)

