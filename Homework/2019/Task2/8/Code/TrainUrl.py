# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:50:06 2019

@author: Birdman
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import csv
import tldextract
import urllib.parse
import re

#清洗数据
def clean_url(url):
    URL = 'http://' + url if re.match(r'^(?:http|ftp)s?://', url) is None else url
    return URL
#数点
def count_dots(url): 
    return url.count('.')
#计算url长度
def url_length(url):
    return len(str(url))
#是否存在顶级域名
def has_subdomain(url):
    return 1 if tldextract.extract(url)[0] is not None else 0
#求出domain的长度
def domain_lentgh(url):
    return len(tldextract.extract(url)[1])
#求出域名后缀的长度
def suffix_length(url):
    return len(tldextract.extract(url)[2])
#是否存有路径
def has_path(url):
    return 1 if urllib.parse.urlparse(url)[2] is not None else 0
#是否查询context
def has_query(url):
    return 1 if urllib.parse.urlparse(url)[4] is not None else 0
#计算路径长度
def path_length(url):
    return len(urllib.parse.urlparse(url)[2])
#计算查询内容长度
def query_length(url):
    return len(urllib.parse.urlparse(url)[4])
#数/
def count_SubDir(url): 
    return url.count('/')
#数@
def count_At(url): 
    return url.count('@')

#放弃
def has_IP(url):
	compile_rule = re.compile(r'(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])')
	match_list = re.findall(compile_rule, url)
	if match_list:
		return 1
	else:
		return 0

def get_features(temp,url):
    temp.append(count_dots(url))
    temp.append(url_length(url))
    temp.append(has_subdomain(url))
    temp.append(has_path(url))
    temp.append(has_query(url))
    temp.append(path_length(url))
    temp.append(query_length(url))
    temp.append(domain_lentgh(url))
    temp.append(suffix_length(url))
    temp.append(count_SubDir(url))
    temp.append(has_IP(url))
    temp.append(count_At(url))
    return temp

if __name__ == '__main__':
    url=[]
    label = []
    temp = []
    data = []
    LaBel = []
    
    with open(r'C:\Users\Birdman\Desktop\data\data.csv','r',encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            url.append(clean_url(row['url']))
            if row['label'] == 'good':
                label.append(0)
            elif row['label'] == 'bad':
                label.append(1)
        csvfile.close()
    #print(label[:10])
    #print(countdots(url[1]))
    for i in range(0,len(url)):
        temp = get_features(temp,url[i])
        data.append(temp)
        LaBel.append(label[i])
        temp = []
    
    data=np.array(data)
    LaBel=np.array(LaBel)
    
    X_train,X_test,y_train,y_test=train_test_split(data,LaBel,test_size=0.3,random_state=0)
    sc = StandardScaler()

    sc.fit(X_train)
    #对数据集进行处理
    url_tree = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=10, random_state=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
        

    print("***************start****************")
    url_tree_save=url_tree.fit(X_train,y_train)
    print("**************saveing***************")
    joblib.dump(url_tree_save,'C:/Users/Birdman/Desktop/data/url_tree.model')
    print("Training score:%f" % (url_tree.score(X_train, y_train)))
    print("Testing score:%f" % (url_tree.score(X_test, y_test)))
    ''' clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    print("Training score:%f" % (clf.score(X_train, y_train)))
    print("Testing score:%f" % (clf.score(X_test, y_test)))'''
    #test

    url="www.baidu.com"
    test=[]
    test = np.array(get_features(test,url))
    if url_tree.predict([test]) == 1:
        print("test result:  GOOD")
    else:
        print("test result:  BAD")
    '''if clf.predict([test]) == 1:
        print("test result:  GOOD")
    else:
        print("test result:  BAD")'''

