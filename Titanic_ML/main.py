# main.py
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 시각화 하는 함수
def bar_chart(data, feature) :
    # 특성의 이름에 따라 생존한 사람을 카운트 한다.
    a1 = data[data['Survived'] ==1][feature].value_counts()
    # 특성의 이름에 따라 사망한 사람을 카운트 한다.
    a2 = data[data['Survived'] ==0][feature].value_counts()

    df = pd.DataFrame([a1, a2])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()

# nan 검사
import math

def check_nan(data, name) :
    count = 0
    for value in data[name] :
        if math.isnan(value) :
            count += 1
    print('nan :', count)

# step1: 데이터를 준비한다.
def step1_get_data() :
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

# step2 : 데이터에 대한 학습
def step2_learning_data(data) :
    # Pclass : 등급(1, 2, 3)
    #          3등석 일수록 사망할 확률이 높다
    # bar_chart(data, 'Pclass')
    # Name : 승객의 이름
    #        승객의 이름은 사망과 생존과 관련이 없다.
    # bar_chart(data, 'Name')
    # Sex : 성별(male, female)
    #       남자가 여자보다 사망률이 높다.
    # bar_chart(data, 'Sex')
    # Age : 나이
    #       청년층일 수록 사망률이 높다.
    # bar_chart(data, 'Age')
    # SibSp : 같이 탑승한 형제자매수
    # bar_chart(data, 'SibSp')
    # Parch : 같이 탑승한 부모자식의 수
    # bar_chart(data, 'Parch')
    # Ticket : 승선한 사람들의 티켓 번호
    #          티켓 번호의 구조가 연관성이 전혀 없다.
    # bar_chart(data, 'Ticket')
    # Fare : 요금 (등급에 따라 가격이 다르다)
    # bar_chart(data, 'Fare')
    # Cabin : 객실 정보(구역+번호)
    # bar_chart(data, 'Cabin')
    # Embarked : 탑승 선착장 정보
    # bar_chart(data, 'Embarked')
    pass


# step3: 데이터에 대한 전처리
def step3_data_preprocessing(train, test) :
    train_test_data = [train, test]

    # 성별 데이터 전처리 : male => 0, female => 1
    sex_map = {'male' : 0, 'female' : 1}
    for dataset in train_test_data :
        dataset['Sex'] = dataset['Sex'].map(sex_map)

    # print(train['Sex'])
    # print(test['Sex'])

    # 이름 데이터 전처리
    # 나이 데이터에 비어있는 데이터를 채워주기 위해
    # 이름에 있는 Mr, Mrs, Miss 데이터를 활용하기 위함.
    for dataset in train_test_data :
        dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

    title_Map = {
        'Mr' : 0, 'Miss' : 1, 'Mrs' : 2, 'Master' : 3,
        'Dr' : 3, 'Rev' : 3, 'Col' : 3, 'Mlle' : 3,
        'Major' : 3, 'Countess' : 3, 'Sir' : 3,
        'Don' : 3, 'Jonkheer' : 3, 'Lady' : 3,
        'Capt' : 3, 'Ms' : 3, 'Mme' : 3, 'Dona' : 3
    }
    for dataset in train_test_data :
        dataset['Title'] = dataset['Title'].map(title_Map)

    # print(train['Title'])
    # print(test['Title'])
    # print(train['Title'])
    # print(train['Title'].value_counts())
    # print(test['Title'].value_counts())
    # Name 데이터를 제거한다.
    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    # Age(나이) 전처리
    # 비어있는 곳은 Title을 기준으로 그룹을 묶고
    # 그 그룹내의 평균 값을 셋팅한다.
    train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
    test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
    # 나이를 그룹으로 묶는다.
    # 16세 이하 : 0, 17 ~ 26 : 1, 27 ~ 36 : 2
    # 37 ~ 62 : 3, 63세 이상 : 4
    for dataset in train_test_data :
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
        dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
        dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
        dataset.loc[dataset['Age'] > 62, 'Age'] = 4

    # bar_chart(train, 'Age')

    # 가족수 데이터 전처리
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    test['FamilySize'] = train['SibSp'] + train['Parch'] + 1

    family_map = {
        1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0,
        7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4
    }
    for dataset in train_test_data :
        dataset['FamilySize'] = dataset['FamilySize'].map(family_map)

    # bar_chart(train, 'FamilySize')

    # Fare 요금 전처리
    # check_nan(train, 'Fare')
    # check_nan(test, 'Fare')
    # nan 데이터 셋팅: 등급을 기준으로 그룹을 묶고, 각 그룹별 중간 값을 넣어준다.
    test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)

    for dataset in train_test_data :
        dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
        dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Age'] = 1,
        dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Age'] = 2,
        dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3

    # bar_chart(train, 'Fare')

    # 객실
    # 첫 글자만 가지고 와서 다시 셋팅한다.
    for dataset in train_test_data :
        dataset['Cabin'] = dataset['Cabin'].str[:1]

    # bar_chart(train, 'Cabin')
    cabin_map = {
        'A':0, 'B':0.4, 'C':0.8, 'D':1.2,
        'E':1.6, 'F':2.0, 'G':2.4, 'T':2.8
    }
    for dataset in train_test_data :
        dataset['Cabin'] = dataset['Cabin'].map(cabin_map)

    # 비어있는 곳은 등급별 중간값으로 셋팅한다.
    train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
    test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

    # bar_chart(train, 'Cabin')
    # 선착장
    # nan 데이터는 S로 채운다. (s가 많아서)
    for dataset in train_test_data :
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    embarked_map = {'S':0, 'C':1, 'Q':2}
    for dataset in train_test_data :
        dataset['Embarked'] = dataset['Embarked'].map(embarked_map)

    # bar_chart(train, 'Embarked')

    # 불필요한 데이터 삭제
    drop_list = ['Ticket', 'SibSp', 'Parch']
    train = train.drop(drop_list, axis=1)
    test = test.drop(drop_list, axis=1)
    train = train.drop(['PassengerId'], axis=1)

    return train, test

# step4-1: 학습 모델을 결정하지 못했다면 교차 검증을 통해 학습 모델을 결정한다.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import collections

def step41_cross_validation(train_data, target) :
    # 10개의 Fold로 나눈다.
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    # 사용할 머신러닝 알고리즘 객체를 만든다.
    clf1 = KNeighborsClassifier(n_neighbors=13)
    clf2 = DecisionTreeClassifier()
    clf3 = RandomForestClassifier(n_estimators=13)
    clf4 = GaussianNB()
    clf5 = SVC(C=1, kernel='rbf', coef0=1)
    # 교차 검증을 실시한다.
    score1 = cross_val_score(clf1, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
    score2 = cross_val_score(clf2, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
    score3 = cross_val_score(clf3, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
    score4 = cross_val_score(clf4, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
    score5 = cross_val_score(clf5, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')

    avg1 = round(np.mean(score1) * 100, 2)
    avg2 = round(np.mean(score2) * 100, 2)
    avg3 = round(np.mean(score3) * 100, 2)
    avg4 = round(np.mean(score4) * 100, 2)
    avg5 = round(np.mean(score5) * 100, 2)

    print('avg1 :', avg1)
    print('avg2 :', avg2)
    print('avg3 :', avg3)
    print('avg4 :', avg4)
    print('avg5 :', avg5)

    # 제일 높게 나온 객체를 반환한다.
    return clf5

# step4-2: 결정된 학습 모델을 사용해 학습을 한다. 학습이 완료된 모델은 반드시 저장한다.
import pickle
def step42_learning(clf, train_data, target) :
    # 학습한다.
    clf.fit(train_data, target)
    # 저장한다.
    with open('ml.dat', 'wb') as fp :
        pickle.dump(clf, fp)

    print("저장완료")

# step5 : 학습된 모델을 사용한다.
def step5_using(test) :
    # 학습이 완료된 객체를 복원한다.
    with open('ml.dat', 'rb') as fp :
        clf = pickle.load(fp)

    # 학습이 완료된 객체를 통해 예측 결과를 만든다.
    # PassengerId가 제거된 객체를 생성한다.
    test_data = test.drop('PassengerId', axis=1)
    y_pred = clf.predict(test_data)

    # 결과를 출력한다.
    result = collections.Counter(y_pred)
    print('사망: :', result[0])
    print('생존: :', result[1])

    # 결과 저장
    submission = pd.DataFrame({
        'PassengerId' : test['PassengerId'],
        'Survived' : y_pred
    })
    submission.to_csv('submission.csv')
    print('결과 저장 완료')


# 데이터를 받아온다.
train, test = step1_get_data()
# print(train)
# print(test)

# 시각화를 통해 데이터를 학습한다.
# step2_learning_data(train)

# 데이터 전처리
train, test = step3_data_preprocessing(train, test)
# print(train.head())
# print(test())

# 학습데이터와 결과 데이터로 나눈다.
# target = train['Survived']
# train_data = train.drop(['Survived'], axis=1)

# 교차 검증
# clf = step41_cross_validation(train_data, target)

# 학습
# step42_learning(clf, train_data, target)

# 사용
step5_using(test)