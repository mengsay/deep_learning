import random
import numpy as np
import cv2

PATH_AND_LABEL = []

def load_self_data(test_ratio):
    # パスとラベルが記載されたファイルから、それらのリストを作成する
    with open("data_label.txt", mode='r') as file :
        for line in file :
            # 改行を除く
            line = line.rstrip()
            # スペースで区切られたlineを、リストにする
            line_list = line.split()
            PATH_AND_LABEL.append(line_list)
            # 同じジャンルのサムネイルが、かたまらないように、シャッフルする
            random.shuffle(PATH_AND_LABEL)

    DATA_SET = []

    for path_label in PATH_AND_LABEL :

        tmp_list = []

        # 画像を読み込み、サイズを変更する
        #print(path_label[0])
        img = cv2.imread(path_label[0])
        img = cv2.resize(img, (28, 28))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # (28, 28, 3)のN次元配列を一次元に、dtypeをfloat32に、０〜１の値に正規化する
        img = img.flatten().astype(np.float32)/255.0

        #print(img)
        
        tmp_list.append(img)
        
        

        # 分類するクラス数の長さを持つ仮のN次元配列を作成する
        classes_array = np.zeros(10, dtype = 'float64')
        # ラベルの数字によって、リストを更新する
        classes_array[int(path_label[1])] = 1

        tmp_list.append(classes_array)
        
        DATA_SET.append(tmp_list)
        
    TRAIN_DATA_SIZE = int(len(DATA_SET) * test_ratio)
    TRAIN_DATA_SET = (DATA_SET[:TRAIN_DATA_SIZE])
    TEST_DATA_SET = (DATA_SET[TRAIN_DATA_SIZE:])
    
    x_train=[];t_train=[];x_test=[];t_test=[]
    
    for i in TRAIN_DATA_SET:
        x_train.append(i[0])
    x_train=np.array(x_train)
    
    for i in TRAIN_DATA_SET:
        t_train.append(i[1])
    t_train=np.array(t_train)
    
    for i in TEST_DATA_SET:
        x_test.append(i[0])
    x_test=np.array(x_test)
    
    for i in TEST_DATA_SET:
        t_test.append(i[1])
    t_test=np.array(t_test)
    
    
    return (x_train,t_train),(x_test,t_test)
