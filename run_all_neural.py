from __future__ import print_function
import os
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
gpu_use = sys.argv[1]#input('Use gpu: ')
if gpu_use == '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use
import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
tf.Session(config=tf_config)
import numpy as np
import pandas as pd
from random import shuffle
import random
import hickle as hkl
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

sys.path.append('/home/chenshengquan/')



import keras
from keras.datasets import cifar10
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

def evaluate_performance(y_truth, y_pred, p_y_given_x):
    auROC = metrics.roc_auc_score(y_truth, p_y_given_x)
    print('auROC = {}'.format(auROC))
    auPR = metrics.average_precision_score(y_truth, p_y_given_x)
    print('auPR  = {}'.format(auPR))
    acc = metrics.accuracy_score(y_truth, y_pred)
    print('ACC   = {}'.format(acc))

data_type  = 'residual'
pair_feature_number = 100
feature_number = 100
test_name=" "
if int(sys.argv[2])==1:
    test_name = "xin"
if int(sys.argv[2])==2:
    test_name = "baron"
if int(sys.argv[2])==3:
    test_name = "muraro"
if int(sys.argv[2])==4:
    test_name = "segerstolpe"
if int(sys.argv[2])==5:
    test_name = "romanov"
if int(sys.argv[2])==6:
    test_name = "zeisel"
if int(sys.argv[2])==7:
    test_name = "macosko"
if int(sys.argv[2])==8:
    test_name = "shekhar"
if int(sys.argv[2])==9:
    test_name = "darmanis" 
if int(sys.argv[2])==10:
    test_name = "retina"
if int(sys.argv[2])==11:
    test_name = "baron2" 
method = " "
if int(sys.argv[3])==1:
    method = "PCA"
if int(sys.argv[3])==2:
    method = "myself"
if int(sys.argv[3])==3:
    method = "460147"
if int(sys.argv[3])==4:
    method = "scmap"


#pair_id = int(sys.argv[2])
print(test_name)
#batch = int(sys.argv[5])

data  = hkl.load('/home/chenxiaoyang/program/scmap/Data/processed/%s/data_train_%d_sample_%s.hkl'%(test_name, feature_number,method))
data_test = hkl.load(
        '/home/chenxiaoyang/program/scmap/Data/processed/%s/data_test_%d_sample_%s.hkl' % (test_name, feature_number,method))
label_train  = hkl.load('/home/chenxiaoyang/program/scmap/Data/processed/%s/sorted_%s/data_train_label.hkl'%(test_name,data_type))
label_test = hkl.load('/home/chenxiaoyang/program/scmap/Data/processed/%s/sorted_%s/data_test_label.hkl'%(test_name,data_type))
if 1==1:

    few_class_label = []
    for i in range(min(5,len(pd.value_counts(label_train)))):
        if pd.value_counts(label_train)[len(pd.value_counts(label_train))-1-i] < 0.005*len(label_train):
            few_class_label.append(pd.value_counts(label_train).index[len(pd.value_counts(label_train))-1-i])

    few_data = []
    few_label = []
    label =label_train
    for i in range(len(label)):
        if label[i] in few_class_label:
            few_data.append(data[i,:])
            few_label.append(label[i])
    pair_sample_train = []
    pair_label_train = []
    for i in range(len(few_label)):
        for j in range(len(few_label)):
            pair_sample_train.append([few_data[i], few_data[j]])
            if few_label[i] == few_label[j]:
                pair_label_train.append(1)
            else:
                pair_label_train.append(0)
    pair_sample_test = []
    pair_label_test = []
    for i in range(len(label_test)):
        for j in range(len(few_label)):
            pair_sample_test.append([data_test[i,:], few_data[j]])
            if label_test[i] ==few_label[j]:
                pair_label_test.append(1)
            else:
                pair_label_test.append(0)
    if len(few_class_label)>1:
        pair_sample_test = np.array(pair_sample_test).reshape(-1,200)
        pair_sample_train = np.array(pair_sample_train).reshape(-1,200)
        model_ind = 4

        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=200))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.25))
        #model.add(Dense(16, activation='relu'))
        #model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        batch_size = 16
        epochs = 30
        save_dir = os.path.join('/home/chenxiaoyang/program/scmap/Data/processed/segerstolpe/', 'saved_models')
        model_name = 'direct_%02d_trained_%s_model.h5'%(model_ind, data_type)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_acc', verbose=0, patience=3, mode='max')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        save_best = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
        if os.path.exists(model_path):
            print('*** WARNING: the model already exists. Are you sure about overwriting??? ***')
        print('Save trained model at %s ' % model_path)
        model.fit(pair_sample_train, pair_label_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True,
                  callbacks=[early_stopping, save_best])
        y_prob = model.predict(pair_sample_test, batch_size=batch_size, verbose=1)
        from scipy.stats import spearmanr
        from scipy.stats import pearsonr
        pearsonr_list = []
        for i in range(pair_sample_test.shape[0]):
            item = pair_sample_test[i]
            pearsonr_list.append(pearsonr(item[:100], item[100:])[0])
        pearsonr_list = np.around(pearsonr_list, decimals=3)
        pearsonr_list = pearsonr_list.reshape(-1 , len(few_label))
        pred_res_each_cell = y_prob.reshape(-1,len(few_label))
        normalized_pred_res_each_cell = pred_res_each_cell * pearsonr_list
        label_index_dict = dict()
        for label in (set(few_label)):
            label_index_dict[label] = [ i for i in range(len(few_label)) if few_label[i] == label]
        pred_label_names = []
        for i in range(normalized_pred_res_each_cell.shape[0]):
            ref_label_names = []
            ref_label_scores = []
            for key, value in label_index_dict.items():
                ref_label_scores.append(np.mean(normalized_pred_res_each_cell[i,value]))
                ref_label_names.append(key)


            if np.std(ref_label_scores) == 0 or 0.4 > np.max(ref_label_scores):
                pred_label_names.append('unassigned')
            else:
                #ref_label_scores = ref_label_scores.tolist()
                pred_label_names.append(ref_label_names[ref_label_scores.index(max(ref_label_scores))])
        pred_num = []
        pred_assigned_names = []
        for i in range(len(pred_label_names)):
            if pred_label_names[i] != 'unassigned':
                pred_assigned_names.append(pred_label_names[i])
                pred_num.append(i)
    else:
        pred_num = []
        pred_assigned_names = []
        pred_label_names = ["unassigned"]*len(label_test)

LE = LabelEncoder()
LE.fit(label_train+label_test)
y_ = LE.transform(label_train+label_test)
y_train = LE.transform(label_train)
y_test = LE.transform(label_test)
pred_assigned_names = LE.transform(pred_assigned_names)
print(y_train)
y_train = keras.utils.to_categorical(y_train, np.unique(y_).shape[0])
data = data.reshape(-1, 100)
print(data.shape[0])
print(y_train.shape[0])
model_ind = 2
X_train = data
Y_train = y_train

for batch in range(5):

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=100))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(np.unique(y_).shape[0], activation='softmax'))

    batch_size = 16
    epochs = 30
    save_dir = os.path.join('/home/chenxiaoyang/program/scmap/Data/processed/%s/'%test_name, 'neural','%d'%batch)
    model_name = 'dense_%02d_trained_model_%s.h5'%(model_ind,method)

    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_acc', verbose=0, patience=3, mode='max')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    save_best = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
    if os.path.exists(model_path):
        print('*** WARNING: the model already exists. Are you sure about overwriting??? ***')
    print('Save trained model at %s ' % model_path)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True,
          callbacks=[early_stopping, save_best])
for batch in range(5):

    pred = []

    save_dir = os.path.join('/home/chenxiaoyang/program/scmap/Data/processed/%s/' % test_name, 'neural','%d' % batch)
    model_path = os.path.join(save_dir, model_name)
    model.load_weights(model_path)
    y_prob = model.predict(data_test, batch_size=batch_size, verbose=1)
    #hkl.dump(y_prob,'/home/chenxiaoyang/program/scmap/Data/processed/%s/%d/%d/y_prob_sample.hkl' % (test_name, pair_id, batch))
    for item in y_prob:
        if np.max(item) > 0.7:
            pred.append(np.argmax(item))
        else:
            pred.append(100)
    pd.value_counts(pred)
    pred_names_list = pred
    print(y_prob)
    pred_label_names = pred
    k = y_prob.shape[0]
    label = y_test.tolist()

    num = 0
    for i in range(k):
        if i in pred_num:
            pred_names_list[i] = pred_assigned_names[num]
            num = num+1
    print(len(pred_names_list))
    A = metrics.cohen_kappa_score(label, pred_names_list)
    pred = pred_names_list

    assigned_ind = []
    for i in range(len(pred)):
        if pred[i] != 100:
            assigned_ind.append(i)
    assigned_ind = []

    for i in range(len(pred)):
        if pred[i] != 100:
            assigned_ind.append(i)
    filter_unassgin_label = [label[i] for i in assigned_ind]
    filter_unassgin_pred = [pred[i] for i in assigned_ind]
    B = metrics.cohen_kappa_score(filter_unassgin_label, filter_unassgin_pred)
    assigned_rate = len(assigned_ind) / len(label)
    file = open('/home/chenxiaoyang/program/scmap/Data/processed/%s/sample_%s_%d_neural.txt' % (test_name,method, batch), 'w')
    file.write("%f\n%f\n%f" % (A, B, assigned_rate))