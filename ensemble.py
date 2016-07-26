import numpy as np
import pandas as pd
import random 
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile 
from sklearn.feature_selection import f_classif 
from sklearn.feature_selection import SelectFpr 

from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from Ndcg import *
from OwnPreProcess import *

import re

np.random.seed(0)

#Loading data
#df_all = pd.read_csv('tmp_nosession.csv')
#df_all = pd.read_csv('tmp_session_src.csv')
df_all = pd.read_csv('src_session_feateng.csv')



#add session data 
#add_session_data = pd.DataFrame()
#sessions_data = pd.read_csv('data/session_feat.csv')
#sessions_data['id'] = sessions_data['user_id'];
#add_session_data['id'] = sessions_data['user_id'];
#sessions_data = sessions_data.drop(['user_id'],axis=1)
#add_session_data['actiondetail_account_payment_methods'] = sessions_data['actiondetail_account_payment_methods'];
#add_session_data['actiondetail_account_payout_preferences'] = sessions_data['actiondetail_account_payout_preferences'];
#add_session_data['actiondetail_delete_payment_instrument'] = sessions_data['actiondetail_delete_payment_instrument'];
#add_session_data['mainaction_ajax_payout_edit'] = sessions_data['mainaction_ajax_payout_edit'];
#add_session_data['mainaction_ajax_payout_options_by_country'] = sessions_data['mainaction_ajax_payout_options_by_country'];
#df_all = pd.merge(df_all,add_session_data,left_on='id',right_on='id',how='left',sort=False)
#df_all = pd.merge(df_all,sessions_data,left_on='id',right_on='id',how='left',sort=False)

#drop id feat
df_all = df_all.drop(['id'],axis=1)
#process NA data
df_all['young_or_old'] = df_all['young_or_old'].fillna(-1)
df_all['age'] = df_all['age'].fillna(-1)
df_all['dac_day'] = df_all['dac_day'].fillna(-1)
df_all['dac_month'] = df_all['dac_month'].fillna(-1)
df_all['dac_year'] = df_all['dac_year'].fillna(-1)
df_all['tfa_day'] = df_all['tfa_day'].fillna(-1)
df_all['tfa_month'] = df_all['tfa_month'].fillna(-1)
df_all['tfa_year'] = df_all['tfa_year'].fillna(-1)

#unknown data
#df_all['actiondetail_-unknown-'] = df_all['actiondetail_-unknown-'].fillna(300)
#df_all['actiontype_-unknown-'] = df_all['actiontype_-unknown-'].fillna(300)
#df_all['devicetype_-unknown-'] = df_all['devicetype_-unknown-'].fillna(300)
#df_all['mainaction_-unknown-'] = df_all['mainaction_-unknown-'].fillna(300)

df_all = df_all.fillna(0)
#df_all = df_all.drop(['young_or_old'],axis=1)
df_all = df_all.drop(['2014_before_after','blank','payment_sum','recommend_sum','wishlist_sum','password_sum'],axis=1)
df_all = df_all.drop(['season_autm','season_spring','season_summer','season_winter'],axis=1)
df_all = df_all.drop(['AU_prob_age','CA_prob_age','DE_prob_age','ES_prob_age','FR_prob_age','GB_prob_age','IT_prob_age','NL_prob_age','PT_prob_age','US_prob_age'],axis=1)
df_all = df_all.drop(['AU_prob_gender','CA_prob_gender','DE_prob_gender','ES_prob_gender','FR_prob_gender','GB_prob_gender','IT_prob_gender','NL_prob_gender','PT_prob_gender','US_prob_gender'],axis=1)

#df_all = df_all.drop(['devicetype_-unknown-','devicetype_Android App Unknown Phone/Tablet','devicetype_Android Phone','devicetype_Blackberry','devicetype_Chromebook','devicetype_Linux Desktop','devicetype_Mac Desktop','devicetype_Opera Phone','devicetype_Tablet','devicetype_Windows Desktop','devicetype_Windows Phone','devicetype_iPad Tablet','devicetype_iPhone','devicetype_iPodtouch'],axis=1)

vals = df_all.values
#vals = Kmeans_process_vals_norm(vals)
#vals = Kmeans_process_vals(vals)
#vals = knearst_process_vals(vals)

print 'finsish preprocessing feat'

df_test = pd.read_csv('data/test_users.csv')
id_test = df_test['id']
df_train = pd.read_csv('data/train_users.csv')
df_train_label = pd.read_csv('train_label.csv')
labels = df_train_label['country_destination'].values
piv_train = df_train.shape[0]
#label data
le = LabelEncoder()
y = le.fit_transform(labels)   


#feature selection:
print "feature len:" + str(df_all.shape[1])
#print 'feature selection:'
feat_select = SelectKBest(f_classif, k=350).fit(vals[:piv_train],y)
#feat_select = SelectPercentile(f_classif, percentile=50).fit(vals[:piv_train],y)
vals = feat_select.transform(vals)
print "feature len:" + str(len(vals[0]))

val_num = 10000 
#val_num = 0 
piv_train = piv_train - val_num

X_val = vals[:val_num] 
X_train = vals[val_num:val_num+piv_train] 
X_test = vals[val_num+piv_train:] 

#normalized data
norm_vals = preprocessing.MinMaxScaler().fit_transform(vals)
norm_X_val = norm_vals[:val_num] 
norm_X_train = norm_vals[val_num:val_num+piv_train] 
norm_X_test = norm_vals[val_num+piv_train:]

y_val = y[:val_num]
y_train = y[val_num:val_num+piv_train]

#print 'start LDA'
#clf_lda = LDA(n_components=350).fit(norm_X_train,y_train)
#X_train_lda = clf_lda.transform(norm_X_train)
#X_val_lda = clf_lda.transform(norm_X_val)
#X_test_lda = clf_lda.transform(norm_X_test)
###
#X_val = X_val_lda
#norm_X_val = X_val_lda
#X_train = X_train_lda
#norm_X_train = X_train_lda
#X_test = X_test_lda
#norm_X_test = X_test_lda

#print 'start PCA'
#pca = PCA(n_components=300)
#pca.fit(norm_X_train)
#X_train_pca = pca.transform(norm_X_train)
#X_val_pca = pca.transform(norm_X_val)
#X_test_pca = pca.transform(norm_X_test)
###
#X_val = X_val_pca
#norm_X_val = X_val_pca
#X_train = X_train_pca
#norm_X_train = X_train_pca
#X_test = X_test_pca
#norm_X_test = X_test_pca

#+++++++++++++++++++++++++++++++++++++Classifier+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#boosting 
print 'start boosting'
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0).fit(X_train, y_train)                  
y_pred_boosting = xgb.predict_proba(X_test)  
if val_num != 0:
    y_pred_boosting_val = xgb.predict_proba(X_val)
#
#randomforest
#print 'start random forest'
#clf_randforest = RandomForestClassifier().fit(X_train, y_train)                  
#y_pred_randforest = clf_randforest.predict_proba(X_test)  
#if val_num != 0:
#    y_pred_randforest_val = clf_randforest.predict_proba(X_val)  
#
##bagging
#print 'start bagging'
#clf2 = BaggingClassifier(n_estimators=100).fit(X_train, y_train)                  
#y_pred_bagging = clf2.predict_proba(X_test)
#y_pred_bagging_val = clf2.predict_proba(X_val)
#
#logistic regression
print 'start logistci regression'
clf_lr = LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(norm_X_train,y_train)
y_pred_LR = clf_lr.predict_proba(norm_X_test)
if val_num != 0:
    y_pred_LR_val = clf_lr.predict_proba(norm_X_val)
#
#adaboost
#print 'start adaboost'
#clf_adaboost = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
#y_pred_adaboost = clf_adaboost.predict_proba(X_test)  
#if val_num != 0:
#    y_pred_adaboost_val = clf_adaboost.predict_proba(X_val)  
#
#SVM
#print 'start SVM'
#clf_svm = SVC()
#clf_svm.fit(X_train_pca,y_train)
#y_pre_svm = clf_svm.predict_proba(X_test_pca)
#y_pre_svm_val = clf_svm.predict_proba(X_val_pca)
#

#print 'start decision tree'
#clf_dt = DecisionTreeClassifier().fit(X_train,y_train)
#y_pred_DT = clf_dt.predict_proba(X_test)
#if val_num != 0:
#    y_pred_DT_val = clf_dt.predict_proba(X_val)
print '+++++++++++++++++++++start predict+++++++++++++++++++++++'
#y_pred = np.add(np.add(np.add(y_pred_randforest,y_pred_bagging),np.add(y_pred_LR,y_pred_boosting)),y_pred_adaboost)
#y_pred_val = np.add(np.add(np.add(y_pred_randforest_val,y_pred_bagging_val),np.add(y_pred_LR_val,y_pred_boosting_val)),y_pred_adaboost_val)
#y_pred = np.add(y_pred_boosting,np.add(y_pred_LR,y_pred_adaboost))
#if val_num != 0:
#    y_pred_val = np.add(y_pred_adaboost_val,np.add(y_pred_LR_val,y_pred_boosting_val))
#y_pred = y_pred_boosting
#if val_num != 0:
#    y_pred_val = y_pred_boosting_val
#find best proportation for ensenmble weights:
max_LB = 0
max_NDCG = 0
max_LB_w = [];
max_NDCG_w = [];
if val_num != 0:
    #for i in range(0,1):
    for i in range(0,11,1):
        w1 = float(i)/10
        w2 = 1 - w1 
        print "w1:"+str(w1)+" w2:"+str(w2)
        y_pred_LR_val_tmp = [elem *w1 for elem in y_pred_LR_val]
        y_pred_boosting_val_tmp = [elem *w2 for elem in y_pred_boosting_val]
        y_pred_val = np.add(y_pred_LR_val_tmp,y_pred_boosting_val_tmp)
        #y_pred_val = y_pred_boosting_val
        #y_pred_val = y_pred_LR_val 
#LB validate data
        true_num = 0
        for ii in range(0,val_num):
            if np.argsort(y_pred_val[ii])[::-1][0]==y_val[ii]:
                true_num = true_num + 1
        tmp_LB = float(true_num)/val_num
        print "validate LB precision: " + str(tmp_LB)
        if(tmp_LB>max_LB):
            max_LB_w = [w1,w2]
            max_LB = tmp_LB
#NDCG validate data
        labels_pred = [] #list of validate 
        labels_val = le.inverse_transform(y_val).tolist()
        for ii in range(len(y_pred_val)):
            tmp_pred_label = le.inverse_transform(np.argsort(y_pred_val[ii])[::-1])[:5].tolist() 
            labels_pred.append(tmp_pred_label)
        labels_val = pd.Series(labels_val)
        labels_pred = pd.DataFrame(labels_pred)
        tmp_NDCG = score_predictions(labels_pred,labels_val)
        print "validate NDCG score: " + str(tmp_NDCG)
        if(tmp_NDCG>max_NDCG):
            max_NDCG_w = [w1,w2]
            max_NDCG = tmp_NDCG
    print "max_LB_w:"
    print max_LB_w
    print "max_NDCG_w:"
    print max_NDCG_w
    exit()

#Taking the 5 classes with highest probabilities
w = 0.55
#y_pred_LR = [elem*w for elem in y_pred_LR]
#y_pred_boosting = [elem*(1-w) for elem in y_pred_boosting]
#y_pred = np.add(y_pred_LR,y_pred_boosting)
y_pred = y_pred_boosting
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('ret_ensemble_0204.csv',index=False)
