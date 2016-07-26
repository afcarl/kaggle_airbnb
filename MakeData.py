import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

from ConvertStatstic2Feat import *

np.random.seed(0)

#Loading data
df_train = pd.read_csv('data/train_users.csv')
df_train = df_train.iloc[np.random.permutation(len(df_train))]
df_label = pd.DataFrame();
df_label['country_destination'] = df_train['country_destination']
df_label.to_csv('train_label.csv',index=False)
df_test = pd.read_csv('data/test_users.csv')
labels = df_train['country_destination'].values
df_labels = pd.DataFrame()
df_labels['country_destination'] = df_train['country_destination']
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
dfb_all = df_all['date_first_booking'].values
dfb_year = []
dfb_mon = []
dfb_day = []
for dfb in dfb_all:
    if str(dfb) != 'nan':
        tmp_dfb = str(dfb).split('-')
        dfb_year.append(int(tmp_dfb[0]))
        dfb_mon.append(int(tmp_dfb[1]))
        dfb_day.append(int(tmp_dfb[2]))
    else:
        dfb_year.append(np.NAN)
        dfb_mon.append(np.NAN)
        dfb_day.append(np.NAN)
#Removing id and date_first_booking
df_all = df_all.drop(['date_first_booking'], axis=1)
#Filling nan
#df_all = df_all.fillna(-1)

#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#add new feature: before or after 2014
df_before_after = []
av = df_all.dac_year.values
for i in av:
    if np.isnan(i):
        df_before_after.append(np.nan)
    elif i<2014:
        df_before_after.append(1)
    else:
        df_before_after.append(0)
df_all['2014_before_after'] = df_before_after 

av = df_all.dac_month.values
df_season = []
for i in av:
    if i>=3 and i<=5:
        df_season.append('spring')
    elif i>=6 and i<=8:
        df_season.append('summer')
    elif i>=9 and i<=11:
        df_season.append('autm')
    elif i==12 or (i>=1 and i<=2):
        df_season.append('winter')
    else:
        df_season.append('unknow')
df_all['season'] = df_season 



#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#transfor Date to Nunmber
#df_all['tfa_number'] = (df_all['tfa_year']-2010)*365 + df_all['tfa_month']*30 + df_all['tfa_day']
#df_all = df_all.drop(['tfa_year','tfa_month','tfa_day'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<15, av>99), np.nan, av)

#add new feature: young_old
df_young_old = []
df_young = []
df_old = []
df_puber = []
av = df_all.age.values
for i in range(0,len(av)):
    if np.isnan(av[i]):
        df_young_old.append(np.nan)
        #df_young.append(0)
        #df_old.append(0)
        #df_puber.append(0)
    else:
        if av[i]>45:
            df_young_old.append(1)
            #df_young.append(0) 
            #df_old.append(1) 
            #df_puber.append(0) 
        #elif av[i]<22:
            #df_young.append(1) 
            #df_old.append(0) 
            #df_puber.append(0) 
        else:
            df_young_old.append(0)
            #df_young.append(0) 
            #df_old.append(0) 
            #df_puber.append(1)
#df_all['age_peroied'] = df_young_old
df_all['young_or_old'] = df_young_old
#add new feature: age_bkts_dstcountries
age_dstcountries = GetStatsticAgeData()
av = df_all.age.values
feat_age_dst = []
for i in range(0,len(av)):
    if np.isnan(av[i]):
        feat_age_dst.append([np.nan for i in range(len(age_dstcountries[0]))])
    else:
        feat_age_dst.append(age_dstcountries[int((99-av[i])/5)])
dst_countreis_prob_age = ['AU_prob_age','CA_prob_age','DE_prob_age','ES_prob_age','FR_prob_age','GB_prob_age','IT_prob_age','NL_prob_age','PT_prob_age','US_prob_age']
df_age_dstcountries = pd.DataFrame(feat_age_dst,columns=dst_countreis_prob_age)
df_all = pd.concat((df_all, df_age_dstcountries), axis=1)

#gender
df_all.gender.replace(np.nan,'-unknown-',inplace=True)
#add new feature: gender_dstcountries
gender_dstcountries = GetStatsticAgeData()
av = df_all.gender.values
feat_gender_dst = []
for i in range(0,len(av)):
    if av[i]=='MALE':
        feat_gender_dst.append(gender_dstcountries[0])
    elif av[i]=='FEMALE':
        feat_gender_dst.append(gender_dstcountries[1])
    else:
        feat_gender_dst.append([np.nan for i in range(len(gender_dstcountries[0]))])
dst_countreis_prob_gender = ['AU_prob_gender','CA_prob_gender','DE_prob_gender','ES_prob_gender','FR_prob_gender','GB_prob_gender','IT_prob_gender','NL_prob_gender','PT_prob_gender','US_prob_gender']
df_gender_dstcountries = pd.DataFrame(feat_gender_dst,columns=dst_countreis_prob_gender)
df_all = pd.concat((df_all, df_gender_dstcountries), axis=1)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser','season']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#pre process countries data
df_countries = pd.read_csv('data/countries.csv')
df_countries_id = pd.merge(df_labels,df_countries,left_on='country_destination',right_on='country_destination',how='left',sort=False)
df_countries_id = df_countries_id.drop(['country_destination'], axis=1)
df_all_dummy = pd.get_dummies(df_countries_id['destination_language'], prefix='destination_language')
df_countries_id = df_countries_id.drop(['destination_language'],axis=1)
df_countries_id = pd.concat((df_countries_id,df_all_dummy),axis=1)
#combine countries data
#df_all = pd.concat((df_all,df_countries_id),axis=1)

df_all.to_csv('tmp_nosession.csv',index=False)
#pre process session data
print "start session data process"
sessions_data = pd.read_csv('data/session_feat.csv')
sessions_data['id'] = sessions_data['user_id'];
sessions_data = sessions_data.drop(['user_id'],axis=1)
#combine sessions data
df_all = pd.merge(df_all,sessions_data,left_on='id',right_on='id',how='left',sort=False)
print "session data process finish"
df_all.to_csv('tmp_session_src.csv',index=False)

print "make data done"
exit()

