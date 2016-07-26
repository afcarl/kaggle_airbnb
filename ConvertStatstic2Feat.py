import numpy as np
import pandas as pd

age_buckets = ['95-99','90-94','85-89','80-84','75-79','70-74','65-69','60-64','55-59','50-54','45-49','40-44','35-39','30-34','25-29','20-24','15-19']
dst_countreis = ['AU','CA','DE','ES','FR','GB','IT','NL','PT','US']
gender_buckets = ['male','female']
df_all = pd.read_csv('data/age_gender_bkts.csv')
def GetStatsticAgeData():
    age_dstcountries = []
    for tmp_age in age_buckets:
        tmp_df = df_all.loc[df_all['age_bucket'] == tmp_age]
        tmp_age_dstcountries = []
        for tmp_countries in dst_countreis:
            tmp_elm = 0;
            tmp_elm = tmp_elm + np.sum(np.array(tmp_df.loc[tmp_df['country_destination'] == tmp_countries].values[:,3]))
            tmp_age_dstcountries.append(tmp_elm)
        age_dstcountries.append(tmp_age_dstcountries)
    return age_dstcountries
def GetStatsticGenderData():
    gender_dstcountries = []
    for tmp_gender in gender_buckets:
        tmp_df = df_all.loc[df_all['gender'] == tmp_gender]
        tmp_gender_dstcountries = []
        for tmp_countries in dst_countreis:
            tmp_elm = 0;
            tmp_elm = tmp_elm + np.sum(np.array(tmp_df.loc[tmp_df['country_destination'] == tmp_countries].values[:,3]))
            tmp_gender_dstcountries.append(tmp_elm)
        gender_dstcountries.append(tmp_gender_dstcountries)
    return gender_dstcountries
#df_test = pd.DataFrame(GetStatsticGenderData(),columns=dst_countreis)
#print df_test


