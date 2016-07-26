import numpy as np
import pandas as pd
import re


df_all = pd.read_csv('tmp_session_src.csv')

#feature engerning
translate_index = []
payment_index = []
wishlist_index = []
recommend_index = []
password_index = []
colums_name = df_all.columns
for i in range(0,len(colums_name)):
    if(None!=re.match(r'.*?translate.*?',colums_name[i])):
            translate_index.append(i)
    if(None!=re.match(r'.*?payment.*?',colums_name[i])):
            payment_index.append(i)
    if(None!=re.match(r'.*?wishlist.*?',colums_name[i])):
            wishlist_index.append(i)
    if(None!=re.match(r'.*?recommend.*?',colums_name[i])):
            recommend_index.append(i)
    if(None!=re.match(r'.*?password.*?',colums_name[i])):
            password_index.append(i)
vals = df_all.values
translate_values = []
payment_values = []
wishlist_values = []
recommend_values = []
password_values = []
blank_values = []
keep_threshold = 5000
for i in range(0,len(vals)):
    print str(i) + '/' + str(len(vals))
    tmp_val = np.sum(vals[i][translate_index])
    if(tmp_val==np.nan):
        translate_values.append(np.nan)
    elif tmp_val<keep_threshold*len(translate_index):
        translate_values.append(0)
    else:
        translate_values.append(1)
    tmp_val = np.sum(vals[i][payment_index])
    if(tmp_val==np.nan):
        payment_values.append(np.nan)
    elif tmp_val<keep_threshold*len(payment_index):
        payment_values.append(0)
    else:
        payment_values.append(1)
    tmp_val = np.sum(vals[i][wishlist_index])
    if(tmp_val==np.nan):
        wishlist_values.append(np.nan)
    elif tmp_val<keep_threshold*len(wishlist_index):
        wishlist_values.append(0)
    else:
        wishlist_values.append(1)
    tmp_val = np.sum(vals[i][recommend_index])
    if(tmp_val==np.nan):
        recommend_values.append(np.nan)
    elif tmp_val<keep_threshold*len(recommend_index):
        recommend_values.append(0)
    else:
        recommend_values.append(1)
    tmp_val = np.sum(vals[i][password_index])
    if(tmp_val==np.nan):
        password_values.append(np.nan)
    elif tmp_val<keep_threshold*len(password_index):
        password_values.append(0)
    else:
        password_values.append(1)
    blank_values.append(0.5)
df_all['translate_sum'] = translate_values
df_all['payment_sum'] = payment_values
df_all['recommend_sum'] = recommend_values
df_all['wishlist_sum'] = wishlist_values
df_all['password_sum'] = password_values
df_all['blank'] = blank_values 

print 'Make data done'
df_all.to_csv('src_session_feateng.csv',index=False)
exit()
