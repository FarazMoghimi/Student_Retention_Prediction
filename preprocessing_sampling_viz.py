import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from zipfile import ZipFile
from io import StringIO
import fnmatch
from pandas import ExcelWriter
from dirty_cat import GapEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df_s2018 = pd.read_spss("2018 Spring_Enrollment Census.sav")
df_f2018 = pd.read_spss("2018 Fall_Enrollment Census.sav")
df_s2019 = pd.read_spss("2019 Spring_Enrollment Census.sav")
df_f2019 = pd.read_spss("2019 Fall_Enrollment Census.sav")
df_s2020 = pd.read_spss("2020 Spring_Enrollment Census.sav")
df_f2020 = pd.read_spss("2020 Fall_Enrollment Census.sav")
df_s2021 = pd.read_spss("2021 Spring_Enrollment Census.sav")
df_lookup= pd.read_spss("Lookup STRM_SemNo.sav")
df_dsa= pd.read_csv('DSA Coding Workbook.csv')
pd.options.display.max_columns = None  
    
#all fall data and spring 2018 data have the same column structure so does the s19, However, the s20, 



#removing unwanted columns 
df_s2018= df_s2018.loc[:,~df_s2018.columns.str.startswith('student_group_long')]
df_s2018= df_s2018.loc[:,~df_s2018.columns.str.startswith('SEM')]

col_del= ['FstDegree_Type','FirstGen','NatAM_tribe' , 'CHINESE', 'AnyCC', 'previous_deg1', 'FstDegree', 'race_federal', 'BachDegree'
          ,'BachCollege', 'BachCollege', 'BachMajor1', 'strm_demo' , 'athlete_flag', 'HawPac', 'FirstEnrollSemNo',
         'Sem', 'age', 'FstSemMajor', 'LastGPA','LastCollege', 'LastMajor2', 'LastMajor1', 'OneYrRetention', 'TwoYrRetention', 
         'BachCohortYear', 'census_term_short', 'Cohort', 'STRM', 'ThreeYrGradRate', 'FourYrGradRate', 'FiveYrGradRate', 'SixYrGradRate', 
          'census_term_demo', 'LastSTRM', 'BachSTRM', 'LastYear', 'LastTotCredits']

#STRM_SemNo is the starting semester code

df_s2018= df_s2018.drop(columns= col_del)

#creating a benchmark column list based on s2018 that could be used to standardize the entire data
bench_col=df_s2018.columns.values


#standardizing all census columns : 
df_s2021.rename(columns={"pell_eligible": "fa_eligible"}, inplace= True)
df_s2020.rename(columns={"pell_eligible": "fa_eligible"}, inplace= True)
df_s2019.rename(columns={"pell_eligible": "fa_eligible"}, inplace= True)
df_f2019.rename(columns={"pell_eligible": "fa_eligible"}, inplace= True)
df_f2020.rename(columns={"pell_eligible": "fa_eligible"}, inplace= True)
df_f2018.rename(columns={"pell_eligible": "fa_eligible"}, inplace= True)
df_s2019= df_s2019[bench_col]
df_s2020= df_s2020[bench_col]
df_s2021= df_s2021[bench_col]
df_f2019= df_f2019[bench_col]
df_f2018= df_f2018[bench_col]
df_f2020= df_f2020[bench_col]


#for initial stats related to student affairs only
dsa_list=df_dsa['WISER name ']
dsa_list.dropna(inplace= True )
dsa_list.reset_index(inplace=True, drop=True)


#sorting out the student affairs data function 
sa_col= df_s2018.columns[pd.Series(df_s2018.columns).str.startswith('student_group_code')]
def sta_func(df):
    df['USES']=0
    df['HRES']=0
    df['SASE']=0
    df['issta']=0
    for col in sa_col:
        df['USES'] = np.where(df[col].str.contains("USES") , 1,df['USES'])
        df['HRES'] = np.where(df[col].str.contains("HRES") , 1,df['HRES'])
        df['SASE'] = np.where(df[col].str.contains("SASE") , 1,df['SASE'])
        df['issta'] = np.where(df[col].str.contains("USG") , 1,df['issta'])
        df['issta'] = np.where(df[col].str.contains("CADS") , 1,df['issta'])                           
        df['issta'] = np.where(df[col].str.contains("FLI") , 1,df['issta'])
        df['issta'] = np.where(df[col].str.contains("RLRA") , 1,df['issta'])
        df['issta'] = np.where(df[col].str.contains("SPA") , 1,df['issta'])
        df['issta'] = np.where(df[col].str.contains("UASC") , 1,df['issta'])
        df['issta'] = np.where(df[col].str.contains("UASL") , 1,df['issta'])
        
    df= df.drop(columns= sa_col)
    return df
#adding the sem census sem number to everything

df_s2018['censusSemNo']= 68
df_f2018['censusSemNo']= 69
df_s2019['censusSemNo']= 70
df_f2019['censusSemNo']= 71
df_s2020['censusSemNo']= 72
df_f2020['censusSemNo']= 73
df_s2021['censusSemNo']= 74

def label_func(df):
    df['label'] = np.where((df['LastSemNo']> df['censusSemNo']+1) |(df['BachSemNo'].notna()), 1,0)
    return df

def fix_minmax(df,col, min, max):
    df[col] = np.where((df[col]< min), min, df[col])
    df[col] = np.where((df[col]> max), max, df[col])
    return df
  
  #creating a label column
df_s2018= df_s2018.pipe(label_func)
df_s2019= df_s2019.pipe(label_func)
df_s2020= df_s2020.pipe(label_func) #uncomment after you receive the full s2020 data 
df_s2021= df_s2021.pipe(label_func)
df_f2018= df_f2018.pipe(label_func)
df_f2019= df_f2019.pipe(label_func)
df_f2020= df_f2020.pipe(label_func)
#preprocessing student affairs data based the created function
df_s2018= df_s2018.pipe(sta_func)
df_s2019= df_s2019.pipe(sta_func)
df_s2020= df_s2020.pipe(sta_func) #uncomment after you receive the full s2020 data 
df_s2021= df_s2021.pipe(sta_func)
df_f2018= df_f2018.pipe(sta_func)
df_f2019= df_f2019.pipe(sta_func)
df_f2020= df_f2020.pipe(sta_func)

#getting everything concatnated
df=pd.concat([df_s2018, df_f2018,df_s2019, df_f2019,df_s2020 ], sort=False, ignore_index= True)

# handling the pre umass scores : sat, transfer, ....

# fixing the borders
fix_minmax(df, 'gpa_hs',0, 4)
fix_minmax(df, 'SAT_MATH_score',200, 800)
fix_minmax(df, 'SAT_VERB_score',200, 800)
fix_minmax(df, 'SAT_MSS_score',400, 1600)
fix_minmax(df, 'SAT_ERWS_score',400, 1600)
fix_minmax(df, 'ACT_verbal_score',1, 36)
fix_minmax(df, 'ACT_MATH_score',1, 36)

#scaling the columns
a, b = 0, 4
x, y = 200, 800
df['SAT_MATH_score']=(df['SAT_MATH_score'] - x) / (y - x) * (b - a) + a
df['SAT_VERB_score']=(df['SAT_VERB_score'] - x) / (y - x) * (b - a) + a

x, y = 400, 1600
df['SAT_MSS_score']=(df['SAT_MSS_score'] - x) / (y - x) * (b - a) + a
df['SAT_ERWS_score']=(df['SAT_ERWS_score'] - x) / (y - x) * (b - a) + a

x, y = 1, 36
df['ACT_verbal_score']=(df['ACT_verbal_score'] - x) / (y - x) * (b - a) + a
df['ACT_MATH_score']=(df['ACT_MATH_score'] - x) / (y - x) * (b - a) + a

# creating a prior score feature out of all the relevant prior scores: standardized test, prior gpa, hs gpa, transfer, etc.
from copy import deepcopy
df['prior_score']=df[['SAT_MATH_score','SAT_VERB_score','SAT_MSS_score', 'SAT_ERWS_score', 'gpa_hs', 'ACT_verbal_score', 'ACT_MATH_score', 'transfer_GPA' ]].mean(axis=1).copy()

df.isna().sum()

#checking the correlation of missing EOT gpa credit

df['missing_eot']= np.where((df['credits_taken_EOT'].isna()) |(df['credits_total_EOT'].isna()) |(df['gpa_cumulative_EOT'].isna()), 1,0)


df.loc[df.missing_eot==1].label.value_counts()
df.loc[df.missing_eot==1].census_level.value_counts()
#pvalue of the relation between missing eot and label
from scipy.stats import pearsonr
pearsonr(df['missing_eot'], df['label'])

#more cleaning....



# df= df.drop(columns= ['BachSemNo', 'LastSemNo' ])
df= df.drop(columns= ['SAT_MATH_score','SAT_VERB_score','SAT_MSS_score', 'SAT_ERWS_score', 'gpa_hs', 'ACT_verbal_score', 'ACT_MATH_score', 'transfer_GPA' ])

df["census_level"].replace({"Unclassified": "Unknown"}, inplace=True)
df["fa_eligible"].replace({"Y": 1 , "N": 0}, inplace=True)
df['FstLevel'].fillna("U", inplace= True)
df['FstResidency_tuition'].replace({"UNK": "IS", "REGNL":"IS" }, inplace=True)
df['FstLevel'].fillna("IS", inplace= True)
df['Admit_type_official'].fillna("UMT", inplace= True)
df['AnyCOL'].fillna(0, inplace= True)
df['Gender'].fillna('F', inplace= True)

df_final=df.copy()
df_final=df_final.drop(columns= ['BachSemNo', 'LastSemNo', 'missing_eot','FstDegree_Seeking'])
df_final['fa_eligible']= df_final['fa_eligible'].fillna(0)
df_final['NRA']= df_final['NRA'].fillna(0)
df_final['HSDegYr'] = np.where(df_final['HSDegYr'].isna() ,df_final['birthyear']+18 ,df['HSDegYr'])
df_final= df_final.dropna()
df_final=df_final.reset_index()
df_final= df_final.drop(columns= 'index')

df_sampled['FstResidency_tuition']= np.where((df_sampled['FstResidency_tuition']==""), "IS", df_sampled['FstResidency_tuition'])
df_sampled['HSDegYr']= np.where((df_sampled['HSDegYr']<1960),df_sampled['birthyear']+18 , df_sampled['HSDegYr'])
df_sampled= df_sampled.loc[df_sampled['FstFtpt_flag']!=""]
df_sampled= df_sampled.loc[df_sampled['FstLevel']!=""]
df_sampled= df_sampled.loc[df_sampled['fa_eligible']!=""]



#hopefully the preprocessing block getting handling of the non numerical columns
df_sampled=df_sampled.drop(columns= ['census_term' ]) #maybe add first dgeree seeking
df_sampled['FstFtpt_flag'].replace({'full time': 1, 'part time': 0.5} , inplace= True)
df_sampled['FstLevel'].replace({'Freshmen': 1, 'Sophomore': 2, 'Junior':3, 'Senior':4, 'U':0 } , inplace= True)
df_sampled['census_level'].replace({'Freshmen': 1, 'Sophomore': 2, 'Junior':3, 'Senior':4, 'Unknown':0 } , inplace= True)
df_sampled['FstResidency_tuition'].replace({'IS': 3, 'FORGN': 2, 'OS':1} , inplace= True)


#one hot encoding the rest...
categorical_transformer = OneHotEncoder()
coder=  OneHotEncoder()
coded=coder.fit_transform(df_sampled.select_dtypes('object')).toarray()
col= coder.categories_
col = np.concatenate( col, axis=0 )
df_coded= pd.DataFrame(data= coded, columns= col)
df_coded.index= df_sampled.index
df_sampled=df_sampled.merge(df_coded, left_index=True, right_index= True, validate= 'one_to_one')
df_sampled=df_sampled.drop(columns= df_sampled.select_dtypes('object').columns)
# df_sampled=df_sampled.drop(columns= '')
label= df_sampled['label']
df_sampled=df_sampled.drop(columns= 'label')
df_sampled['label']=label


df_sampled= df_sampled.drop(columns= ['CAPS', 'CPCS', 'SFE', 'MGS', 'CEHD'])
df_sampled= df_sampled.drop(columns= ['birthyear','White','Black','Hispanic', 'Asian','NatAm','NRA','AFRICAN','MIDEAST'])



#cleaning more... the sampled one this time
# df_sampled=df_final.groupby('emplid').last()
df_sampled['FstResidency_tuition']= np.where((df_sampled['FstResidency_tuition']==""), "IS", df_sampled['FstResidency_tuition'])
df_sampled['HSDegYr']= np.where((df_sampled['HSDegYr']<1960),df_sampled['birthyear']+18 , df_sampled['HSDegYr'])
df_sampled= df_sampled.loc[df_sampled['FstFtpt_flag']!=""]
df_sampled= df_sampled.loc[df_sampled['FstLevel']!=""]
df_sampled= df_sampled.loc[df_sampled['fa_eligible']!=""]

#hopefully the preprocessing block getting handling of the non numerical columns
df_sampled=df_sampled.drop(columns= ['census_term' ]) #maybe add first dgeree seeking
df_sampled['FstFtpt_flag'].replace({'full time': 1, 'part time': 0.5} , inplace= True)
df_sampled['FstLevel'].replace({'Freshmen': 1, 'Sophomore': 2, 'Junior':3, 'Senior':4, 'U':0 } , inplace= True)
df_sampled['census_level'].replace({'Freshmen': 1, 'Sophomore': 2, 'Junior':3, 'Senior':4, 'Unknown':0 } , inplace= True)
df_sampled['FstResidency_tuition'].replace({'IS': 3, 'FORGN': 2, 'OS':1} , inplace= True)


#one hot encoding the rest...
categorical_transformer = OneHotEncoder()
coder=  OneHotEncoder()
coded=coder.fit_transform(df_sampled.select_dtypes('object')).toarray()
col= coder.categories_
col = np.concatenate( col, axis=0 )
df_coded= pd.DataFrame(data= coded, columns= col)
df_coded.index= df_sampled.index
df_sampled=df_sampled.merge(df_coded, left_index=True, right_index= True, validate= 'one_to_one')
df_sampled=df_sampled.drop(columns= df_sampled.select_dtypes('object').columns)
# df_sampled=df_sampled.drop(columns= '')
label= df_sampled['label']
df_sampled=df_sampled.drop(columns= 'label')
df_sampled['label']=label
df_sampled= df_sampled.drop(columns= ['CAPS', 'CPCS', 'SFE', 'MGS', 'CEHD'])
df_sampled= df_sampled.drop(columns= ['birthyear','White','Black','Hispanic', 'Asian','NatAm','NRA','AFRICAN','MIDEAST'])
###Here the data is ready for ML, cleaned up and ready
df_sampled.to_csv('tot_sample.csv')

#the remainder of the scripts works on getting some insights regarding the data using the train segment of our data:
#train test split 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_gpa.iloc[:,:29]
                                                    , df_gpa['label'] , test_size=0.25, random_state=42, stratify=df_gpa['label'])

scaler = MinMaxScaler(feature_range=[0, 1])
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#Drwaing hearmap for correlations Viz porpuses only
x_train['label']= y_train.values
x_train['label'].value_counts(normalize=True)
import matplotlib.pyplot as plt
corr=x_train.corr()#["Survived"]
plt.figure(figsize=(30, 30))

sns.heatmap(corr, vmax=1, linewidths=0.08,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=30)# adjust yourself
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
# pca_importance_array = pd.DataFrame(pca.components_,columns=df_sampled.iloc[:,:45],index = ['PC-1','PC-2', 'PC-3','PC-4'])
# pca_importance_array=pca_importance_array.abs()
# selected_features= pca_importance_array.idxmax(axis=1)
import matplotlib.pyplot as plt
# Plotting starts here
#Fitting the PCA with our entire data
pca_all = PCA().fit(x_train)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca_all.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title( 'Feature Variance:Feature Importance')
plt.show()

#plotting class distributions for different features on the train sample:
sns.displot(df_sampled, x="prior_score", hue="label", kind="kde")
sns.countplot(x="FstCollege", hue="label", data=df_final)
sns.countplot(x="label", hue="census_level", data=df_sampled)
sns.countplot(x="label", hue="FstResidency_tuition", data=df_sampled)
sns.countplot(x="FstResidency_tuition", hue="label", data=df_sampled)
sns.displot(df, x="gpa_cumulative_EOT", hue="label", kind="kde")



