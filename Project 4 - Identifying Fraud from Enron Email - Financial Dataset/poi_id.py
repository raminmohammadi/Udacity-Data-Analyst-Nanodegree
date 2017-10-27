import matplotlib.pyplot as plt
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from feature_format import featureFormat
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tools import get_k_best,plotfunction, make_data_test



### Load the dictionary containing the dataset
path = r"../ramin/pickle_files/final_project_dataset.pkl"
with open(path, "rb") as data_file:
    enron_data = pickle.load(data_file)


# In[51]:


## Data Wrangling step

print("Length of data is: ", len(enron_data))
print("Number of records : " , enron_data['SKILLING JEFFREY K']['bonus'])
print("Number of features(variables) for each user is: ",len(enron_data['SKILLING JEFFREY K']))


## Finding number of user of intrestes
count = 0
for user in enron_data:
    if enron_data[user]['poi'] == True:
        count +=1
print("Number of users which are preson oof intrest:  " ,count , "\n")

# Number of available features
features = enron_data['JACKSON CHARLENE R'].keys()

print("........Information about Enron:........\n"
      "  COE during scandel : SKILLING JEFFREY\n"
      "  Chairman : KENNETH LAY\n"
      "  financial officer : ANDREW FASTOW\n")

poi =['SKILLING JEFFREY K','LAY KENNETH L','FASTOW ANDREW S']

print("**considering the COE, Chairman, financial officer as POI and checking their Total payment**\n")
for name in poi:
    x = enron_data[name]['total_payments']
    print("      POI: {0} has total patyment of: {1}".format(name,x))


print("\nWe can see Lay KENNETH has the highest total amount of payment which requires to be consider for outlier detection.\n")





print("                                    ****NaN/Missing values****\n"
      "Looking for NaN Values in dataset, i have created a dict with name of columns and two values for each coolumn\n"
      "one is total of missing values and percentage oof missing values, i am intrested to find variables with more\n"
      "than 50% missing values.\n")

NaN_list = dict((el,{'total missing':0, 'percentage missing':0}) for el in features)

for key in enron_data.keys():
    for column in features:
        if enron_data[key][column] == 'NaN':
            NaN_list[column]['total missing']+=1

for key in NaN_list:
    a = NaN_list[key]['total missing']
    NaN_list[key]['percentage missing']=float(a/146)
    if NaN_list[key]['percentage missing'] > 0.5:
        print("{0}: {1}".format(key, NaN_list[key]))


# In[53]:


# plotfunction(enron_data, 'total_payments' , 'total_stock_value' , True)
# plotfunction(enron_data,'salary', 'bonus', True)

print("\nwe can see we have an outlier which can be for spreadsheet quirk and need to be manually removed.\n"
      "there are Two more outliers (SKILLING JEFFREY and LAY KENNETH) I keep in data-set as these values are\n"
      "real and actually they are already a sign of these two managers being involved in the fraud.")

#
import random
random.seed(450)
enron_data.pop('TOTAL', 0)  # returns item and drop from the frame - raise error if not exist

from operator import itemgetter

### remove NAN's from dataset
outliers= []
for key in enron_data:
    val = enron_data[key]['salary']
    if (val == 'NaN') or (val==0.):
        continue
    key_dict = {"Name":key, "Salary":val}
    outliers.append(key_dict)
outliers_final = sorted(outliers,key=itemgetter('Salary'),reverse=True)


### print top 4 salaries
print("\n Outliers are: \n",outliers_final[:3])

def without_key(d, keys):
    return {key: d[key] for key in d if key not in keys}

def remove_oulier(dict_file, keys):
    for key in keys:
        dict_file.pop(key,0)

outlier_keys = {'SKILLING JEFFREY K','LAY KENNETH L','FREVERT MARK A','PICKERING MARK R'}
enron_data = without_key(enron_data, outlier_keys)

outliers = ['THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_oulier(enron_data,outliers)

# plotfunction(enron_data,'salary', 'bonus' ,  False )
# plotfunction(enron_data, 'total_payments' , 'total_stock_value', False)


# In[49]:


my_dataset = enron_data


# In[32]:


features_list = ['poi','salary', 'total_payments', 'loan_advances', 'other','deferred_income', 'from_messages',
                 'from_this_person_to_poi','shared_receipt_with_poi', 'deferral_payments',
                 'bonus','to_messages', 'director_fees', 'exercised_stock_options','expenses',
                 'total_stock_value','restricted_stock_deferred','from_poi_to_this_person',
                 'long_term_incentive', 'restricted_stock']


print("\n   ********** I am considering top 10 features ***********\n")
best_features = get_k_best(my_dataset, features_list, 10)
# MY new feature List
new_feature_list = ['poi'] + best_features
make_data_test(DecisionTreeClassifier(), my_dataset, features_list, 10)


### create new features

for key, row in my_dataset.items():

    new_value = float(row["from_poi_to_this_person"])/float(row["to_messages"])
    my_dataset[key]['from_poi_ratio'] = new_value if not np.isnan(new_value) else 0;

    new_value = float(row["from_this_person_to_poi"])/float(row["from_messages"])
    my_dataset[key]['to_poi_ratio'] = new_value if not np.isnan(new_value) else 0;


features_list = ["poi", "from_poi_ratio", "to_poi_ratio"]
data = featureFormat(my_dataset, features_list, sort_keys = True)

# plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    x1 = plt.scatter( from_poi, to_poi, color='b')
    if point[0] == 1:
        x2 = plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.legend((x1,x2), ("Not POI", "POI"), loc='upper right')
plt.xlabel("fraction of emails this person gets from poi")
plt.show()


features_list = ['poi','to_poi_ratio', 'salary', 'total_payments', 'loan_advances', 'other','deferred_income',
                 'from_messages','from_this_person_to_poi','shared_receipt_with_poi','from_poi_ratio', 'deferral_payments',
                 'bonus','to_messages', 'director_fees', 'exercised_stock_options','expenses', 'total_stock_value',
                 'restricted_stock_deferred','from_poi_to_this_person', 'long_term_incentive', 'restricted_stock']


# chosing features with highest score after crteating two new features and run the model to compare the results

print("\n   ********** I am considering top 10 features ***********\n")
# best_features = get_k_best(my_dataset, features_list, 14)
# # MY new feature List
# new_feature_list = ['poi'] + best_features
# make_data_test(DecisionTreeClassifier(), my_dataset, new_feature_list, 10)

best_features = get_k_best(my_dataset, features_list, 10)
# MY new feature List
new_feature_list = ['poi'] + best_features
clf_decision_tree = make_data_test(DecisionTreeClassifier(), my_dataset, new_feature_list, 10)

# best_features = get_k_best(my_dataset, features_list, 7)
# # MY new feature List
# new_feature_list = ['poi'] + best_features
# clf_decision_tree = make_data_test(DecisionTreeClassifier(), my_dataset, new_feature_list, 10)

# best_features = get_k_best(my_dataset, features_list, 4)
# # MY new feature List
# new_feature_list = ['poi'] + best_features
# make_data_test(DecisionTreeClassifier(), my_dataset, new_feature_list, 10)




logistic = Pipeline(steps = [
    ('scalar', StandardScaler()),
    ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty='l2',random_state=20))])
clf_regression = make_data_test(logistic, my_dataset, new_feature_list, 10)

# logistic = Pipeline(steps = [
#     ('scalar', StandardScaler()),
#     ('classifier', LogisticRegression(tol = 0.01, C = 10**-8, penalty='l2',random_state=20))])
# clf2 = make_data_test(logistic, my_dataset, new_feature_list, 10)

# logistic = Pipeline(steps = [
#     ('scalar', StandardScaler()),
#     ('classifier', LogisticRegression(tol = 0.1, C = 10**-8, penalty='l2',random_state=20))])
#
# clf2 = make_data_test(logistic, my_dataset, new_feature_list, 10)



from sklearn.ensemble import RandomForestClassifier

# clf4 = RandomForestClassifier(max_depth =7 , max_features = 'sqrt', n_estimators = 10, random_state = 42)
# clf4 = make_data_test(clf4, my_dataset ,new_feature_list, 10)

clf_random_forest = RandomForestClassifier(max_depth =9 , max_features = 'sqrt', n_estimators = 10, random_state = 42)
clf_random_forest = make_data_test(clf_random_forest, my_dataset ,new_feature_list, 10)

# clf4 = RandomForestClassifier(max_depth =11 , max_features = 'sqrt', n_estimators = 10, random_state = 42)
# clf4 = make_data_test(clf4, my_dataset ,new_feature_list, 10)



from sklearn.cluster import KMeans
clf_KMeans = KMeans(n_clusters=2, tol=0.1,random_state=42, max_iter=1000)
clf_KMeans = make_data_test(clf_KMeans,my_dataset ,new_feature_list, 10)


#
#

print("\n ***** I finally chose GuassianNB() as my final model *****")
clf_Guassian = GaussianNB()
clf_Guassian = make_data_test(clf_Guassian,my_dataset ,new_feature_list, 10)

pickle.dump(clf_Guassian, open("my_classifier_guassian.pkl", "wb") )
pickle.dump(clf_random_forest, open("my_classifier_random.pkl", "wb") )
pickle.dump(clf_KMeans, open("my_classifier_KMeans.pkl", "wb") )
pickle.dump(clf_decision_tree, open("my_classifier.pkl", "wb") )
pickle.dump(clf_regression, open("my_classifier_regression.pkl", "wb") )

pickle.dump(my_dataset, open("my_dataset.pkl", "wb") )
pickle.dump(new_feature_list, open("my_feature_list.pkl", "wb") )


