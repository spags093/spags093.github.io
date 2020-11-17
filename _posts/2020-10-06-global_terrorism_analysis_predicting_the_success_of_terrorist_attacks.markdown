---
layout: post
title:      "Global Terrorism Analysis"
date:       2020-10-06 23:01:48 -0400
permalink:  global_terrorism_analysis_predicting_the_success_of_terrorist_attacks
---

## Predicting the Success of Terrorist Attacks

**Author**: Jeff Spagnola


![](https://github.com/spags093/dsc-mod-3-project-v2-1-onl01-dtsc-ft-070620/blob/master/main-image.jpeg)


Terrorism is a worldwide problem.  Between 1970 and 2017, there were 181,691 terrorist attacks recorded globally.  Attacks have been recorded on every continent and in over 180 countries.  In the interest of national security, we will analyze terrorist attacks between 1997 and 2017 to figure out the specifc factors that determine whether a terrorist attack will be successeful.  By knowing the patterns, strenghts and weakness of terrorists and terrorist organizations, we can be more prepared to prevent new attacks in the future.  

Throughout the course of this notebook, we will attempt to determine what factors are most important in ensuring the success of a terrorist attack.  Utilizing the OSEMN data science process, we will analyze the data and then employ various machine learning algorithms to determine the importance of features.  

 <div class="alert alert-info" role="alert">
<center><b><u>Defining "Success"</u></b></center>
<u>According to the Global Terrorist Database: </u><br>
Success of a terrorist strike is defined according to the tangible effects of the attack.
Success is not judged in terms of the larger goals of the perpetrators. For example, a
bomb that exploded in a building would be counted as a success even if it did not
succeed in bringing the building down or inducing government repression. <br><br>
The definition of a successful attack depends on the type of attack. Essentially, the
key question is whether or not the attack type took place. If a case has multiple
attack types, it is successful if any of the attack types are successful, with the
exception of assassinations, which are only successful if the intended target is killed.
    </div>


### Data
The data used in this analysis is from the Global Terrorism Database.  This database was compiled by the National Consortium for the Study of Terrorism and Responses to Terrorism.  Aside from coming up with a catchy name for their organization, they've also compiled data on over 180,000 terrorist attacks worldwide between 1997 and 2017.  The full dataset contains 181,691 rows and 135 columns.  


## Methods
This notebook was created using the OSEMN data science method and, below, we will walk through each step of this process as well as share the results.

### Obtain
The first step in the process, as usual, is to import the Global Terrorism Database using pandas.  The dataset is very large but we decided to import the whole thing and then start to pare it down from there during the scrubbing proces.  The original dataframe is over 180,000 rows and 135 columns.  

```python

df = pd.read_csv('global_terrorism.csv', engine = 'python')
print(df.shape)
df.head()

```

### Scrub
Upon initial inspection, there were several methods we were able to use to make the dataset more manageable.  First, we decided to focus on just the most recent 20 years of the dataset (1997-2017) and this brought the number of rows down to just over 117,000.  Several columns were irrelevant to the target, several others were redundant, and many were able to be condensed into a single feature.  The final pruned dataset was 117,381 rows and 42 columns.  

From here, we split the data into training and testing sets using train_test_split.  

```python

X = df.drop('success', axis = 1)
y = df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 30)

```

Next, we separated the training data into numerical and categorical columns as well as created Pipelines to transform the data for modeling.  

```python

num_transformer = Pipeline(steps = [('imputer', KNNImputer(n_neighbors = 2)), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')), 
    ('encoder', OneHotEncoder(handle_unknown = 'ignore', sparse = False))])

preprocessing = ColumnTransformer(transformers = [
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)])

X_train_trans = preprocessing.fit_transform(X_train)
X_test_trans = preprocessing.transform(X_test)

```

### Explore
We employed a variety of plots to find interesting trends within the data.  

#### Map of  Attempted Terrorist Attacks 
<center><img src="Terrorism Map.png" alt="Terrorism Map" width = '600' height = '400'></center>
<div class="alert alert-info" role="alert">
    Above is a map indicating the location of every terrorist attack from 1970 to 2017.  
</div>

#### Geographic Region
<center><img src="geographic-attacks.png" alt="Geographic Location"></center>
<div class="alert alert-info" role="alert">
    The plot above shows that an overwhelming majority of terrorist attacks occur in the <b>Middle East/North Africa and South Asia. </b> 
    </div>

#### Attack Type
<center><img src="attack-type.png" alt="Attack Type"></center>
<div class="alert alert-info" role="alert">
  The plot above shows that an most attempted terrorist attacks are either <b>bombings or armed assaults. </b> 
</div>

#### Target Type
<center><img src="target-type.png" alt="Target Type"></center>
<div class="alert alert-info" role="alert">
  The plot above shows that the top targets for a terrorist attack are <b>Private Property, Military, Police, Government, and Business.</b>
</div>

### Model
The modeling phase was an iterative process where we ran several different types of models in order to determine the best fit for the data.  

#### Logistic Regression
First on the list is Logistic Regression.  First, we run a base model and then performed a GridSearchCV in order to tune the hyperparameters.  After getting the optimal hyperparameters, we fit a seocnd model and reviewed our results. 

```python
params = {'class_weight': ['balanced'],
          'solver': ['lbfgs', 'liblinear'],
          'C': [1.0, 3.0, 5.0]}
grid = GridSearchCV(estimator = LogisticRegression(), param_grid = params, cv = 3, n_jobs = -1)
grid.fit(X_train_trans_df, y_train)

logreg2 = LogisticRegression(class_weight = 'balanced',
                             C = 1.0,
                             solver = 'lbfgs')
logreg2.fit(X_train_trans_df, y_train)
```

<center><img src="LOGREG-RESULTS.png" alt="Logistic Regression"></center>

#### Decision Tree
Next, we ran a base decision tree classifier to see if this would return better results.  Again, we then performed a GridSearchCV to find the optimal hyperparameters and then fit a second model.  

```python
params = {'class_weight': [None, 'balanced'],
          'criterion': ['gini', 'entropy'],
          'max_depth': [1, 3, 5], 
          'min_samples_leaf': [1, 3, 5]}
grid = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = params, cv = 3, n_jobs = -1)
grid.fit(X_train_trans_df, y_train)

decision_tree2 = DecisionTreeClassifier(class_weight = 'balanced', 
                                        criterion = 'entropy', 
                                        max_depth = 5, 
                                        min_samples_leaf = 5)
decision_tree2.fit(X_train_trans_df, y_train)
```

<center><img src="DecisionTreeResults.png" alt="Decision Tree"></center>

#### Random Forest
After the decision tree, we decided to try an ensemble method and used the random forest algorithm.  Once again, we performed a GridSearchCV and fit the model with the tuned hyperparameters.

```python
params = {'class_weight': [None, 'balanced'],
          'criterion': ['gini', 'entropy'],
          'max_depth': [1, 3, 5], 
          'min_samples_leaf': [1, 5, 10]}
grid = GridSearchCV(estimator = RandomForestClassifier(), param_grid = params, cv = 3, n_jobs = -1)
grid.fit(X_train_trans_df, y_train)

random_forest2 = RandomForestClassifier(class_weight = 'balanced', 
                                        criterion = 'entropy', 
                                        max_depth = 5, 
                                        min_samples_leaf = 10)
random_forest2.fit(X_train_trans_df, y_train)
```

<center><img src="Random Forest Results.png" alt="Random Forest"></center>

#### XGBoost
Next, we wanted to experiment with using XGBoost on the data.  In case you're not noticing the pattern yet, we performed a GridSearchCV and fit the model with the tuned hyperparameters.

```python
params = {'gamma': [0.5, 1, 2, 5],
          'min_child_weight': [1, 5, 10],
          'max_depth': [1, 3, 5]}
grid = GridSearchCV(estimator = xgb.XGBClassifier(), param_grid = params, cv = 3, n_jobs = -1)
grid.fit(X_train_trans_df, y_train)

xgboost2 = xgb.XGBClassifier(gamma = 1,
                             min_child_weight = 1,  
                             max_depth = 5)
xgboost2.fit(X_train_trans_df, y_train)
```
<center><img src="XGBoost Results.png" alt="XGBoost"></center>

#### Stacking Ensemble
The results are very good, but we wanted to see if we would be able to get a bit more accuracy by using a Stacking Ensemble made up of our decision tree, random forest, and XGBoost.  

```python
estimators = [('dt', DecisionTreeClassifier(class_weight = 'balanced', 
                                            criterion = 'entropy',
                                            max_depth = 5, 
                                            min_samples_leaf = 5)),
              ('rf', RandomForestClassifier(class_weight = 'balanced', 
                                            criterion = 'entropy', 
                                            max_depth = 5, 
                                            min_samples_leaf = 10)), 
              ('xg', xgb.XGBClassifier(gamma = 1,
                                       min_child_weight = 1,  
                                       max_depth = 5))]

stack = StackingClassifier(estimators = estimators, cv = 3, n_jobs = -1)
stack.fit(X_train_trans_df, y_train)
```

<center><img src="Stacking Results.png" alt="Stacking Classifier"></center>

### Interpret
In terms of trying to prevent terrorist attacks, it is extremely important that we limit the false negatives in our model.  Therefore, the metric that we chose to use for scoring the model performence is the Recall.  

<center><img src="Recall Scores.png" alt="Recall Scores"></center>

Though all the models performed well, we can see that XGBoost and the Stacking Classifier performed the best in terms of Recall Score.  Though the Stacking Classifier performed better in other ways, we will be moving forward with the XGBoost model.  The reason for this is that we will be using SHAP for further analysis and at this moment, SHAP does not support Stacking Classifiers.  

Speaking of SHAP...

#### SHAP Summary Bar Plot
<img src="Shap-summary plot.png" alt="Shap Summary Plot Bar">
> Bar plot showing the most important features as per SHAP calculations.

#### SHAP Summary Plot
<img src="Shap-summaryplot2.png" alt="Shap Summary Plot Dot">
> Plot that shows the positive and negative affect on the target per each important feature.

#### Model Feature Importances 
<img src="featureImportances.png" alt="Model Feature Importances">
> Bar plot showing the most important features as per the feature importances from the model.

## Results

 <div class="alert alert-info" role="alert">
    <b><u>Top 5 Feature Importances from Sklearn </u></b>- targtype1_txt_Unknown, ishostkid, property, nkill, attacktype1_txt_Assassination<br><br>
<b><u>Top 5 Feature Importances from SHAP</u></b> - nkill, property, nwound, attacktype1_txt_Assassination, ishostkid<br><br>
    <b><u> Overlapping Important Features </u></b>- nkill, ishostkid, attacktype1_txt_Assassination, property
 </div>

<div class="alert alert-success" role="alert">
    <b><u><center>Overlapping Important Features:</center></u></b><br><br>
1. <b><u>nkill </u></b>- Total Number of Fatalities - High numbers of fatalities positively affect the success and low numbers of fatalities negatively affect the success of attack.<br><br>
2. <b><u>ishostkid</u></b> - Hostages or Kidnapping Victims - Very positively affects the success of attack.<br><br>
3. <b><u>attacktype1_txt_Assassination</u></b> - Attempted Assassination - Very negatively affects the success of attack.<br><br>
    4. <b><u>property</u></b> - Property Damage (Evidence of property damage from attack) - Slightly positively affects the success of attack.
    </div>


## Recommendations:

<b>Based on the results of this analysis, we can make the following recommnedations: </b>
 - Maintain high levels of intel and security in the Middle East and South Asia.
 - Develop better methods and/or technology for bomb detection and disarmament.  
 - Focus intel on target areas that have the highest concentration of people.
 - Increase security for high level targets for potential kidnapping or hostage situations.  


## Limitations & Next Steps

### Future Work
With more time, we can gain even more insight into what can make a terrorist attack successful in order to create better security measures.  

 - Time:  We can increase the range of years of the data in our analysis.  For example, the full dataset ranges from 1970 to 2017.  
 - Models:  We can increase the size and complexiy of our models in order to increase the accuracy of our results. 
 - Data:  We can research and compile additonal data from other resources for a more well rounded dataset.  


### For further information
Please review the narrative of our analysis in [our jupyter notebook](./student-Copy2.ipynb) or review our [presentation](./Mod-3 Project Presentation.pdf)

For any additional questions, please contact **jeff.spags@gmail.com.

