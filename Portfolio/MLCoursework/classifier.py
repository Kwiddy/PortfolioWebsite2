###############################################
# RUNNING THIS SCRIPT
###############################################
# - Run with "python3 classifier.py"
#   The classifier.py file must be in the same file location
#   As the datasets which must be named identically to those
#   provided by the coursework's OULAD link
#
#   Tested working versions (retrieve via pip install):
#   - Python 3.5.3 / 3.6.3 / 3.8.2
#   - Pandas 0.25.3 / 1.0.1 / 1.0.3
#   - Numpy 1.18.1 / 1.18.2 / 1.18.3
#   - Seaborn 0.9.1 / 0.10.0
#   - Matplotlib.pylab 1.18.1 / 1.18.2 / 1.18.3
#
#   - The following need to pip installed (using pip3):
#       Seaborn, pandas, sklearn, numpy, matplotlib, scipy
#
# - Select a method using the options list
#   For example, enter "D" for Decision Tree
#
# - Select a no. of categories from the list
#   in a similar style
#
# - Correllation, importances, and descriptions
#   will be printed for development purposes
#
# - It will then begin calculating folds
#
# - Appropritate graphs will be be created and saved to files
#
# - Finally, score metrics will be printed and
#   user will be prompted for another method
#   or exiting
###############################################

#Required Imports
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import math
import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

#Function for Restructuring Data
def restructData(data, option):

    #Creating a copy of the original data to be manipulated
    new_data = data.copy()

    #Detecting and manipulating the self-merged weightTotal Dataframe
    if "weightTotal" in new_data.columns:
        #Creating a new column containing the mean of: a student's summative grades multiplied by its associated weight
        weightedScore = (new_data.groupby("id_student")["weightTotal"].mean())
        new_data = pd.merge(weightedScore, new_data, on="id_student")

        #Removing the redundant column from merge
        del new_data['weightTotal_y']

        #Renaming the new column
        new_data = new_data.rename(columns={"weightTotal_x": "weightedTotal"})

        #Removing multiple entries for a single student id
        #Each row with the same student ID will have identical data for this dataframe
        new_data = new_data.drop_duplicates(subset='id_student', keep="first")

        #Removing redundant columns
        #These columns will be used in the merging of later dataframes but not this one
        del new_data["date_submitted"]
        del new_data["is_banked"]
        del new_data["score"]
        del new_data["code_module"]
        del new_data["code_presentation"]
        del new_data["assessment_type"]
        del new_data["weight"]
        del new_data["date"]
        del new_data["id_assessment"]

        #Return the newly structure dataframe
        return new_data

    #Replacing final_result with numerical values
    if "final_result" in new_data.columns:
        #Using all four categories
        if option.upper() == "A":
            new_data["final_result"] = new_data["final_result"].replace("Withdrawn", 1)
            new_data["final_result"] = new_data["final_result"].replace("Fail", 2)
            new_data["final_result"] = new_data["final_result"].replace("Pass", 3)
            new_data["final_result"] = new_data["final_result"].replace("Distinction", 4)

        #Grouping withdrawn and fail
        elif option.upper() == "B":
            new_data["final_result"] = new_data["final_result"].replace("Withdrawn", 1)
            new_data["final_result"] = new_data["final_result"].replace("Fail", 1)
            new_data["final_result"] = new_data["final_result"].replace("Pass", 2)
            new_data["final_result"] = new_data["final_result"].replace("Distinction", 3)

        #Grouping pass and distinction
        elif option.upper() == "C":
            new_data["final_result"] = new_data["final_result"].replace("Withdrawn", 1)
            new_data["final_result"] = new_data["final_result"].replace("Fail", 2)
            new_data["final_result"] = new_data["final_result"].replace("Pass", 3)
            new_data["final_result"] = new_data["final_result"].replace("Distinction", 3)

        #Grouping withdrawn/fail, pass/distinction
        elif option.upper() == "D":
            new_data["final_result"] = new_data["final_result"].replace("Withdrawn", 1)
            new_data["final_result"] = new_data["final_result"].replace("Fail", 1)
            new_data["final_result"] = new_data["final_result"].replace("Pass", 2)
            new_data["final_result"] = new_data["final_result"].replace("Distinction", 2)


    #Replacing highest_education with dummies. Creating a new boolean (1 or 0) column for each option
    if "highest_education" in new_data.columns:
        new_data = new_data.join(pd.get_dummies(new_data["highest_education"]))
        del new_data["highest_education"]

    #Replacing gender values to make the column numeric
    if "gender" in new_data.columns:
        new_data["gender"] = new_data["gender"].replace("M", 0)
        new_data["gender"] = new_data["gender"].replace("F", 1)

    #Replacing Region column with dummies. Creating a new boolean (1 or 0) column for each option
    if "region" in new_data.columns:
        new_data = new_data.join(pd.get_dummies(new_data["region"]))
        del new_data["region"]

    #Replacing Disability values to make teh column numeric
    if "disability" in new_data.columns:
        new_data["disability"] = new_data["disability"].replace("N", 0)
        new_data["disability"] = new_data["disability"].replace("Y", 1)

    #Replacing Age Band column with dummies. Creating a new boolean (1 or 0) column for each option
    if "age_band" in new_data.columns:
        new_data = new_data.join(pd.get_dummies(new_data["age_band"]))
        del new_data["age_band"]

    #Replacing IMD Band column with dummies. Creating a new boolean (1 or 0) column for each option
    if "imd_band" in new_data.columns:
        new_data = new_data.join(pd.get_dummies(new_data["imd_band"]))
        del new_data["imd_band"]

    #Replacing NaN values in generated columns of particular dataframe with Median score:
    if "meanScore" in new_data.columns:
        new_data["meanScore"] = new_data["meanScore"].fillna(new_data["meanScore"].median())
        new_data["numRes"] = new_data["numRes"].fillna(new_data["numRes"].median())
        
    #Restructuring information from the spreadsheet of student scores
    if "score" in new_data.columns:
        #Creating a new column to store the number of entries per student
        new_data["numRes"] = new_data.pivot_table(index=['id_student'], aggfunc='size')

        #Creating a mean score column, a mean score of 0 will be replaced by the median value of the column
        totalScore = (new_data.groupby("id_student")["score"].sum())
        new_data["meanScore"] = totalScore
        new_data["meanScore"] = new_data["meanScore"]/new_data["numRes"]
        new_data["meanScore"] = new_data["meanScore"].replace(0, new_data["meanScore"].median())

        #Resetting the dataframe's index for merging purposes
        new_data = new_data.reset_index()

        #Removing the original score column
        del new_data["score"]

        #Removing duplicate rows of a single student as none but the first will be needed
        new_data = new_data.drop_duplicates(subset='id_student', keep="first")

    #Restructuring the dataframe of assessment weights
    if "weight" in new_data.columns:
        #Removing redundant columns
        del new_data["code_module"]
        del new_data["code_presentation"]

        #Adding new columns using dummies of each assessment type. Will create additional "Boolean" (1 or 0) columns
        new_data = new_data.join(pd.get_dummies(new_data["assessment_type"]))

        #Removing the original assessment_type column
        del new_data["assessment_type"]
    
    #Detecting the initial studentRegistration DataFrame
    if "date_registration" in new_data.columns:
        #Removing redundant columns or columns I have chosen not to use for training and testing purposes
        del new_data["code_module"]
        del new_data["code_presentation"]
        del new_data["date_unregistration"]

        #Creating a new column describing the number of enrollments / registrations per student
        new_data["numEnroll"] = new_data.pivot_table(index=['id_student'], aggfunc='size')

        #Finding the mean date_registration per student
        dateSum = (new_data.groupby("id_student")["date_registration"].sum())
        new_data["date_registration"] = dateSum
        new_data["date_registration"] = new_data["date_registration"] / new_data["numEnroll"]

        #Resetting the datagram index for later merging purposes
        new_data = new_data.reset_index()

        #Removing redundant duplicates
        new_data = new_data.drop_duplicates(subset='id_student', keep="first")

    #Restructuring data from the studentVle spreadsheet
    #PLEASE NOTE THAT I WOULD HAVE PREFERRED TO MANIPULATE THIS DATA FURTHER ALONG WITH THE VLE.CSV DATA
    #   BUT HAD TO RESTRICT IT DUE TO LACK OF COMPUTATIONAL POWER, THIS HAS BEEN NOTED IN MY REPORT
    if "sum_click" in new_data.columns:
        #Counting the number of sites visited per student
        new_data["numSites"] = new_data.pivot_table(index=['id_student'], aggfunc='size')

        #Finding the total number of clicks across all sites per student
        sumConcat = (new_data.groupby("id_student")["sum_click"].sum())
        new_data["sumConcat"] = sumConcat

        #Finding the mean date from site clicks and creating a new column
        dateConcat = (new_data.groupby("id_student")["date"].sum())
        new_data["dateConcat"] = dateConcat
        new_data["dateConcat"] = new_data["dateConcat"] / new_data["numSites"]

        #Resetting index for merging purposes
        new_data = new_data.reset_index()
        
        #Removing redundant duplicates
        new_data = new_data.drop_duplicates(subset='id_student', keep="first")

        #Removing redundant column
        del new_data["id_site"]
        del new_data["date"]
        del new_data["sum_click"]
        del new_data["code_module"]
        del new_data["code_presentation"]

    #Returning the restructured Datafram
    return new_data

#Generating a list of predictors to be used
def genColumns(columns, method):
    #Adding all columns created by dummies in the data restructuring to list of predictors
    regions = ['East Anglian Region', 'Scotland', 'North Western Region', 'South East Region', 'West Midlands Region', 'North Region', 'South Region', 'Wales', 'Ireland', 'Yorkshire Region', 'London Region', 'South West Region', 'East Midlands Region']
    imd_bands = ['0-10%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    age_band = ['0-35', '35-55', '55<=']
    assessment_type = ['CMA', 'Exam', 'TMA']
    highest_education = ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification']
    for item in regions:
        columns.append(item)
    for item in imd_bands:
        columns.append(item)
    for item in age_band:
        columns.append(item)
    for item in assessment_type:
        columns.append(item)
    for item in highest_education:
        columns.append(item)

    #Adding additional predictors to the list
    columns.append('meanScore')
    columns.append('gender')
    columns.append('numRes')
    columns.append('disability')
    columns.append('weight')
    columns.append('is_banked')
    columns.append("sumConcat")
    columns.append("date_submitted")
    columns.append("dateConcat")
    columns.append("numSites")
    columns.append("date")
    columns.append("date_registration")
    columns.append("numEnroll")
    columns.append('module_presentation_length')
    columns.append('weightedTotal')

    #Adding id_assessment and id_student for Classifier models but not for Regression Models
    #   They have been shown to have negative effect on regression models
    if method not in ["L", "C"]:
        columns.append('id_assessment')
        columns.append('id_student')

    #Returning the new list of predictors
    return columns

#Function for Confusion Matrix Plotting
def plot_confusion_matrix(cm, ctgNames, title='Confusion matrix', cmap=plt.cm.Blues,):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    markers = np.arange(len(ctgNames))
    plt.xticks(markers, ctgNames, rotation=45)
    plt.yticks(markers, ctgNames)
    plt.tight_layout()
    plt.xlabel('Predicted Final Result')
    plt.ylabel('True Final Result')
    #Output Confusion Matrix
    plt.savefig('ConfusionMatrix.png')
    plt.clf()

def genGraphs(tar_test, predictions, corr, completeDF, option):
    #Set params depending on number of categories chosen
    if option.upper() == "A":
        ctgList = ["Withdrawn", "Fail", "Pass", "Distinction"]
    elif option.upper() == "B":
        ctgList = ["Withdrawn/Fail", "Pass", "Distinction"]
    elif option.upper() == "C":
        ctgList = ["Withdrawn", "Fail", "Pass/Distinction"]
    elif option.upper() == "D":
        ctgList = ["Withdrawn/Fail", "Pass/Distinction"]

    #Print Confusion Matrix
    try:
        cm = sklearn.metrics.confusion_matrix(tar_test, predictions)
        #Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plot_confusion_matrix(cm_normalized, ctgList, title='Normalized confusion matrix')
    except Exception as e:
        print("Create Confusion Matrix... Failed")
        print("Error: ", e)

    #Print correlation heatmap
    try:
        #Set colouring
        cmap = sns.cm.rocket_r
        #Create Heatmap
        ax = sns.heatmap(corr, cmap=cmap)
        plt.savefig("heatmap.png")
        plt.clf()
    except Exception as e:
        print("Create Correlation Heatmap... Failed")
        print("Error: ", e)

    #Print Final Result vs dateConcat
    try:
        xy = np.vstack([completeDF["final_result"], completeDF["dateConcat"]])
        z = gaussian_kde(xy)(xy)
    
        m, c = np.polyfit(completeDF["final_result"], completeDF["dateConcat"], 1)
        plt.scatter( completeDF["final_result"], completeDF["dateConcat"], c=z, alpha=0.4)

        plt.ylabel("dateConcat")
        plt.xlabel("final_result")

        plt.title("Correlation between final_result and dateConcat")
        plt.legend()
        plt.colorbar()

        plt.plot(completeDF["final_result"], m*completeDF["final_result"] + c)
        plt.savefig("FRvDateConcat.png")
        plt.clf()
    except Exception as e:
        print("Create final_result vs DateConcat... Failed")
        print("Error: ", e)

    #Print Final Result vs weightedTotal
    try:
        xy = np.vstack([completeDF["final_result"], completeDF["weightedTotal"]])
        z = gaussian_kde(xy)(xy)

        m, c = np.polyfit(completeDF["final_result"], completeDF["weightedTotal"], 1)
        plt.scatter( completeDF["final_result"], completeDF["weightedTotal"], c=z, alpha=0.4)

        plt.ylabel("weightedTotal")
        plt.xlabel("final_result")

        plt.title("Correlation between final_result and weightedTotal")
        plt.legend()
        plt.colorbar()
    
        plt.plot(completeDF["final_result"], m*completeDF["final_result"] + c)
        plt.savefig("FRvweightedTotal.png")
        plt.clf()
    except Exception as e:
        print("Create final_result vs DateConcat... Failed")
        print("Error: ", e)

    #Print Final Result vs meanScore
    try:
        xy = np.vstack([completeDF["final_result"], completeDF["meanScore"]])
        z = gaussian_kde(xy)(xy)

        m, c = np.polyfit(completeDF["final_result"], completeDF["meanScore"], 1)
        plt.scatter( completeDF["final_result"], completeDF["meanScore"], c=z, alpha=0.4)

        plt.ylabel("meanScore")
        plt.xlabel("final_result")

        plt.title("Correlation between final_result and meanScore")
        plt.legend()
        plt.colorbar()

        plt.plot(completeDF["final_result"], m*completeDF["final_result"] + c)
        plt.savefig("FRvmeanScore.png")
        plt.clf()
    except Exception as e:
        print("Create final_result vs DateConcat... Failed")
        print("Error: ", e)

def genVCurve(clf, tar_test, tar_train, pred_test, pred_train, param_range, predictors, targets, paramName):
    #Generate scores for graphing
    train_scores, test_scores = validation_curve(clf, predictors, targets, param_name=paramName, param_range=param_range, scoring="accuracy", n_jobs=-1)
    
    #Print Scores for development purposes
    print("train_scores: ", train_scores)
    print("test_scores: ", test_scores)

    ## DISCLAIMER: PLEASE NOTE THAT THE REMAINDER OF THIS FUNCTION IS DERIVED FROM AN EXAMPLE AT
    ##  https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    #   HOWEVER I COULD THINK OF NO OTHER WAY TO CREATE THE VALIDATION CURVE APART FROM AS BELOW
    #   I HAVE ADDED SIMPLE COMMENTS TO DEMONSTRATE MY UNDERSTANDING

    #Generate Stats for graphing 
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    #Set graph's titles and labels
    plt.title("Validation Curve of RFC")
    plt.xlabel(paramName)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2

    #Change the x-axis to log scaling  for training score
    plt.semilogx(param_range, train_mean, label="Training score",
                color="darkorange", lw=lw)

    #Fill the area between the two lines to demonstrate mean and std
    plt.fill_between(param_range, train_mean - train_std,
                    train_mean + train_std, alpha=0.2,
                    color="darkorange", lw=lw)

    #Change the x-axis to log scaling for CV score
    plt.semilogx(param_range, test_mean, label="Cross-validation score",
                color="navy", lw=lw)

    #Fill the area between the two lines to demonstrate mean and std
    plt.fill_between(param_range, test_mean - test_std,
                    test_mean + test_std, alpha=0.2,
                    color="navy", lw=lw)

    #Final output of graph
    plt.legend(loc="best")
    fileName = "ValCurve" + paramName + ".png" 
    plt.savefig(fileName)
    plt.clf()

#Generate a decision tree from the split data
def genDecisionTree(pred_train, pred_test, tar_train, tar_test, columns, corr, completeDF, option):
    #Create the decision tree classifier
    clf = DecisionTreeClassifier()

    #Cross validating the model
    scores = cross_val_score(clf, pred_train, tar_train, cv=5)
    print()
    print("Scores:")
    print(scores)
    print()

    #Fit model to training data
    clf.fit(pred_train, tar_train)

    #Generate predictions
    predictions = clf.predict(pred_test)

    # Create a list of features with their importances, for development and analytical purposes
    featureVals = list(clf.feature_importances_)
    feature_importances = [(feature, round(fVal, 5)) for feature, fVal in zip(columns, featureVals)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1])
    [print('Variable: {:35} Importance: {}'.format(*pair)) for pair in feature_importances]

    #Generate Graphs
    genGraphs(tar_test, predictions, corr, completeDF, option)

    #Generate a classification report and other metrics to determin performance
    print(classification_report(tar_test, predictions))
    print("Accuracy: %.3f"%sklearn.metrics.accuracy_score(tar_test, predictions))
    print("f1 score: %.3f"%sklearn.metrics.f1_score(tar_test, predictions, average="weighted"))
    print("MSE: %.3f"%sklearn.metrics.mean_squared_error(tar_test, predictions))
    print("MAE: %.2f"%sklearn.metrics.mean_absolute_error(tar_test, predictions))
    print("Explained Variance Score: %.3f"%sklearn.metrics.explained_variance_score(tar_test, predictions))
    print()

#Generate a random forest from the split data
def genRandomForest(pred_train, pred_test, tar_train, tar_test, columns, corr, completeDF, option, predictors, targets):
    #Create the random forest classifier
    clf = RandomForestClassifier() #(n_estimators=100)

    #Generate validation curve
    try:
        # Set parameters for validation curve
        max_depth = [int(n) for n in np.linspace(10, 100, num=10)]
        n_estimators = [int(n) for n in np.linspace(start=100, stop=200, num=10)]
        min_samples_split = [int(n) for n in np.linspace(2, 10, num=5)]
        min_samples_leaf = [int(n) for n in np.linspace(2, 10, num=5)]

        # Create validation curves
        try:
            genVCurve(clf, tar_test, tar_train, pred_test, pred_train, max_depth, predictors, targets, "max_depth")
        except Exception as e:
            print("max_depth Validation curve...failed")
            print(e)
        try:
            genVCurve(clf, tar_test, tar_train, pred_test, pred_train, n_estimators, predictors, targets, "n_estimators")
        except Exception as e:
            print("n_estimators Validation curve...failed")
            print(e)
        try:
            genVCurve(clf, tar_test, tar_train, pred_test, pred_train, min_samples_split, predictors, targets, "min_samples_split")
        except Exception as e:
            print("min_samples_split Validation curve...failed")
            print(e)
        try:
            genVCurve(clf, tar_test, tar_train, pred_test, pred_train, min_samples_leaf, predictors, targets, "min_samples_leaf")
        except Exception as e:
            print("min_samples_leaf Validation curve...failed")
            print(e)

    except Exception as e:
        print("Create Validation Curve... failed")
        print("Error: ", e)

    #Initial cross validation
    try:
        scores = cross_val_score(clf, pred_train, tar_train, cv=5)
        print()
        print("Scores:")
        print(scores)
        print()
    except Exception as e:
        print("Cross Validation failed: ", e)

    ######################
    ## Create a list of features with their importances, for development and analytical purposes
    # clf.fit(pred_train, tar_train)
    # predictions = clf.predict(pred_test)
    # featureVals = list(clf.feature_importances_)
    # feature_importances = [(feature, round(fVal, 5)) for feature, fVal in zip(columns, featureVals)]
    # feature_importances = sorted(feature_importances, key=lambda x: x[1])
    # [print('Variable: {:35} Importance: {}'.format(*pair)) for pair in feature_importances]
    # exit()
    ######################

    #Use the random grid function with GridSearchCV for hyper parameter tuning and cross validation
    param_grid = genRandomGrid("R")
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)

    ## Randomized search (previous version replaced by GridSearch)
    #clf_random = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
    
    #Fit the training data to the model and generate predictions
    grid_search.fit(pred_train, tar_train)
    predictions = grid_search.predict(pred_test)

    #Generate Graphs
    genGraphs(tar_test, predictions, corr, completeDF, option)

    #Create a classification report and associated metrics
    try:
        print("r2 score: %.3f"%sklearn.metrics.r2_score(tar_test, predictions))
    except Exception as e:
        print("Printing r2 Score... Failed")
        print("Error: ", e)
    print(classification_report(tar_test, predictions))
    print("Accuracy: %.3f"%sklearn.metrics.accuracy_score(tar_test, predictions))
    print("f1 score: %.3f"%sklearn.metrics.f1_score(tar_test, predictions, average="weighted"))
    print("MSE: %.3f"%sklearn.metrics.mean_squared_error(tar_test, predictions))
    print("MAE: %.2f"%sklearn.metrics.mean_absolute_error(tar_test, predictions))
    print("Explained Variance Score: %.3f"%sklearn.metrics.explained_variance_score(tar_test, predictions))
    print()

#Generate a linear regression model from the split data
def genLinearRegression(pred_train, pred_test, tar_train, tar_test):
    #Create the model
    regr = LinearRegression()

    #Cross validating the model
    scores = cross_val_score(regr, pred_train, tar_train, cv=5)
    print()
    print("Scores:")
    print(scores)
    print()

    #Fit model to training data
    regr.fit(pred_train, tar_train)

    #Generate predictions
    predictions = regr.predict(pred_test)

    #print(classification_report(tar_test, predictions))
    #print("Accuracy: %.3f"%sklearn.metrics.accuracy_score(tar_test, predictions))

    #Print accuracy metrics
    print()
    print("r2 score: %.3f"%sklearn.metrics.r2_score(tar_test, predictions))
    print("MSE: %.3f"%sklearn.metrics.mean_squared_error(tar_test, predictions))
    print("MAE: %.2f"%sklearn.metrics.mean_absolute_error(tar_test, predictions))
    #print("Explained Variance Score: %.3f"%sklearn.metrics.explained_variance_score(tar_test, predictions))
    #print("Coefficients: \n", regr.coef_)
    print()
    
#Generate a logistic regression model from the split data
def genLogisticRegression(pred_train, pred_test, tar_train, tar_test, corr, completeDF, option, predictors, targets):
    #Create a logistic regression model with random state and appropriate max_iter
    regr = LogisticRegression(random_state=42, max_iter=50000)

    #Initial cross validation
    try:
        scores = cross_val_score(regr, pred_train, tar_train, cv=5)
        print()
        print("Scores:")
        print(scores)
        print()
    except Exception as e:
        print("Cross Validation failed: ", e)

    #Reformat the data so that its within the bounds of 0-1, this improves performance
    min_max_scaler = MinMaxScaler()
    pred_train_mms = min_max_scaler.fit_transform(pred_train)
    pred_test_mms = min_max_scaler.fit_transform(pred_test)

    #use the genRandomGrid function and grid search for hyper parameter tuning and cross validation
    param_grid = genRandomGrid("L")
    grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(pred_train_mms, tar_train)
    predictions = grid_search.predict(pred_test_mms)

    #Generate Graphs
    genGraphs(tar_test, predictions, corr, completeDF, option)

    #Print appropriate accuracy metrics
    try:
    #print(classification_report(tar_test, predictions))
        print("Accuracy: %.3f"%sklearn.metrics.accuracy_score(tar_test, predictions))
    except Exception as e:
        print("Printing Accuracy... Failed")
        print("Error: ", e)
    try:
        print("f1 score: %.3f"%sklearn.metrics.f1_score(tar_test, predictions, average="weighted"))
    except Exception as e:
        print("Printing f1 Score... Failed")
        print("Error: ", e)
    print()
    print("r2 score: %.3f"%sklearn.metrics.r2_score(tar_test, predictions))
    print("MSE: %.3f"%sklearn.metrics.mean_squared_error(tar_test, predictions))
    print("MAE: %.2f"%sklearn.metrics.mean_absolute_error(tar_test, predictions))
    print("Explained Variance Score: %.3f"%sklearn.metrics.explained_variance_score(tar_test, predictions))
    #print("Coefficients: \n", regr.coef_)
    print()
    
#Generate an SVC model from the split data
def genSVC(pred_train, pred_test, tar_train, tar_test):
    #Create the model
    clf = SVC(gamma='auto')

    #Cross validating the model
    scores = cross_val_score(clf, pred_train, tar_train, cv=5)
    print()
    print("Scores:")
    print(scores)
    print()

    #Fit model to training data
    clf.fit(pred_train, tar_train)

    # Generate predictions
    predictions = clf.predict(pred_test)
    
    #Print accuracy metrics
    try:
        print(classification_report(tar_test, predictions))
    except Exception as e:
        print(e)
    try:
        print("Accuracy: %.3f"%sklearn.metrics.accuracy_score(tar_test, predictions))
    except Exception as e:
        print(e)
    print()
    print("f1 score: %.3f"%sklearn.metrics.f1_score(tar_test, predictions, average="weighted"))
    print("MSE: %.3f"%sklearn.metrics.mean_squared_error(tar_test, predictions))
    print("MAE: %.2f"%sklearn.metrics.mean_absolute_error(tar_test, predictions))
    print("Explained Variance Score: %.3f"%sklearn.metrics.explained_variance_score(tar_test, predictions))
    print()

#Main computational process
def main(choices):
    #Store user inputs
    method = choices[0]
    option = choices[1]
    
    #Retrieve data from OULAD spreadsheets
    DATA = pd.read_csv("studentInfo.csv")
    DATA2 = pd.read_csv("studentAssessment.csv")
    DATA3 = pd.read_csv("assessments.csv")
    DATA4 = pd.read_csv("courses.csv")
    DATA5 = pd.read_csv("studentVle.csv")
    #DATA6 = pd.read_csv("vle.csv")
    DATA7 = pd.read_csv("studentRegistration.csv")

    #Renaming the data and setting indexes where appropriate
    studentInfo = DATA#.set_index("id_student")
    studentAssessment = DATA2.set_index("id_student")
    assessment = DATA3.set_index("id_assessment")
    courses = DATA4#.set_index(["code_module", "code_presentation"])
    studentVle = DATA5.set_index("id_student")
    #vle = DATA6
    studentRegistration = DATA7.set_index("id_student")

    #Creating a newly merged datafram to later be merged
    # This dataframe will add a column with the mean of each student's scores multiplied by its associated assignment's weight
    weightTotal = pd.merge(DATA2, DATA3, on="id_assessment")
    weightTotal["weightTotal"] = weightTotal["weight"] * weightTotal["score"]
    weightTotal = restructData(weightTotal, option)

    #Restructuring initial dataframes
    studentAssessment = restructData(studentAssessment, option)
    studentRegistration = restructData(studentRegistration, option)
    assessment = restructData(assessment, option)

    #For development purposes
    studentInfo.dtypes
    studentAssessment.dtypes

    #Initialising the dataframe into which others will be merged
    concatResult = studentInfo

    #Restructuring studentVle dataframe
    studentVle = restructData(studentVle, option)

    #Restructuring the merge of studentInfo and studentAssesment which have been left joined on student id
    #   It is from here that many of the predictors come
    concatResult = pd.merge(studentInfo, studentAssessment, on="id_student", how="left")
    concatResult = restructData(concatResult, option)

    #Merging the restructured courses dataframe
    new_merge = pd.merge(concatResult, courses, how='inner', on=["code_module", "code_presentation"])
    concatResult = new_merge

    #Merging the restructured assessment dataframe
    weightAdd = pd.merge(concatResult, assessment, how='left', on="id_assessment")
    concatResult = weightAdd

    #Merging the restructured studentVle datafram
    stuVleAdd = pd.merge(concatResult, studentVle, how='left', on="id_student")
    concatResult = stuVleAdd

    #Merging the restructured studentRegistration dataframe
    regDate = pd.merge(concatResult, studentRegistration, how='left', on="id_student")
    concatResult = regDate

    #Merging the created mean of summative weights dataframe
    addWeightedTotal = pd.merge(concatResult, weightTotal, how='left', on="id_student")
    concatResult = addWeightedTotal

    #Replacing most NaN values with the median of their columns
    concatResult["weight"] = concatResult["weight"].fillna(concatResult["weight"].median())
    concatResult["weightedTotal"] = concatResult["weightedTotal"].fillna(concatResult["weightedTotal"].median())
    concatResult["id_assessment"] = concatResult["id_assessment"].fillna(concatResult["id_assessment"].median())
    concatResult["date_submitted"] = concatResult["date_submitted"].fillna(concatResult["date_submitted"].median())
    concatResult["date"] = concatResult["date"].fillna(concatResult["date"].median())
    concatResult["numSites"] = concatResult["numSites"].fillna(concatResult["numSites"].median())
    concatResult["dateConcat"] = concatResult["dateConcat"].fillna(concatResult["dateConcat"].median())
    concatResult["is_banked"] = concatResult["is_banked"].fillna(concatResult["is_banked"].median())
    concatResult["sumConcat"] = concatResult["sumConcat"].fillna(concatResult["sumConcat"].median())

    #Replacing dummy NaN values with 0
    concatResult["CMA"] = concatResult["CMA"].fillna(0)
    concatResult["TMA"] = concatResult["TMA"].fillna(0)
    concatResult["Exam"] = concatResult["Exam"].fillna(0)

    #Use the genColumns function to generate a list of predictors
    columns = genColumns(['num_of_prev_attempts','studied_credits'], method)

    #Define the predictors and targets to be used
    predictors = concatResult[columns]
    print("PREDICTORS: ", predictors.columns)
    targets = concatResult.final_result

    #Split the dataset
    pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.3, random_state=42)

    #Print feature correlations, for development purposes
    #   This indicates which predictors may contribute more
    corr = concatResult.corr()
    print(corr["final_result"].sort_values())
    print(concatResult.describe())

    #Choose appropriate ML method based on user's initial input
    methodPrompt(pred_train, pred_test, tar_train, tar_test, columns, method, corr, concatResult, option, predictors, targets)

#Apply appropriate ML method and give the option to apply additional methods
def methodPrompt(pred_train, pred_test, tar_train, tar_test, columns, method, corr, completeDF, option, predictors, targets):
    #Choose appropriate ML Method
    if method.upper() == "D":
        genDecisionTree(pred_train, pred_test, tar_train, tar_test, columns, corr, completeDF, option)
    elif method.upper() == "R":
        genRandomForest(pred_train, pred_test, tar_train, tar_test, columns, corr, completeDF, option, predictors, targets)
    elif method.upper() == "L":
        genLinearRegression(pred_train, pred_test, tar_train, tar_test)
    elif method.upper() == "S":
        genSVC(pred_train, pred_test, tar_train, tar_test)    
    elif method.upper() == "C":
        genLogisticRegression(pred_train, pred_test, tar_train, tar_test, corr, completeDF, option, predictors, targets)    

    #Allow for the use of additional ML methods before exiting
    yn = input("Input another command [Y/N]? ")
    if yn.upper() == "Y":
        start()
    else:
        exit()

#Generate a grid of parameters to try
def genRandomGrid(method):
    #Create a parameter grid for tuning logistic regression
    if method == "L":
        param_grid = {
            'penalty' : ['l1', 'l2'],
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'tol': [0.001, 0.0001, 0.00001],
            'solver' : ['liblinear', 'lbfgs'] 
            # As for Multi_class default seems to perform the best, I have removed this parameter to reduce runtime
        }

    #Create a parameter grid for tuning Random Forest Classifier
    elif method == "R":
        max_depth = [int(n) for n in np.linspace(10, 100, num=10)]
        max_depth.append(None)
        param_grid = {
            'n_estimators': [int(n) for n in np.linspace(start=100, stop=200, num=10)],
            'max_depth': max_depth,
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }

    #Return the parameter grid for hyper parameter tuning
    return param_grid

#Prompt user for their preferences
def chooseMethod():
    #Allow user to input their chosen ML method
    print()
    print("Decision Tree - D")
    print("Random Forest - R")
    print("Linear Regression - L")
    print("Logistic Regression - C")
    print("SVC - S")
    method = input("Enter a command above: ")
    #Catching invalid input
    if method.upper() not in ["D", "R", "L", "C", "S"]: 
        print("Incorrect Input")
        chooseMethod()

    #Allow user to input their choesn number of categories
    print()
    print("4 Categories (Withdrawn, Fail, Pass, Distinction) - A")
    print("3 Categories (Withdrawn/Fail, Pass, Distinction) - B")
    print("3 Categories (Withdrawn, Fail, Pass/Distinction) - C")
    print("2 Categories (Withdrawn/Fail, Pass/Distinction) - D")
    option = input("Enter an option above: ")
    #Catching invalid input
    if option.upper() not in ["A", "B", "C", "D"]: 
        print("Incorrect Input")
        chooseMethod()
    
    #Returning user inputs
    return [method, option]

#Initiate user input and main function
#   This has been made a function so that it can be called again from the methodPrompt function
def start():
    choices = chooseMethod()
    main(choices)

print("Python: ", sys.version)
print("Pandas: ", pd.__version__)
print("Numpy: ", np.__version__)
print("Seaborn: ", sns.__version__)
print("matplotlib.pylab: ", plt.__version__)

#Begin the program
start()