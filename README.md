# Kaggle-Prudential-Life-Insurance-Assesment

https://www.kaggle.com/c/prudential-life-insurance-assessment

**Final private leader board ranking: 48/2619 (top 2%)**

My solution and ideas to start and finish a kaggle competition in a week.


###First, the solution.

To run **Final_Submission.py**, please first download two pre-processed data files - **train_processed.csv** and **test_processed.csv**, and put them in the same folder. Then change the variable **FOLDER** in the **Final_Submission.py** to the directory that you've downloaded the data files. This script will generate a file **submission.csv** as the final prediction results.

The two processed data files are produced using the following steps:

1) count the number of null feature values row-wise

2) generate two additional features seperating the character and number parts of the feature "Product_Info_2"

3) add interaction term between BMI and Age

4) add counts of medical keywords

5) one-hot-encoding for 'Product_Info_2', 'Product_Info_2_char', 'Product_Info_2_num'

6) encode all the categorical features with the out-of-fold mean of the target

The final submission is composed of two seperate models with slightly different paramters. The model weights are 2 and 1 respectively. One extra step is added after the model training to find the optimal splits. This step maximizes the quadratic weighted kappa with respect to 7 cut-off points. With those optimal cut-off points, the final prediction is then caculated by mapping the raw regression results to the oridinal risk measures.


###Second, the ideas to complete a competition in a short time.

I started this competition a week before the deadline and had about 5 weekday nights plus the weekends to figure things out. I'll list the tasks for each day and hopefully that would provide a framework to manage the time and resources if a short turnaround is required.

**Please note that this is just for data competitions like Kaggle where the training and test sets have been cleaned. In real world, the data munging part should take a significant amout of time and would be absolute key in machine learning pipeline.**


**- Day 1:**

1) Download the data

2) Exploratory analysis and feature engineering

Day 1 is crucial. You should perform descriptive analysis to get an idea of what the data look like and hence choose the right tools, procedures, and models. The things that need to check including but not limited to:

-- Data size and feature dimension (fit into the memory or not)

-- Missing values (how to impute the null values)

-- Distributions of the target (skewed or not)

-- Non-numeric features (encoding strategy)

-- Variance by feature (redundent or not)

-- Correlations between features

-- Interaction terms

After exploring the data, you should be able to conduct feature preprocessing and engineering if necessary. In my case, feature engineering always makes the difference and brings in siginifant improvement in model performance.


**- Day 2:**

1) Understand the evaluation metric

2) Set up the cross validation correctly

Day 2 is all about model evaluation. You should build a mechanism to evaluate the model performance locally. The first key is the evaluation metric. Understanding the metric that is being used is helpful for both the coding and the model selection. The second key is the cross validation. Setting it up correctly is not easy. Tools like numpy and scikit-learn are your friends. You can use many utility functions from these tools to implement your cross validation. 


**- Day 3:**

1) Try different modeling algorithms 

2) Benchmark different models using cross-validation 

In day 3 we finally get to the modeling part. There are lots of excellent open source tools we can choose from, such as Scikit-learn, XGBoost, Keras, Vowpal Wabbit, etc. We need to take time to implement each of the tools to the specific data. After that, we should benchmark the model performance using cross-validation that we set up in day 2.


**- Day 4:**

1) Focus on a couple of modeling methods that outporm others

2) Grid-search for good set of parameters

Due to the limited time, we need to focus on a couple of modeling methods after benchmarking in day 3. These two modeling methods are the ones we want to improve on. One direction is to find a better parameter grid. We can set up the grid search of model parameters. Again, with short turnaround, the randomized parameter optimization is preferred over the exhaustive grid search.


**- Day 5 & 6:**

1) Try ensemble

At this point, you should have a number of good models with proper parameters. Also, you've got the local cross-validation set up. The next step for improving the model performance is to try ensemble of models. There are many ways that you can go. You can combine models with different specifications, sampling strategies, objective funcions, encoding methods, etc. The goal is to develop many well-performed yet diversified models and combine them. 


**- Day 7:**

1) Try stacking

Stacking is another way of combining the models. It is a multi-layer model framework and would work well if models that are being stacked are diversified. 


**This is a tight schedule, without doubt. I haven't listed all the technical details behind these bullet points. But I hope this would be helpful as a framework, especially when you face a scenario with reletively cleaned data, but short turnaround.**


Please do not hesitate to reach me at **feifeiyu1204@gmail.com** for comments & suggestions.


Best regards,

Feifei Yu
