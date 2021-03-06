# A Simple Web App for Predicting Live-birth Occurence

## Background

- Infertility, among other factors, is considered the cause of unsuccessful conception. Advancements in medical procedure and physiology enable the fertilization process to occur outside of the human body, which is generally referred to as in-vitro fertilization (IVF). 

- IVF does not guarantee pregnancy even if the couple passes the assessment process. The factors that cause conception failure come in a broad spectrum, ranging from low to high significance. There could be some combination of factors or underlying factors that contribute to the successful conception in IVF, something that is missing if the assessment is made based on experience or incomplete statistical information. Moreover, IVF is well known to be a high-cost medical service with great uncertainty. 

- We aim to create a model that can predict live-birth occurrences based on patients’ medical records. The final model will be deployed as a web application, complete with a simple UI that shows prediction results.

## Objectives

- The main goal is to predict the live-birth occurrence based on the input given by the users

- The second goal is to give some consideration to the patients whether it is worth it or not to continue the IVF program. For example, the medical practitioner has a pessimistic hope, and the model predicts that the live-birth occurrence is low. The patients could stop the IVF program and use the money for other purposes

## Methods

- The overall system architecture is shown below:
![](https://github.com/harry-bpm/plds_teamc/blob/master/method.png)

- The following deep learning model is used to predict the live-birth occurrence:
![](https://github.com/harry-bpm/plds_teamc/blob/master/model.png)

|       Parameters      |             Value            |
|:---------------------:|:----------------------------:|
|       Batch size      |              256             |
|        Optimizer      |             Adam             |
|          Loss         |      Binary crossentropy     |
|         Epochs        |              200             |
|     Callbacks used    | Early stopping (patience=10) |

## About the dataset
From the dataset, 23 features were selected based on the reviewed paper (Ratna et. al). All types of infertilities (6) and causes of infertilities (11) are categorical data. Further processing was not necessary since there are not any abnormalities in those features. Other features were processed and explained briefly as follows:
- Age range: First make sure all data is in the same format (18-34, 35-37, 38-39, 40-42, 43-44, 45-50). Then assign the following values (0,1,2,3,4,5) to each category respectively
- Total Number of Previous treatments, Both IVF and DI at the clinic: Most data is already categorical (ranging from 0 to 5). If the data is “>5”, replace it with “6”
- Total Number of IVF pregnancies: Most data is already categorical (ranging from 0 to 5)
- Total Number of Previous IVF cycles: Most data is already categorical (ranging from 0 to 5). If the data is “>5”, replace it with “6”
- Embryos Transferred: Most data is already categorical (ranging from 0 to 3). 
- Total Embryos Created: Numerical data


## Dashboard
- The web app can be accessed here: https://ivf-livebirth-test.herokuapp.com/

- Simply fill in the form and press the predict button
![](https://github.com/harry-bpm/plds_teamc/blob/master/dashboard.gif)

## Analysis
Prediction tends to give a negative result. This is probably due to an unbalanced dataset (i.e., more negative samples than positive samples). 

## Conclusion
- Deployed model in the web app could give live-birth occurrence prediction. Simple UI makes anyone could use the app without any problem.

- Though the app works as intended, the prediction result must not be used as the main reason when deciding to do IVF.

## References
- Goyal, A., Kuchana, M. & Ayyagari, K.P.R. Machine learning predicts live-birth occurrence before in-vitro fertilization treatment. Sci Rep 10, 20925 (2020). https://doi.org/10.1038/s41598-020-76928-z

- Ratna MB, Bhattacharya S, Abdulrahim B, McLernon DJ. A systematic review of the quality of clinical prediction models in in vitro fertilisation. Hum Reprod. 2020 Jan 1;35(1):100-116. doi: 10.1093/humrep/dez258. PMID: 31960915.

- https://github.com/FChmiel/ivf_embryo_prediction

