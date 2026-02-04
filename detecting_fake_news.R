##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##
## Data Science and Public Policy
## Professor Tamar Mitts
## Columbia University
##----------------------------------------------------------------------##
## Detecting Fake News
## Data source: LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION
## More data can be found here: 
## https://paperswithcode.com/paper/liar-liar-pants-on-fire-a-new-benchmark
##----------------------------------------------------------------------##
## The code below can give you a head start on some ways you can examine the data:
## 1) Describe the dataset, exmamine who makes the most false statements
## 2) Use the text of the statements to build a fake news classifier
## 3)
##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##

library(quanteda)
library(quanteda.textmodels)
library(ggplot2)
library(readr)
library(caret)

setwd("")

train = read_tsv("train.tsv")
test = read_tsv("test.tsv")
valid = read_tsv("valid.tsv")

##---------------------##
## Describe the dataset
##---------------------##

## Combine all datasets to one for descriptive statistics:
all_data = rbind(train, test, valid)

## Note that each statement is labeled as one of 6 categories that go from
## most false to most true: pants-fire, false, barely-true, half-true, mostly-true, and true.

## You can create a variable that converts these labels to a scale:
all_data$truth_scale = NA
all_data$truth_scale[all_data$label=="pants-fire"] = 1
all_data$truth_scale[all_data$label=="FALSE"] = 2
all_data$truth_scale[all_data$label=="barely-true"] = 3
all_data$truth_scale[all_data$label=="half-true"] = 4
all_data$truth_scale[all_data$label=="mostly-true"] = 5
all_data$truth_scale[all_data$label=="TRUE"] = 6

## Examine the distribution of the scale:
hist(all_data$truth_scale)

## You can also create a binary version of this variable:
all_data$fake = ifelse(all_data$truth_scale %in% c(1,2,3), 1, 0)
# (here we code that half-true, mostly-true and true as not fake, but you can change of course!)

table(all_data$fake)

## Who tends to make more false statements?

fake_by_party = as.data.frame(prop.table(table(all_data$fake, all_data$party), margin=2))
colnames(fake_by_party) = c("fake", "party", "proportion")

ggplot(fake_by_party,aes(x=party,y=proportion,fill=fake)) + 
  geom_col() + coord_flip()

##-----------------------------##
## Build a fake news classifier
##-----------------------------##

## Here is some code you can use to build a simple Naive Bayes classifier using the text
## of the statements to to predict  fake news:

## First, create the variable "fake" for the train, test, and validation datasets
train$fake = ifelse(train$label %in% c("pants-fire","FALSE","barely-true"), 1, 0)
test$fake = ifelse(test$label %in% c("pants-fire","FALSE","barely-true"), 1, 0)
valid$fake = ifelse(valid$label %in% c("pants-fire","FALSE","barely-true"), 1, 0)

## Preprocess the train and test datasets:

train_dfm = corpus(train, text_field = "statement") %>%
  tokens(remove_numbers = TRUE, remove_punct = TRUE, remove_url = TRUE,
         remove_symbols = TRUE) %>% 
  tokens_wordstem(language = "en") %>% 
  tokens_tolower() %>%
  tokens_select(pattern = stopwords('en'), selection = 'remove') %>%
  dfm()

test_dfm = corpus(test, text_field = "statement") %>%
  tokens(remove_numbers = TRUE, remove_punct = TRUE, remove_url = TRUE,
         remove_symbols = TRUE) %>% 
  tokens_wordstem(language = "en") %>% 
  tokens_tolower() %>%
  tokens_select(pattern = stopwords('en'), selection = 'remove') %>%
  dfm() 

# The function textmodel_nb() classifies text using a Naive Bayes model.
# In the function, x is the DFM of the training data and y is the training labels associated
# with each training document:
classify_nb = textmodel_nb(x = train_dfm, y = docvars(train_dfm, "fake"))

# Naive Bayes can only consider words that occur both in the training set and the test set.
# Let's use the function dfm_match() to make the features identical 
# in the training and test sets:
test_dfm_matched = dfm_match(test_dfm, features = featnames(train_dfm))

## Let's predict the classes in the test set using the predict() function:
predicted_class = predict(classify_nb, newdata = test_dfm_matched, type="class")

## To evaluate performance, we examine the confusion matrix:
## confusionMatrix() is a powerful function from the 'caret' package that provides information
## on many model performance metrics:
actual_class = docvars(test_dfm_matched, "fake")
tab_class = table(predicted_class, actual_class)
confusionMatrix(tab_class, mode = "everything", positive="1")

# What is the accuracy? Precision? Recall? Specificity? F1?

## Additional ideas: you can build a model to see if other features in the data are better 
## at predicting fake news. Is it the speaker? The party of the speaker? The area where they
## come from? The context in which the statement is made? etc.


