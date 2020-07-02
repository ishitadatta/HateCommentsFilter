# Hate Comments Filter
A comment filtering algorithm to detect the polarity in words and phrases using sentiment analysis techniques in NLP.

## Purpose
This project was created as part of the hackathon - Hashcode by Microsoft Innovation Lab

## Project Description
Hate speech is currently of broad and current interest in the domain of social media. The anonymity and flexibility afforded by the Internet has made it easy for users to communicate in an aggressive manner. And as the amount of online hate speech is increasing, methods that automatically detect hate speech is very much required.  In order to create a hate-speech-detecting algorithm, we have used Python-based NLP machine learning techniques. We collected data (say, the data from a social media site that may have comments / tweets) and trained it. For data collection, we used web crawling. Using the beautifulsoup library , we collected the comments of popular posts on a website. 

## Twitter analysis
For sentiment analysis on tweets we have used Tweepy by making use of twittter credentials to access keys and api and achieved the following details:
* Streaming Tweets
* Accessing published tweets by pagination
* Analyzing Twitter data
* Visualising Twitter data
* Running sentiment analysis on Twitter data

## Implementation in-progress
Using a technique called Tf-Idf vectorization, we intend on extracting keywords that convey importance within hate speech. After collecting all the texts within those tags, we created a hate speech dataset. We are using logistic regression to train the computer to classify hate speech using the data extracted.


