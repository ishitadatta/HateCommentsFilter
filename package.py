import pandas as pd
import numpy as np
import random
import seaborn as sns
class pred:

	#Constructor To Accept Values From The Form
	def __init__(self, comment):
		self.comment = comment

	#Predicts Hate Speech Comments
	def predictor(self):
		stock=pd.read_csv('bad-words.csv')
        self.comment
        username = input()
        li = list(username.split(" "))
        a=open('bad-words.csv').read()
        list1 = list(a.split("\n"))
        for x in li:
            if x in list1:
                flag = 1
                print(x)
                break
            else:
                flag = 0
                if flag:
                    print("True")
                    else:
                        print("False")
    	return [df.to_html()]    
"""        #Obtaining The Dataframe Values For The Company Specified In The Form
		a=Hate_Comments_Filter.loc[Hate_Comments_Filter['Symbol']==self.comment]
		#Obtaining Only Those Values From The Datframe Which Are Necessary For The Predictions
		a=a[['Words']] 
		#Renaming The Columns So That It's Understandable By The Learning Model
		a=a.rename(columns={'Words':'wd'})

		#Renaming The Columns Of The Prediction Dataframe So That It's Understandable By The User
		columns_renamed={
			'Comment':'Comment',
			'Hate_Word':'Hate Word Detected'
			'Hate_Speech':'Is the following comment a Hate Speech?',
		}

		#Taking Only Those Columns Which Are To Be Displayed On The Web Page
		columns_for_prediction=[
			'Comment',
			'Hate Word Detected',
			'Is the following comment a Hate Speech?'
		]

		#Obtaining Plot For Stock Price Versus Date (Shows The Stock Prices Change Over The Intervals Of 1 Year)
		figure = m.plot(df, xlabel='Date', ylabel='Price')
		#Address Of Where To Save The Graph
		url_figure=f".\\static\\assets\\img\\graphs\\figure\\{self.cmp1}_{self.time}.png"
		plt.savefig(url_figure)

		#Obtaining Plots For Weekly Changes, Yearly Changes And Trend Changes (In Intervals Of 1 Year)
		figure3 = m.plot_components(df)
		url_figure3=f".\\static\\assets\\img\\graphs\\figure3\\{self.cmp1}_{self.time}.png"
		plt.savefig(url_figure3)

		#Renaming The Columns Of The Prediction Dataframe
		df=df.rename(columns=columns_renamed)

		#Converting The Dataframe To An HTML Table (Stored As A String Object)
		#index=False Removes The Row Values Of The Datarframe
		df=df.to_html(columns=columns_for_prediction, index=False, border=2)
"""
	