import pandas as pd
import numpy as np
import statsmodels.api as sm

#Reading from the CSV file
googFile = "C:/Users/walajas/Documents/logistic.csv"
goog = pd.read_csv(googFile,sep=",",usecols=[0, 1, 2],names=['Date','Goog','SP500'],header=0)

#sort the details in ascending order of dates
goog['Date'] = pd.to_datetime(goog['Date'], format='%m/%d/%Y')
goog = goog.sort_values(['Date'], ascending=[True])
goog = goog[:-1]
goog['Goog'] = pd.to_numeric(goog['Goog'])
goog['SP500'] = pd.to_numeric(goog['SP500'])
returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]].pct_change()
returns['Intercept'] = 1
xData = np.array(returns[["Goog","SP500",'Intercept']][1:-1])
yData = (returns["Goog"] > 0)[2:]

logit = sm.Logit(yData, xData)
result = logit.fit()
result
#the predict below is ran over xData to know how it performs
k=result.predict(xData)
count = 0
for i in range(144):
    print(k[i], yData[i])
    if (k[i] == yData[i]):
        count = count +1
print(count)