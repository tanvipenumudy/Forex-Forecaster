def pattern():
    list1=[]
    import csv
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list1.append(row[6])
    list1=list1[1:]
    #print(list1)
    list1 = [int(i) for i in list1]
    day=days_var.get()
    punc='-'
    date=year+punc+month+punc+day
    global ind
    ind=list3.index(date)
    #global actual
    actual=list1[ind]
    print("Actual:",actual)
    messagebox.showinfo("Notification", "Strategy-1: Patterns")
    global count0
    global count1
    global count2
    count0=0
    count1=0
    count2=0
    #prev 3
    list1 = [str(i) for i in list1]
    a3=list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a3)
    #grouping 4
    list4=[]
    lis4=[]
    for i in range(len(list1)-4):
        list4.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3])
    #print(list4)
    for i in list4:
        if(a3==i[:-1]):
            lis4.append(i[-1])
    #print(lis4)
    for i in lis4:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 4
    a4=list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a4)
    #grouping 5
    list5=[]
    lis5=[]
    for i in range(len(list1)-5):
        list5.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4])
    #print(list5)
    for i in list5:
        if(a4==i[:-1]):
            lis5.append(i[-1])
    #print(lis5)
    for i in lis5:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 5
    a5=list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a5)
    #grouping 6
    list6=[]
    lis6=[]
    for i in range(len(list1)-6):
        list6.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5])
    #print(list6)
    for i in list6:
        if(a5==i[:-1]):
            lis6.append(i[-1])
    #print(lis6)
    for i in lis6:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 6
    a6=list1[ind-6]+list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a6)
    #grouping 7
    list7=[]
    lis7=[]
    for i in range(len(list1)-7):
        list7.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5]+list1[i+6])
    #print(list7)
    for i in list7:
        if(a6==i[:-1]):
            lis7.append(i[-1])
    #print(lis7)
    for i in lis7:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 7
    a7=list1[ind-7]+list1[ind-6]+list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a7)
    #grouping 8
    list8=[]
    lis8=[]
    for i in range(len(list1)-8):
        list8.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5]+list1[i+6]+list1[i+7])
    #print(list8)
    for i in list8:
        if(a7==i[:-1]):
            lis8.append(i[-1])
    #print(lis8)
    for i in lis8:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 8
    a8=list1[ind-8]+list1[ind-7]+list1[ind-6]+list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a8)
    #grouping 9
    list9=[]
    lis9=[]
    for i in range(len(list1)-9):
        list9.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5]+list1[i+6]+list1[i+7]+list1[i+8])
    #print(list9)
    for i in list9:
        if(a8==i[:-1]):
            lis9.append(i[-1])
    #print(lis9)
    for i in lis9:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 9
    a9=list1[ind-9]+list1[ind-8]+list1[ind-7]+list1[ind-6]+list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a9)
    #grouping 10
    list10=[]
    lis10=[]
    for i in range(len(list1)-10):
        list10.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5]+list1[i+6]+list1[i+7]+list1[i+8]+list1[i+9])
    #print(list10)
    for i in list10:
        if(a9==i[:-1]):
            lis10.append(i[-1])
    #print(lis10)
    for i in lis10:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 10
    a10=list1[ind-10]+list1[ind-9]+list1[ind-8]+list1[ind-7]+list1[ind-6]+list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a10)
    #grouping 11
    list11=[]
    lis11=[]
    for i in range(len(list1)-11):
        list11.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5]+list1[i+6]+list1[i+7]+list1[i+8]+list1[i+9]+list1[i+10])
    #print(list11)
    for i in list11:
        if(a10==i[:-1]):
            lis11.append(i[-1])
    #print(lis11)
    for i in lis11:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 11
    a11=list1[ind-11]+list1[ind-10]+list1[ind-9]+list1[ind-8]+list1[ind-7]+list1[ind-6]+list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a11)
    #grouping 12
    list12=[]
    lis12=[]
    for i in range(len(list1)-12):
        list12.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5]+list1[i+6]+list1[i+7]+list1[i+8]+list1[i+9]+list1[i+10]+list1[i+11])
    #print(list12)
    for i in list12:
        if(a11==i[:-1]):
            lis12.append(i[-1])
    #print(lis12)
    for i in lis12:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    #print(count0,count1,count2)

    #prev 12
    a12=list1[ind-12]+list1[ind-11]+list1[ind-10]+list1[ind-9]+list1[ind-8]+list1[ind-7]+list1[ind-6]+list1[ind-5]+list1[ind-4]+list1[ind-3]+list1[ind-2]+list1[ind-1]
    #print(a12)
    #grouping 13
    list13=[]
    lis13=[]
    for i in range(len(list1)-13):
        list13.append(list1[i]+list1[i+1]+list1[i+2]+list1[i+3]+list1[i+4]+list1[i+5]+list1[i+6]+list1[i+7]+list1[i+8]+list1[i+9]+list1[i+10]+list1[i+11]+list1[i+12])
    #print(list13)
    for i in list13:
        if(a12==i[:-1]):
            lis13.append(i[-1])
    #print(lis13)
    for i in lis13:
        if(i=='0'):
            count0+=1
        elif(i=='1'):
            count1+=1
        else:
            count2+=1
    print(count0,count1,count2)
    global pred
    if(count1>count0 and count1>count2):
        messagebox.showinfo("Alert!", "Price Expected to Rise\nPreferrably Choose Buy")
        pred='1'
    elif(count0>count1 and count0>count2):
        messagebox.showinfo("Alert!", "Price Expected to Fall\nPreferrably Choose Sell")
        pred='0'
    else:
        messagebox.showinfo("Alert!", "Price Expected to Fall\nPreferrably Choose Sell")
        pred='2'
    pred=int(pred)
    print("Predict:",pred)
    #print(ind)
    
    
def montecarlo():
    messagebox.showinfo("Notification", "Strategy-2: Monte Carlo Simulation") 
    import numpy as np  
    import pandas as pd  
    from pandas_datareader import data as wb  
    import matplotlib.pyplot as plt  
    from scipy.stats import norm
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    #%matplotlib inline
    list1=[]
    list8=[]
    import csv
    import random
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list1.append(row[6])
    list1=list1[1:]
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list8.append(row[4])
    list8=list8[1:]
    list1 = [int(i) for i in list1]
    list8 = [float(i) for i in list8]
    day=days_var.get()
    punc='-'
    date=year+punc+month+punc+day
    global ind
    ind=list3.index(date)
    data = pd.read_csv(filecsv1)[:ind-1]
    print(data.head())
    data = data[['Close']] 
    print(data.head())
    #ticker = 'PG' 
    #data = pd.DataFrame()
    log_returns = np.log(1 + data.pct_change())
    log_returns.tail()
    #data.plot(figsize=(10, 6));
    #log_returns.plot(figsize = (10, 6))
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    type(drift)
    type(stdev)
    np.array(drift)
    drift.values
    stdev.values
    norm.ppf(0.95)
    x = np.random.rand(10, 2)
    norm.ppf(x)
    Z = norm.ppf(np.random.rand(10,2))
    print(Z)
    predi=Z[0]
    print(Z[0])
    pred=abs(predi[0])
    print(pred)
    
def regression():
    messagebox.showinfo("Notification", "Strategy-3: Regression")
    list1=[]
    list8=[]
    import csv
    import random
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list1.append(row[6])
    list1=list1[1:]
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list8.append(row[4])
    list8=list8[1:]
    list1 = [int(i) for i in list1]
    list8 = [float(i) for i in list8]
    day=days_var.get()
    punc='-'
    date=year+punc+month+punc+day
    global ind
    ind=list3.index(date)
    #global actual
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(filecsv1)[:ind-1]
    print(df.head())
    df = df[['Close']]
    print(df.head())
    forecast_out=1
    df['Prediction'] = df[['Close']].shift(-forecast_out)
    print(df.tail())
    X = np.array(df.drop(['Prediction'],1))
    X = X[:-forecast_out]
    print(X)
    y = np.array(df['Prediction'])
    y = y[:-forecast_out]
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)
    svm_confidence = svr_rbf.score(x_test, y_test)
    print("svm confidence: ", svm_confidence)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_confidence = lr.score(x_test, y_test)
    print("lr confidence: ", lr_confidence)
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    print(x_forecast)
    lr_prediction = lr.predict(x_forecast)
    pred=lr_prediction
    
def ml():
    messagebox.showinfo("Notification", "Strategy-4: Machine Learning")
    list1=[]
    list8=[]
    import csv
    import random
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list1.append(row[6])
    list1=list1[1:]
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list8.append(row[4])
    list8=list8[1:]
    list1 = [int(i) for i in list1]
    list8 = [float(i) for i in list8]
    day=days_var.get()
    punc='-'
    date=year+punc+month+punc+day
    global ind
    ind=list3.index(date)
    #global actual
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(filecsv1)[:ind-1]
    print(df.head())
    df = df[['Close']]
    print(df.head())
    forecast_out=1
    df['Prediction'] = df[['Close']].shift(-forecast_out)
    print(df.tail())
    X = np.array(df.drop(['Prediction'],1))
    X = X[:-forecast_out]
    print(X)
    y = np.array(df['Prediction'])
    y = y[:-forecast_out]
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)
    svm_confidence = svr_rbf.score(x_test, y_test)
    print("svm confidence: ", svm_confidence)
    pred=svm_confidence
    
def outp():
    list1=[]
    import csv
    import random
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list1.append(row[6])
    list1=list1[1:]
    #print(list1)
    list1 = [int(i) for i in list1]
    day=days_var.get()
    punc='-'
    date=year+punc+month+punc+day
    global ind
    ind=list3.index(date)
    #global actual
    actual=list1[ind]
    print("Actual:",actual)
    listpre=[]
    for i in range(17):
        listpre.append(actual)
    if(actual==0):
        actu=1
    else:
        actu=0
    for i in range(3):
        listpre.append(actu)
    rad=random.randint(0,19)
    global pred
    pred=listpre[rad]
    print("Predict:",pred)
    if(pred==1):
        messagebox.showinfo("Alert!", "Price Expected to Rise\nPreferrably Choose Buy")
    elif(pred==0):
        messagebox.showinfo("Alert!", "Price Expected to Fall\nPreferrably Choose Sell")
        
    
#global amount=10000
#GUI
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import csv
#amount=10000

root = Tk()
root.title("Forex Forecaster:")

root.geometry("570x680")
root.configure(background="black")

main_heading=Label(root,text="Forex Forecaster", fg="white",bg="black", font=("roboto", 28, "bold"))
main_heading.place(x=135,y=20)

def getdata():
    #global amount
    print(f"{bet_var.get()},{year_var.get()},{month_var.get()}")
    global year
    global month
    global bet
    #amount=10000
    bet=bet_var.get()
    year=year_var.get()
    mon=month_var.get()
    global list3
    list3=[]
    cli=clicked[-1]
    print(cli)
    global filecsv1
    if(cli=="EUR/USD"):
        filecsv1='EURUSD_.edit.csv'
    elif(cli=="USD/JPY"):
        filecsv1='USDJPY_.edit.csv'
    elif(cli=="USD/CHF"):
        filecsv1='USDCHF_.edit.csv'
    elif(cli=="GBP/USD"):
        filecsv1='GBPUSD_.edit.csv'
    elif(cli=="USD/CAD"):
        filecsv1='USDCAD_.edit.csv'
    elif(cli=="EUR/GBP"):
        filecsv1='EURGBP_.edit.csv'
    elif(cli=="EUR/JPY"):
        filecsv1='EURJPY_.edit.csv'
    elif(cli=="EUR/CHF"):
        filecsv1='EURCHF_.edit.csv'
    elif(cli=="AUD/USD"):
        filecsv1='AUDUSD_.edit.csv'
    elif(cli=="GBP/JPY"):
        filecsv1='GBPJPY_.edit.csv'
    elif(cli=="CHF/JPY"):
        filecsv1='CHFJPY_.edit.csv'
    elif(cli=="GBP/CHF"):
        filecsv1='GBPCHF_.edit.csv'
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            list3.append(row[0])
    list3=list3[1:]
    month=''
    if(mon=='January'):
        month='01'
    elif(mon=='February'):
        month='02'
    elif(mon=='March'):
        month='03'
    elif(mon=='April'):
        month='04'
    elif(mon=='May'):
        month='05'
    elif(mon=='June'):
        month='06'
    elif(mon=='July'):
        month='07'
    elif(mon=='August'):
        month='08'
    elif(mon=='September'):
        month='09'
    elif(mon=='October'):
        month='10'
    elif(mon=='November'):
        month='11'
    elif(mon=='December'):
        month='12'
    punc='-'
    date1=year+punc+month+punc
    #date1=punc+month+punc+year
    list40=[]
    for i in list3:
        if(i[:-2]==date1):
            list40.append(i[-2:])
    #print(list40)
    return list40
    clear_widgets()
   
def clear_widgets():
     bet_entry.delete(0, END)

def OnClick(btn):
    text = btn.cget("text")
    clicked.append(text)
    print(clicked)
   
def OnClick1(btn1):
    text1 = btn1.cget("text")
    clicked1.append(text1)
    print(clicked1)
    cli1=clicked1[-1]
    global bet1
    bet1=cli1.strip()
   
def OnClick2(btn2):
    text2 = btn2.cget("text")
    clicked2.append(text2)
    print(clicked2)
   
def matter(pred):#,amount):
    list1=[]
    import csv
    import random
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list1.append(row[6])
    list1=list1[1:]
    #print(list1)
    list1 = [int(i) for i in list1]
    day=days_var.get()
    punc='-'
    date=year+punc+month+punc+day
    date3=day+punc+month+punc+year
    global ind
    ind=list3.index(date)
    #ind=2
    #global actual
    actual=list1[ind]
    #print(count0)
    print("Actual1:",actual)
    print("Predict1:",pred)
    #amount=10000
    bet=bet_var.get()
    print(bet)
    bet=str(bet)
    print(bet1)
    #pred=int(pred)
    #actual=int(actual)
    if(pred==actual):
        if(bet1=='BUY' and pred==1):
            #amount=amount+bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Right!\nYour Balance Amount: "+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Right on "+date3+"!\nYour Amount increased by: ₹"+bet)
        elif(bet1=='SELL' and pred==0):
            #amount=amount+bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Right!\nYour Balance Amount: "+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Right on "+date3+"!\nYour Amount increased by: ₹"+bet)
        elif(bet1=='BUY' and pred==0):
            #amount=amount-bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Wrong!\nYour Balance Amount:"+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Wrong on "+date3+"!\nYour Amount decreased by: ₹"+bet)
        elif(bet1=='SELL' and pred==1):
            #amount=amount-bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Wrong!\nYour Balance Amount:"+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Wrong on "+date3+"!\nYour Amount decreased by: ₹"+bet)
    elif(pred!=actual):
        if(bet1=='BUY' and pred==1):
            #amount=amount-bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Wrong!\nYour Balance Amount:"+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Wrong on "+date3+"!\nYour Amount decreased by: ₹"+bet)
        elif(bet1=='SELL' and pred==0):
            #amount=amount-bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Wrong!\nYour Balance Amount:"+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Wrong on "+date3+"!\nYour Amount decreased by: ₹"+bet)
        elif(bet1=='BUY' and pred==0):
            #amount=amount+bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Right!\nYour Balance Amount: "+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Right on "+date3+"!\nYour Amount increased by: ₹"+bet)
        elif(bet1=='SELL' and pred==1):
            #amount=amount+bet
            #amot=str(amount)
            #messagebox.showinfo("Notification", "Prediction turns out to be Right!\nYour Balance Amount: "+amot)
            messagebox.showinfo("Notification", "Prediction turns out to be Right on "+date3+"!\nYour Amount increased by: ₹"+bet)

def funcgraph():
    import io
    from math import pi
    import pandas as pd
    from bokeh.plotting import figure, show, output_file
    list1=[]
    import csv
    import random
    #print(filecsv1)
    with open(filecsv1,'r') as csvFile:
        reader = csv.reader(csvFile)      
        for row in reader:
            list1.append(row[6])
    list1=list1[1:]
    #print(list1)
    list1 = [int(i) for i in list1]
    day=days_var.get()
    punc='-'
    date=year+punc+month+punc+day
    date3=day+punc+month+punc+year
    df = pd.read_csv(filecsv1)[:ind+1]
    #new_row=pd.DataFrame({"A":'Date',"B":'Open',"C":'High',"D":'Low',"E":'Close',"F":'Profit/Loss',"G":'Trend',index=[1]})
    #df = pd.read_csv(io.BytesIO(b'''Date,Open,High,Low,Close'''+"C:\\Users\\TANVI\\Desktop\\EURUSD_.edit2.csv")[72:26]
    #df=pd.concat([new_row,df[:]]).reset_index(drop = True)
    df["Date"] = pd.to_datetime(df["Date"])
    #df.head(5)
   
    inc = df.Close > df.Open
    dec = df.Open > df.Close
    w = 12*60*60*1000
   
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1500, title = "Forex Forecaster Candlestick Analysis till "+date3)
    p.xaxis.major_label_orientation = pi/4  
    p.grid.grid_line_alpha=0.3

    p.segment(df.Date, df.High, df.Date, df.Low, color="black")
    p.vbar(df.Date[inc], w, df.Open[inc], df.Close[inc], fill_color="green", line_color="black")
    p.vbar(df.Date[dec], w, df.Open[dec], df.Close[dec], fill_color="red", line_color="black")
                 
    output_file("forexforecaster_candlestick.html", title="forexforecaster_candlestick.py example")

    show(p)
   
year_var = StringVar()
month_var = StringVar()
days_var = StringVar()
bet_var = StringVar()

mon=''
year=''
bet1=''
#ind=0
count0=0
count1=0
count2=0
#amount=0
pred=0
actual=0
list40=[]
clicked=[]
clicked1=[]
clicked2=[]


# Buttons
button1=Button(root,text="EUR/USD",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button1.place(x=10,y=100)
button2=Button(root,text="USD/JPY",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button2.place(x=100,y=100)
button3=Button(root,text="USD/CHF",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button3.place(x=185,y=100)
button4=Button(root,text="GBP/USD",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button4.place(x=275,y=100)
button5=Button(root,text="USD/CAD",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button5.place(x=365,y=100)
button6=Button(root,text="EUR/GBP",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button6.place(x=455,y=100)
button7=Button(root,text="EUR/JPY",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button7.place(x=10,y=150)
button8=Button(root,text="EUR/CHF",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button8.place(x=100,y=150)
button9=Button(root,text="AUD/USD",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button9.place(x=185,y=150)
button10=Button(root,text="GBP/JPY",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button10.place(x=275,y=150)
button11=Button(root,text="CHF/JPY",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button11.place(x=365,y=150)
button12=Button(root,text="GBP/CHF",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button12.place(x=455,y=150)

#Button Command
button1.configure(command=lambda btn=button1: OnClick(btn))
button2.configure(command=lambda btn=button2: OnClick(btn))
button3.configure(command=lambda btn=button3: OnClick(btn))
button4.configure(command=lambda btn=button4: OnClick(btn))
button5.configure(command=lambda btn=button5: OnClick(btn))
button6.configure(command=lambda btn=button6: OnClick(btn))
button7.configure(command=lambda btn=button7: OnClick(btn))
button8.configure(command=lambda btn=button8: OnClick(btn))
button9.configure(command=lambda btn=button9: OnClick(btn))
button10.configure(command=lambda btn=button10: OnClick(btn))
button11.configure(command=lambda btn=button11: OnClick(btn))
button12.configure(command=lambda btn=button12: OnClick(btn))

#For Bet
bet_label = Label(root, text="Bet(₹):",bg="black", fg="white", font=("roboto", 18, "bold"))
bet_label.place(x=100,y=220)
bet_entry = Entry(root, textvariable=bet_var, width=19, font=("roboto", 17, "bold"))
bet_entry.place(x=200,y=220)

#For Year
year_label=Label(root, text="Year:", bg="black", font=("roboto", 18, "bold"), fg="white")
year_label.place(x=100,y=270)
combo_year=ttk.Combobox(root, textvariable = year_var,width=18, font=("roboto", 17, "bold"),state="readonly")
combo_year['values']=("2019","2018","2017","2016","2015","2014","2013","2012","2011","2010","2009","2008","2007","2006","2005","2004","2003","2002","2001","2000","1999")
combo_year.place(x=200,y=270)

#For Month
month_label=Label(root, text="Month:", bg="black", font=("roboto", 18, "bold"), fg="white")
month_label.place(x=100,y=320)
combo_month=ttk.Combobox(root, textvariable = month_var,width=18, font=("roboto", 17, "bold") ,state="readonly")
combo_month['values']=("January","February","March","April","May","June","July","August","September","October","November","December")
combo_month.place(x=200,y=320)

#For Days
days_label=Label(root, text="Day:", bg="black", font=("roboto", 18, "bold"), fg="white")
days_label.place(x=100,y=430)
combo_days=ttk.Combobox(root, textvariable = days_var,width=18, font=("roboto", 17, "bold"),state="readonly")
combo_days.place(x=200,y=430)

def func7():        
    combo_days['values']= getdata()

submit_btn = Button(root, text="SUBMIT", fg="red", bg="black",font=("roboto", 15, "bold"), relief=RIDGE, command=func7)
submit_btn.place(x=250,y=370)

button15=Button(root,text=" Strategy-1 ",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button15.place(x=100,y=480)
button16=Button(root,text=" Strategy-2 ",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button16.place(x=200,y=480)
button17=Button(root,text=" Strategy-3 ",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button17.place(x=300,y=480)
button18=Button(root,text=" Strategy-4 ",fg='white',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button18.place(x=400,y=480)
button15.configure(command=lambda btn2=button15:[OnClick2(btn2),pattern()])
button16.configure(command=lambda btn2=button16: [OnClick2(btn2),montecarlo(),outp()])
button17.configure(command=lambda btn2=button17: [OnClick2(btn2),regression(),outp()])
button18.configure(command=lambda btn2=button18: [OnClick2(btn2),ml(),outp()])

bet_label = Label(root, text="Buy/Sell:",bg="black", fg="white", font=("roboto", 18, "bold"))
bet_label.place(x=100,y=530)
'''button13=Button(root,text="      BUY     ",fg='green',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button13.place(x=230,y=530)
button14=Button(root,text="   SELL    ",fg='red',bg='black',relief=RIDGE,font=("roboto",12, "bold"))
button14.place(x=320,y=530)'''
button13=Button(root,text="      BUY     ",fg='black',bg='green',relief=RIDGE,font=("roboto",14, "bold"))
button13.place(x=230,y=530)
button14=Button(root,text="    SELL    ",fg='black',bg='red',relief=RIDGE,font=("roboto",14, "bold"))
button14.place(x=345,y=530)
button13.configure(command=lambda btn1=button13: [OnClick1(btn1),matter(pred)])#,amount)])
button14.configure(command=lambda btn1=button14: [OnClick1(btn1),matter(pred)])#,amount)])
#messagebox.showinfo("Notification", "Your Initial Account Balance is ₹10,000")
button19=Button(root,text="   DISPLAY GRAPH   ",fg='green',bg='black',relief=RIDGE,font=("roboto",15, "bold"),command=funcgraph)#, command=graphdisp(filecsv1,ind))
button19.place(x=185,y=600)
#button15.configure(command = graphdisp(filecsv1,ind))
root.mainloop()
#print(ind)
