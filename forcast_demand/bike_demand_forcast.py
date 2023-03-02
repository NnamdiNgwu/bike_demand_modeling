import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import Lasso, PoissonRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,KBinsDiscretizer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def pandas_csv_reader(csv_file=''):
    """function reads csv into pandas
    input: csv file
    oupt: pandas dataframe"""
    # paser_dates converts the date from string to datetime
    train_df = pd.read_csv(csv_file, parse_dates=True, index_col=0)
    return train_df

def feature_expansion(train_df1):
    """#expanding the columns through day time index.
    this will create columns
     containing individual time data 
     input: dataframe
     output: dataframe with some extral features"""
    #creating hours of the day
    train_df1['hour_of_day'] = train_df1.index.hour
    #creating days of the month
    train_df1['day_of_month'] = train_df1.index.day
    #creating days of the week
    train_df1['day_of_week'] = train_df1.index.dayofweek
    train_df1['day_of_the_year'] = train_df1.index.dayofyear
    train_df1['month'] = train_df1.index.month
    train_df1['year'] = train_df1.index.year
    train_df1 = train_df1
    return train_df1
   

def data_wangring(train_df1):
    """ """
    #droping the registered and casual since they are represented by count to avoid feature leakage
    train_df1.drop(['casual', 'registered'],axis=1, inplace=True)
    #atemp and temp has Multi-Collinearity, we drop it
    train_df1.drop('atemp', axis=1, inplace=True)

    train_df1.corr() # check for data correlation
    train_df1.info() #checking the table information
    train_df1.isnull().sum() #check the null values
    train_df1.duplicated() #check for duplicates
    train_df1.groupby('weather')[['count']].count()#Checking the impact of weather on demand
    return train_df1.groupby('weather')[['count']].count().plot.bar()


def train_model(X_train, y_train):
    """this function fits and train model using pipeline
    to apply preprocessing for model optimization
     returns fitted model """
    # pipeline to use standardscaler and KBinsdiscre
    my_pipe = Pipeline([('my_scaler', StandardScaler()),
                        ('my_kbins', KBinsDiscretizer())])
    #featture engineering and transform using columtransformer
    ct = ColumnTransformer([('my_pipe', my_pipe,['temp', 'humidity', 'windspeed']),
                            ('my_onehotencoder', OneHotEncoder(handle_unknown='ignore'),
                              ['season', 'weather', 'hour_of_day', 'day_of_week',
                               'day_of_month', 'year', 'month', 'holiday', 
                               'day_of_the_year','workingday'])])
    
    #fit and transform X_train using columntransfor
    fit_trans_X_train = ct.fit_transform(X_train)
    trans_X_test = ct.transform(X_test) 
    # Modeling using Poisson Regressor
    poisson_model = Pipeline([('ct', ct), ('polynomial', PolynomialFeatures(degree=2, include_bias=False)), 
                              ('poissonRegressor', PoissonRegressor(alpha=0.05))])
    fitted_model = poisson_model.fit(X_train, y_train)
    return fitted_model


def make_prediction(X_test, fitted_model):
    """ the function takes a fitted poissonRegressor 
    and return predictions (hard and soft)
    input: fitted_model
    input: X_test
    output: hard and soft predictions"""
    X_test = X_test
    y_predictions = fitted_model.predict(X_test) # y_prediction
    return y_predictions

    


if __name__ == '__main__':
    
    df = pandas_csv_reader('./forcast_demand/train.csv')
    y = df['count']
        
    expandedcolumn =feature_expansion(df)
    X = expandedcolumn.drop('count', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=48)

    fitted_mode = train_model(X_train, y_train)
    
    y_pred = make_prediction(X_test, fitted_mode)

    model_score = f'Model_score: {round(fitted_mode.score(X_test, y_test),3 )}'
    RSMLE_score = f'RSMLE: {round(mean_squared_log_error(y_test, y_pred),3)}'
    
    print(X)
    print(y)
    print('model predictions is :', y_pred)
    print(model_score)
    print(RSMLE_score)
