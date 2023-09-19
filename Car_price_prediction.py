import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


#With the use of the library pandas we load the data from the csv file
car_data = pd.read_csv(r"C:\Users\milto\Desktop\Github\Tensorflow\Car_Price_prediction\car data.csv")

##With the use of the instruction shape we checking the numbers of rows and columns
##The reason fot that is because we need to know the number of inputs
#print(car_data.shape)

##We need to check how many missing values our csv file has
#print(car_data.isnull().sum())


###Print the first five data to see the structure that our dataset has
#print(car_data.head())

#We need our data to be in number form
#So, Fuel_Type and Seller_Type column must be change into 0 and 1

###Reconstructing the columns with 0 and 1
car_data.replace({'Fuel_Type':{'Petrol': 0, 'Diesel' : 1, 'CNG' : 2},\
                  'Seller_Type':{'Dealer': 0, 'Individual' : 1},\
                  'Transmission': {'Manual' : 0, 'Automatic': 1}}, inplace=True)

#print(car_data.head())

##We need remove car_name in an other project we can give values to every car and
##improve our prediction about each one of them.
input_x = car_data.drop(['Car_Name','Selling_Price'],axis=1)
output_y = car_data['Selling_Price'] #Output(Y) = Selling_Price

#We split the data into train and test
#The parametre test_size is specifies the number of datas that we gonna use for test
#Random_state = 2 is used to ensures that the dataset is split in the same way every time we run the code
x_train, x_test, y_train, y_test = train_test_split(input_x, output_y, test_size = 0.1, random_state=2)


####  1# LinearRegression()
##############################
LinRegModel = LinearRegression()
LinRegModel.fit(x_train,y_train)
#Prediction on Training data
training_data_pred = LinRegModel.predict(x_train)


#r2_score is a function from sklearn that computes the r-squared score
train_error_1 = metrics.r2_score(y_train, training_data_pred)
print("\nMethod #1: Linear Regression")
print("Train_data: Square_Error = " , train_error_1)

#The data tha we prepare for test
test_data_pred = LinRegModel.predict(x_test)
test_error_1 = metrics.r2_score(y_test, test_data_pred)
print("Test_data: Square_Error = ", test_error_1)

#Plot for the Linear Regression example
#plt.scatter(y_test, test_data_pred)
#plt.xlabel("Actual_Price")
#plt.ylabel("Predicted_Price")
#plt.title(" Actual Prices vs Predicted Prices")
#plt.show()


####  2# Lasso()
##############################
LassoModel = Lasso()
LassoModel.fit(x_train,y_train)

# prediction on Training data
Lasso_training_data_pred = LassoModel.predict(x_train)
train_error_2 = metrics.r2_score(y_train, Lasso_training_data_pred)
print("\nMethod #2: Lasso Regression")
print("Train_data: Square_Error = ", train_error_2)
# R squared Error
error_score = metrics.r2_score(y_train, Lasso_training_data_pred)

# prediction on Training data
Lasso_test_data_pred = LassoModel.predict(x_test)

# R squared Error
test_error_2 = metrics.r2_score(y_test, Lasso_test_data_pred)
print("Test_data: Square_Error = ", test_error_2)


#Plot for the Lasso Regression example 
#plt.scatter(Y_test, test_data_prediction)
#plt.xlabel("Actual Price")
#plt.ylabel("Predicted Price")
#plt.title(" Actual Prices vs Predicted Prices")
#plt.show()
