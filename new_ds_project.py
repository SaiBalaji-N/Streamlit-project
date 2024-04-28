import pandas as pd
import streamlit as st
st.markdown("# Importing Dataset")
st.markdown("### please select cleaned dataset with last column as target")
data=st.file_uploader("upload the csv file",type=["csv","xlsx"])
if data is not None:
    df=pd.read_csv(data)
    st.table(df.head())
else:
    st.write("No file uploaded.")
if data is not None:
  st.markdown("select it is a classification or regression problem")
  type=st.checkbox("Classification")
  type1=st.checkbox("Regression")
  print(type)
  if type:
        st.write("we are dealing with Classification problem")
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
  if type1:
        st.write("we are dealing with Regression problems")
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
if data is not None:
  if type or type1:
    inputs=df.iloc[:,:-1]
    outputs=df.iloc[:,-1]
    st.markdown("### inputs to our model")
    st.table(inputs.head())
    st.markdown("### outputs to our model")
    st.table(outputs.head())
    size=st.slider("select train and test split percentage",10,100,step=5)/100
    x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=1-size)
    if type:
      selected=st.selectbox("select one model",["Logistic Regression","Support Vector Machine Classifier",
                                          "Decision Tree Classifier","KNN Classifier"])
    if type1:
      selected=st.selectbox("select one model",["Linear Regression","Support Vector Machine Regressor",
                                          "Decision Tree Regressor","KNN Regressor"])
    if type or type1:
      if selected=="Logistic Regression":
        model=LogisticRegression()
      elif selected=="Support Vector Machine Classifier":
        model=SVC()
      elif selected == "Decision Tree Classifier":
        model=DecisionTreeClassifier()
      elif selected == "KNN Classifier":
        neighbors=st.number_input("enter no.of neighbors",10,100,step=5)
        model=KNeighborsClassifier(n_neighbors=neighbors)
      elif selected == "Linear Regression":
        model=LinearRegression()
      elif selected=="Support Vector Machine Regressor":
        model=SVR()
      elif selected == "Decision Tree Regressor":
        model=DecisionTreeRegressor()
      elif selected=="KNN Regressor":
        neighbors=st.number_input("enter no.of neighbors",10,100,step=5)
        model=KNeighborsRegressor(n_neighbors=neighbors)
      else:
          st.write("select a model")
if data is not None:
  if type or type1:
    model.fit(x_train,y_train)
    predictions=model.predict(x_test)
    accuracy=model.score(x_test,y_test)
    st.success(f"the accuracy of our model is {accuracy}")
    st.write("Model actual and predictions are :")
    final={
      "Actual":y_test,
      "Predicted":predictions
    }
    st.table(final)