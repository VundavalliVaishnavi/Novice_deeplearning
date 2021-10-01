from flask import *
import pandas as pd
import os
from flask_cors import CORS
from flask_jsonpify import jsonpify
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app=Flask("__name__")
CORS(app)

@app.route('/upload', methods = ['POST'])  
def upload():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        # print(request.get_data())

        # print("fine")
        try:
            df=pd.read_csv(f.filename)
        except:
            try:
                df=pd.read_excel(f.filename)
            except:
                return jsonify("Please upload a valid file. i.e the file should be CSV or Xlsx")
        print(df)
        return jsonify("File uploaded sucessfully")


@app.route('/<filename>/head')
def head(filename):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)

    print("---------------------------------")
  
    head = df.head().values.tolist()
    print(head)
    JSONP_data = jsonpify(head)
    return JSONP_data

@app.route('/<filename>/describe')
def desc(filename):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)
    print("---------------------------------")
    desc = df.describe().values.tolist()
    print(desc)
    JSONP_data = jsonpify(desc)
    return JSONP_data

@app.route('/<filename>/plot/<x>/<y>')
def plotgraph (filename,x,y):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)
    
    X= df[int(x)].values.tolist()   
    Y= df[int(y)].values.tolist() 
    plt.scatter(X,Y)
    plt.title("distribution")
    plt.xlabel(x)
    plt.ylabel(y)
    print("---------------------------------")
    # print(type(image))
    # return render_template('untitled1.html', name = plt.show())
    plt.savefig("plotimage.png")
    return jsonify("okay")

@app.route('/<filename>/shape')
def shape(filename):
    try:
        df=pd.read_csv(filename, header=None)
    except:
        df=pd.read_excel(filename, header=None)

    print("---------------------------------")

    x,y= df.shape
    print(x,y)
    dictin={"rows":x,"columns":y}
    return jsonify(dictin)


@app.route('/<filename>/<predfile>/linearregnovice')
def linearregnovice(filename,predfile):
    try:
        data=pd.read_csv(filename, header=None)
    except:
        data=pd.read_excel(filename, header=None)
    
    print("---------------------------------")

    try:
        pred=pd.read_csv(predfile, header=None)
    except:
        pred=pd.read_excel(predfile, header=None)
        
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = np.array(data.iloc[:,:-1].values)  
    y = np.array(data.iloc[:,-1].values)
    # X=data.iloc[:,:-1].values # print(len(X)) y=df.iloc[:,-1].values
    
    
    dictin={}
    # Splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
    print("*************************")

    regr = LinearRegression()

    regr.fit(X_train, y_train)
    
    # X_pred=np.array([[8],[6],[5]])
    X_pred = np.array(pred.iloc[:,:].values) 
    # X_pred=np.array([[8,2],[6,5],[5,6]])

    y_pred = regr.predict(X_pred)
    # print(X_test,y_pred)
    print()
    print(regr.score(X_test, y_test))
    dictin["r2_value"]=regr.score(X_test, y_test)
    dictin["X_pred"]=X_pred.tolist()
    dictin["y_pred"]=y_pred.tolist()

    return jsonify(dictin)

@app.route('/<filename>/<predfile>/linearR2')
def linearR2(filename,predfile):
    try:
        data=pd.read_csv(filename, header=None)
    except:
        data=pd.read_excel(filename, header=None)
    
    print("---------------------------------")

    try:
        pred=pd.read_csv(predfile, header=None)
    except:
        pred=pd.read_excel(predfile, header=None)
        
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = np.array(data.iloc[:,:-1].values)  
    y = np.array(data.iloc[:,-1].values)
    # X=data.iloc[:,:-1].values # print(len(X)) y=df.iloc[:,-1].values
    
    
    dictin={}
    # Splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
    print("*************************")
    regressionObj = LinearRegression()
    regressionObj.fit(X_train, y_train)

    # validating the regression model with train and test sets
    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = regressionObj.predict(X_train)
    r_squared_score_train = r2_score(y_train, y_train_pred)
    print(" r squared score for train data set is ", r_squared_score_train)
    print()

    # validating the model on test set
    y_test_pred = regressionObj.predict(X_test)
    r_squared_score_test = r2_score(y_test, y_test_pred)
    print(" r squared score for test data set is ", r_squared_score_test)
    print()
    dictin["train"]=r_squared_score_train
    dictin["test"]=r_squared_score_test
    return jsonify(dictin)


@app.route('/<filename>/<predfile>/logisticnovice')
def logisticnovice(filename,predfile):
    try:
        data=pd.read_csv(filename, header=None)
    except:
        data=pd.read_excel(filename, header=None)
    
    print("---------------------------------")

    try:
        pred=pd.read_csv(predfile, header=None)
    except:
        pred=pd.read_excel(predfile, header=None)
        
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.array(data.iloc[:,:-1].values)  
    y = np.array(data.iloc[:,-1].values)
    # X=data.iloc[:,:-1].values # print(len(X)) y=df.iloc[:,-1].values
    
    dictin={}
    # Splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
    print("*************************")
    logisticObj = LogisticRegression(random_state=0)  # default l2 regularisation is applied
    logisticObj.fit(X_train, y_train.ravel())

    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = logisticObj.predict(X_train)
    f_score_train = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for logistic classification model on training data is ", f_score_train)
    print()

    # validating the model on test set
    y_test_pred = logisticObj.predict(X_test)
    f_score_test = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for logistic classification model on test data is ", f_score_test)
    print()
    dictin["train"]=f_score_train
    dictin["test"]=f_score_test
    return jsonify(dictin)

@app.route('/<filename>/<predfile>/SVMnovice')
def SVMnovice(filename,predfile):
    try:
        data=pd.read_csv(filename, header=None)
    except:
        data=pd.read_excel(filename, header=None)
    
    print("---------------------------------")

    try:
        pred=pd.read_csv(predfile, header=None)
    except:
        pred=pd.read_excel(predfile, header=None)
        
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SVC
    import numpy as np

    X = np.array(data.iloc[:,:-1].values)  
    y = np.array(data.iloc[:,-1].values)
    # X=data.iloc[:,:-1].values # print(len(X)) y=df.iloc[:,-1].values
    
    dictin={}
    # Splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
    print("*************************")
    # svcObject = SVC(C=0.1, kernel="linear")
    svcObject = SVC(kernel='rbf')
    svcObject.fit(X_train, y_train.ravel())

    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = svcObject.predict(X_train)
    fbeta_score_train=fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print(" F-Score (0-1) for the svm classification model on training data is : ",fbeta_score_train)
    print()

    # validating the model on test set
    y_test_pred = svcObject.predict(X_test)
    fbeta_score_test=fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print(" F-Score (0-1) for the svm classification model test data is : ",fbeta_score_test)
    print()

    dictin["train"]=fbeta_score_train
    dictin["test"]=fbeta_score_test
    return jsonify(dictin)

@app.route('/<filename>/<predfile>/deeplearning')
def deeplearning(filename,predfile):
    try:
        data=pd.read_csv(filename, header=None)
    except:
        data=pd.read_excel(filename, header=None)
    
    print("---------------------------------")

    try:
        pred=pd.read_csv(predfile, header=None)
    except:
        pred=pd.read_excel(predfile, header=None)
        
    
    from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.array(data.iloc[:,:-1].values)  
    y = np.array(data.iloc[:,-1].values)
    # X=data.iloc[:,:-1].values # print(len(X)) y=df.iloc[:,-1].values
    
    dictin={}
    # Splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
    print("*************************")
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    len_y = len(y_train)
    # y_train = y_train.values
    y_train = y_train.reshape(len_y, 1)

    len_test_y = len(y_test)
    # y_test = y_test.values
    y_test = y_test.reshape(len_test_y, 1)

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    def ann_mlp():
        features=data.shape[1]-1
        X = tf.placeholder(shape=[None, features], dtype=tf.float32)
        Y = tf.placeholder(tf.float32, [None, 1])

        # input
        W1 = tf.Variable(tf.random_normal([features,features*2], seed=0), name='weight1')
        b1 = tf.Variable(tf.random_normal([features*2], seed=0), name='bias1')
        layer1 = tf.nn.sigmoid(tf.matmul(X,W1) + b1)

        # hidden1
        W2 = tf.Variable(tf.random_normal([features*2,features*2], seed=0), name='weight2')
        b2 = tf.Variable(tf.random_normal([features*2], seed=0), name='bias2')
        layer2 = tf.nn.sigmoid(tf.matmul(layer1,W2) + b2)

        # hidden2
        result=features*2
        W3 = tf.Variable(tf.random_normal([features*2,result*2], seed=0), name='weight3')
        b3 = tf.Variable(tf.random_normal([result*2], seed=0), name='bias3')
        layer3 = tf.nn.sigmoid(tf.matmul(layer2,W3) + b3)

        # output
        W4 = tf.Variable(tf.random_normal([result*2,1], seed=0), name='weight4')
        b4 = tf.Variable(tf.random_normal([1], seed=0), name='bias4')
        logits = tf.matmul(layer3,W4) + b4
        hypothesis = tf.nn.sigmoid(logits)

        cost_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=Y)
        cost = tf.reduce_mean(cost_i)

        train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

        prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
        correct_prediction = tf.equal(prediction, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        # print("\n============Processing============")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(10001):
                sess.run(train, feed_dict={X: X_train, Y: y_train})
                if step % 1000 == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={X: X_train, Y: y_train})
                    print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

            train_acc = sess.run(accuracy, feed_dict={X: X_train, Y: y_train})
            test_acc,test_predict,test_correct = sess.run([accuracy,prediction,correct_prediction], feed_dict={X: X_test, Y: y_test})
            
            # print("\n============Results============")
            # print("test_predict values: ",test_predict)
            print("Model Prediction =", train_acc)
            print("Test Prediction =", test_acc)
            
            return train_acc,test_acc,test_predict
        
    ann_mlp_train_acc, ann_mlp_test_acc, test_predict = ann_mlp()
    res = fbeta_score(y_test, test_predict, average='binary', beta=0.5)
    # res_1=fbeta_score(y_test, test_predict, average='binary', beta=0.5)
    dictin["test"] = res
    return jsonify(dictin)

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host= "0.0.0.0", port=port)







