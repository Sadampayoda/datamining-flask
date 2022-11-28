from flask import Flask , render_template ,request ,redirect ,url_for,jsonify
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier# Create KNN classifier
from sklearn.model_selection import train_test_split #split dataset into train and test data
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier



app = Flask(__name__,template_folder='templete')

df = pd.read_csv('heart_cleveland_upload.csv')



@app.route('/', methods=["POST","GET"])
def Home():
    return render_template('index.html')

@app.route('/search',methods=["POST","GET"])
def Search():
    # return render_template('search.html')
    
    return render_template('search.html')



@app.route('/K-Nearest-Neighbor',methods=['POST','GET'])
def iterasi():
    
    
    if request.method == "POST":
        
        nama = request.form['nama']
        umur = float(request.form['umur'])
        kelamin = float(request.form['kelamin'])
        nyeri = float(request.form['nyeri-data'])
        tekanan = float(request.form['tekanan-darah'])
        kadar = float(request.form['kadar-kolestrol'])
        gula = float(request.form['gula-darah'])
        elektro = float(request.form['elektro'])
        detak = float(request.form['detak-jantung'])
        fisik = float(request.form['fisik-berat'])
        st = float(request.form["ST-Depression"])
        segmen = float(request.form['segmen']) 
        kapal = float(request.form['kapal'])
        cacar = float(request.form['cacar'])

        
        X = df.drop(columns=["condition"])
        y = df["condition"].values
        percent_amount_of_test_data = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = percent_amount_of_test_data, random_state=1, stratify=y)
        amount_of_neighbor = 3
        knn = KNeighborsClassifier(n_neighbors = amount_of_neighbor)

        knn.fit(X_train,y_train)
        knn.score(X_test, y_test)
        hasil = knn.predict([[5,umur,kelamin,nyeri,tekanan,kadar,gula,elektro,detak,fisik,st,segmen,kapal,cacar]])
        return render_template('hasil.html',hasil=hasil , nama=nama)



       
        
    else :
        return render_template('knn.html')


@app.route('/deskripsi')
def deskripsi():
    
    return render_template('deskripsi.html',df=df)

@app.route('/preprosesing')
def preprosesing():
    
    df_for_minmax_scaler=pd.DataFrame(df, columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','condition'])
    df_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)
    df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','condition'])
    data = df_hasil_minmax_scaler
    return render_template('preprosesing.html', df=data)



@app.route('/naive-bayes-Classifer',methods=['POST','GET'])
def code():
    if request.method == "POST":
        
        nama = request.form['nama']
        umur = float(request.form['umur'])
        kelamin = float(request.form['kelamin'])
        nyeri = float(request.form['nyeri-data'])
        tekanan = float(request.form['tekanan-darah'])
        kadar = float(request.form['kadar-kolestrol'])
        gula = float(request.form['gula-darah'])
        elektro = float(request.form['elektro'])
        detak = float(request.form['detak-jantung'])
        fisik = float(request.form['fisik-berat'])
        st = float(request.form["ST-Depression"])
        segmen = float(request.form['segmen']) 
        kapal = float(request.form['kapal'])
        cacar = float(request.form['cacar'])

        # separate target 
        #create a dataframe with all training data except the target column
        X = df.drop(columns=["condition"])#check that the target variable has been removed
        X.head()
        # values
        X=df.iloc[:,0:14].values

        # classes
        y=df.iloc[:,14].values

        clf = GaussianNB()
        clf.fit(X, y)
        clf_pf = GaussianNB()
        clf_pf.partial_fit(X, y, np.unique(y))
        tes = clf_pf.predict([[5,umur,kelamin,nyeri,tekanan,kadar,gula,elektro,detak,fisik,st,segmen,kapal,cacar]])


        return render_template('hasil.html',resultNb = tes , nama=nama)

    return render_template('naive-bayes-Classifer.html')

@app.route('/modeling')
def implement():
    #KNN

    X = df.drop(columns=["condition"])
    y = df["condition"].values
    percent_amount_of_test_data = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = percent_amount_of_test_data, random_state=1, stratify=y)
    amount_of_neighbor = 3
    knn = KNeighborsClassifier(n_neighbors = amount_of_neighbor)
    knn.fit(X_train,y_train)
    #check accuracy of our model on the test data
    HasilKnn = knn.score(X_test, y_test)

    #Naive bayer
    gaussian = GaussianNB()
    # 
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test) 
    HasilNaive = accuracy_score(y_test,Y_pred)

    #Random forest
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)

    Y_pred = gaussian.predict(X_test) 
    HasilRandom = accuracy_score(y_test,Y_pred)

    #HasilEnsamble
    layer_one_estimators = [
                        ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
                        ('knn_1', KNeighborsClassifier(n_neighbors=10))             
                       ]
    layer_two_estimators = [
                            ('dt_2', DecisionTreeClassifier()),
                            ('rf_2', RandomForestClassifier(n_estimators=50, random_state=42)),
                        ]
    layer_two = StackingClassifier(estimators=layer_two_estimators, final_estimator=GaussianNB())

    # Create Final model by 
    clfs = StackingClassifier(estimators=layer_one_estimators, final_estimator=layer_two)
    HasilEnsamble = clfs.fit(X_train, y_train).score(X_test, y_test)

    return render_template('modeling.html',df=df,HasilKnn=HasilKnn,HasilNaive=HasilNaive,HasilRandom=HasilRandom,HasilEnsamble=HasilEnsamble)


@app.route('/Decision-Tree')
def akurasi():
    return render_template('Decision-Tree.html')

if __name__ == "__main__":
    app.run(debug=True,port=5000)