from flask import Flask , render_template ,request ,redirect ,url_for
import pandas as pd
import numpy as np



app = Flask(__name__,template_folder='templete')

@app.route('/', methods=["POST","GET"])
def Home():
    return render_template('index.html')

@app.route('/search',methods=["POST","GET"])
def Search():
    # return render_template('search.html')
    
    return render_template('search.html')



@app.route('/K-Nearest-Neighbor',methods=['POST','GET'])
def iterasi():
    from sklearn.model_selection import train_test_split #split dataset into train and test data
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    from sklearn.neighbors import KNeighborsClassifier# Create KNN classifier
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

        df = pd.read_csv('heart_cleveland_upload.csv')
        tess = df.head
        X = df.drop(columns=["condition"])
        y = df["condition"].values
        percent_amount_of_test_data = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = percent_amount_of_test_data, random_state=1, stratify=y)
        amount_of_neighbor = 3
        knn = KNeighborsClassifier(n_neighbors = amount_of_neighbor)

        knn.fit(X_train,y_train)
        knn.score(X_test, y_test)
        hasil = knn.predict([[umur,kelamin,nyeri,tekanan,kadar,gula,elektro,detak,fisik,st,segmen,kapal,cacar,1]])
        return render_template('hasil.html',hasil=hasil , nama=nama)



       
        
    else :
        return render_template('knn.html')



@app.route('/code/Diskritisasi')
def code():
    return render_template('code.html')

@app.route('/Diskritisasi')
def implement():
    return render_template('dsk.html')

@app.route('/akurasi')
def akurasi():
    return render_template('akurasi.html')

if __name__ == "__main__":
    app.run(debug=True,port=5000)