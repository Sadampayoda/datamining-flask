from flask import Flask , render_template ,request ,redirect ,url_for

app = Flask(__name__,template_folder='templete')

@app.route('/', methods=["POST","GET"])
def Home():
    return render_template('index.html')

@app.route('/search',methods=["POST","GET"])
def Search():
    # return render_template('search.html')
    if request.method == "POST":
        search = request.form['cari'];
        return render_template('search.html',data=search)
    else:
        return redirect(url_for('Home'))


@app.route('/iterasi/K-Nearest-Neighbor')
def iterasi():
    return render_template('iterasi.html')

@app.route('/code/Diskritisasi')
def code():
    return render_template('code.html')

@app.route('/impl/K-Nearest-Neighbor')
def implement():
    return render_template('implement.html')

@app.route('/akurasi')
def akurasi():
    return render_template('akurasi.html')

if __name__ == "__main__":
    app.run(debug=True,port=5000)