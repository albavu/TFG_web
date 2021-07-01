from flask import Flask, render_template, jsonify, request
from flask.wrappers import Request
import joblib
from keras.models import load_model
from numpy.lib.npyio import save
from model import resultado
from model import leer
from model import deepL



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/machine')
def machine():
    return render_template('machine.html')

@app.route('/deep')
def deep():
    return render_template('deep.html')

@app.route('/modelo')
def modelo():
    return render_template('modelo.html')

def value_predict(to_predict_list):
    clf=load_model('modeloDL_OHE_final_dinuc.h5')
    prediccion=clf.predict(to_predict_list)
    return prediccion[0]

@app.route('/sub',methods=['POST'])
def submit():
    if request.method == "POST":
        to_predict_list= request.form.get("seq")
        to_predict_list = resultado(to_predict_list)

        try:
            result = value_predict(to_predict_list)
            if float(result[0]) >=0.5:
                prediction = 'No nucleosoma'
            elif float(result[0]) <=0.5:
                prediction = 'Nucleosoma'
        except ValueError:
            prediction = 'Error formato datos'

    return render_template('modelo.html', prediction=prediction)

@app.route('/subir',methods=['POST'])
def subir():
    file = None
    resultado = None
    if request.method == "POST":
        file = request.files['file']
        file.save('/Users/alba_vu/Desktop/python_web/src/'+file.filename)
        leer(file)

        resultado = deepL().to_dict(orient='records')
        print(resultado)

    return render_template('modelo.html', prediccion = resultado)


if __name__ == '__main__':
    app.run(debug=True)
