from flask import Flask , render_template , request
import pickle

CV = pickle.load(open('Pickel_File/Count_vect.pkl','rb'))
model = pickle.load(open('Pickel_File/sentiment_analysis.pkl','rb'))


app = Flask(__name__)


@app.route("/")
@app.route("/home",methods = ['GET','POST'])
def home():

    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        text = request.form['ftext']
        data = [text]
        vect = CV.transform(data).toarray()
        my_pred = model.predict(vect)
        return render_template('home.html', prediction=my_pred)





if __name__ == '__main__':
    app.run(debug=True)
