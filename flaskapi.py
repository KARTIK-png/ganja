import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask
from flask_restful import Api, Resource

url = "https://drive.google.com/uc?id=15b1AvhMiAYhzvIuvzmuMD5Z9wPKih2nF"
# features = ['thumb', 'index finger', 'middle finger', 'ring finger', 'pinky finger', 'pitch', 'roll', 'class']
# dataset = pd.read_csv(url, names=features)
dataset = pd.read_csv(url)


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
sc = StandardScaler()
x = sc.fit_transform(x)
classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)
classifier.fit(x, y)

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self, thumb, index, middle, ring, pinky, pitch, roll):
        x_test = [[thumb, index, middle, ring, pinky, pitch, roll]]
        x_test = np.array(x_test)
        x_test = sc.transform(x_test)
        y_pred = classifier.predict(x_test)
        return jsonify({"class": y_pred[0]})


api.add_resource(HelloWorld, "/hello/<int:thumb>/<int:index>/<int:middle>/<int:ring>/<int:pinky>/<int:pitch>/<int:roll>")

if __name__ == "__main__":
    app.run(debug=False)
