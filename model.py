from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

x, y=load_iris(return_X_y=True)
model=LogisticRegression()
model.fit(x,y)
joblib.dump(model, "model.pkl")

print("Model is trained and saved as model.pkl")