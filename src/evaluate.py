import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
x_test,y_test = joblib.load('set.pkl')

prediction = model.predict(x_test)
acc = accuracy_score(prediction,y_test)

print('metrics',acc)

if acc >= 0.90:
    print("model was better at performing")
    exit(0)
else:
    print("model was worst condition")
    exit(1)