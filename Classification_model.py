import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle



with open('asl_dataset.pickle', 'rb') as x:
    X, y = pickle.load(x)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

valid_data = [sample for sample in X if len(sample) == 63]
valid_labels = [y[i] for i in range(len(X)) if len(X[i]) == 63]

X = np.array(valid_data, dtype=np.float32)
y = np.array(valid_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify = y)

model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

print(Counter(y))
print(y_pred)


f = open('model.p', 'wb')
pickle.dump((model), f)
f.close()