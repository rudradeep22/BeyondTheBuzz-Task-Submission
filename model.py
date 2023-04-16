import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/train.csv')
print(df.isnull().any().sum())
X = df.drop(['VERDICT'], axis=1)
y = df.get(['VERDICT'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

predictions = model.predict(X_test)

# Convert predictions to binary data
binary_predictions = (predictions > 0.5).astype(int)
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(y_test, binary_predictions)
print('Accuracy:', accuracy.result().numpy())

X_submit = pd.read_csv('./data/test.csv').drop(['Id'], axis=1)
predictions = model.predict(X_submit)

# Convert predictions to binary labels
binary_predictions = (predictions > 0.5).astype(int)
c =0
for i in binary_predictions:
    if i == 0:
        c+=1
print('No. of zeros: ',c)
data = { 'Id' : np.arange(1,58922),
        'VERDICT':  np.ndarray.flatten(binary_predictions)}
ans = pd.DataFrame(data)
ans.to_csv('predictions.csv', index=False)