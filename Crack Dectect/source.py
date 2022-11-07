#import library
import tensorflow as tf

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns

import plotly.express as px

from pathlib import Path  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#load data
possitive_dir = Path('E:/My file/File in here/archive/Positive')
negative_dir = Path('E:/My file/File in here/archive/Negative')

#Create DataFrames
def generate_df(image_dir, label):
	filepaths = pd.Series(list(image_dir.glob(r'*.jpg')),name='Filepath').astype(str)
	labels = pd.Series(label, name='Label', index=filepaths.index)

	df = pd.concat([filepaths, labels], axis=1)
	return df

possitive_df = generate_df(possitive_dir, label="POSITIVE")
negative_df = generate_df(negative_dir, label="NEGATIVE")

all_df = pd.concat([possitive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
print(all_df)

train_df, test_df = train_test_split(
	all_df.sample(6000, random_state=1),
	train_size=0.7,
	shuffle=True,
	random_state=1
)

print(train_df)
print(test_df)

#Loading image data

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=1./255,
	validation_split=0.2
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=1./255
)

train_data = train_gen.flow_from_dataframe(
	train_df,
	x_col='Filepath',
	y_col='Label',
	target_size=(120,120),
	color_mode='rgb',
	class_mode='binary',
	batch_size=32,
	shuffle=True,
	seed=42,
	subset='training'
)

val_data = train_gen.flow_from_dataframe(
	train_df,
	x_col='Filepath',
	y_col='Label',
	target_size=(120,120),
	color_mode='rgb',
	class_mode='binary',
	batch_size=32,
	shuffle=True,
	seed=42,
	subset='validation'
)

test_data = test_gen.flow_from_dataframe(
	test_df,
	x_col='Filepath',
	y_col='Label',
	target_size=(120,120),
	color_mode='rgb',
	class_mode='binary',
	batch_size=32,
	shuffle=False,
	seed=42
)

inputs = tf.keras.Input(shape=(120,120,3))
x=tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs)
x=tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
x=tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)

model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy']
)

print(model.summary())

history = model.fit(
	train_data,
	validation_data=val_data,
	epochs=100,
	callbacks=[
		tf.keras.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=3,
			restore_best_weights=True
		)
	]

)


fig = px.line(
	history.history,
	y=['loss', 'val_loss'],
	labels={'index':"Epoch",'value':"Loss"},
	title="Training and Validation Loss Over Time"
)


fig.show()


#Results

def evaluate_model(model, test_data):
	results = model.evaluate(test_data, verbose=0)
	loss = results[0]
	acc = results[1]

	print("Test Loss: {:.5f}%",format(loss))
	print("Test Accuracy: {:.2f}%",format(acc*100))

	y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(np.int))
	cm = confusion_matrix(test_data.labels, y_pred)
	clr = classification_report(test_data.labels, y_pred, target_names=["NEGATIVE","POSITIVE"])

	plt.figure(figsize=(6,6))
	sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
	plt.xticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE","POSITIVE"])
	plt.yticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE","POSITIVE"])
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.title("Confusion Matrix")
	plt.show()

	print("Classificationn Report:\n--------------", clr)

evaluate_model(model, test_data)






