import  tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split = 0.3)
train_set= train_datagen.flow_from_directory(
        "C:/Users/abhis/data/machine learning/rock_paper_scissors/image_dataset",
        target_size=(150,150),
        color_mode="grayscale",
        batch_size=32,
        class_mode='categorical',
        subset = 'training')
test_set = train_datagen.flow_from_directory(
    "C:/Users/abhis/data/machine learning/rock_paper_scissors/image_dataset",
    target_size = (150,150),
    color_mode="grayscale",
    batch_size=32,
    class_mode = 'categorical',
    subset = 'validation'
)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=(150,150,1)),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=4,activation="softmax")
])
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=["accuracy"])
Val_ACCURACY_THRESHOLD = 0.95
ACCURACY_THRESHOLD = 0.98
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > Val_ACCURACY_THRESHOLD):
            self.model.stop_training = True
        elif(logs.get('accuracy') > ACCURACY_THRESHOLD):
            self.model.stop_training = True
callbacks = myCallback()
model.fit(
    train_set,
    epochs = 20,
    validation_data = test_set,
    callbacks=[callbacks]
    )

model.save_weights('model.h5')

with open("model.json", "w") as json_file:
    json_file.write(model.to_json())