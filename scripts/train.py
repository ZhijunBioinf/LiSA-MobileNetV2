import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os


from model.lisa_mobilenetv2 import OptimizedMobileNetV2

# ========== ScopeLoss（optional） ==========
def ScopeLoss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.numpy()
        print(f"当前学习率: {lr}")


train_dir = "data/train"
val_dir = "data/val"

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)


model = OptimizedMobileNetV2(input_shape=(224, 224, 3), num_classes=10, alpha=1.0)
model.compile(optimizer='adam', loss=ScopeLoss, metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('model/best_model.h5', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True),
    TimeHistory(),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=1e-6),
    LearningRateLogger()
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=80,
    callbacks=callbacks
)
