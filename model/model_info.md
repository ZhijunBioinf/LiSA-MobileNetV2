# LiSA-MobileNetV2 Model Information

## Model Overview

LiSA-MobileNetV2 is an improved lightweight model based on MobileNetV2, tailored for rice disease classification tasks. The improvements include

- Modified inverted residual block
- Swish activation function (replacing ReLU6)
- Integration of SE (Squeeze-and-Excitation) attention mechanism
- Custom loss function ScopeLoss

The model achieves state-of-the-art accuracy on a 10-class rice disease dataset while maintaining high inference speed and a compact size, making it suitable for mobile deployment.

---

##  Architecture Details

 Component                      Description                                         
-----------------------------------------------------------------------------------
 Base architecture             MobileNetV2                                         
 Activation function           Swish                                               
 Attention mechanism           Squeeze-and-Excitation (SE)                         
 Final output layer            Dense (10 classes with softmax)                     
 Model output format           TensorFlow Lite (`LiSA-MobileNetV2.tflite`)         

---

##  Training Configuration

 Parameter           Value                
-----------------------------------------
 Epochs             80                   
 Optimizer          Adam                 
 Initial LR         0.001                
 Batch size         (based on generator) 
 Loss function      ScopeLoss            
 Metrics            Accuracy             
 Validation split   via `val_generator`  

---

## 🔁 Learning Rate Scheduling & Callbacks

- `ReduceLROnPlateau`  
  Automatically reduces learning rate by a factor of 0.1 if `val_loss` does not improve for 5 consecutive epochs. Minimum learning rate `1e-6`.

- `EarlyStopping`  
  Stops training if no improvement after 10 consecutive epochs on validation loss, and restores the best model.

- `ModelCheckpoint`  
  Saves the best model (lowest validation loss) to `best_model.h5`.

- `TimeHistory`  
  Custom callback to record epoch duration.

- `LearningRateLogger`  
  Logs learning rate at the end of each epoch for monitoring.



##  File Description

- `LiSA-MobileNetV2.tflite` Final quantized model used in the Android application.
- `best_model.h5` Best model checkpoint during training (Keras format).
- `train.py` Contains training logic including all callbacks.
