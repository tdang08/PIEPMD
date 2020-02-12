# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:42:21 2020

@author: Based Gpd
"""

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator




# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = 2
model_mode = 'inference'


# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
#weights_path = 'D:\Desktop\Python\pierluigiferrari-keras-ssd-master\saved models\ssd300_pascal_07+12_epoch-11_loss-13.3436_val_loss-5.7738.h5'

#model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)





# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'D:\Desktop\Python\pierluigiferrari-keras-ssd-master\saved models\ssd300_pascal_07+12_epoch-11_loss-13.3436_val_loss-5.7738.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})





dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
Pascal_VOC_dataset_images_dir = "D:\Desktop\Python\Testing Data"
Pascal_VOC_dataset_annotations_dir = "D:\Desktop\Python\Data\Annotations"
Pascal_VOC_dataset_image_set_filename = "D:\Desktop\Python\Testing image set\Testing_image_set.txt"

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['crack', 'pothole']

dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)


# evaluator = Evaluator(model=model,
#                       n_classes=n_classes,
#                       data_generator=dataset,
#                       model_mode=model_mode)

# results = evaluator(img_height=img_height,
#                     img_width=img_width,
#                     batch_size=8,
#                     data_generator_mode='resize',
#                     round_confidences=False,
#                     matching_iou_threshold=0.5,
#                     border_pixels='include',
#                     sorting_algorithm='quicksort',
#                     average_precision_mode='sample',
#                     num_recall_points=11,
#                     ignore_neutral_boxes=True,
#                     return_precisions=True,
#                     return_recalls=False,
#                     return_average_precisions=True,
#                     verbose=True)

# mean_average_precision, average_precisions, precisions, recalls = results


# for i in range(1, len(average_precisions)):
#     print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
# print()
# print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))



# m = max((n_classes + 1) // 2, 2)
# n = 2

# fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
# for i in range(m):
#     for j in range(n):
#         if n*i+j+1 > n_classes: break
#         cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
#         cells[i, j].set_xlabel('recall', fontsize=14)
#         cells[i, j].set_ylabel('precision', fontsize=14)
#         cells[i, j].grid(True)
#         cells[i, j].set_xticks(np.linspace(0,1,11))
#         cells[i, j].set_yticks(np.linspace(0,1,11))
#         cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)





# 1: Set the generator for the predictions.

val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

predict_generator = val_dataset.generate(batch_size=1,
                                          shuffle=True,
                                          transformations=[convert_to_3_channels,
                                                          resize],
                                          label_encoder=None,
                                          returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                          keep_images_without_gt=False)


# 2: Generate samples.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))


# 3: Make predictions.

y_pred = model.predict(batch_images)


# 4: Decode the raw predictions in `y_pred`.

y_pred_decoded = decode_detections(y_pred,
                                    confidence_thresh=0.5,
                                    iou_threshold=0.4,
                                    top_k=200,
                                    normalize_coords=normalize_coords,
                                    img_height=img_height,
                                    img_width=img_width)


# 5: Convert the predictions for the original image.

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])




# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['crack', 'pothole']

plt.figure(figsize=(20,12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()

# for box in batch_original_labels[i]:
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})