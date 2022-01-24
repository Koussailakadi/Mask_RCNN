# -*- coding: utf-8 -*-

"""
Author: koussaila KADI 
Email: koussaila.kadi@outlook.fr
"""



import os
import sys
sys.path.append("./")
print(os.getcwd())
print(os.listdir(os.getcwd()))
from mrcnn import *
from mrcnn.m_rcnn import *



#---------------------------------------------------
# dataset:
#---------------------------------------------------

# load dataset: 
images_path = "with_cracks.zip"
annotations_path = "annotations_coco.json"
extract_images(images_path, "./dataset")

# split dataset: 
"""
train: 80%
val: 10%
test: 10%
"""
dataset_train = load_image_dataset( annotations_path, "dataset", "train")
dataset_val = load_image_dataset( annotations_path, "dataset", "val")
dataset_test = load_image_dataset(annotations_path, "dataset", "test")
class_number = dataset_train.count_classes()
print('All images :',len(dataset_train.image_ids)+len(dataset_val.image_ids)+len(dataset_test.image_ids))
print('Train:',len(dataset_train.image_ids))
print('Validation:',len(dataset_val.image_ids))
print('test: ',len(dataset_test.image_ids))
print("Classes: {}".format(class_number))

# show dataset samples:
display_image_samples(dataset_train)

#---------------------------------------------------
# ConstumConfiguration:
#---------------------------------------------------

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 2  # Background + Car and truck

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 20

    # Images size:  
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 384

    #mini mask:
    USE_MINI_MASK = False

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # learning rate
    LEARNING_RATE = 0.001

    
# Load Configuration
config = CustomConfig()
config.display()


#---------------------------------------------------
# Inference and load model weights:
#---------------------------------------------------

#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Get path to saved weightsµ
model_path = "./logs/mask_rcnn_object_0018.h5"

# Load trained weights
model.load_weights(model_path, by_name=True)


#---------------------------------------------------
# Prediction:
#---------------------------------------------------


image_id = random.choice(dataset_test.image_ids)
#image_id=22
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset_test, config,
                            image_id)

print('original image')
plt.imshow(original_image)
plt.show()


# original masks:
print("Ground Truth")
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_val.class_names, figsize=(8, 8),show_bbox=False)


# Model result
print("Prediction")
results = model.detect([original_image], verbose=0)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_test.class_names, r['scores'], ax=get_ax(), show_bbox=False)
