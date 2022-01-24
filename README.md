# MASK RCNN Architecture (Costum):
The aim of this work is to use the mask RCNN neural network architecture for image segmentation. 

## Train model:  GPU is required for training
to use this model, you just have to open the Jupyter notebook file : 
[click here](https://github.com/Koussailakadi/Mask_RCNN/blob/main/Train_Demo_Mask_RCNN.ipynb) on google colab or jupyter notebook and load your data (all the dataset in one  zip file, and the annotations in Json, or coco Json format)

## Inference model: 
to use this model, you just have to open the Jupyter notebook file : 
[click here](https://github.com/Koussailakadi/Mask_RCNN/blob/main/Inference_Demo_Mask_RCNN.ipynb) on google colab or jupyter notebook and load your data (all the dataset in one  zip file, and the annotations in Json, or coco Json format)

## Dataset: 
the split of the dataset is done like the following: 
train : 80% of the dataset 
validation : 10% of the dataset
test: 10% of the dataset 

dataset example: 
![alt text](https://github.com/Koussailakadi/Mask_RCNN/blob/main/img/train_images.png)

## Demo:
original image:
![alt text](https://github.com/Koussailakadi/Mask_RCNN/blob/main/img/original_image.png)

ground truth: 
![alt text](https://github.com/Koussailakadi/Mask_RCNN/blob/main/img/ground_truth_image.png)
 
the prediction:
![alt text](https://github.com/Koussailakadi/Mask_RCNN/blob/main/img/predicted_image.png)



to get more information about the mask RCNN architecture, please go to the refrence repository on github linked in the refrences bellow. 
# référence:
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}