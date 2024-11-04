# ALNPR using Yolov5 model

## Description
This documentation covers the training process for YOLOv5 models for automatic number plate recognition, including setup, training, monitoring, and inference.

## Dependencies
- Python 3.x
- PyTorch
- YOLOv5 requirements


## Training Process

### 1. Setup Environment
1. Clone YOLOv5 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt comet_ml
   ```

2. Prepare your custom_data.yaml file:
   ```yaml
   path: ../dataset  # dataset root dir
   train: images/train  # train images
   val: images/val  # val images
   
   # Classes
   nc: 1  # number of classes
   names: ['number_plate']  # class names
   ```

### 2. Training
Start training with:
```bash
python train.py --img 640 --batch 16 --epochs 50 --data custom_data.yaml --weights yolov5s.pt --cache
```

Parameters explained:
- `--img 640`: Input image size
- `--batch 16`: Batch size
- `--epochs 50`: Number of training epochs
- `--data`: Path to data.yaml file
- `--weights`: Initial weights (yolov5s.pt for small model)
- `--cache`: Cache images for faster training

Training outputs:
- Results saved to `runs/train/exp*`
- Best weights saved as `best.pt`
- Last weights saved as `last.pt`


### 3. Validation
Validate your model:
```bash
python val.py --weights runs/train/exp/weights/best.pt --data custom_data.yaml --img 640
```

### 4. Inference
Run detection on new images:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images --img 640 --conf 0.25
```

## Training Results
Results are saved in `runs/train/exp*` including:
- Training curves
- Confusion matrix
- Precision-Recall curves
- Validation images with predictions
- Mosaic training images

## Model Export
Export your trained model:
```bash
python export.py --weights runs/train/exp/weights/best.pt --include torchscript onnx
```

## Result

Video 1:- 
![b087d9ff-d1b8-47ce-8e58-fcd0a820fc20 (1)](https://github.com/user-attachments/assets/af5c5b77-6341-4d29-aefe-a415ca3d3a29)

Image 1:-

![TEST](https://github.com/user-attachments/assets/f91f1e80-e4e8-425b-87c8-7aec561dba44)


## Dataset Link

Link :- [Dataset_Link](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection).

Note:-  1.Annotations(labels) in the dataset are in ".xml" format. 
        2.Using python script i have converted it into Yolo format and stored in ".txt" documents.
        3.Lastly, I have split them in 80:20 % ratio into train and val folder inside the train_data folder 
