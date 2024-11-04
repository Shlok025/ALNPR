# YOLOv5 Training Pipeline

## Description
This documentation covers the training process for YOLOv5 models, including setup, training, monitoring, and inference.

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

### 3. Monitor Training
- Training metrics logged in real-time
- View results in Tensorboard:
  ```bash
  tensorboard --logdir runs/train
  ```
- Metrics tracked:
  - mAP (mean Average Precision)
  - Precision
  - Recall
  - Loss values (box, objectness, classification)

### 4. Validation
Validate your model:
```bash
python val.py --weights runs/train/exp/weights/best.pt --data custom_data.yaml --img 640
```

### 5. Inference
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

## Notes
- Training progress is automatically logged
- Best model weights are saved based on validation performance
- Use `--resume` flag to continue training from a checkpoint
- Adjust batch size based on available GPU memory
- Monitor GPU memory usage during training

## License
This project uses YOLOv5 which is licensed under the GPL-3.0 license.