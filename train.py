from ultralytics import YOLO

# Step 1: Define the path to your data configuration file and model
data_path = 'data.yaml'  # Path to your data.yaml file
model_path = 'yolov8n.pt'  # YOLOv8 nano model (you can change this to yolov8s.pt, yolov8m.pt, etc.)

# Step 2: Create a YOLO model object
model = YOLO(model_path)

# Step 3: Train the model
model.train(
    data=data_path,  # Path to data configuration file
    epochs=100,  # Number of epochs to train for
    batch=16,  # Batch size for training
    imgsz=640,  # Image size (640x640 is the default for YOLOv8)
    name='snail-detection',  # Name of the training run
    save_period=1,  # Save checkpoint every epoch
    device='cpu'  # Use CPU since no CUDA device is available
)

# Step 4: Evaluate the model
metrics = model.val()

# Step 5: Export the trained model for inference (optional)
model.export(format='onnx')  # Export the trained model to ONNX format (you can also export to 'torchscript', 'coreml', etc.)

# Step 6: Print results
print(f"Training completed. Checkpoints are saved in the 'runs' directory.")
print(f"Validation metrics: {metrics}")
