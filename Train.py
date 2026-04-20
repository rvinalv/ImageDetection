from ultralytics import YOLO
import os
import torch
import time
import yaml
import glob

def train_custom_model():
    DATASET_ROOT = r"C:\Users\abdol\dataset\projector_aircon"
    
    TRAIN_IMAGES = r"C:\Users\abdol\dataset\projector_aircon\train\images"
    TRAIN_LABELS = r"C:\Users\abdol\dataset\projector_aircon\train\labels"
    VAL_IMAGES = r"C:\Users\abdol\dataset\projector_aircon\val\images"
    VAL_LABELS = r"C:\Users\abdol\dataset\projector_aircon\val\labels"
    DATA_YAML = r"C:\Users\abdol\dataset\projector_aircon\data.yaml"
    
    EPOCHS = 50
    BATCH_SIZE = 16
    IMG_SIZE = 640
    MODEL_NAME = "yolov8n.pt"
    
    print("Checking dataset structure...")
    
    paths_to_check = [
        (DATASET_ROOT, "Dataset root"),
        (TRAIN_IMAGES, "Train images"),
        (TRAIN_LABELS, "Train labels"),
        (VAL_IMAGES, "Validation images"),
        (VAL_LABELS, "Validation labels"),
        (DATA_YAML, "Data YAML file")
    ]
    
    all_paths_exist = True
    for path, description in paths_to_check:
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  ✓ {description}: {path} ({file_count} images)")
            else:
                print(f"  ✓ {description}: {path}")
        else:
            print(f"  ✗ {description} NOT FOUND: {path}")
            all_paths_exist = False
    
    if not all_paths_exist:
        print("\nError: Some dataset paths are missing!")
        return None
    
    print("\nChecking data.yaml file...")
    try:
        with open(DATA_YAML, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        print(f"  Classes: {data_config.get('names', 'Not found')}")
        print(f"  Number of classes: {data_config.get('nc', 'Not found')}")
        
    except Exception as e:
        print(f"  Error reading data.yaml: {e}")
        yaml_content = f"""path: {DATASET_ROOT}
train: train/images
val: val/images

nc: 2
names: ['projector', 'air conditioner']
"""
        with open(DATA_YAML, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"  Created data.yaml at: {DATA_YAML}")
    
    print("\nChecking hardware...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
        device = 'cuda'
    else:
        print("  No GPU found, using CPU")
        device = 'cpu'
    
    print("\nLoading model...")
    try:
        model = YOLO(MODEL_NAME)
        print(f"  Model loaded: {MODEL_NAME}")
    except Exception as e:
        print(f"  Error loading model: {e}")
        try:
            model = YOLO('yolov8n.yaml')
            print("  Created YOLOv8n model from scratch")
        except Exception as e2:
            print(f"  Error creating model: {e2}")
            return None
    
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    config_summary = f"""
    Dataset: {DATASET_ROOT}
    Epochs: {EPOCHS}
    Batch Size: {BATCH_SIZE}
    Image Size: {IMG_SIZE}
    Device: {device}
    Model: {MODEL_NAME}
    """
    print(config_summary)
    
    print("\nStarting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            patience=15,
            save=True,
            save_period=10,
            resume=False,
            amp=True,
            
            val=True,
            plots=True,
            cos_lr=True,
            
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            
            workers=4 if device == 'cuda' else 2,
            verbose=True,
            seed=42,
            deterministic=True,
            
            name='projector_aircon_v1',
            project='runs/train',
            exist_ok=True,
            
            cache=False,
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.1f} minutes!")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\nEvaluating trained model...")
    
    try:
        best_model_path = os.path.join('runs', 'train', 'projector_aircon_v1', 'weights', 'best.pt')
        
        if os.path.exists(best_model_path):
            trained_model = YOLO(best_model_path)
            print(f"  Loaded best model: {best_model_path}")
            
            metrics = trained_model.val(
                data=DATA_YAML,
                batch=BATCH_SIZE,
                imgsz=IMG_SIZE,
                conf=0.25,
                iou=0.45,
                save_json=True,
                save_hybrid=True,
                plots=True
            )
            
            print("\n" + "=" * 40)
            print("EVALUATION RESULTS")
            print("=" * 40)
            
            if hasattr(metrics, 'box'):
                print(f"  mAP50: {metrics.box.map50:.4f}")
                print(f"  mAP50-95: {metrics.box.map:.4f}")
                print(f"  Precision: {metrics.box.mp:.4f}")
                print(f"  Recall: {metrics.box.mr:.4f}")
            
        else:
            print(f"  Best model not found at: {best_model_path}")
            
    except Exception as e:
        print(f"  Evaluation error: {e}")
    
    print("\nTesting inference...")
    
    try:
        sample_images = []
        for root, dirs, files in os.walk(TRAIN_IMAGES):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    sample_images.append(os.path.join(root, file))
                    if len(sample_images) >= 3:
                        break
            if sample_images:
                break
        
        if sample_images:
            print(f"  Found {len(sample_images)} sample images for testing")
            
            if 'trained_model' not in locals():
                best_model_path = os.path.join('runs', 'train', 'projector_aircon_v1', 'weights', 'best.pt')
                trained_model = YOLO(best_model_path)
            
            sample_image = sample_images[0]
            print(f"  Testing on: {os.path.basename(sample_image)}")
            
            results = trained_model(sample_image, conf=0.25, save=True, save_txt=True)
            
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    print(f"  Detected {len(r.boxes)} objects:")
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = trained_model.names[cls_id]
                        print(f"    - {cls_name}: {conf:.2f}")
            
            result_dir = os.path.join('runs', 'detect', 'predict')
            if os.path.exists(result_dir):
                print(f"  Results saved in: {result_dir}")
                
    except Exception as e:
        print(f"  Inference test error: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    model_info = f"""
    Total Epochs: {EPOCHS}
    Training Time: {training_time/60:.1f} minutes
    Device Used: {device}
    
    Best Model: runs/train/projector_aircon_v1/weights/best.pt
    Last Model: runs/train/projector_aircon_v1/weights/last.pt
    Logs: runs/train/projector_aircon_v1/
    """
    
    print(model_info)
    
    try:
        report_path = os.path.join('runs', 'train', 'projector_aircon_v1', 'training_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLOv11 Custom Training Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Training completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epochs: {EPOCHS}\n")
            f.write(f"Training time: {training_time/60:.1f} minutes\n")
            f.write(f"Dataset: {DATASET_ROOT}\n")
            f.write(f"Device: {device}\n")
        
        print(f"\nTraining report saved: {report_path}")
        
    except Exception as e:
        print(f"Could not save report: {e}")
    
    print("\n" + "=" * 60)
    print("Training process completed!")
    print("=" * 60)
    
    return model

def verify_dataset():
    DATASET_ROOT = r"C:\Users\abdol\dataset\projector_aircon"
    
    try:
        yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        classes = data_config.get('names', [])
        print(f"Classes in dataset: {classes}")
        
        train_images = glob.glob(os.path.join(DATASET_ROOT, 'train', 'images', '*.jpg')) + \
                      glob.glob(os.path.join(DATASET_ROOT, 'train', 'images', '*.png'))
        print(f"Train images: {len(train_images)}")
        
        train_labels = glob.glob(os.path.join(DATASET_ROOT, 'train', 'labels', '*.txt'))
        print(f"Train labels: {len(train_labels)}")
        
        val_images = glob.glob(os.path.join(DATASET_ROOT, 'val', 'images', '*.jpg')) + \
                     glob.glob(os.path.join(DATASET_ROOT, 'val', 'images', '*.png'))
        print(f"Validation images: {len(val_images)}")
        
        val_labels = glob.glob(os.path.join(DATASET_ROOT, 'val', 'labels', '*.txt'))
        print(f"Validation labels: {len(val_labels)}")
        
        total_images = len(train_images) + len(val_images)
        if total_images > 0:
            train_ratio = len(train_images) / total_images
            print(f"Train/Val split: {train_ratio:.1%}/{1-train_ratio:.1%}")
        
        if train_labels:
            sample_label = train_labels[0]
            with open(sample_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    print(f"Sample label format: {lines[0].strip()}")
        
        print("Dataset verification completed")
        
    except Exception as e:
        print(f"Dataset verification error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv11 CUSTOM MODEL TRAINER")
    print("Epochs: 50")
    print("=" * 60)
    
    import sys
    
    try:
        from ultralytics import YOLO
        print("ultralytics installed")
    except ImportError:
        print("ultralytics not installed!")
        print("Install with: pip install ultralytics")
        sys.exit(1)
    
    try:
        import torch
        print(f"torch {torch.__version__} installed")
    except ImportError:
        print("torch not installed!")
        print("Install with: pip install torch torchvision")
        sys.exit(1)
    
    response = input("\nVerify dataset before training? (y/n): ").lower()
    if response == 'y':
        verify_dataset()
    
    response = input("\nStart training with 50 epochs? (y/n): ").lower()
    if response != 'y':
        print("\nTraining cancelled.")
        sys.exit(0)
    
    model = train_custom_model()
    
    if model:
        print("\n" + "=" * 60)
        print("TRAINING SUCCESSFULLY COMPLETED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)