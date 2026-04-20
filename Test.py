from ultralytics import YOLO
import cv2
import time
import os

def test_video_all_objects():
    VIDEO_PATH = r"C:\Users\abdol\Downloads\projector_aircon\telegram_video(1).mp4"
    STANDARD_MODEL = "yolov8n.pt"
    CUSTOM_MODEL_PATH = r"C:\Users\abdol\dataset\projector_aircon\runs\train\projector_aircon_v1\weights\best.pt"
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file not found: {VIDEO_PATH}")
        return
    
    if not os.path.exists(CUSTOM_MODEL_PATH):
        print(f"Custom model not found: {CUSTOM_MODEL_PATH}")
        return
    
    print(f"Testing: {VIDEO_PATH}")
    
    standard_model = YOLO(STANDARD_MODEL)
    custom_model = YOLO(CUSTOM_MODEL_PATH)
    
    TARGET_CLASSES = {
        63: 'laptop',
        62: 'tv',
        66: 'keyboard',
        56: 'chair',
    }
    
    print(f"Custom model classes: {custom_model.names}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video file")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, Total frames: {total_frames}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"test_output_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_objects = 0
    objects_by_class = {
        'laptop': 0,
        'tv': 0,
        'keyboard': 0,
        'chair': 0,
        'projector': 0,
        'air_conditioner': 0
    }
    
    colors = {
        'laptop': (0, 255, 0),
        'tv': (0, 165, 255),
        'keyboard': (0, 255, 255),
        'chair': (180, 105, 255),
        'projector': (255, 0, 0),
        'air_conditioner': (0, 0, 255)
    }
    
    start_time = time.time()
    print("\nStarting detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - FPS: {fps_current:.1f}")
            
            frame_detections = []
            
            # Detect standard objects
            results = standard_model(frame, conf=0.25, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        
                        if class_id in TARGET_CLASSES:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_name = TARGET_CLASSES[class_id]
                            
                            if confidence >= 0.25:
                                total_objects += 1
                                objects_by_class[class_name] += 1
                                frame_detections.append((x1, y1, x2, y2, confidence, class_name))
            
            # Detect custom objects
            custom_results = custom_model(frame, conf=0.1, verbose=False)
            
            for result in custom_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = custom_model.names[class_id]
                        
                        if class_name == 'air_conditioner' and confidence >= 0.1:
                            total_objects += 1
                            objects_by_class['air_conditioner'] += 1
                            frame_detections.append((x1, y1, x2, y2, confidence, class_name))
                        elif class_name == 'projector' and confidence >= 0.5:
                            total_objects += 1
                            objects_by_class['projector'] += 1
                            frame_detections.append((x1, y1, x2, y2, confidence, class_name))
            
            # Draw detections
            for detection in frame_detections:
                x1, y1, x2, y2, confidence, class_name = detection
                
                if class_name == 'air conditioner':
                    class_name = 'air_conditioner'
                
                color = colors.get(class_name, (255, 255, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                cv2.rectangle(frame, 
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1),
                            color, -1)
                
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (255, 255, 255), 2)
            
            # Add statistics overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Objects: {len(frame_detections)}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Total: {total_objects}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, "Detections:", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            y_offset = 155
            for class_name in ['laptop', 'tv', 'keyboard', 'chair', 'projector', 'air_conditioner']:
                count = objects_by_class[class_name]
                if count > 0:
                    color = colors.get(class_name, (0, 255, 255))
                    cv2.putText(frame, f"{class_name}: {count}", (30, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset += 30
            
            out.write(frame)
            
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    
    elapsed_time = time.time() - start_time
    processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    cap.release()
    out.release()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"Frames processed: {frame_count}")
    print(f"Processing time: {elapsed_time:.1f}s")
    print(f"Average FPS: {processing_fps:.1f}")
    print(f"Total objects detected: {total_objects}")
    
    print("\nDetection summary:")
    print("-" * 40)
    for class_name, count in objects_by_class.items():
        if count > 0:
            print(f"{class_name:20} : {count:4d}")
    
    print(f"\nOutput saved: {output_path}")
    
    return {
        'total_frames': frame_count,
        'total_objects': total_objects,
        'objects_by_class': objects_by_class,
        'output_video': output_path
    }

if __name__ == "__main__":
    print("=" * 60)
    print("OBJECT DETECTION TEST")
    print("=" * 60)
    
    test_video_all_objects()