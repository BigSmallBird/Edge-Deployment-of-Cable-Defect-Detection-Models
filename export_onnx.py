from ultralytics import YOLO

# Load your YOLO model
model = YOLO("best.pt")

# Export to ONNX format first (this is cross-platform)
model.export(format="onnx", imgsz=640)

print("ONNX导出完成！")



