from ultralytics import YOLO

# Train all 5 folds automatically
for i in range(5):
    print(f"\n========== Training Fold {i} ==========\n")

    model = YOLO("models/pothole.pt")  # your trained model

    model.train(
        data=f"data/kfold_{i}/dataset.yaml",
        epochs=5,
        imgsz=640,
        project="kfold_results",
        name=f"fold_{i}"
    )

print("\nAll folds trained successfully!")