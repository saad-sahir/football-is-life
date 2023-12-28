from ultralytics import YOLO
import cv2
import os

# Load the pre-trained model
model = YOLO('best.pt')  # Replace with your model path

# List of image file paths (will be video here and take each frame)
image_files = ['images/test.jpg', 'images/test2.jpg']  # Add your image paths here

# Folder to save processed images
save_folder = 'results'
os.makedirs(save_folder, exist_ok=True)

# Desired size for the resized image (width, height)
desired_size = (900, 600)  # You can adjust this as needed

# Process each image
for image_path in image_files:
    # Load the image
    image = cv2.imread(image_path)

    # Run inference on the image
    results = model(image)

    # Assuming results[0] contains the detections for the first (and only) image
    detections = results[0]

    # Use 'Boxes.data' for bounding boxes
    if hasattr(detections, 'boxes') and hasattr(detections.boxes, 'data'):
        for data in detections.boxes.data:
            # Extract bounding box coordinates
            x1, y1, x2, y2, conf, class_id = data
            label = detections.names[int(class_id)]

            # Draw rectangle on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Put label text above the rectangle
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resize the image
    resized_image = cv2.resize(image, desired_size)

    # Display the resized image
    cv2.imshow('YOLO Detection', resized_image)
    cv2.waitKey(0)

    # Save the original sized image (not resized)
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_folder, base_name)
    cv2.imwrite(save_path, image)

# Close OpenCV windows
cv2.destroyAllWindows()