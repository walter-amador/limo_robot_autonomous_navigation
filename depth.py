import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
model_type = "DPT_LeViT_224"  # MiDaS v3.1 - LeViT-224 (fastest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Select device: CUDA > MPS (Mac) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

midas.to(device)
midas.eval()

print(f"Using device: {device}")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
elif model_type == "DPT_LeViT_224":
    transform = midas_transforms.levit_transform
else:
    transform = midas_transforms.small_transform


def init_camera(source=0):  # Changed default to 0
    """Initialize video capture."""
    cap = cv2.VideoCapture(source)
    return cap


def main():
    cap = init_camera(2)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        # Resize frame for display and processing consistency
        frame = cv2.resize(frame, (640, 480))

        # Transform input for model
        input_batch = transform(frame).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Normalize the output to 0-255 for display
        output_norm = cv2.normalize(
            output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply a colormap for better visualization (optional, but looks nice)
        output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

        # Show original and depth map
        cv2.imshow("Original", frame)
        cv2.imshow("Depth Map", output_color)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
