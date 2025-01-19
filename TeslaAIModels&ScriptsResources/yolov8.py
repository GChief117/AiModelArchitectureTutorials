# Created by Gunnar Beck Nelson
# 1/10/2024
#
import torch
import torch.nn as nn

# Description of the Model
"""
This implementation represents a simplified version of the YOLO (You Only Look Once) object detection model, with approximately 75 million parameters. 
The architecture consists of two main components:
1. Feature Extraction Layers: Convolutional layers extract hierarchical spatial features from input images. These layers identify edges, textures, shapes, and higher-level features necessary for detecting objects in the image.
   - Number of Layers in Feature Extraction: 10 layers (5 convolutional layers and 5 max-pooling layers).
2. Bounding Box Prediction: Fully connected layers use the extracted features to predict bounding box coordinates, class probabilities, and confidence scores for each grid cell.
   - Number of Layers in Bounding Box Prediction: 2 fully connected layers.

Neural Network Design Considerations:
- Number of Layers: This model has a total of 12 layers (10 convolutional/pooling layers and 2 fully connected layers).
  - Deep Networks: More layers allow the model to learn complex, abstract patterns but may require more data and computational resources.
  - Shallow Networks: Fewer layers are computationally efficient but may struggle with complex tasks.
- Benefits of Deep Networks: 
  - Better feature representation.
  - Improved accuracy for complex tasks like object detection.
- Drawbacks of Deep Networks:
  - Prone to overfitting if training data is insufficient.
  - Higher computational cost and memory usage.

Input:
- A batch of RGB images of size `(Batch_Size, 3, 224, 224)`.

Output:
- A tensor of size `(Batch_Size, 1470)`, representing a 7x7 grid with 30 values for each grid cell (20 classes, 4 bounding box coordinates, and 1 confidence score).

Activation Function:
- ReLU (Rectified Linear Unit) is used for non-linearity, as it is computationally efficient and avoids vanishing gradient problems.

Optimizer:
- Adam optimizer is used for its adaptive learning rate and computational efficiency.
"""

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        
        # Feature Extraction Layers
        """
        The feature extraction component consists of convolutional layers to learn hierarchical features, 
        followed by max-pooling layers to reduce spatial dimensions and retain important information.
        - Total Convolutional Layers: 5
        - Total Pooling Layers: 5
        
         1) Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
           - Weights: (3×3×3) × 64 = 1,728
           - Biases: 64
           - Total: 1,728 + 64 = 1,792

        2) Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
           - Weights: (64×3×3) × 128 = 73,728
           - Biases: 128
           - Total: 73,728 + 128 = 73,856

        3) Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
           - Weights: (128×3×3) × 256 = 294,912
           - Biases: 256
           - Total: 294,912 + 256 = 295,168

        4) Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
           - Weights: (256×3×3) × 512 = 2,359,296
           - Biases: 512
           - Total: 2,359,296 + 512 = 2,359,808

        5) Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
           - Weights: (512×3×3) × 1024 = 4,718,592
           - Biases: 1024
           - Total: 4,718,592 + 1024 = 4,719,616

        6) Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
           - Weights: (1024×3×3) × 1024 = 9,437,184
           - Biases: 1024
           - Total: 9,437,184 + 1024 = 9,438,208
        
        """
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Layer 1: Input: 3x224x224 -> Output: 64x224x224
            nn.ReLU(),  # Non-linearity
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Layer 2: Output: 128x224x224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 3: Downsample: 128x224x224 -> 128x112x112
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Layer 4: Output: 256x112x112
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Layer 5: Output: 512x112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 6: Downsample: 512x112x112 -> 512x56x56
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # Layer 7: Output: 1024x56x56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 8: Downsample: 1024x56x56 -> 1024x28x28
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),  # Layer 9: Output: 1024x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Layer 10: Downsample: 1024x28x28 -> 1024x14x14
        )
        
        
        # ------------------------------------------------------------------
        # Fully Connected Layers
        # ------------------------------------------------------------------
        """
        After the final MaxPool, the feature map is (1024 x 14 x 14) per sample.
        We flatten to (1024 * 14 * 14) = 200,704 features.

        1) Linear(200,704 -> 4,096)
           - Weights: 200,704 × 4,096
           - Biases: 4,096

        2) Linear(4,096 -> 7×7×30 = 1470)
           - Weights: 4,096 × 1,470
           - Biases: 1,470
        """
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 14 * 14, 4096),
            nn.ReLU(),
            nn.Linear(4096, 7 * 7 * 30)
        )

    def forward(self, x):
        """
        Forward Pass:
        1. Feature extraction via convolution + pooling.
        2. Flatten and pass through fully connected layers.
        3. Output shape: (Batch_Size, 1470)
        """
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------
# MODEL CREATION & PARAMETER COUNT
# ------------------------------------------------------------------
yolo_model = YOLO()
total_params = sum(p.numel() for p in yolo_model.parameters())
print(f"YOLO Total Parameters: {total_params:,}")  # ~75 million

# ------------------------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------------------------
"""
- Batch Size: 32
- Activation Function: ReLU
- Optimizer: Adam
"""
optimizer = torch.optim.Adam(yolo_model.parameters(), lr=1e-3)
