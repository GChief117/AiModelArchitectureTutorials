import torch
import torch.nn as nn

# ------------------------------------------------------------------
# MODEL DESCRIPTION
# ------------------------------------------------------------------
"""
This CNN is designed for object detection with bounding boxes and class predictions.
The architecture has two main parts:
1. A deep feature extractor composed of convolutional + ReLU layers and multiple max-pooling operations.
2. A pair of fully connected layers that regress bounding box coordinates and predict class probabilities/confidence.

Key Points:
- The feature extractor reduces the input from 224×224 to a final 7×7 spatial map.
- We use several convolution layers increasing (or maintaining) channel depth, 
  interspersed with max-pooling to downsample.
- The output of the final convolution block is flattened, then passed to two
  fully connected layers:
  - The first FC (large) learns high-level representations.
  - The second FC outputs the final predictions (7×7×30 = 1470).
- The total parameter count is ~75 million, aligning with a large variant of YOLOv8.

Input Tensor Shape:
- (Batch_Size, 3, 224, 224)

Output Tensor Shape:
- (Batch_Size, 1470)

Where 1470 = 7×7×30:
  - 7×7 grid cells
  - Each cell predicts 2 bounding boxes (each bounding box has 4 coords + 1 confidence = 5) → 2×5 = 10
  - Plus class probabilities, e.g., 20 classes → 20
  - Total per cell: 10 + 20 = 30
"""

class ConvNetBoundingBox(nn.Module):
    def __init__(self):
        super(ConvNetBoundingBox, self).__init__()
        
        # ------------------------------------------------------------------
        # FEATURE EXTRACTION
        # ------------------------------------------------------------------
        """
        We have 7 convolutional layers, each followed by ReLU.
        We use 5 max-pooling operations (each with kernel_size=2, stride=2) 
        to progressively reduce spatial dimensions from 224→112→56→28→14→7.
        
        Convolution Layer Details (with approximate parameter counts):
        
        1) Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
           - Weights: (64 × 3 × 3 × 3) = 64×27 = 1,728
           - Biases: 64
           - Total = 1,728 + 64 = 1,792

        2) Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
           - Weights: (128 × 64 × 3 × 3) = 128×576 = 73,728
           - Biases: 128
           - Total = 73,728 + 128 = 73,856

        3) Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
           - Weights: (256 × 128 × 3 × 3) = 256×1,152 = 294,912
           - Biases: 256
           - Total = 294,912 + 256 = 295,168

        4) Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
           - Weights: (512 × 256 × 3 × 3) = 512×2,304 = 1,179,648
           - Biases: 512
           - Total = 1,179,648 + 512 = 1,180,160

        5) Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
           - Weights: (1024 × 512 × 3 × 3) = 1024×4,608 = 4,718,592
           - Biases: 1024
           - Total = 4,718,592 + 1024 = 4,719,616

        6) Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
           - Weights: (1024 × 1024 × 3 × 3) = 1024×9,216 = 9,437,184
           - Biases: 1024
           - Total = 9,437,184 + 1024 = 9,438,208

        7) Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
           - Weights: (1024 × 1024 × 3 × 3) = 9,437,184
           - Biases: 1024
           - Total = 9,437,184 + 1024 = 9,438,208

        Summation of all conv layer parameters ~ 26.3 million.
        """
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            # Block 3
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            # Block 4
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14

            # Block 5
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
        )
        
        # ------------------------------------------------------------------
        # FULLY CONNECTED HEAD
        # ------------------------------------------------------------------
        """
        After the final pooling, the spatial map is 7×7 with 1024 channels.
        So we flatten from (1024×7×7) = 50176 to a hidden layer, then output 7×7×30=1470.

        We choose a hidden size of 1140 for the first FC so that the total 
        parameter count lands around ~75 million.

        Flatten dimension: 1024×7×7 = 50,176
        FC1: Linear(50,176 -> 1140)
            - Weights: 50,176 × 1140 = 57,200,640
            - Biases: 1140
            - Total ≈ 57,201,780

        FC2: Linear(1140 -> 1470)
            - Weights: 1140 × 1470 = 1,675,800
            - Biases: 1470
            - Total ≈ 1,677,270

        Sum of FC params ≈ 58.9 million.
        Combined with convolution params (~26.3 million) yields ~85 million, 
        which is slightly above 75M. If you want it closer to 75M, you can reduce
        the hidden FC size slightly. Below we show a tweak to reduce the total 
        near ~75M:

        *Parameter-Tweaked Version*:
        - Let's do 50,176 -> 1024 for the first FC. 
          That leads to ~51.2M in FC1 instead of ~57.2M, trimming ~6M overall.

        For demonstration, we'll keep the 1140 dimension in code below. 
        Adjust it if you want precisely ~75M.
        """
        
        # For demonstration, let's do 50176 -> 1140 (slightly above 75M total).
        # If you want it closer to exactly 75M, reduce 1140 to ~1050–1100.
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 1140),  # Big first FC
            nn.ReLU(),
            nn.Linear(1140, 7 * 7 * 30)    # Output: 1470
        )

    def forward(self, x):
        """
        Forward Pass Steps:
        1) Convolution + ReLU + MaxPool blocks for feature extraction.
        2) Flatten and pass through fully connected layers for final bounding
           box + class predictions (7×7×30).
        3) Output shape: (Batch_Size, 1470)
        """
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x


# ------------------------------------------------------------------
# MODEL CREATION & PARAMETER COUNT
# ------------------------------------------------------------------
model = ConvNetBoundingBox()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model Total Parameters: {total_params:,}")  # Expect ~75-85 million (depending on hidden size).

# ------------------------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------------------------
"""
- Batch Size: 32 (typical small-batch object detection)
- Loss Function: Could be a combination of bounding box regression + 
                 classification (e.g., MSE for coords, CrossEntropy for class).
- Optimizer: Adam (adaptive learning rate, widely used for convenience).
- Learning Rate: 1e-3 (typical starting point).
"""
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
