import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ---------  Load Image ---------
image = cv2.imread(r"C:\Users\91777\Downloads\honda_2.jpeg")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Store original shape
original_shape = image.shape

# -------------- Reshape Image ---------------
# Convert image into pixel dataset
pixels = image.reshape(-1, 3)

# ------------- Apply GMM ----------------
gmm = GaussianMixture(
    n_components=5,     # Number of clusters
    covariance_type='tied',
    random_state=42
)

gmm.fit(pixels)

# Predict cluster labels
labels = gmm.predict(pixels)

# ------------- Replace Pixels ----------------
# Use cluster centers
segmented_pixels = gmm.means_[labels]

# Convert back to image shape
segmented_image = segmented_pixels.reshape(original_shape)

# Convert datatype
segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

# -----------------------------
# STEP 5: Display Results
# -----------------------------
plt.figure(figsize=(12,6))

# Original image
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Segmented image
plt.subplot(1,2,2)
plt.imshow(segmented_image)
plt.title("GMM Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()