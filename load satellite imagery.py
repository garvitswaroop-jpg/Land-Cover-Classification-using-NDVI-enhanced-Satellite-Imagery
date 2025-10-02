import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# --- Load multispectral image (example: 4 bands = R,G,B,NIR) ---
image_path = "satellite_image.tif"
label_path = "landcover_labels.tif"

with rasterio.open(image_path) as src:
    red = src.read(1).astype('float32')
    green = src.read(2).astype('float32')
    blue = src.read(3).astype('float32')
    nir = src.read(4).astype('float32')

with rasterio.open(label_path) as src:
    labels = src.read(1).astype('int32')

# --- Compute NDVI ---
ndvi = (nir - red) / (nir + red + 1e-6)

# --- Stack features: R,G,B,NDVI ---
H, W = red.shape
features = np.stack([red, green, blue, ndvi], axis=-1).reshape(-1, 4)
targets = labels.reshape(-1)

# --- Remove nodata pixels ---
mask = targets > 0
X = features[mask]
y = targets[mask]

# --- Train simple classifier ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Visualization ---
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].imshow(np.dstack([red, green, blue]) / 10000)  # RGB image
axs[0].set_title("RGB Satellite Image")
axs[0].axis("off")

axs[1].imshow(ndvi, cmap="RdYlGn")
axs[1].set_title("NDVI Map")
axs[1].axis("off")

axs[2].imshow(labels, cmap="tab20")
axs[2].set_title("Land Cover Map")
axs[2].axis("off")
plt.show()

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.show()
