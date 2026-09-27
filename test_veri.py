import matplotlib.pyplot as plt
from src.data import FilamentDataset

ds = FilamentDataset(
    images_dir="data/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    annotations_json="data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
)

print(f"Toplam ornek sayisi: {len(ds)}")

image, target = ds[0]
print(f"Goruntu boyutu: {image.shape}, filament sayisi: {target['masks'].shape[0]}")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image[0], cmap="gray")
ax[0].set_title("H-Alpha goruntu")
ax[1].imshow(image[0], cmap="gray")
combined_mask = target["masks"].sum(dim=0)
ax[1].imshow(combined_mask, cmap="Reds", alpha=0.5)
ax[1].set_title(f"{target['masks'].shape[0]} filament maskesi ustte")
plt.show()