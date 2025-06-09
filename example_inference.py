import cv2
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch

from uniflowmatch.models.ufm import (
    UniFlowMatchClassificationRefinement,
    UniFlowMatchConfidence,
)
from uniflowmatch.utils.viz import warp_image_with_flow

if __name__ == "__main__":

    USE_REFINEMENT_MODEL = False

    if USE_REFINEMENT_MODEL:
        model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
    else:
        model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")

    # === Load and Prepare Images ===
    source_path = "examples/image_pairs/fire_academy_0.png"
    target_path = "examples/image_pairs/fire_academy_1.png"

    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # === Predict Correspondences ===
    result = model.predict_correspondences_batched(
        source_image=torch.from_numpy(source_image),
        target_image=torch.from_numpy(target_image),
    )

    flow_output = result.flow.flow_output[0].cpu().numpy()
    covisibility = result.covisibility.mask[0].cpu().numpy()

    # === Visualize Results ===
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    axs[0, 0].imshow(source_image)
    axs[0, 0].set_title("Source Image")

    axs[0, 1].imshow(target_image)
    axs[0, 1].set_title("Target Image")

    # Warp the image using flow
    warped_image = warp_image_with_flow(source_image, None, target_image, flow_output.transpose(1, 2, 0))
    warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(
        warped_image
    )
    warped_image /= 255.0

    axs[0, 2].imshow(warped_image)
    axs[0, 2].set_title("Warped Image")

    # Flow visualization
    flow_vis_image = flow_vis.flow_to_color(flow_output.transpose(1, 2, 0))
    axs[1, 0].imshow(flow_vis_image)
    axs[1, 0].set_title("Flow Output (Valid at covisible region)")

    # Covisibility mask
    axs[1, 1].imshow(covisibility > 0.5, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Covisibility Mask (Thresholded by 0.5)")

    heatmap = axs[1, 2].imshow(covisibility, cmap="gray", vmin=0, vmax=1)
    axs[1, 2].set_title("Covisibility Mask")
    plt.colorbar(heatmap, ax=axs[1, 2])

    plt.tight_layout()
    plt.savefig("ufm_output.png")
    plt.show()
    print("Saved ufm_output.png")
