"""Automated RECIST 1.1 measurement from segmentation masks."""

import numpy as np
from scipy import ndimage


class RECISTMeasurer:
    """Compute RECIST 1.1 measurements from 3D binary segmentation masks.

    RECIST 1.1 criteria:
        - Complete Response (CR): Disappearance of all target lesions
        - Partial Response (PR): >= 30% decrease in sum of longest diameters
        - Progressive Disease (PD): >= 20% increase in sum of longest diameters
        - Stable Disease (SD): Neither PR nor PD criteria met
    """

    CR_THRESHOLD = 0.0  # Complete disappearance
    PR_THRESHOLD = -0.30  # 30% decrease
    PD_THRESHOLD = 0.20  # 20% increase
    PD_ABSOLUTE_INCREASE_MM = 5.0  # Minimum absolute increase for PD (RECIST 1.1 §4.3)

    # Target lesion eligibility (RECIST 1.1 §3.1.1, §3.1.2)
    MIN_TARGET_DIAMETER_MM = 10.0  # Minimum longest diameter to qualify as target lesion
    MAX_TARGET_LESIONS = 5  # Maximum total target lesions (whole-body cap)

    def longest_axial_diameter(
        self, mask: np.ndarray, pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> float:
        """Compute longest axial diameter of a lesion from its 3D mask.

        Per RECIST 1.1 the longest diameter is the longest *in-plane* extent of
        the lesion. We therefore measure the maximum Feret diameter on every
        axial slice and return the global maximum, rather than only the
        largest-area slice: a lesion whose longest extent lies on a smaller-area
        slice would otherwise be silently under-measured.

        Args:
            mask: Binary 3D mask [H, W, D]
            pixdim: Voxel spacing in mm (H, W, D)

        Returns:
            Longest axial diameter in mm.
        """
        if mask.sum() == 0:
            return 0.0

        max_diameter = 0.0
        scale = np.array([pixdim[0], pixdim[1]])

        # Scan every axial slice; the longest in-plane diameter is not
        # necessarily on the largest-area slice.
        for d in range(mask.shape[2]):
            axial_mask = mask[:, :, d]
            if axial_mask.sum() == 0:
                continue

            coords = np.argwhere(axial_mask > 0)
            if len(coords) < 2:
                continue

            # Max pairwise (Feret) distance on this slice, scaled by spacing.
            scaled_coords = coords.astype(float) * scale
            for i in range(len(scaled_coords)):
                dists = np.sqrt(np.sum((scaled_coords[i:] - scaled_coords[i]) ** 2, axis=1))
                max_diameter = max(max_diameter, dists.max())

        return float(max_diameter)

    def volume_mm3(
        self, mask: np.ndarray, pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> float:
        """Compute lesion volume in mm^3."""
        voxel_vol = pixdim[0] * pixdim[1] * pixdim[2]
        return float(mask.sum() * voxel_vol)

    def measure_lesions(
        self, mask: np.ndarray, pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> list[dict]:
        """Detect and measure individual lesions in a multi-lesion mask.

        Args:
            mask: Binary 3D mask (may contain multiple connected components)
            pixdim: Voxel spacing in mm

        Returns:
            List of dicts with lesion measurements, sorted by size (largest first).
        """
        labeled_array, num_features = ndimage.label(mask > 0)
        lesions = []

        for i in range(1, num_features + 1):
            lesion_mask = (labeled_array == i).astype(np.uint8)
            lesions.append(
                {
                    "id": i,
                    "longest_diameter_mm": self.longest_axial_diameter(lesion_mask, pixdim),
                    "volume_mm3": self.volume_mm3(lesion_mask, pixdim),
                    "voxel_count": int(lesion_mask.sum()),
                }
            )

        # RECIST 1.1 §3.1.1: drop sub-threshold lesions (longest diameter < 10 mm)
        lesions = [
            les for les in lesions if les["longest_diameter_mm"] >= self.MIN_TARGET_DIAMETER_MM
        ]

        # RECIST 1.1 §3.1.2: keep at most MAX_TARGET_LESIONS, ranked by longest diameter
        lesions.sort(key=lambda x: x["longest_diameter_mm"], reverse=True)
        lesions = lesions[: self.MAX_TARGET_LESIONS]
        return lesions
