import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from cv2 import dnn_superres


def main():
    img1_path = "train/stairs/stairs_split_1_1710453620694_cropped.png"
    img2_path = "train/stairs/stairs_split_1_1710453576271_cropped.png"
    path = "./model.pb"  # e.g., "EDSR_x4.pb"
    sr.readModel(path)
    sr = dnn_superres.DnnSuperResImpl_create()
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    """
  # Load images
    

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(desc1, desc2)
    matches_2 = bf.match(desc2, desc1)
    matches_sorted = sorted(matches_2, key=lambda x: x.distance)
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None)
    img_matches_2 = cv2.drawMatches(
        img2,
        kp2,
        img1,
        kp1,
        matches_sorted[:5],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    ) """

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    img_kp_1 = cv2.drawKeypoints(
        img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    img_kp_2 = cv2.drawKeypoints(
        img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

    # Display the first image in the first subplot
    axes[0].imshow(img_kp_1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")  # Hide axes

    # Display the second image in the second subplot
    axes[1].imshow(img_kp_2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")  # Hide axes

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

    """ # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img3), plt.show()

    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.title("Top 50 Feature Matches")
    plt.axis("off")
    plt.show() """


if __name__ == "__main__":
    main()
