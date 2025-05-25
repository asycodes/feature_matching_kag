import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering

excel_file = "actual_submission.xlsx"
df = pd.DataFrame(
    columns=[
        "image_id",
        "dataset",
        "scene",
        "image",
        "rotation_matrix",
        "translation_vector",
    ],
)


def ratio_test(matches, ratio=0.75):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def main():
    # FEATURE EXTRACTION START
    datasets = {}
    for dataset in os.listdir("train/"):
        features = {}
        dataset_path = os.path.join("train/", dataset)
        for image in os.listdir(dataset_path):

            pred = [image + "_id", dataset, None, image, None, None]
            df.loc[len(df)] = pred
            if image.endswith(".png"):
                img_path = os.path.join(dataset_path, image)
                # Read the image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                sift = cv2.ORB_create(nfeatures=500)
                kp, desc = sift.detectAndCompute(img, None)
                if desc is not None:
                    features[image] = (kp, desc)
        datasets[dataset] = features
    # FEATURE EXTRACTION COMPLETED

    # FEATURE MATCHING START
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    match_scores = defaultdict(
        dict
    )  # match_scores[img1][img2] = number of good matches

    for dataset_path, features in datasets.items():
        image_list = list(features.keys())
        total_pairs = (len(image_list) * (len(image_list) - 1)) // 2
        pair_count = 0
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):  # skip redundant and self-match
                pair_count += 1
                img1_name = image_list[i]
                img2_name = image_list[j]
                # extract features once
                kp1, desc1 = features[img1_name]
                kp2, desc2 = features[img2_name]
                if desc1 is not None and desc2 is not None:
                    matches = bf.knnMatch(desc1, desc2, k=2)
                    good_matches = ratio_test(matches)
                    match_scores[img1_name][img2_name] = len(good_matches)
                    match_scores[img2_name][img1_name] = len(good_matches)
                progress = (pair_count / total_pairs) * 100
                print(
                    f"Matching {img1_name} ↔ {img2_name} | Progress: {progress:.2f}%".ljust(
                        100
                    ),
                    end="\r",
                )
        print()
        images_features = list(features.keys())
        n = len(images_features)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    score = match_scores[images_features[i]].get(images_features[j], 0)
                    similarity_matrix[i, j] = score

        distance_matrix = 1 / (similarity_matrix + 1e-5)  # Avoid divide-by-zero
        np.fill_diagonal(distance_matrix, 0)

        clustering = AgglomerativeClustering(
            n_clusters=3, metric="precomputed", linkage="average"
        )

        labels = clustering.fit_predict(distance_matrix)

        # Show results
        for i, img in enumerate(images_features):
            df.loc[df["image"] == img, "scene"] = labels[i]

        df.to_excel(excel_file, index=False)
        print(f"Results saved to {excel_file}")


if __name__ == "__main__":
    main()
