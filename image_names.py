import os

def generate_image_lists(train_dir="train", output_dir="Patch-NetVLAD/patchnetvlad/dataset_imagenames"):
    os.makedirs(output_dir, exist_ok=True)

    for dataset in os.listdir(train_dir):
        dataset_path = os.path.join(train_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        image_paths = []
        for filename in os.listdir(dataset_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(f"./{dataset}/{filename}")

        if image_paths:
            output_file = os.path.join(output_dir, f"{dataset}.txt")
            with open(output_file, "w") as f:
                f.write("\n".join(image_paths))
            print(f"[✓] Saved {len(image_paths)} paths to {output_file}")

if __name__ == "__main__":
    generate_image_lists()
