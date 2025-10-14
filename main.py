import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import MDS
import networkx as nx
from joblib import Parallel, delayed
from itertools import combinations
from tqdm import tqdm
from shapely.geometry import Point, Polygon
import multiprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import pycolmap
from scripts.database import *
import sqlite3
from time import time, sleep
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scripts.database import COLMAPDatabase, image_ids_to_pair_id
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pathlib


#### DISCONTINUEDDDDDDDDDDDD
multiprocessing.set_start_method("spawn", force=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
print(torch.cuda.is_available())
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).to(device)
model.fc = torch.nn.Identity()

def shortlist_images(images_dir,query_image_path,k=50):
    image_dir = images_dir
    query_image_path = query_image_path
    output_feature_dir = os.path.join(image_dir, query_image_path)

    img = cv2.imread(output_feature_dir)

    if img is None:
        raise ValueError(f"Failed to load image: {query_image_path}")

    img_height, img_width = img.shape[:2]
    top_k = k  # Number of similar images to shortlist

    
    model.eval()

    # Preprocessing (auto handles resize, normalize, etc.)
    transform = weights.transforms()

    def extract_feature(img_path):
        image = Image.open(img_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
        with torch.no_grad():
            features = model(tensor)
        return features.squeeze(0).cpu().numpy()  # Shape: (2048,)

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    features = []
    for path in image_paths:
        features.append(extract_feature(path))
    features = np.array(features)

    query_feature = extract_feature(output_feature_dir).reshape(1, -1)

    similarities = cosine_similarity(query_feature, features).flatten()
    top_k_indices = similarities.argsort()[::-1][:top_k]

    return top_k_indices


FORCE_RERUN = True  # set to True if you want to rerun extraction manually

def build_similarity_graph(image_names, similarity_matrix, threshold):
    G = nx.Graph()
    n = len(image_names)
    
    for i in range(n):
        G.add_node(image_names[i])
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                G.add_edge(image_names[i], image_names[j], weight=similarity_matrix[i][j])
    return G

def draw_graph(G, clusters,title):
    color_map = {}
    for idx, cluster in enumerate(clusters):
        for node in cluster:
            color_map[node] = idx
    colors = [color_map[n] for n in G.nodes()]
    nx.draw(G, node_color=colors, with_labels=True, node_size=200)
    plt.title(title)
    plt.show(block=True)

def extract_keypoint_coords(serialized_kps):
    return [kp["pt"] for kp in serialized_kps]

def keypoint_inside_any_box(pt, regions):
    x, y = pt
    for region in regions:
        # Ensure valid shape and dtype
        if len(region) < 1:
            continue
        region = np.array(region, dtype=np.int32).reshape(-1, 1, 2)
        x_, y_, w, h = cv2.boundingRect(region)
        if x_ <= x <= x_ + w and y_ <= y <= y_ + h:
            return True
    return False

def filter_keypoints_by_mser(keypoints, descriptors, mser_regions):
    # Convert MSER regions to bounding boxes
    bounding_boxes = [cv2.boundingRect(region) for region in mser_regions]

    filtered_kps = []
    filtered_desc = []

    for i, kp in enumerate(keypoints):
        pt = kp["pt"] if isinstance(kp, dict) else kp.pt
        if keypoint_inside_any_box(pt, bounding_boxes):
            filtered_kps.append(kp)
            filtered_desc.append(descriptors[i])

    if len(filtered_desc) == 0:
        return [], None

    return filtered_kps, np.array(filtered_desc)


def remove_nested_regions(regions):
    bboxes = [cv2.boundingRect(r) for r in regions]
    keep = []

    for i, (x1, y1, w1, h1) in enumerate(bboxes):
        rect1 = (x1, y1, x1 + w1, y1 + h1)
        nested = False
        for j, (x2, y2, w2, h2) in enumerate(bboxes):
            if i == j:
                continue
            rect2 = (x2, y2, x2 + w2, y2 + h2)
            if (rect1[0] >= rect2[0] and rect1[1] >= rect2[1] and
                rect1[2] <= rect2[2] and rect1[3] <= rect2[3]):
                nested = True
                break
        if not nested:
            keep.append(regions[i])

    return keep



def compute_pair(img1_name, img2_name, desc1_proj, desc2_proj, kp1_pts, kp2_pts):
    if desc1_proj is None or desc2_proj is None:
        return (img1_name, img2_name, 0, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    nn_matches = bf.knnMatch(desc1_proj, desc2_proj, k=2)
    good_matches = [
    m for match in nn_matches if len(match) == 2
    for m, n in [match] if m.distance < 0.9 * n.distance]

    if len(good_matches) <= 4:
        return (img1_name, img2_name, 0, None)

    src_pts = np.float32([kp1_pts[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2_pts[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return (img1_name, img2_name, 0, None)

    matchesMask = mask.ravel().tolist()
    match_array = np.array(
        [[m.queryIdx, m.trainIdx] for m, keep in zip(good_matches, matchesMask) if keep],
        dtype=np.uint32
    )
    score = len(match_array)
    return (img1_name, img2_name, score, match_array)


def serialize_keypoints2(keypoints):
    return [
        {
            "pt": kp.pt,
            "size": kp.size,
            "angle": kp.angle,
            "response": kp.response,
            "octave": kp.octave,
            "class_id": kp.class_id,
        }
        for kp in keypoints
    ]



def extract_features_akaze(datasetname, imagename, dataset_image):
    try:
        img_path = os.path.join("test", datasetname, imagename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            return (datasetname, imagename, [], None)
        mask = np.zeros(img.shape, dtype=np.uint8)
        for region in dataset_image[datasetname][imagename]["mser"]:
            cv2.drawContours(mask, [region], -1, 255, -1)
        akaze = cv2.AKAZE_create(threshold=0.0005)
        kp, desc = akaze.detectAndCompute(img, None)

        if desc is not None:
            kp_serialized = serialize_keypoints2(kp)
            return (datasetname, imagename, kp_serialized, desc)
        else:
            print(f"[WARN] Desc is none: {img_path}")
            return (datasetname, imagename, [], None)

    except Exception as e:
        print(f"[ERROR] Exception in {datasetname}/{imagename}: {e}")
        return (datasetname, imagename, [], None)

    


def min_max_normalize(score_dict):
    # Flatten all scores into a list
    all_scores = [v for subdict in score_dict.values() for v in subdict.values()]
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)

    # Normalize each score
    normalized = {
        k1: {
            k2: (v - min_score) / (max_score - min_score + 1e-6)
            for k2, v in v1.items()
        }
        for k1, v1 in score_dict.items()
    }

    return normalized

MAX_PIXELS = 12_000_000
MAX_WIDTH = 2000
MAX_HEIGHT = 1500

def safe_resize(img):
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS or h > MAX_HEIGHT or w > MAX_WIDTH:
        scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, (MAX_PIXELS / (h * w))**0.5)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


#MSER region helps to detect 
def extract_mser_regions_multi_resolution(datasetname, imagename, scales=[0.5, 1.0, 1.5]):
    try:
        img_path = os.path.join("test", datasetname, imagename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            return (datasetname, imagename, [])
        img = safe_resize(img)
        h, w = img.shape[:2]
        mser = cv2.MSER_create(delta=3, min_area=int(30), max_area=int(0.25 * h * w))
        all_regions = []

        for scale in scales:
            scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            regions, _ = mser.detectRegions(scaled_img)
            for region in regions:
                # Rescale region back to original resolution
                region = (region / scale).astype(np.int32)
                all_regions.append(region)
        filtered_regions= remove_nested_regions(all_regions)
        reshaped_regions = [
            np.array(region, dtype=np.int32).reshape(-1, 1, 2)
            for region in filtered_regions if len(region) > 0
        ]
        return (datasetname, imagename, reshaped_regions)

    except Exception as e:
        print(f"[ERROR] Failed for {imagename} in {datasetname}: {e}")
        return (datasetname, imagename, [])
    
""" 
def load_patch_netvlad_descriptors(folder_path):
    descriptors = {}
    for file in os.listdir(folder_path):
        if not file.endswith(".npy") or file == "globalfeats.npy":
            continue  # Skip non-npy and globalfeats.npy

        path = os.path.join(folder_path, file)
        arr = np.load(path)

        if arr.ndim == 1:
            print(f"[WARNING] Skipping 1D descriptor: {file} with shape {arr.shape}")
            continue  # Skip invalid shape

        descriptors[file] = arr
    return descriptors """

""" def normalize_descriptors(desc_dict):
    names = list(desc_dict.keys())
    vectors = np.stack([desc_dict[name] for name in names])
    normalized_vectors = normalize(vectors, axis=1)  # L2 normalization
    return names, normalized_vectors

def get_top_k_similar(descriptors, top_percent=0.3):
    image_names = list(descriptors.keys())
    vectors = np.array([descriptors[name] for name in image_names])
    for name, desc in descriptors.items():
        if desc.ndim != 2:
            print(f"[ERROR] {name} has invalid descriptor shape {desc.shape}")
    
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(vectors)
    
    num_images = len(image_names)
    num_top = int(top_percent * num_images)
    
    top_similar = {}

    for idx, row in enumerate(sim_matrix):
        # Get indices of top k most similar (excluding self)
        top_k_idx = np.argsort(row)[::-1][1:num_top + 1]
        top_similar[image_names[idx]] = [image_names[i] for i in top_k_idx]

    return top_similar, sim_matrix
 """


""" 
def build_pairwise_list(top_similar):
    seen_pairs = set()
    pairs = []

    for img1, similars in top_similar.items():
        for img2 in similars:
            pair = tuple(sorted([img1, img2]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(pair)
    
    return pairs
 """

def load_match_graph(db_path, min_matches=15):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM images")
    id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
    
    cursor.execute("SELECT pair_id, rows FROM matches")
    matches = cursor.fetchall()

    def pair_id_to_image_ids(pair_id):
        MAX_IMAGE_ID = 2**31 - 1
        image_id1 = pair_id // MAX_IMAGE_ID
        image_id2 = pair_id % MAX_IMAGE_ID
        return image_id1, image_id2

    G = nx.Graph()
    for pair_id, num_matches in matches:
        if num_matches < min_matches:
            continue
        id1, id2 = pair_id_to_image_ids(pair_id)
        name1 = id_to_name.get(id1)
        name2 = id_to_name.get(id2)
        if name1 and name2:
            G.add_edge(name1, name2, weight=num_matches)

    return G
def rotation_matrix_to_quaternion(R_mat):
    return R.from_matrix(R_mat).as_quat()

def estimate_two_view_geometry(kp1, kp2, match_array, img_size):
    valid_matches = [m for m in match_array if m[0] < len(kp1) and m[1] < len(kp2)]
    if len(valid_matches) < len(match_array):
        print(f"[WARN] {len(match_array) - len(valid_matches)} invalid matches dropped")
    if len(valid_matches) < 8:
        return None, None, None, None, None
    # Extract the matched keypoints
    pts1 = np.float32([kp1[m[0]]['pt'] for m in valid_matches])
    pts2 = np.float32([kp2[m[1]]['pt'] for m in valid_matches])

    # Estimate Fundamental matrix using RANSAC
    F, inliers = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC,
                                        ransacReprojThreshold=1.0, confidence=0.999)
    if inliers is None:
        return None, None, None, None, None
    inliers = inliers.ravel().astype(bool)
    valid_matches = np.array(valid_matches, dtype=np.uint32)
    inlier_matches = valid_matches[inliers]

    # Estimate Essential matrix (assuming fx = fy and principal point at center)
    focal = 1000 
    cx, cy = img_size[0] / 2, img_size[1] / 2
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0,     0,  1]])

    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None, None, None
    # Recover relative pose (qvec, tvec)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    qvec = rotation_matrix_to_quaternion(R)
    tvec = t.flatten()

    return F, E, qvec, tvec, inlier_matches


def create_colmap_db(dataset_images, match_pairs, datasetname,database_path="colmap.db"):
    if os.path.exists(database_path):
        os.remove(database_path)

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    seen = set()
    unique_results_akaze = []
    

    image_id_map = {}

    # Insert images and keypoints
    for img_name, data in dataset_images[datasetname].items():
        if not data.get("valid", True):
            continue
        camera_model = "SIMPLE_PINHOLE"
        img_path = os.path.join("test", datasetname, img_name)
        print(img_path)
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        height, width = img.shape[:2] 
        focal_length = 1000 
        cx, cy = width / 2, height / 2
        camera_id = db.add_camera(camera_model, width, height, [focal_length, cx, cy])
        image_id = db.add_image(img_name, camera_id)
        image_id_map[img_name] = image_id
        keypoints = np.array([kp["pt"] for kp in data["kp"]], dtype=np.float32)
        db.add_keypoints(image_id, keypoints)

    # Insert matches
    for img1, img2, match_score, match_array in match_pairs:
        if match_score < 1 or match_array is None:
            continue

        id1 = image_id_map[img1]
        id2 = image_id_map[img2]

        if id1 > id2:
            image_id1, image_id2 = id2, id1
            matches = match_array[:, [1, 0]]
            kp1 = dataset_images[datasetname][img2]["kp"]
            kp2 = dataset_images[datasetname][img1]["kp"]
        else:
            image_id1, image_id2 = id1, id2
            matches = match_array
            kp1 = dataset_images[datasetname][img1]["kp"]
            kp2 = dataset_images[datasetname][img2]["kp"]


        print(f"Added {len(match_array)} matches between {img1} and {img2}")
        db.add_matches(image_id1, image_id2, matches)
        F, E, qvec, tvec, inlier_matches = estimate_two_view_geometry(kp1, kp2, matches, (width, height))
        if inlier_matches is None or F is None:
            continue
        H = np.eye(3)

        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        if F.shape == (9,3):
            F = F[0:3]
            print(F)
        assert F.shape == (3, 3) and F.dtype == np.float64
        assert E.shape == (3, 3) and E.dtype == np.float64
        assert H.shape == (3, 3) and H.dtype == np.float64
        assert qvec.shape == (4,) and qvec.dtype == np.float64
        assert tvec.shape == (3,) and tvec.dtype == np.float64
        # Config 2 = two-view geometry
        db.add_two_view_geometry(
            image_id1,
            image_id2,
            inlier_matches,
            F=F,
            E=E,
            H=H,
            qvec=qvec,
            tvec=tvec,
            config=2
        )

    db.commit()
    db.close()

def generate_pairs_for_image(query_image_name, image_paths, dataset_path, k):
    try:
        # Get top-k indices for the current image
        top_k_indices = shortlist_images(dataset_path, query_image_name, k=k)

        all_image_paths = [os.path.join(dataset_path, f) for f in image_paths]
        shortlist_paths = [all_image_paths[i] for i in top_k_indices]
        shortlist_filenames = [os.path.basename(p) for p in shortlist_paths]

        return [(query_image_name, other) for other in shortlist_filenames if other != query_image_name]
    except Exception as e:
        print(f"Error processing {query_image_name}: {e}")
        return []
    

def create_submission(results):
    with open(submission_file, 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in results:
            for scene in results[dataset]:
                for image in results[dataset][scene]:
                    r = results[dataset][scene][image]['R']
                    t = results[dataset][scene][image]['t']
                    f.write(f'{image},{dataset},{scene},{r},{t}\n')
def main():
    # FEATURE EXTRACTION START
    jobs = []
    predictions = defaultdict(lambda: defaultdict(dict))
    dataset_images = defaultdict(dict)
    for dataset_name in os.listdir("test"):
        dataset_path = "test/" + dataset_name
        for img_file in os.listdir(dataset_path):
            if img_file.endswith((".jpg", ".png")):
                dataset_images[dataset_name][img_file] = {}
                jobs.append((dataset_name, img_file))
    print("ok")
    
    mser_results = Parallel(n_jobs=6, prefer= "processes")(delayed(extract_mser_regions_multi_resolution)(ds, img) for ds, img in tqdm(jobs, desc="Finding Regions"))
    for ds, img, regions in mser_results:
        dataset_images[ds][img]= {
            "mser" : regions
        }
    results = Parallel(n_jobs=6,prefer="processes")(delayed(extract_features_akaze)(ds, img, dataset_images) for ds, img in tqdm(jobs, desc="Finding KEYPOINTS"))

    for ds, img, kp, desc in results:
        dataset_images[ds][img]["kp"] = kp
        dataset_images[ds][img]["desc"] = desc
        
    print("Extraction done saving now")


    

    print("finish inserting the regions")
    # FEATURE EXTRACTION COMPLETED

    # FEATURE MATCHING START
    match_scores_akaze = defaultdict(
        dict
    )  # match_scores[img1][img2] = number of good matches
    #match_scores_sift = defaultdict(
        #dict
    #)
    output_feature_dir = os.path.join("features_filtered", dataset_name)
    os.makedirs(output_feature_dir, exist_ok=True)

    for dataset_name in os.listdir("test"):
        similarity_matrices = {}
        image_list_akaze = list(dataset_images[dataset_name].keys())

        dataset_path = "test/" + dataset_name
        image_paths = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

        image_paths = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

        # --- Parallel Execution ---
        results = Parallel(n_jobs=6, prefer="processes")(
            delayed(generate_pairs_for_image)(query_image_name, image_paths, dataset_path, k=50)
            for query_image_name in tqdm(image_paths, desc="Shortlisting in parallel")
        )

        # Flatten the list of lists
        all_pairs_akaze = [pair for sublist in results for pair in sublist]

        """ print("CURRENT DATASET: ",dataset_name)
        for img in dataset_images[dataset_name]:
            kps = dataset_images[dataset_name][img]["kp"]
            desc = dataset_images[dataset_name][img]["desc"]
            mser = dataset_images[dataset_name][img]["mser"]
            filtered_kps, filtered_desc = filter_keypoints_by_mser(kps, desc, mser)


            dataset_images[dataset_name][img]["kp"] = filtered_kps
            dataset_images[dataset_name][img]["desc"] = filtered_desc
            if filtered_desc is None or len(filtered_desc) == 0 or len(filtered_kps) != len(filtered_desc):
                dataset_images[dataset_name][img]["valid"] = False
                continue
            dataset_images[dataset_name][img]["valid"] = True """


        results_akaze = Parallel(n_jobs=6, prefer= "processes")(
            delayed(compute_pair)(
                img1, img2,
                dataset_images[dataset_name][img1]["desc"],
                dataset_images[dataset_name][img2]["desc"],
                extract_keypoint_coords(dataset_images[dataset_name][img1]["kp"]),
                extract_keypoint_coords(dataset_images[dataset_name][img2]["kp"]),
            )
            for img1, img2 in tqdm(all_pairs_akaze, desc="Matching pairs")
        )

        seen = set()
        unique_results_akaze = []

        for img1, img2, score, matches in results_akaze:
            if img1 > img2:
                img1, img2 = img2, img1  # Force order

            pair_key = (img1, img2)
            if pair_key not in seen:
                seen.add(pair_key)
                unique_results_akaze.append((img1, img2, score, matches))

        

        # after image matching, we filter using MSER


        
        match_index_dict = {}
        # Store results
        for img1, img2, score, match_array in unique_results_akaze:
            if score < 1:
                continue
            match_scores_akaze[img1][img2] = score
            match_scores_akaze[img2][img1] = score
            match_index_dict[(img1, img2)] = match_array
            match_index_dict[(img2, img1)] = match_array[:, [1, 0]]  # reversed


        #scores_akaze_norm = min_max_normalize(score)
        create_colmap_db(dataset_images, unique_results_akaze , dataset_name, database_path="colmap.db")


        mapper_options_tt = pycolmap.IncrementalPipelineOptions()
        mapper_options_tt.min_model_size = 3
        mapper_options_tt.max_num_models = 25
        mapper_options_tt.min_num_matches = 5
        maps = pycolmap.incremental_mapping(
            database_path="colmap.db",
            image_path="test/" + dataset_name,
            output_path="output/reconstruction",
            options=mapper_options_tt
        )
        print(maps)
        print("Number of Reconstructions: ",len(maps))
        for i in range(len(maps)):
            print("Reconstruction number: ",i)
            print(maps[i].summary())
            print()

        print(maps)

        n = len(image_list_akaze)

        for map_index, cur_map in maps.items():
            for image_id, image in cur_map.images.items():
                image_name = image.name
                print(image.cam_from_world.rotation.matrix())
                R_mat = image.cam_from_world.rotation.matrix()
                tvec = image.cam_from_world.translation
        
                if dataset_name not in predictions:
                    predictions[dataset_name] = {}
                if map_index not in predictions[dataset_name]:
                    predictions[dataset_name][map_index] = {}
        
                predictions[dataset_name][map_index][image_name] = {
                    'R': R_mat,
                    't': np.array(tvec)
                }
        print("end dataset")

    # Add outliers (unmapped images)
    for dataset_name in os.listdir("test"):
        if dataset_name not in predictions:
            predictions[dataset_name] = {}
        if 'outliers' not in predictions[dataset_name]:
            predictions[dataset_name]['outliers'] = {}
    
        dataset_path = f"test/{dataset_name}"
        for img_file in os.listdir(dataset_path):
            if img_file.endswith((".jpg", ".png")):
                # Check if the image is in any scene
                already_present = any(
                    img_file in predictions[dataset_name][scene]
                    for scene in predictions[dataset_name]
                )
                if not already_present:
                    predictions[dataset_name]['outliers'][img_file] = {
                        'R': np.full((3, 3), np.nan),
                        't': np.full((3,), np.nan)
                    }
    return predictions


if __name__ == "__main__":
    submission_file = 'submission.csv'
    src = 'test'
    predictions = main() # get predicitons here
    create_submission(predictions)
    print('congrats')
