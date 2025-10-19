# download_images.py (v4 - Added Sofa, Stool, Cupboard)
import pandas as pd
import ast
import os
import requests
from tqdm import tqdm
import re # Import regular expressions for keyword matching

# --- 1. Define More Specific Category Mapping & Keywords ---
SPECIFIC_CATEGORY_MAP = {
    # Chairs
    'Chairs': 'Chair',
    'Barstools': 'Stool', # Changed Barstools to Stool
    'Living Room Furniture': 'Chair', # Ambiguous, but often chairs
    'Kitchen & Dining Room Furniture': 'Chair',
    'Accent Chairs': 'Chair',
    'Dining Chairs': 'Chair',
    'Folding Chairs': 'Chair',
    'Office Chairs': 'Chair',
    'Stools': 'Stool', # Added Stools

    # Tables
    'Tables': 'Table',
    'TV Trays': 'Table',
    'Accent Furniture': 'Table', # Often tables
    'End Tables': 'Table',
    'Coffee Tables': 'Table',
    'Desks & Workstations': 'Table',
    'Home Office Furniture': 'Table',
    'Console Tables': 'Table',
    'Folding Tables': 'Table',

    # Racks
    'Free Standing Shoe Racks': 'Shoe Rack',
    'Coat Racks': 'Rack',
    'Garment Racks': 'Rack',
    'Shoe Organizers': 'Shoe Rack',
    'Plate Racks': 'Rack', # Added Plate Racks

    # Mats
    'Doormats': 'Mat',
    'Bath Rugs': 'Mat',
    'Area Rugs': 'Mat',

    # Sofas
    'Sofas & Couches': 'Sofa',
    'Loveseats': 'Sofa',
    'Futons': 'Sofa',

    # Cupboards / Cabinets
    'Cabinets': 'Cupboard',
    'Storage Cabinets': 'Cupboard',
    'Pantry Cabinets': 'Cupboard',
    'Bathroom Storage & Organization': 'Cupboard', # Often includes cabinets
    'Kitchen Storage & Organization': 'Cupboard',
    'Home Office Cabinets': 'Cupboard',
    
    # Add more specific mappings if needed...
}

# Keywords to look for in the product title (case-insensitive)
TITLE_KEYWORDS = {
    'Chair': ['chair', 'seating', 'bench', 'ottoman'], # Removed 'stool'
    'Stool': ['stool'], # Added Stool keywords
    'Table': ['table', 'desk', 'stand', 'tray', 'console'],
    'Rack': ['rack', 'shelf', 'organizer'],
    'Shoe Rack': ['shoe rack', 'shoe shelf', 'shoe organizer'],
    'Mat': ['mat', 'rug'],
    'Sofa': ['sofa', 'couch', 'loveseat', 'futon', 'settee'],
    'Cupboard': ['cupboard', 'cabinet', 'pantry', 'storage unit', 'hutch'],
}

def get_clean_label_v4(row):
    """
    Smarter labeling v4: Uses title keywords and checks all categories
    against a specific map. Includes Sofa, Stool, Cupboard.
    """
    categories_str = row['categories']
    title = str(row['title']).lower() # Convert title to lowercase

    # --- Step 1: Check Specific Category Map ---
    try:
        categories_list = ast.literal_eval(categories_str)
        if isinstance(categories_list, list):
            # Check categories in reverse (most specific first)
            for category in reversed(categories_list):
                category_name = category.strip()
                if category_name in SPECIFIC_CATEGORY_MAP:
                    return SPECIFIC_CATEGORY_MAP[category_name]
    except:
        pass # Ignore parsing errors

    # --- Step 2: If no specific category match, use Title Keywords ---
    found_labels = []
    for label, keywords in TITLE_KEYWORDS.items():
        if any(re.search(r'\b' + keyword + r'\b', title) for keyword in keywords):
            found_labels.append(label)

    # Decide based on keywords found (with priority):
    if len(found_labels) == 1:
        return found_labels[0]
    elif len(found_labels) > 1:
        # Priority: Sofa > Cupboard > Table > Chair > Stool > Shoe Rack > Rack > Mat
        if 'Sofa' in found_labels: return 'Sofa'
        if 'Cupboard' in found_labels: return 'Cupboard'
        if 'Table' in found_labels: return 'Table'
        if 'Chair' in found_labels: return 'Chair'
        if 'Stool' in found_labels: return 'Stool'
        if 'Shoe Rack' in found_labels: return 'Shoe Rack'
        if 'Rack' in found_labels: return 'Rack'
        if 'Mat' in found_labels: return 'Mat'

    # --- Step 3: If still no match, return 'Other' ---
    return 'Other'

def get_first_image_url(images_str):
    """Parses the string list of images and returns the first URL."""
    try:
        images_list = ast.literal_eval(images_str)
        if isinstance(images_list, list) and len(images_list) > 0:
            return images_list[0].strip()
    except:
        pass
    return None

def download_images():
    print("Loading dataset...")
    df = pd.read_csv('intern_data_ikarus.csv')

    print("Cleaning data and mapping categories (v4 - Sofa, Stool, Cupboard)...")

    df['clean_label'] = df.apply(get_clean_label_v4, axis=1)
    df['image_url'] = df['images'].apply(get_first_image_url)

    df_clean = df.dropna(subset=['image_url'])
    df_clean = df_clean[df_clean['clean_label'] != 'Other']

    min_images_per_category = 5
    label_counts = df_clean['clean_label'].value_counts()
    labels_to_keep = label_counts[label_counts >= min_images_per_category].index.tolist()
    df_clean = df_clean[df_clean['clean_label'].isin(labels_to_keep)]
    print(f"\nApplied filter: Keeping categories with >= {min_images_per_category} images.")

    print(f"\nFound {len(df_clean)} images to download.")
    print("\nFinal categories we will be training on:")
    print(df_clean['clean_label'].value_counts())

    base_dir = 'images_v3' # New folder name
    os.makedirs(base_dir, exist_ok=True)

    for label in df_clean['clean_label'].unique():
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)

    print(f"\nDownloading images to '{base_dir}' folder...")
    for _, row in tqdm(df_clean.iterrows(), total=df_clean.shape[0]):
        try:
            url = row['image_url']
            label = row['clean_label']
            filename = f"{row['uniq_id']}.jpg"
            save_path = os.path.join(base_dir, label, filename)

            if not os.path.exists(save_path):
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                if 'image' in response.headers.get('Content-Type', '').lower():
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
        except:
            pass # Skip errors silently

    print("\n--- Image Download Complete! ---")
    print(f"Check the '{base_dir}' folder. Review the images for accuracy.")

if __name__ == "__main__":
    download_images()