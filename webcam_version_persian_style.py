# Fixed version of webcam_version.py with improved lookup and quit functionality

import torch
import torch.nn as nn
import torchvision.models as models
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import torch
from torchvision import transforms
from IPython.display import display
import pandas as pd
import torch.utils.data as data
import torch
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import models
import cv2
import arabic_reshaper
from bidi.algorithm import get_display

import gdown
import time
import numpy as np
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Persian to English translation for display
persian_to_english_display = {
    # Names
    'علی احمدی': 'Ali Ahmadi',
    'فاطمه محمدی': 'Fatemeh Mohammadi', 
    'حسن رضایی': 'Hassan Rezaei',
    'مریم کریمی': 'Maryam Karimi',
    'محمد جعفری': 'Mohammad Jafari',
    'زهرا نوری': 'Zahra Nouri',
    'احمد صادقی': 'Ahmad Sadeghi',
    'نرگس احمدی': 'Narges Ahmadi',
    'رضا محمدی': 'Reza Mohammadi',
    'سارا کریمی': 'Sara Karimi',
    'امیر رضایی': 'Amir Rezaei',
    'نازنین جعفری': 'Nazanin Jafari',
    'حسین صادقی': 'Hussein Sadeghi',
    'الهام نوری': 'Elham Nouri',
    'مهدی احمدی': 'Mehdi Ahmadi',
    
    # Vehicle models
    'پژو 206': 'Peugeot 206',
    'پژو 207': 'Peugeot 207', 
    'پژو 405': 'Peugeot 405',
    'سمند': 'Samand',
    'تیبا': 'Tiba',
    
    # Colors
    'سفید': 'White',
    'مشکی': 'Black',
    'آبی': 'Blue',
    'قرمز': 'Red',
    
    # Status messages
    'NOT FOUND IN DATABASE': 'NOT FOUND IN DATABASE',
    'DETECTION READY': 'DETECTION READY',
    'PROCESSING...': 'PROCESSING...',
    'Press q to quit, s to save frame': 'Press q to quit, s to save frame'
}

def translate_for_display(persian_text):
    """Translate Persian text to English for display purposes"""
    if persian_text in persian_to_english_display:
        return persian_to_english_display[persian_text]
    return persian_text

#  version info
VERSION = '12'
MODEL_SIZE = 'n'  # Options: n, s, m, l, x
EPOCHS = 20
IMGSZ = 640
BATCH = 16
DEVICE = '0'

# Loading the trained model:
LPD_model_trained = YOLO(f"yolo{VERSION}{MODEL_SIZE}_trained.pt")

BASE_PATH = "."
LPD_RELATIVE_PATH = 'LPD_FILES'
LPR_RELATIVE_PATH = 'LPR_FILES'


# transform for persian language
digit_vocabulary = "0123456789"
persian_letters = "آ ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی".split()



digit_to_idx = {char: idx for idx, char in enumerate(digit_vocabulary)}
letter_to_idx = {char: idx for idx, char in enumerate(persian_letters)}
idx_to_digit = {idx: char for idx, char in enumerate(digit_vocabulary)}
idx_to_letter = {idx: char for idx, char in enumerate(persian_letters)}

persian_to_english_digits = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}

persian_letter_normalization = {
    "الف": "آ",
    "ا" : "آ",
    "ژ (معلولین و جانبازان)": "ژ",
    "ه‍" : "ه"
}


def translate(label):
    first_two_digits = ''.join([persian_to_english_digits.get(char, char) for char in label[:2]])
    persian_letter = label[2]
    remaining_digits = ''.join([persian_to_english_digits.get(char, char) for char in label[3:]])
    return first_two_digits + persian_letter + remaining_digits


def preprocess_sample(image_path,
                      label,
                      full_transform=True,
                      log=False,
                      dir_path=BASE_PATH ,
                      relative_path=f'{LPR_RELATIVE_PATH}/detections',
                      language='en'):
    if language == 'fa':
        label = translate(label)

    elif language != 'en':
        raise Exception('Un-supported language!')

    path = f'{dir_path}/{relative_path}'

    train_transform = transforms.Compose([
        transforms.Resize((100, 400)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((100, 400)),
        transforms.ToTensor(),
    ])

    transform = train_transform if full_transform else test_transform
    image = Image.open(f'{path}/{image_path}').convert("RGB")
    image = transform(image)

    for key, value in persian_letter_normalization.items():
        label = label.replace(key, value)


    if log:
        print(label)
        for i , c in enumerate(label):
            if c in persian_letters:
                c = '*'
            print(f"{i}:{c}", end=' | ')
        print()


    digits = torch.tensor([digit_to_idx[char] for char in label if char.isdigit()])
    letter = letter_to_idx[label[2]]

    return image, digits, letter


    # Utility function to plot an image
def plot_image(img, title, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.axis(False)
    plt.show()


# Utility function to print final result of LPD model:
def final_results_PLD(LPD_model):
    metrics = LPD_model.val()
    print("Final Results: ")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")


def render_persian_text_on_image(image, text, position, font_size=40, color=(255, 2, 2)):
    """
    Render Persian text on an image with proper text shaping and direction
    
    Args:
        image: PIL Image object
        text: Persian text string
        position: (x, y) tuple for text position
        font_size: Font size (default: 40)
        color: RGB color tuple (default: red)
    
    Returns:
        PIL Image with Persian text rendered
    """
    try:
        # Load Persian font
        font_path = os.path.join(BASE_PATH, 'fonts', 'Vazirmatn-Medium.ttf')
        if not os.path.exists(font_path):
            print(f"Warning: Font file not found at {font_path}")
            # Try alternative font path
            font_path = 'fonts/Vazirmatn-Medium.ttf'
            if not os.path.exists(font_path):
                print(f"Warning: Alternative font path also not found: {font_path}")
                return image
        
        font = ImageFont.truetype(font_path, font_size, encoding='unic')
        draw = ImageDraw.Draw(image)
        
        # Reshape Persian text for proper display
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        
        # Debug print
        print(f"Rendering text: '{text}' -> '{reshaped_text}' -> '{bidi_text}' at position {position}")
        
        # Draw the text
        draw.text(position, bidi_text, color, font=font)
        
        return image
        
    except Exception as e:
        print(f"Error rendering Persian text: {e}")
        import traceback
        traceback.print_exc()
        return image


def convert_to_persian_digits(text):
    """
    Convert English digits to Persian digits for better display
    """
    persian_digits = {
        '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
        '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
    }
    
    result = ""
    for char in text:
        if char in persian_digits:
            result += persian_digits[char]
        else:
            result += char
    
    return result


def display_persian_text_example():
    """
    Example function to demonstrate Persian text rendering
    """
    try:
        # Create a sample image or load an existing one
        file_name = 'hesam.jpg'
        if os.path.exists(file_name):
            image = Image.open(file_name)
        else:
            # Create a blank image if file doesn't exist
            image = Image.new('RGB', (400, 200), color='white')
        
        # Example Persian text
        text = 'سلام من امین هستم!'
        
        # Render Persian text
        image_with_text = render_persian_text_on_image(image, text, (50, 100))
        
        # Display debug information
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        print(f"""
{text} 
{reshaped_text} 
{bidi_text}
bidi without reshape: {get_display(text)}""")
        
        return image_with_text
        
    except Exception as e:
        print(f"Error in Persian text example: {e}")
        return None


# Function to plot saved images by ultralytics during training and validation of LPD Model:
def plot_LPD_info(train_info=True, val_info=False):
    train_results_path = os.path.join(BASE_PATH, "runs/detect/train/")
    val_results_path = os.path.join(BASE_PATH, "runs/detect/val/")
    results_img_paths = []
    if train_info:
        results_img_paths.extend([os.path.join(train_results_path, f) for f in os.listdir(train_results_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if val_info:
        results_img_paths.extend([os.path.join(val_results_path, f) for f in os.listdir(val_results_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
         
    for img_path in results_img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_title = img_path.split("/")[-1].split(".")[0]
        plot_image(img, img_title)


# Utility function to plot recorded log by ultralytics during training and validation:
def plot_LPD_training_info():
     results_path = os.path.join(BASE_PATH, "runs/detect/train/results.csv")
     results_csv = pd.read_csv(results_path)
     epochs = len(results_csv)
     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 14))

     ax[0].plot(results_csv['epoch'], results_csv['train/box_loss'], label='Train: Box Loss', color="red")
     ax[0].plot(results_csv['epoch'], results_csv['train/cls_loss'], label='Train: Classification Loss', color="green")
     ax[0].plot(results_csv['epoch'], results_csv['train/dfl_loss'], label='Train: DFL Loss', color="blue")
     ax[0].plot(results_csv['epoch'], results_csv['val/box_loss'], label='Validation: Box Loss', color="red", alpha=0.5)
     ax[0].plot(results_csv['epoch'], results_csv['val/cls_loss'], label='Validation: Classification Loss', color="green", alpha=0.5)
     ax[0].plot(results_csv['epoch'], results_csv['val/dfl_loss'], label='Validation: DFL Loss', color="blue", alpha=0.5)
     ax[0].set_xlabel('Epoch')
     ax[0].set_ylabel('Loss')
     ax[0].set_title('Train Loss over Epochs')
     ax[0].set_xlim([1, epochs])
     ax[0].grid()
     ax[0].legend()

     ax[1].plot(results_csv['epoch'], results_csv['metrics/precision(B)'], label='Precision')
     ax[1].plot(results_csv['epoch'], results_csv['metrics/recall(B)'], label='Recall')
     ax[1].set_xlabel('Epoch')
     ax[1].set_ylabel('Precision / Recall')
     ax[1].set_xlim([1, epochs])
     ax[1].grid()
     ax[1].legend()
     
     ax[2].plot(results_csv['epoch'], results_csv['metrics/mAP50(B)'], label=' mAP50')
     ax[2].plot(results_csv['epoch'], results_csv['metrics/mAP50-95(B)'], label='mAP50-95')
     ax[2].set_xlabel('Epoch')
     ax[2].set_ylabel('mAP')
     ax[2].set_xlim([1, epochs])
     ax[2].grid()
     ax[2].legend()
     plt.tight_layout()
     plt.show()
     

# Function to get random image and label from splitted test set
def get_random_test_image():
    test_images = sorted([f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    test_labels = sorted([f for f in os.listdir(val_labels_dir) if f.endswith(('.txt'))])

    idx = random.randint(0, len(test_images) - 1)
    
    image_path = os.path.join(val_images_dir, test_images[idx])
    label_path = os.path.join(val_labels_dir, test_labels[idx])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with open(label_path, 'r') as file:
        label = file.read()
        
    return image, label



class FCNPLPRModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0"):
        super(FCNPLPRModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier[1] = nn.Identity()
        
        self.digit_classifier = nn.Linear(1280, 7 * 10)
        self.letter_classifier = nn.Linear(1280, len(persian_letters))
        
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        features = self.backbone(x)
        
        digit_outputs = self.digit_classifier(features)
        digit_outputs = digit_outputs.view(digit_outputs.size(0), 7, 10)
        digit_outputs = self.softmax(digit_outputs) # digit_outputs  -> [batch_size, 7, 10]
        
        letter_output = self.letter_classifier(features)
        letter_output = self.softmax(letter_output) # letter_output -> [batch_size, num_persian_letters]
        return digit_outputs, letter_output

def decode_predictions(digit_outputs, letter_output):
    digit_predictions = torch.argmax(digit_outputs, dim=2)  # [batch_size, 7]
    batch_size = digit_predictions.size(0)

    digits = []
    for b in range(batch_size):  # Iterate over batch
        sample_digits = []
        for i in range(7):  # Iterate over 7 digits
            try:
                sample_digits.append(idx_to_digit[digit_predictions[b][i].item()])
            except KeyError as e:
                print(f"KeyError: {e} (digit_predictions[{b}][{i}] = {digit_predictions[b][i].item()})")
                sample_digits.append("?")  # Use a placeholder for invalid indices
        digits.append(sample_digits)

    letter_prediction = torch.argmax(letter_output, dim=1)  # [batch_size]
    letters = []
    for b in range(batch_size):
        try:
            letters.append(idx_to_letter[letter_prediction[b].item()])
        except KeyError as e:
            print(f"KeyError: {e} (letter_prediction[{b}] = {letter_prediction[b].item()})")
            letters.append("?")  # Use a placeholder for invalid indices

    labels = []
    for b in range(batch_size):
        label = "".join(digits[b][:2]) + letters[b] + "".join(digits[b][2:])
        labels.append(label)

    return labels



def evaluate_misclassification(gt, pred):
    assert len(gt) == len(pred), "GT and Pred must have the same length."

    # Collect misclassified characters
    misclassified = []
    char_error_count = 0

    for i, (gt_char, pred_char) in enumerate(zip(gt, pred)):
        if gt_char != pred_char:
            misclassified.append(f"{gt_char} with {pred_char} at pos {i}")
            char_error_count += 1

    # Format the output
    misclassified_str = " , ".join(misclassified) if misclassified else "None"
    result = (f"GT: {gt} | Pred: {pred} | "
              f"Misclassified: {misclassified_str} | "
              f"Char error_count: {char_error_count}")

    return result , char_error_count




def calculate_accuracy(model, df, device, dir_path=BASE_PATH, relative_path=f'{LPR_RELATIVE_PATH}/detections', language='en', log=False , cer = False):
    model.eval()
    FP= []
    CE = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), ncols=100):
            image_path = row["image_path"]
            label = row["label"]
            try:
                image, true_digits, true_letter = preprocess_sample(image_path, label, full_transform=False, dir_path=dir_path, relative_path=relative_path, language=language)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
            true_label = "".join([digit_vocabulary[d] for d in true_digits[:2]]) \
                         + persian_letters[true_letter] \
                         + "".join([digit_vocabulary[d] for d in true_digits[2:]])
            image = image.unsqueeze(0).to(device)
            digit_outputs, letter_output = model(image)

            predicted_labels = decode_predictions(digit_outputs, letter_output)

            if predicted_labels[0] == true_label:
                correct += 1
            elif log:
                report , char_error_count = evaluate_misclassification(true_label,predicted_labels[0])
                CE += char_error_count
                FP.append(report)
            total += 1


    accuracy = correct / total * 100

    if cer:
        print(f'CER: {100* CE/(8*df.shape[0]): .4f}%')

    if log:
         for false_positive in FP:
            print(false_positive)



    return accuracy


# Define the base path where your model is located
BASE_PATH = "."

# Path to the saved model
LPR_model_path = os.path.join(BASE_PATH, "PLPR-CNN.pth")

# Create the model instance with the same architecture
LPR_model_final = FCNPLPRModel().to(device)

# Load the saved weights with map_location to handle CPU/GPU mismatch
LPR_model_final.load_state_dict(torch.load(LPR_model_path, map_location=device))
LPR_model_final.eval()

LPD_model_path = os.path.join(BASE_PATH, "yolo12n_trained.pt")


LPD_model_final = YOLO(LPD_model_path)


COLOR = (0, 255, 0)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1

# Load the license plate database
def load_license_plate_database():
    """Load the license plate database from CSV file"""
    try:
        db_path = os.path.join(BASE_PATH, "license_plate_database.csv")
        if os.path.exists(db_path):
            return pd.read_csv(db_path)
        else:
            print("Warning: license_plate_database.csv not found. Creating sample database...")
            # Create sample database if file doesn't exist
            sample_data = {
                'plate_number': ['11ل111', '12ل222', '13ل333'],
                'owner_name': ['علی احمدی', 'فاطمه محمدی', 'حسن رضایی'],
                'vehicle_model': ['پژو 206', 'سمند', 'پژو 405'],
                'vehicle_color': ['سفید', 'مشکی', 'آبی'],
                'registration_date': ['1400/01/15', '1399/06/20', '1401/03/10'],
                'phone_number': ['09123456789', '09187654321', '09351234567']
            }
            df = pd.DataFrame(sample_data)
            df.to_csv(db_path, index=False)
            return df
    except Exception as e:
        print(f"Error loading database: {e}")
        return pd.DataFrame()

def lookup_license_plate(plate_number, database):
    """Look up license plate information in the database with improved matching"""
    if database.empty:
        return None
    
    # Search for exact match
    match = database[database['plate_number'] == plate_number]
    if not match.empty:
        return match.iloc[0].to_dict()
    
    # If no exact match, try to find similar plates (for cases with recognition errors)
    # This is a more flexible matching system for Persian license plates
    for _, row in database.iterrows():
        db_plate = row['plate_number']
        
        # Check if lengths match
        if len(db_plate) == len(plate_number):
            # Count matching characters
            matches = sum(1 for a, b in zip(db_plate, plate_number) if a == b)
            similarity = matches / len(db_plate)
            
            # Allow up to 2 character differences for better matching
            if similarity >= 0.7:  # 70% similarity threshold
                print(f"🔍 Fuzzy match found: '{plate_number}' ~ '{db_plate}' (similarity: {similarity:.2f})")
                return row.to_dict()
    
    # Also try to match with different Persian letter variations
    # Some plates might use different Persian letters (ب, ل, etc.)
    for _, row in database.iterrows():
        db_plate = row['plate_number']
        
        # Normalize Persian letters for comparison
        normalized_db = db_plate.replace('ب', 'ل').replace('ل', 'ب')
        normalized_input = plate_number.replace('ب', 'ل').replace('ل', 'ب')
        
        if normalized_db == normalized_input:
            print(f"🔍 Letter variation match found: '{plate_number}' ~ '{db_plate}'")
            return row.to_dict()
    
    return None

def draw_license_plate_overlay(frame, bbox, plate_number, confidence, owner_info=None):
    """Draw a professional-looking license plate overlay with owner information"""
    x1, y1, x2, y2 = bbox
    
    # Draw main bounding box with corner markers
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    
    # Draw corner markers for better visibility
    corner_length = 20
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), (0, 255, 0), 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), (0, 255, 0), 3)
    # Top-right corner
    cv2.line(frame, (x2 - corner_length, y1), (x2, y1), (0, 255, 0), 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), (0, 255, 0), 3)
    # Bottom-left corner
    cv2.line(frame, (x1, y2 - corner_length), (x1, y2), (0, 255, 0), 3)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), (0, 255, 0), 3)
    # Bottom-right corner
    cv2.line(frame, (x2 - corner_length, y2), (x2, y2), (0, 255, 0), 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), (0, 255, 0), 3)
    
    # Display plate number above the rectangle
    plate_y_above = y1 - 20  # Position above the rectangle
    
    # Convert frame to PIL Image for Persian text rendering
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Convert plate number to Persian digits and render above rectangle
    persian_plate_number = convert_to_persian_digits(plate_number)
    plate_text_above = f"پلاک: {persian_plate_number}"
    
    # Debug print
    print(f"Rendering plate number: {plate_text_above} at position ({x1}, {plate_y_above})")
    
    pil_image = render_persian_text_on_image(pil_image, plate_text_above, (x1, plate_y_above), font_size=30, color=(0, 255, 0))
    
    # Convert back to OpenCV format
    frame_with_plate = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Copy the plate number area back to the original frame
    # Increase the area to ensure we capture the text properly
    text_height = 40  # Increased height to capture text
    text_width = 300  # Increased width to capture text
    frame[plate_y_above-text_height//2:plate_y_above+text_height//2, x1:x1+text_width] = frame_with_plate[plate_y_above-text_height//2:plate_y_above+text_height//2, x1:x1+text_width]
    
    if owner_info:
        # Plate found in database - green theme
        text_y_start = y1 - 100
        # Draw background rectangle for text (reduced height since plate number is above rectangle)
        cv2.rectangle(frame, (x1-5, text_y_start-5), (x1+350, y1+5), (0, 100, 0), -1)
        cv2.rectangle(frame, (x1-5, text_y_start-5), (x1+350, y1+5), (0, 255, 0), 2)
        
        # Add a small icon/indicator
        cv2.circle(frame, (x1-15, text_y_start+35), 8, (0, 255, 0), -1)
        cv2.putText(frame, "✓", (x1-20, text_y_start+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display owner information with Persian text rendering
        display_plate = translate_for_display(plate_number)
        
        # Convert frame to PIL Image for Persian text rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Render Persian text for owner information with better spacing
        persian_texts = [
            (f"مالک: {owner_info['owner_name']}", (x1, text_y_start+15)),
            (f"خودرو: {owner_info['vehicle_model']}", (x1, text_y_start+40)),
            (f"رنگ: {owner_info['vehicle_color']}", (x1, text_y_start+65))
        ]
        
        for text, pos in persian_texts:
            pil_image = render_persian_text_on_image(pil_image, text, pos, font_size=25, color=(255, 255, 255))
        
        # Plate number is now displayed above the rectangle, so we don't need it here
        
        # Convert back to OpenCV format
        frame_with_persian = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Copy the Persian text areas back to the original frame
        frame[text_y_start-5:y1+5, x1-5:x1+350] = frame_with_persian[text_y_start-5:y1+5, x1-5:x1+350]
        
        # Add confidence score in English (smaller and at the bottom)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, text_y_start+115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        
    else:
        # Plate not found in database - red theme
        text_y_start = y1 - 60
        # Draw background rectangle for text (reduced height since plate number is above rectangle)
        cv2.rectangle(frame, (x1-5, text_y_start-5), (x1+350, y1+5), (0, 0, 100), -1)
        cv2.rectangle(frame, (x1-5, text_y_start-5), (x1+350, y1+5), (0, 0, 255), 2)
        
        # Add a small icon/indicator
        cv2.circle(frame, (x1-15, text_y_start+25), 8, (0, 0, 255), -1)
        cv2.putText(frame, "✗", (x1-20, text_y_start+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display "not found" message with Persian text
        display_plate = translate_for_display(plate_number)
        
        # Convert frame to PIL Image for Persian text rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Render Persian text for "not found" message with better spacing
        not_found_text = "در پایگاه داده یافت نشد"
        pil_image = render_persian_text_on_image(pil_image, not_found_text, (x1, text_y_start+40), font_size=25, color=(255, 255, 255))
        
        # Plate number is now displayed above the rectangle, so we don't need it here
        
        # Convert back to OpenCV format
        frame_with_persian = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Copy the Persian text areas back to the original frame
        frame[text_y_start-5:y1+5, x1-5:x1+350] = frame_with_persian[text_y_start-5:y1+5, x1-5:x1+350]
        
        # Add confidence score in English (smaller and at the bottom)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, text_y_start+65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

# End-to-End model for Licence Plate Detection and Prediction
class E2E_LPDR(nn.Module):
    def __init__(self, LPD_model, LPR_model):
        super(E2E_LPDR, self).__init__()
        self.LPD_model = LPD_model
        self.LPR_model = LPR_model
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 400)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

    def forward(self, x):
        print("Processing: Licence Plate Detection Phase...")
        results = self.LPD_model(x)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        conf = results.boxes.conf.cpu().numpy()
        
        print(f"🔍 YOLO detected {len(boxes)} potential license plates")
        
        if len(boxes) == 0:
            print("❌ No license plates detected by YOLO model")
            return []
        
        print("\nProcessing: Licence Plate Recognition Phase...")
        self.LPR_model.eval()
        detected_plates = []
        
        for i, (box, proba) in enumerate(zip(boxes, conf)):
            print(f"📝 Processing detection {i+1}/{len(boxes)} (confidence: {proba:.2f})")
            x1, y1, x2, y2 = box
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            licence_plate_img = x[y1:y2, x1:x2]
            licence_plate_img_tensor = self.test_transform(licence_plate_img).to(device)
            
            with torch.no_grad():  
                digits, letter = self.LPR_model.forward(licence_plate_img_tensor)
                
            digits_tensor = digits.clone().cpu().squeeze()
            letter_tensor = letter.clone().cpu().squeeze()

            digits_idx = torch.argmax(digits_tensor, dim=1).numpy().tolist()
            letter_idx = torch.argmax(letter_tensor, dim=0).numpy().tolist()

            output = {
                'first_2_digits': digits_idx[:2],
                'middle_3_digits': digits_idx[2:5],
                'last_2_digits': digits_idx[5:],
                'letter': persian_letters[letter_idx],
                'confidence': proba,
                'bbox': box
            }
            
            # Construct the full plate number
            plate_number = "".join([str(d) for d in digits_idx[:2]]) + persian_letters[letter_idx] + "".join([str(d) for d in digits_idx[2:5]]) + "".join([str(d) for d in digits_idx[5:]])
            output['full_plate_number'] = plate_number
            
            detected_plates.append(output)
            print(f"✅ Detected Plate: {plate_number} (Confidence: {proba:.2f}) at bbox: {box}")
            
        print(f"🎯 Total plates processed: {len(detected_plates)}")
        return detected_plates

def run_webcam_detection():
    """Run real-time license plate detection using webcam"""
    print("Starting webcam license plate detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Load the database
    database = load_license_plate_database()
    if database.empty:
        print("Error: Could not load license plate database")
        return
    
    print(f"Loaded database with {len(database)} license plates")
    
    # Initialize webcam
    print("🔍 Attempting to open webcam...")
    cap = cv2.VideoCapture(0)
    
    # Try different webcam indices if 0 doesn't work
    if not cap.isOpened():
        print("⚠️  Webcam index 0 failed, trying index 1...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("⚠️  Webcam index 1 failed, trying index 2...")
        cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam on any index")
        print("💡 Troubleshooting tips:")
        print("   - Make sure your webcam is connected and not in use by another application")
        print("   - Try running as administrator")
        print("   - Check if your webcam drivers are properly installed")
        return
    
    print("✅ Webcam opened successfully!")
    
    # Set webcam properties
    print("⚙️  Setting webcam properties...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get actual webcam properties
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 Webcam resolution: {actual_width}x{actual_height}")
    print(f"🎬 Webcam FPS: {fps}")
    
    # Test reading a frame
    ret, test_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read test frame from webcam")
        cap.release()
        return
    else:
        print(f"✅ Test frame read successfully: {test_frame.shape}")
    
    # Initialize the LPDR model
    print("🤖 Initializing LPDR model...")
    LPDR_model = E2E_LPDR(LPD_model_final, LPR_model_final)
    print("✅ LPDR model initialized successfully!")
    
    frame_count = 0
    last_detection_time = 0
    detection_cooldown = 2.0  # seconds between detections
    detected_plates = []  # Initialize detected_plates variable
    
    print("🚀 Starting webcam detection loop...")
    print("💡 Press 'q' to quit, 's' to save frame")
    print("⏱️  Detection runs every 2 seconds")
    print("=" * 50)
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from webcam")
                print("💡 Trying to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Failed to reconnect to webcam")
                    break
                continue
            
            # No flip - keep original orientation for better recognition
            # frame = cv2.flip(frame, 1)  # Commented out to fix recognition
            # Note: Horizontal flip was causing recognition issues because
            # the model was trained on non-mirrored images
            
            # Add orientation indicator for debugging
            cv2.putText(frame, "ORIGINAL ORIENTATION", (frame.shape[1] - 300, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Get current time first
            current_time = time.time()
            
            # Add frame counter and instructions with Persian text
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert frame to PIL Image for Persian text rendering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Render Persian instructions
            persian_instructions = "برای خروج q و برای ذخیره s را فشار دهید"
            pil_image = render_persian_text_on_image(pil_image, persian_instructions, (10, 60), font_size=20, color=(255, 255, 255))
            
            # Render Persian status messages
            if current_time - last_detection_time > detection_cooldown:
                # Detection is ready
                cv2.circle(frame, (frame.shape[1] - 50, 30), 15, (0, 255, 0), -1)
                status_text = "آماده تشخیص"
                pil_image = render_persian_text_on_image(pil_image, status_text, (frame.shape[1] - 200, 50), font_size=25, color=(0, 255, 0))
            else:
                # Detection is processing
                cv2.circle(frame, (frame.shape[1] - 50, 30), 15, (0, 165, 255), -1)
                status_text = "در حال پردازش"
                pil_image = render_persian_text_on_image(pil_image, status_text, (frame.shape[1] - 200, 50), font_size=25, color=(0, 165, 255))
            
            # Convert back to OpenCV format
            frame_with_persian = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Copy the Persian text areas back to the original frame
            frame[60:80, 10:400] = frame_with_persian[60:80, 10:400]  # Instructions area
            frame[50:75, frame.shape[1] - 200:frame.shape[1] - 50] = frame_with_persian[50:75, frame.shape[1] - 200:frame.shape[1] - 50]  # Status area
            
            # Add detection info
            if detected_plates:
                cv2.putText(frame, f"Plates: {len(detected_plates)}", (frame.shape[1] - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No plates detected", (frame.shape[1] - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process frame every few seconds to avoid overwhelming the model
            if current_time - last_detection_time > detection_cooldown:
                try:
                    # Convert BGR to RGB for the model
                    # Note: frame is now in original orientation (no flip) for better recognition
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run detection
                    new_detected_plates = LPDR_model(frame_rgb)
                    
                    # Update detected_plates only if we got new results
                    if new_detected_plates:
                        detected_plates = new_detected_plates
                        print(f"🔍 Detection completed: {len(detected_plates)} plate(s) found")
                    
                    last_detection_time = current_time
                    
                except Exception as e:
                    print(f"Error during detection: {e}")
                    last_detection_time = current_time
            
            # Always draw rectangles for previously detected plates (even when not detecting)
            if detected_plates:
                for plate_info in detected_plates:
                    plate_number = plate_info['full_plate_number']
                    confidence = plate_info['confidence']
                    bbox = plate_info['bbox']
                    
                    # Look up plate in database
                    owner_info = lookup_license_plate(plate_number, database)
                    
                    # Draw the professional overlay
                    draw_license_plate_overlay(frame, bbox, plate_number, confidence, owner_info)
                    
                    # Print information only once per detection cycle
                    if current_time - last_detection_time < 0.1:  # Print only right after detection
                        if owner_info:
                            print(f"\n=== LICENSE PLATE FOUND ===")
                            print(f"Plate Number: {plate_number}")
                            print(f"Owner: {owner_info['owner_name']}")
                            print(f"Vehicle: {owner_info['vehicle_model']}")
                            print(f"Color: {owner_info['vehicle_color']}")
                            print(f"Registration Date: {owner_info['registration_date']}")
                            print(f"Phone: {owner_info['phone_number']}")
                            print(f"Confidence: {confidence:.2f}")
                            print("=" * 30)
                        else:
                            print(f"\n=== UNKNOWN LICENSE PLATE ===")
                            print(f"Plate Number: {plate_number}")
                            print(f"Confidence: {confidence:.2f}")
                            print("This plate is not registered in our database.")
                            print("=" * 30)
            
            # Add detection statistics at the bottom with Persian text
            if detected_plates:
                # Count found vs not found plates
                found_count = sum(1 for plate in detected_plates if lookup_license_plate(plate['full_plate_number'], database) is not None)
                not_found_count = len(detected_plates) - found_count
                
                # Draw statistics box
                cv2.rectangle(frame, (10, frame.shape[0] - 80), (300, frame.shape[0] - 10), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, frame.shape[0] - 80), (300, frame.shape[0] - 10), (255, 255, 255), 2)
                
                # Convert frame to PIL Image for Persian text rendering
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Render Persian statistics text
                persian_stats = [
                    (f"تشخیص داده شده: {len(detected_plates)}", (20, frame.shape[0] - 60)),
                    (f"یافت شده در پایگاه: {found_count}", (20, frame.shape[0] - 40)),
                    (f"یافت نشده: {not_found_count}", (20, frame.shape[0] - 20))
                ]
                
                for text, pos in persian_stats:
                    color = (0, 255, 0) if "یافت شده" in text else (0, 0, 255) if "یافت نشده" in text else (255, 255, 255)
                    pil_image = render_persian_text_on_image(pil_image, text, pos, font_size=20, color=color)
                
                # Convert back to OpenCV format
                frame_with_persian = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Copy the Persian text areas back to the original frame
                frame[frame.shape[0] - 80:frame.shape[0] - 10, 10:300] = frame_with_persian[frame.shape[0] - 80:frame.shape[0] - 10, 10:300]
            
            # Display the frame
            cv2.imshow('License Plate Detection - Webcam', frame)
            
            # Handle key presses - FIXED QUIT FUNCTIONALITY
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("🔄 Quit key pressed. Exiting...")
                break
            elif key == ord('s') or key == ord('S'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"webcam_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            
            frame_count += 1
            
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            print("💡 Continuing...")
            continue
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

def test_single_image(image_path):
    """Test the LPDR model on a single image"""
    print(f"Testing image: {image_path}")
    
    # Load the database
    database = load_license_plate_database()
    if database.empty:
        print("Error: Could not load license plate database")
        return
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize the LPDR model
    LPDR_model = E2E_LPDR(LPD_model_final, LPR_model_final)
    
    # Run detection
    detected_plates = LPDR_model(image)
    
    if detected_plates:
        for plate_info in detected_plates:
            plate_number = plate_info['full_plate_number']
            confidence = plate_info['confidence']
            
            print(f"\n=== LICENSE PLATE DETECTED ===")
            print(f"Plate Number: {plate_number}")
            print(f"Confidence: {confidence:.2f}")
            
            # Look up plate in database
            owner_info = lookup_license_plate(plate_number, database)
            
            if owner_info:
                print(f"Owner: {owner_info['owner_name']}")
                print(f"Vehicle: {owner_info['vehicle_model']}")
                print(f"Color: {owner_info['vehicle_color']}")
                print(f"Registration Date: {owner_info['registration_date']}")
                print(f"Phone: {owner_info['phone_number']}")
            else:
                print("This plate is not registered in our database.")
            
            print("=" * 30)
    else:
        print("No license plates detected in the image.")

def test_detection_with_mock_data():
    """Test the detection system with mock data to verify drawing works"""
    print("🧪 Testing detection system with mock data...")
    
    # Create a mock frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some mock license plate detections
    mock_plates = [
        {
            'full_plate_number': '11ل111',
            'confidence': 0.95,
            'bbox': [100, 100, 300, 150]
        },
        {
            'full_plate_number': '12ل222',
            'confidence': 0.87,
            'bbox': [400, 200, 600, 250]
        }
    ]
    
    # Load database
    database = load_license_plate_database()
    
    # Draw overlays for mock plates
    for plate_info in mock_plates:
        plate_number = plate_info['full_plate_number']
        confidence = plate_info['confidence']
        bbox = plate_info['bbox']
        
        # Look up plate in database
        owner_info = lookup_license_plate(plate_number, database)
        
        # Draw the professional overlay
        draw_license_plate_overlay(frame, bbox, plate_number, confidence, owner_info)
    
    # Add some text to the frame
    cv2.putText(frame, "Mock Detection Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Press any key to continue", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Display the frame
    cv2.imshow('Mock Detection Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Mock detection test completed!")
    print("If you saw rectangles and text overlays, the drawing system is working correctly.")

# Main execution
if __name__ == "__main__":
    print("License Plate Detection and Recognition System")
    print("1. Run webcam detection")
    print("2. Test single image")
    print("3. Test detection drawing (mock data)")
    print("4. Test Persian text rendering")
    print("5. Exit")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == "1":
        run_webcam_detection()
    elif choice == "2":
        image_path = input("Enter image path (or press Enter for default '2.jpeg'): ").strip()
        if not image_path:
            image_path = "2.jpeg"
        test_single_image(image_path)
    elif choice == "3":
        test_detection_with_mock_data()
    elif choice == "4":
        print("Testing Persian text rendering...")
        result_image = display_persian_text_example()
        if result_image:
            print("Persian text rendered successfully!")
            # Save the result image
            result_image.save("persian_text_test_result.jpg")
            print("Result saved as 'persian_text_test_result.jpg'")
        else:
            print("Failed to render Persian text")
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")
