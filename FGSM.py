import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

print("Starting script...")

print("Loading MobileNetV2 model...")
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
print("Model loaded successfully")

def preprocess(image):
    print("Preprocessing image...")
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]  
    print("Preprocessing complete")
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    print("Creating adversarial pattern...")
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    print("Adversarial pattern created")
    return signed_grad

IMAGE_DIR = 'data/cityscapes/train/img'  # Directory containing training images
OUTPUT_DIR = 'adversarial_images'  # Local directory for output images

print(f"Setting up directories...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)  # Create the input directory structure if it doesn't exist
print(f"Directories created/verified")

print(f"Looking for images in: {os.path.abspath(IMAGE_DIR)}")
image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.png'))
print(image_paths)

if not image_paths:
    image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    print(f"Found {len(image_paths)} jpg images")
else:
    print(f"Found {len(image_paths)} png images")

if not image_paths:
    print("No images found in the images directory!")
    print("Please add some images to the 'images' directory")
    exit(1)

print(f"Found images: {image_paths}")
epsilons = [0, 0.01, 0.1, 0.15]

for image_path in image_paths:
    print(f"\nProcessing image: {image_path}")
    try:
        print("Reading image file...")
        image_raw = tf.io.read_file(image_path)
        print("Decoding image...")
        image_decoded = tf.image.decode_image(image_raw, channels=3)
        image_decoded.set_shape([None, None, 3])
        print("Preprocessing image...")
        image_preprocessed = preprocess(image_decoded)
        print("Image preprocessing complete")

        print("Getting model predictions...")
        probs = pretrained_model.predict(image_preprocessed)
        predicted_class = tf.argmax(probs, axis=-1).numpy()[0]
        num_classes = probs.shape[-1]
        label = tf.one_hot(predicted_class, num_classes)
        label = tf.reshape(label, (1, num_classes))
        print(f"Predicted class: {predicted_class}")
        
        print("Creating adversarial perturbations...")
        perturbations = create_adversarial_pattern(image_preprocessed, label)
        print("Perturbations created")
        
        # Save original image
        print("Saving original image...")
        orig_np = image_preprocessed[0].numpy()
        orig_np = ((orig_np + 1) / 2.0) * 255.0
        orig_np = tf.cast(orig_np, tf.uint8).numpy()
        orig_pil = Image.fromarray(orig_np)
        base_name = os.path.basename(image_path)
        orig_file_name = os.path.splitext(base_name)[0] + '_original.png'
        orig_save_path = os.path.join(OUTPUT_DIR, orig_file_name)
        orig_pil.save(orig_save_path)
        print(f"Saved original image: {orig_save_path}")
        
        for eps in epsilons:
            print(f"\nProcessing epsilon = {eps}")
            adv_image = image_preprocessed + eps * perturbations
            adv_image = tf.clip_by_value(adv_image, -1, 1)
            
            print("Converting and saving adversarial image...")
            adv_np = adv_image[0].numpy()
            adv_np = ((adv_np + 1) / 2.0) * 255.0
            adv_np = tf.cast(adv_np, tf.uint8).numpy()
            adv_pil = Image.fromarray(adv_np)
            file_name = os.path.splitext(base_name)[0] + f'_eps{eps:.3f}.png'
            save_path = os.path.join(OUTPUT_DIR, file_name)
            adv_pil.save(save_path)
            print(f"Saved adversarial image: {save_path}")
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        continue

print("\nScript completed!")
