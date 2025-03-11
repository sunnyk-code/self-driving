import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, Concatenate

print("Starting script...")

# Define Cityscapes mappings and constants
NUM_CLASSES = 19
ignore_index = 250  # Used for void classes

# Cityscapes classes mapping
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = [
    "unlabelled",
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

# Official Cityscapes colors (same order as class_names)
colors = [
    [0, 0, 0],        # unlabelled
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32]     # bicycle
]

# Create dictionary mapping valid classes to training IDs (0-18)
class_map = dict(zip(valid_classes, range(NUM_CLASSES)))

def encode_segmap(mask):
    """Convert Cityscapes labelIds to training labels."""
    print(f"Input mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Unique values in input mask: {np.unique(mask)}")
    
    # If mask is RGB, convert to single channel since Cityscapes labels are single-channel
    if len(mask.shape) == 3 and mask.shape[-1] == 3:
        print("Converting RGB mask to single channel")
        # Take first channel since Cityscapes label images have same values in all channels
        mask = mask[..., 0]
    
    # Initialize with ignore_index
    label_mask = np.ones_like(mask, dtype=np.int32) * ignore_index
    
    # Map valid classes to their training IDs
    for valid_class, train_id in class_map.items():
        label_mask[mask == valid_class] = train_id
    
    print(f"Unique IDs after mapping: {np.unique(label_mask)}")
    print(f"Output mask shape: {label_mask.shape}")
    return label_mask

def create_colormap(pred):
    """Convert segmentation prediction to RGB colormap using Cityscapes colors."""
    print(f"Creating colormap for prediction shape: {pred.shape}")
    print(f"Unique values in prediction: {np.unique(pred)}")
    
    r = pred.copy()
    g = pred.copy()
    b = pred.copy()
    
    # Set colors for each class
    for l in range(NUM_CLASSES):
        r[pred == l] = colors[l + 1][0]  # l + 1 to skip unlabelled
        g[pred == l] = colors[l + 1][1]
        b[pred == l] = colors[l + 1][2]
    
    # Set color for ignore_index/void
    r[pred == ignore_index] = colors[0][0]
    g[pred == ignore_index] = colors[0][1]
    b[pred == ignore_index] = colors[0][2]
    
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    print(f"Output colormap shape: {rgb.shape}")
    return rgb

def preprocess_mask(mask):
    """Preprocess segmentation mask following Cityscapes format."""
    print("Preprocessing mask...")
    if isinstance(mask, tf.Tensor):
        mask = mask.numpy()
    
    print(f"Original mask shape: {mask.shape}")
    
    # Convert labelIds to trainIds
    label_mask = encode_segmap(mask)
    print(f"After encoding - unique values: {np.unique(label_mask)}")
    
    # Convert to tensor for resizing
    mask = tf.convert_to_tensor(label_mask, dtype=tf.int32)
    
    # Resize using nearest neighbor to preserve label values
    mask = tf.image.resize(mask[..., tf.newaxis], [512, 1024], method='nearest')
    print(f"After resize shape: {mask.shape}")
    
    # Remove extra dimensions
    mask = tf.squeeze(mask)
    print(f"After squeeze shape: {mask.shape}")
    
    # Convert to one-hot, excluding ignore_index
    mask = tf.one_hot(
        tf.where(mask == ignore_index, 0, mask),  # Replace ignore_index with 0 temporarily
        depth=NUM_CLASSES,
        dtype=tf.float32
    )
    print(f"After one-hot shape: {mask.shape}")
    
    # Create valid pixels mask with same shape as one-hot encoded mask
    valid_pixels = tf.cast(tf.not_equal(label_mask, ignore_index), tf.float32)
    valid_pixels = tf.image.resize(valid_pixels[..., tf.newaxis], [512, 1024], method='nearest')
    valid_pixels = tf.squeeze(valid_pixels)[..., tf.newaxis]  # Shape: [512, 1024, 1]
    print(f"Valid pixels shape: {valid_pixels.shape}")
    
    # Broadcast valid_pixels to match one-hot mask shape
    valid_pixels = tf.tile(valid_pixels, [1, 1, NUM_CLASSES])
    print(f"Broadcasted valid pixels shape: {valid_pixels.shape}")
    
    # Zero out the one-hot vectors corresponding to ignore_index
    mask = mask * valid_pixels
    
    # Add batch dimension
    mask = mask[tf.newaxis, ...]
    print(f"Final shape: {mask.shape}")
    
    return mask

def upsample_block(filters, size):
    """Upsampling block for the decoder"""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential([
        UpSampling2D(size=2, interpolation='bilinear'),
        Conv2D(filters, size, padding='same', kernel_initializer=initializer, use_bias=False),
        BatchNormalization(),
        ReLU()
    ])
    return result

def get_model():
    # Use MobileNetV2 as base model (better for real-time applications)
    base_model = tf.keras.applications.MobileNetV2(input_shape=[512, 1024, 3], include_top=False)
    
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 256x512
        'block_3_expand_relu',   # 128x256
        'block_6_expand_relu',   # 64x128
        'block_13_expand_relu',  # 32x64
        'block_16_project',      # 16x32
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    # Create the upsampling path
    up_stack = [
        upsample_block(512, 3),  # 32x64
        upsample_block(256, 3),  # 64x128
        upsample_block(128, 3),  # 128x256
        upsample_block(64, 3),   # 256x512
    ]

    inputs = Input(shape=[512, 1024, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    # This is the last layer of the model (now with upsampling)
    x = Conv2D(NUM_CLASSES, 3, padding='same')(x)
    x = UpSampling2D(size=2, interpolation='bilinear')(x)  # Final upsampling to match input resolution

    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    # Load pre-trained weights if available
    if os.path.exists('cityscapes_weights.h5'):
        print("Loading pre-trained Cityscapes weights...")
        model.load_weights('cityscapes_weights.h5')
    else:
        print("Warning: No pre-trained weights found. Model will use random initialization.")
    
    return model

print("Loading segmentation model...")
pretrained_model = get_model()
print("Model loaded successfully")

def preprocess(image):
    print("Preprocessing image...")
    image = tf.cast(image, tf.float32)
    # First convert to tensor and ensure shape is known
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image)
    
    # Print original shape for debugging
    print(f"Original image shape: {image.shape}")
    
    # Resize with explicit method
    image = tf.image.resize(image, [512, 1024], method='bilinear')
    print(f"After resize shape: {image.shape}")
    
    # Normalize to [-1, 1]
    image = image / 127.5 - 1
    
    # Add batch dimension if not present
    if len(image.shape) == 3:
        image = image[tf.newaxis, ...]
    
    print(f"Final preprocessed shape: {image.shape}")
    return image

# Combined loss: Dice Loss + Cross Entropy
def segmentation_loss(y_true, y_pred):
    """Cross entropy loss with proper handling of ignore_index."""
    print(f"Loss calculation - y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    
    # Get the mask of valid pixels (not ignore_index)
    valid_mask = tf.reduce_any(y_true > 0, axis=-1)
    
    # Flatten the predictions and targets
    y_pred_flat = tf.reshape(y_pred, [-1, NUM_CLASSES])
    y_true_flat = tf.reshape(y_true, [-1, NUM_CLASSES])
    valid_mask_flat = tf.reshape(valid_mask, [-1])
    
    # Only compute loss on valid pixels
    y_pred_valid = tf.boolean_mask(y_pred_flat, valid_mask_flat)
    y_true_valid = tf.boolean_mask(y_true_flat, valid_mask_flat)
    
    # Compute cross entropy loss
    ce_loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y_true_valid, 
            y_pred_valid,
            from_logits=True
        )
    )
    
    return ce_loss

def create_adversarial_pattern(input_image, target_mask):
    print("Creating adversarial pattern...")
    print(f"Input image shape: {input_image.shape}")
    print(f"Target mask shape: {target_mask.shape}")
    
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        print(f"Model prediction shape: {prediction.shape}")
        loss = segmentation_loss(target_mask, prediction)
        
    gradient = tape.gradient(loss, input_image)
    print(f"Gradient shape: {gradient.shape}")
    signed_grad = tf.sign(gradient)
    print(f"Final perturbation shape: {signed_grad.shape}")
    
    # Ensure shapes match
    if signed_grad.shape != input_image.shape:
        print("Shape mismatch detected!")
        signed_grad = tf.image.resize(signed_grad, [input_image.shape[1], input_image.shape[2]], method='bilinear')
        signed_grad = tf.reshape(signed_grad, input_image.shape)
        print(f"Reshaped perturbation to: {signed_grad.shape}")
    
    print("Adversarial pattern created")
    return signed_grad

IMAGE_DIR = 'data/cityscapes/train/img'
MASK_DIR = 'data/cityscapes/train/label'  # Changed from 'mask' to 'label' to match directory structure
OUTPUT_DIR = 'adversarial_images'

def process_all_images():
    print(f"Setting up directories...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    print(f"Directories created/verified")

    print(f"Looking for images in: {os.path.abspath(IMAGE_DIR)}")
    image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.png'))
    if not image_paths:
        image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))

    if not image_paths:
        print("No images found in the images directory!")
        exit(1)

    print(f"Found {len(image_paths)} images")
    epsilons = [0, 0.01, 0.1, 0.15]

    for image_path in image_paths:
        print(f"\nProcessing image: {image_path}")
        try:
            # Get corresponding mask path
            mask_path = image_path.replace(IMAGE_DIR, MASK_DIR)
            if not os.path.exists(mask_path):
                print(f"No corresponding mask found for {image_path}")
                continue

            print("Reading image and mask files...")
            image_raw = tf.io.read_file(image_path)
            mask_raw = tf.io.read_file(mask_path)
            
            print("Decoding image and mask...")
            image_decoded = tf.image.decode_image(image_raw, channels=3)
            mask_decoded = tf.image.decode_image(mask_raw, channels=3)  # Changed to 3 channels for RGB masks
            
            image_decoded.set_shape([None, None, 3])
            mask_decoded.set_shape([None, None, 3])  # Updated shape for RGB mask
            
            print("Preprocessing image and mask...")
            image_preprocessed = preprocess(image_decoded)
            mask_preprocessed = preprocess_mask(mask_decoded)
            
            print("Creating adversarial perturbations...")
            perturbations = create_adversarial_pattern(image_preprocessed, mask_preprocessed)
            
            # Save original image
            print("Saving original image...")
            orig_np = ((image_preprocessed[0].numpy() + 1) * 127.5).astype(np.uint8)
            orig_pil = Image.fromarray(orig_np)
            base_name = os.path.basename(image_path)
            orig_file_name = os.path.splitext(base_name)[0] + '_original.png'
            orig_save_path = os.path.join(OUTPUT_DIR, orig_file_name)
            orig_pil.save(orig_save_path)
            
            # Save ground truth mask with colors
            mask_np = tf.argmax(mask_preprocessed[0], axis=-1).numpy()
            colored_mask = create_colormap(mask_np)
            mask_pil = Image.fromarray(colored_mask)
            mask_file_name = os.path.splitext(base_name)[0] + '_mask.png'
            mask_save_path = os.path.join(OUTPUT_DIR, mask_file_name)
            mask_pil.save(mask_save_path)
            
            for eps in epsilons:
                print(f"\nProcessing epsilon = {eps}")
                print(f"Image shape: {image_preprocessed.shape}")
                print(f"Perturbation shape: {perturbations.shape}")
                adv_image = image_preprocessed + eps * perturbations
                print(f"Adversarial image shape: {adv_image.shape}")
                adv_image = tf.clip_by_value(adv_image, -1, 1)
                
                # Get segmentation prediction for adversarial image
                adv_pred = pretrained_model(adv_image)
                
                # Save prediction visualization with colors
                pred_np = tf.argmax(adv_pred[0], axis=-1).numpy()
                colored_pred = create_colormap(pred_np)
                pred_pil = Image.fromarray(colored_pred)
                pred_file_name = os.path.splitext(base_name)[0] + f'_pred_eps{eps:.3f}.png'
                pred_save_path = os.path.join(OUTPUT_DIR, pred_file_name)
                pred_pil.save(pred_save_path)
                
                # Save adversarial image
                print("Saving adversarial image...")
                adv_np = ((adv_image[0].numpy() + 1) * 127.5).astype(np.uint8)
                adv_pil = Image.fromarray(adv_np)
                file_name = os.path.splitext(base_name)[0] + f'_eps{eps:.3f}.png'
                save_path = os.path.join(OUTPUT_DIR, file_name)
                adv_pil.save(save_path)
                print(f"Saved adversarial image: {save_path}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print("\nScript completed!")

if __name__ == '__main__':
    process_all_images()
