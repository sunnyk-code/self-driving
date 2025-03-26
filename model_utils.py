import os
import tarfile
import numpy as np
import tensorflow as tf

class DeepLabModel:
    """Class to load and run DeepLabV3 model."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    LOGITS_TENSOR_NAME = 'ResizeBilinear_2:0'  # DeepLabV3 logits tensor
    
    def __init__(self, model_path):
        """
        Initialize the DeepLabV3 model.
        
        Args:
            model_path: Path to frozen inference graph
        """
        self.graph = tf.Graph()
        self.session = None
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
        
        # Load model
        self._load_graph(model_path)
        self._init_session()
        
        # Create placeholders for gradient computation
        with self.graph.as_default():
            self.input_image = self.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME)
            self.logits = self.graph.get_tensor_by_name(self.LOGITS_TENSOR_NAME)
            
            # Create a placeholder for target labels (one-hot encoded)
            self.target_labels_ph = tf.compat.v1.placeholder(
                tf.float32, 
                [None, None, None, self.logits.shape[-1]], 
                name='target_labels'
            )
            
            # Define the loss function (cross-entropy)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.target_labels_ph,
                    logits=self.logits
                )
            )
            
            # Compute gradients with respect to input
            self.gradients = tf.gradients(self.loss, self.input_image)[0]
    
    def _load_graph(self, model_path):
        """Load frozen inference graph."""
        with self.graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
    
    def _init_session(self):
        """Initialize TensorFlow session."""
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
    
    def predict(self, image):
        """
        Run inference on an image.
        
        Args:
            image: A numpy array representing the input image
            
        Returns:
            Segmentation map as a numpy array
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Run inference
        segmentation = self.session.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: image}
        )
        
        # Remove batch dimension
        return segmentation[0]
    
    def get_logits(self, image):
        """
        Get logits (pre-softmax outputs) for an image.
        
        Args:
            image: A numpy array representing the input image
            
        Returns:
            Logits as a numpy array
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Run inference to get logits
        logits = self.session.run(
            self.LOGITS_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: image}
        )
        
        return logits
    
    def compute_gradients(self, image, target_labels):
        """
        Compute gradients of the loss with respect to the input image.
        
        Args:
            image: A numpy array representing the input image
            target_labels: Target labels (one-hot encoded)
            
        Returns:
            Gradients as a numpy array
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Add batch dimension to target_labels if needed
        if len(target_labels.shape) == 3:
            target_labels = np.expand_dims(target_labels, axis=0)
        
        # Run the gradient computation
        grads = self.session.run(
            self.gradients,
            feed_dict={
                self.INPUT_TENSOR_NAME: image,
                self.target_labels_ph: target_labels
            }
        )
        
        return grads
    
    def close(self):
        """Close the TensorFlow session."""
        if self.session:
            self.session.close()
            self.session = None

def extract_model(model_filename, model_dir):
    """
    Extract DeepLabV3 model from tar.gz file.
    
    Args:
        model_filename: Filename of the model tar.gz
        model_dir: Directory to extract the model to
        
    Returns:
        Path to the extracted frozen_inference_graph.pb file
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Path to tarball
    tarball_path = os.path.join(model_dir, model_filename)
    
    # Download model if it doesn't exist
    if not os.path.exists(tarball_path):
        model_url = f'http://download.tensorflow.org/models/{model_filename}'
        print(f"Downloading model from {model_url}...")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, tarball_path)
            print(f"Downloaded model to {tarball_path}")
        except Exception as e:
            raise ValueError(f"Failed to download model: {e}")
    
    # Extract tarball
    print(f"Extracting model from {tarball_path}...")
    try:
        with tarfile.open(tarball_path) as tar:
            tar.extractall(model_dir)
    except Exception as e:
        raise ValueError(f"Failed to extract model: {e}")
    
    # Find frozen_inference_graph.pb
    for root, dirs, files in os.walk(model_dir):
        if 'frozen_inference_graph.pb' in files:
            model_path = os.path.join(root, 'frozen_inference_graph.pb')
            print(f"Extracted model to {model_path}")
            return model_path
    
    raise ValueError("Could not find frozen_inference_graph.pb in extracted model") 