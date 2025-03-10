import requests
import os

def download_image():
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # URL of a test image (using a reliable test image)
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
    
    # Download the image
    print("Downloading test image...")
    response = requests.get(url)
    if response.status_code == 200:
        with open('images/test.jpg', 'wb') as f:
            f.write(response.content)
        print("Image downloaded successfully to images/test.jpg!")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

if __name__ == '__main__':
    download_image() 