# CNN-using-JS-with-a-visualizer-
This is an interactive Convolutional Neural Network (CNN) visualizer built entirely from scratch using vanilla JavaScript, HTML, and CSS. No external libraries or frameworks are used - every mathematical operation and visualization is implemented manually.
Features
🧠 Complete CNN Implementation
Convolution Layer: Applies edge detection and feature extraction filters
Activation Layer: ReLU activation function for non-linearity
Pooling Layer: Max pooling for dimensionality reduction
Dense Layer: Fully connected layer with learnable weights
Output Layer: Softmax activation for classification probabilities
📊 Real-time Visualization
Input Processing: Upload any image, automatically resized to 32x32 pixels
Layer-by-layer Visualization: See feature maps, activations, and outputs at each stage
Parameter Counting: Displays number of weights and biases for each layer
Automatic Processing: Complete forward pass with timed visualization
🎨 Educational Interface
Modern UI: Clean, responsive design with card-based layout
Interactive Controls: Image upload and reset functionality
Detailed Output: Shows raw values, probabilities, and class predictions
Real-time Feedback: Watch the CNN process your image step by step
Technical Details
Architecture
Input: 32×32 grayscale images
Convolution: 2 filters (3×3 kernels)
Pooling: 2×2 max pooling
Dense: 8 neurons
Output: 3 classes with softmax probabilities

-> Mathematical Operations
All implemented from scratch:
Matrix convolution
ReLU activation
Max pooling
Matrix multiplication (dense layers)
Softmax normalization
-> Use Cases
Educational Tool: Learn how CNNs process images
Algorithm Understanding: Visualize each step of neural network computation
Prototyping: Test CNN architectures before implementation
Demonstration: Show the inner workings of deep learning
