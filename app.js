// CNN Visualizer from Scratch
// Author: Your Name
// No external libraries used

// --- DOM Elements ---
const inputCanvas = document.getElementById('input-canvas');
const inputCtx = inputCanvas.getContext('2d');
const imageUpload = document.getElementById('image-upload');
const stepBtn = document.getElementById('step-btn');
const resetBtn = document.getElementById('reset-btn');

const convFeatureMaps = document.getElementById('conv-feature-maps');
const activationMaps = document.getElementById('activation-maps');
const poolingMaps = document.getElementById('pooling-maps');
const denseActivations = document.getElementById('dense-activations');
const outputValues = document.getElementById('output-values');
const convParams = document.getElementById('conv-params');
const denseParams = document.getElementById('dense-params');
const outputParams = document.getElementById('output-params');

// --- CNN Parameters ---
const INPUT_SIZE = 32; // We'll resize input to 32x32 for speed
const KERNEL_SIZE = 3;
const NUM_FILTERS = 2;
const POOL_SIZE = 2;
const DENSE_SIZE = 8;
const OUTPUT_SIZE = 3;

// Simple kernels for demonstration (edge detection, blur)
const kernels = [
  [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
  ],
  [
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
  ]
];

// --- State ---
let originalImageData = null;
let currentStep = 0;
let convOut = [];
let reluOut = [];
let poolOut = [];
let denseOut = [];
let outputOut = [];
let lastDenseRaw = [];

// --- Utility Functions ---
function toGrayscale(imageData, w, h) {
  // Returns 2D array of grayscale values
  const gray = [];
  for (let y = 0; y < h; y++) {
    gray[y] = [];
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = imageData.data[idx];
      const g = imageData.data[idx + 1];
      const b = imageData.data[idx + 2];
      gray[y][x] = (r + g + b) / 3 / 255;
    }
  }
  return gray;
}

function drawGrayscaleToCanvas(arr, canvas) {
  const ctx = canvas.getContext('2d');
  const h = arr.length;
  const w = arr[0].length;
  canvas.width = w;
  canvas.height = h;
  const imgData = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = Math.max(0, Math.min(1, arr[y][x]));
      const idx = (y * w + x) * 4;
      imgData.data[idx] = imgData.data[idx + 1] = imgData.data[idx + 2] = v * 255;
      imgData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// --- CNN Layer Functions ---
function convolve(input, kernel) {
  const h = input.length;
  const w = input[0].length;
  const kh = kernel.length;
  const kw = kernel[0].length;
  const out = [];
  for (let y = 0; y < h - kh + 1; y++) {
    out[y] = [];
    for (let x = 0; x < w - kw + 1; x++) {
      let sum = 0;
      for (let i = 0; i < kh; i++) {
        for (let j = 0; j < kw; j++) {
          sum += input[y + i][x + j] * kernel[i][j];
        }
      }
      out[y][x] = sum;
    }
  }
  return out;
}

function relu(input) {
  return input.map(row => row.map(v => Math.max(0, v)));
}

function maxPool(input, size) {
  const h = input.length;
  const w = input[0].length;
  const out = [];
  for (let y = 0; y < h; y += size) {
    const row = [];
    for (let x = 0; x < w; x += size) {
      let max = -Infinity;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          if (y + i < h && x + j < w) {
            max = Math.max(max, input[y + i][x + j]);
          }
        }
      }
      row.push(max);
    }
    out.push(row);
  }
  return out;
}

function flatten(inputs) {
  // inputs: array of 2D arrays
  let arr = [];
  for (const mat of inputs) {
    for (const row of mat) {
      arr = arr.concat(row);
    }
  }
  return arr;
}

function dense(input, weights, biases) {
  // input: 1D array
  // weights: 2D array [output][input]
  // biases: 1D array
  const out = [];
  for (let i = 0; i < weights.length; i++) {
    let sum = biases[i];
    for (let j = 0; j < input.length; j++) {
      sum += input[j] * weights[i][j];
    }
    out.push(sum);
  }
  return out;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// --- Visualization Functions ---
function clearVisualizations() {
  convFeatureMaps.innerHTML = '';
  activationMaps.innerHTML = '';
  poolingMaps.innerHTML = '';
  denseActivations.innerHTML = '';
  outputValues.innerHTML = '';
  convParams.textContent = '';
  denseParams.textContent = '';
  outputParams.textContent = '';
}

function showConvLayer(convOut) {
  convFeatureMaps.innerHTML = '';
  convOut.forEach((map, i) => {
    const canvas = document.createElement('canvas');
    drawGrayscaleToCanvas(map, canvas);
    convFeatureMaps.appendChild(canvas);
  });
}

function showActivationLayer(reluOut) {
  activationMaps.innerHTML = '';
  reluOut.forEach((map, i) => {
    const canvas = document.createElement('canvas');
    drawGrayscaleToCanvas(map, canvas);
    activationMaps.appendChild(canvas);
  });
}

function showPoolingLayer(poolOut) {
  poolingMaps.innerHTML = '';
  poolOut.forEach((map, i) => {
    const canvas = document.createElement('canvas');
    drawGrayscaleToCanvas(map, canvas);
    poolingMaps.appendChild(canvas);
  });
}

function showDenseLayer(denseOut) {
  denseActivations.innerHTML = '';
  denseOut.forEach((v, i) => {
    const div = document.createElement('div');
    div.textContent = v.toFixed(2);
    denseActivations.appendChild(div);
  });
}

function showOutputLayer(outputOut, rawOut) {
  outputValues.innerHTML = '';
  outputOut.forEach((v, i) => {
    const div = document.createElement('div');
    div.innerHTML = `<strong>Class ${i}</strong><br>Raw: ${rawOut[i].toFixed(2)}<br>Prob: ${(v * 100).toFixed(1)}%`;
    outputValues.appendChild(div);
  });
}

function showConvParams() {
  // Conv: (kernel size * kernel size) * num_filters, 1 bias per filter
  const weights = KERNEL_SIZE * KERNEL_SIZE * NUM_FILTERS;
  const biases = NUM_FILTERS;
  convParams.textContent = `Weights: ${weights}, Biases: ${biases}`;
}

function showDenseParams(inputLen) {
  // Dense: inputLen * DENSE_SIZE, DENSE_SIZE biases
  const weights = inputLen * DENSE_SIZE;
  const biases = DENSE_SIZE;
  denseParams.textContent = `Weights: ${weights}, Biases: ${biases}`;
}

function showOutputParams() {
  // Output: DENSE_SIZE * OUTPUT_SIZE, OUTPUT_SIZE biases
  const weights = DENSE_SIZE * OUTPUT_SIZE;
  const biases = OUTPUT_SIZE;
  outputParams.textContent = `Weights: ${weights}, Biases: ${biases}`;
}

// --- CNN Forward Pass ---
function runConvLayer(grayImg) {
  convOut = kernels.map(kernel => convolve(grayImg, kernel));
  showConvLayer(convOut);
}

function runActivationLayer() {
  reluOut = convOut.map(map => relu(map));
  showActivationLayer(reluOut);
}

function runPoolingLayer() {
  poolOut = reluOut.map(map => maxPool(map, POOL_SIZE));
  showPoolingLayer(poolOut);
}

function runDenseLayer() {
  // Random weights for demo
  const flat = flatten(poolOut);
  const weights = Array.from({length: DENSE_SIZE}, () => Array.from({length: flat.length}, () => Math.random() * 0.2 - 0.1));
  const biases = Array.from({length: DENSE_SIZE}, () => Math.random() * 0.2 - 0.1);
  denseOut = dense(flat, weights, biases);
  showDenseLayer(denseOut);
}

function runOutputLayer() {
  // Random weights for demo
  const weights = Array.from({length: OUTPUT_SIZE}, () => Array.from({length: DENSE_SIZE}, () => Math.random() * 0.2 - 0.1));
  const biases = Array.from({length: OUTPUT_SIZE}, () => Math.random() * 0.2 - 0.1);
  lastDenseRaw = dense(denseOut, weights, biases);
  outputOut = softmax(lastDenseRaw);
  showOutputLayer(outputOut, lastDenseRaw);
}

// --- Automatic Forward Pass ---
async function runAllLayers(grayImg) {
  clearVisualizations();
  // 1. Conv
  runConvLayer(grayImg);
  showConvParams();
  await new Promise(r => setTimeout(r, 1000));
  // 2. Activation
  runActivationLayer();
  await new Promise(r => setTimeout(r, 1000));
  // 3. Pooling
  runPoolingLayer();
  await new Promise(r => setTimeout(r, 1000));
  // 4. Dense
  // Calculate input length for dense params
  const flat = flatten(poolOut);
  showDenseParams(flat.length);
  runDenseLayer();
  await new Promise(r => setTimeout(r, 1000));
  // 5. Output
  showOutputParams();
  runOutputLayer();
}

// --- Image Upload Handler ---
imageUpload.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new window.Image();
  img.onload = function() {
    inputCanvas.width = INPUT_SIZE;
    inputCanvas.height = INPUT_SIZE;
    inputCtx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
    originalImageData = inputCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    resetAll();
    // Start automatic forward pass
    const gray = toGrayscale(originalImageData, INPUT_SIZE, INPUT_SIZE);
    runAllLayers(gray);
  };
  img.src = URL.createObjectURL(file);
});

// Remove/disable step button and stepper logic
stepBtn.style.display = 'none';

function resetAll() {
  currentStep = 0;
  // stepBtn.disabled = false; // not needed
  clearVisualizations();
  if (originalImageData) {
    inputCtx.putImageData(originalImageData, 0, 0);
  } else {
    inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
  }
}

resetBtn.addEventListener('click', resetAll);

// --- Initial State ---
resetAll();
