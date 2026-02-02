/**
 * XOR Problem Example
 * 
 * Trains a neural network to solve the XOR problem
 */

const nn = require('../src/index');

console.log('=== Training Neural Network on XOR Problem ===\n');

// Create model
const model = new nn.Sequential(
  new nn.Linear(2, 4),
  new nn.Tanh(),
  new nn.Linear(4, 1),
  new nn.Sigmoid()
);

// Create loss function
const criterion = new nn.MSECriterion();

// XOR training data
const trainingData = [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] }
];

// Training parameters
const learningRate = 0.5;
const epochs = 10000;

console.log('Training...');
console.log('Learning rate:', learningRate);
console.log('Epochs:', epochs);
console.log();

// Training loop
for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;

  for (const data of trainingData) {
    // Forward pass
    const prediction = model.forward(data.input);
    const loss = criterion.forward(prediction, data.target);
    totalLoss += loss;

    // Backward pass
    const gradLoss = criterion.backward(prediction, data.target);
    model.backward(data.input, gradLoss);

    // Update parameters
    const params = model.parameters();
    for (let i = 0; i < params.parameters.length; i++) {
      const param = params.parameters[i];
      const grad = params.gradParameters[i];

      if (Array.isArray(param[0])) {
        // 2D array (weight matrix)
        for (let j = 0; j < param.length; j++) {
          for (let k = 0; k < param[j].length; k++) {
            param[j][k] -= learningRate * grad[j][k];
          }
        }
      } else {
        // 1D array (bias vector)
        for (let j = 0; j < param.length; j++) {
          param[j] -= learningRate * grad[j];
        }
      }
    }

    // Zero gradients
    model.zeroGradParameters();
  }

  // Print progress
  if ((epoch + 1) % 1000 === 0) {
    console.log(`Epoch ${epoch + 1}, Average Loss: ${(totalLoss / trainingData.length).toFixed(6)}`);
  }
}

console.log('\n=== Testing Trained Network ===\n');

// Test the trained network
model.evaluate(); // Set to evaluation mode
for (const data of trainingData) {
  const prediction = model.forward(data.input);
  const predictedValue = prediction[0];
  const targetValue = data.target[0];
  const rounded = Math.round(predictedValue);
  
  console.log(`Input: [${data.input}] -> Prediction: ${predictedValue.toFixed(4)}, Target: ${targetValue}, Rounded: ${rounded}`);
}

console.log('\n✓ Training complete!');
