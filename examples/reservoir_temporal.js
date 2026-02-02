/**
 * Temporal Pattern Recognition with Reservoir Computing
 * 
 * Demonstrates using ReservoirLayer for time series prediction/classification.
 * Reservoir computing excels at processing temporal sequences without training
 * the recurrent weights - only the readout layer is trained.
 */

const nn = require('../src/index');

console.log('=== Temporal Pattern Recognition with Reservoir Computing ===\n');

// Create a model with reservoir + readout
// The reservoir processes temporal dynamics, readout learns the mapping
const inputSize = 2;
const reservoirSize = 50;
const outputSize = 1;

// Build model
const model = new nn.Sequential(
  new nn.ReservoirLayer(inputSize, reservoirSize, {
    spectralRadius: 0.95,  // Close to 1 for long-term memory
    leakRate: 0.3,         // Slow dynamics
    sparsity: 0.1,         // 10% connectivity
    inputScale: 0.5
  }),
  new nn.Linear(reservoirSize, outputSize),
  new nn.Sigmoid()
);

// Loss function
const criterion = new nn.MSECriterion();

// Generate synthetic temporal data: sine wave prediction
// Task: predict the next value in a sine wave
function generateSineSequence(length, frequency = 0.1, phase = 0) {
  const sequence = [];
  for (let t = 0; t < length; t++) {
    const value = Math.sin(2 * Math.PI * frequency * t + phase);
    const noise = (Math.random() - 0.5) * 0.1;
    sequence.push([value + noise, Math.cos(2 * Math.PI * frequency * t + phase)]);
  }
  return sequence;
}

// Training data: multiple sine wave sequences
const numSequences = 5;
const sequenceLength = 20;
const trainingData = [];

for (let i = 0; i < numSequences; i++) {
  const phase = Math.random() * 2 * Math.PI;
  const sequence = generateSineSequence(sequenceLength, 0.1, phase);
  
  // Create input-target pairs
  for (let t = 0; t < sequence.length - 1; t++) {
    trainingData.push({
      input: sequence.slice(0, t + 1),  // Sequence up to time t
      target: [(sequence[t + 1][0] + 1) / 2]  // Next value (normalized to [0,1])
    });
  }
}

console.log('Training data size:', trainingData.length);
console.log('Reservoir size:', reservoirSize);
console.log('Training...\n');

// Training parameters
const learningRate = 0.01;
const epochs = 50;

// Training loop
for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;
  
  for (const data of trainingData) {
    // Reset reservoir state for each sequence
    if (model.modules[0].resetState) {
      model.modules[0].resetState();
    }
    
    // Forward pass - process entire sequence
    // For reservoir, we get sequence output, but we only care about the final state
    const reservoirOutput = model.modules[0].forward(data.input);
    const finalState = Array.isArray(reservoirOutput[0]) ? 
                       reservoirOutput[reservoirOutput.length - 1] : 
                       reservoirOutput;
    
    // Now pass through readout and sigmoid
    const readoutOutput = model.modules[1].forward(finalState);
    const prediction = model.modules[2].forward(readoutOutput);
    
    const loss = criterion.forward(prediction, data.target);
    totalLoss += loss;
    
    // Backward pass
    const gradLoss = criterion.backward(prediction, data.target);
    const gradReadout = model.modules[2].backward(readoutOutput, gradLoss);
    const gradFinalState = model.modules[1].backward(finalState, gradReadout);
    
    // We don't backprop through reservoir (fixed weights)
    
    // Update parameters (only readout layer has learnable parameters)
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
  if ((epoch + 1) % 10 === 0) {
    const avgLoss = totalLoss / trainingData.length;
    console.log(`Epoch ${epoch + 1}, Average Loss: ${avgLoss.toFixed(6)}`);
  }
}

console.log('\n=== Testing Trained Model ===\n');

// Test on a new sequence
const testSequence = generateSineSequence(15, 0.1, Math.PI / 4);
model.evaluate();

console.log('Predicting next values in sine wave sequence:\n');
console.log('Time | Actual | Predicted | Error');
console.log('-' .repeat(45));

// Reset reservoir for test
if (model.modules[0].resetState) {
  model.modules[0].resetState();
}

for (let t = 5; t < testSequence.length - 1; t++) {
  // Reset and process sequence up to time t
  if (model.modules[0].resetState) {
    model.modules[0].resetState();
  }
  
  const input = testSequence.slice(0, t + 1);
  const actual = (testSequence[t + 1][0] + 1) / 2;  // Normalize
  
  // Forward through reservoir
  const reservoirOutput = model.modules[0].forward(input);
  const finalState = Array.isArray(reservoirOutput[0]) ? 
                     reservoirOutput[reservoirOutput.length - 1] : 
                     reservoirOutput;
  
  // Forward through readout and sigmoid
  const readoutOutput = model.modules[1].forward(finalState);
  const prediction = model.modules[2].forward(readoutOutput);
  const predicted = prediction[0];
  const error = Math.abs(actual - predicted);
  
  console.log(`  ${t.toString().padStart(2)} | ${actual.toFixed(4)} | ${predicted.toFixed(4)}   | ${error.toFixed(4)}`);
}

console.log('\n✓ Reservoir computing example complete!');
console.log('\nKey insights:');
console.log('- Reservoir layer provides rich temporal representations');
console.log('- Only readout layer is trained (fast training)');
console.log('- Suitable for time series prediction and classification');
console.log('- Spectral radius controls memory of past inputs');
