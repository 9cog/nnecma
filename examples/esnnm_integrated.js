/**
 * Integrated ESNNM (Echo-State-Neural-Network-Membrane) Example
 * 
 * Demonstrates the combined power of reservoir computing and membrane computing
 * for neuro-symbolic pattern recognition.
 * 
 * The architecture:
 * 1. Reservoir Layer: Extracts temporal/dynamic features from input
 * 2. Membrane Layer: Performs symbolic-numeric computation via P-systems
 * 3. Readout Layer: Maps to output space (trainable)
 */

const nn = require('../src/index');

console.log('=== Integrated ESNNM Example ===\n');
console.log('Neuro-Symbolic Computing: Reservoir + Membrane + Readout\n');

// Create ESNNM model
const inputSize = 3;
const reservoirSize = 20;
const numMembranes = 3;
const outputSize = 2;

const model = new nn.ESNNMContainer(
  inputSize,
  reservoirSize,
  numMembranes,
  outputSize,
  {
    reservoir: {
      spectralRadius: 0.9,
      leakRate: 0.7,
      sparsity: 0.15,
      inputScale: 0.5
    },
    membrane: {
      objectTypes: 8,
      ruleComplexity: 6,
      communicationRate: 0.4,
      fuzzyness: 0.8
    }
  }
);

console.log('Model architecture:');
console.log(model.toString());
console.log();

// Loss function
const criterion = new nn.MSECriterion();

// Generate synthetic classification data
// Task: Classify temporal patterns into categories
function generateTemporalPattern(patternType, length = 10) {
  const sequence = [];
  
  for (let t = 0; t < length; t++) {
    let sample;
    
    if (patternType === 0) {
      // Rising pattern
      sample = [
        t / length,
        Math.sin(t * 0.5),
        (Math.random() - 0.5) * 0.1
      ];
    } else if (patternType === 1) {
      // Oscillating pattern
      sample = [
        Math.sin(t * 0.8),
        Math.cos(t * 0.8),
        (Math.random() - 0.5) * 0.1
      ];
    } else {
      // Decaying pattern
      sample = [
        1.0 - t / length,
        Math.exp(-t / 5),
        (Math.random() - 0.5) * 0.1
      ];
    }
    
    sequence.push(sample);
  }
  
  return sequence;
}

// Generate training data
const trainingData = [];
const samplesPerClass = 10;

for (let patternType = 0; patternType < 2; patternType++) {
  for (let i = 0; i < samplesPerClass; i++) {
    const sequence = generateTemporalPattern(patternType, 10);
    
    // One-hot encoding for target
    const target = [0, 0];
    target[patternType] = 1;
    
    trainingData.push({
      input: sequence,
      target: target
    });
  }
}

// Shuffle training data
for (let i = trainingData.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [trainingData[i], trainingData[j]] = [trainingData[j], trainingData[i]];
}

console.log('Training data: ', trainingData.length, 'samples');
console.log('  - Rising patterns (class 0):', samplesPerClass);
console.log('  - Oscillating patterns (class 1):', samplesPerClass);
console.log();

// Training parameters
const learningRate = 0.05;
const epochs = 30;

console.log('Training ESNNM model...');
console.log('Learning rate:', learningRate);
console.log('Epochs:', epochs);
console.log();

// Training loop
for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;
  let correct = 0;
  
  for (const data of trainingData) {
    // Reset states for each sequence
    model.resetStates();
    
    // Forward pass
    const prediction = model.forward(data.input);
    const loss = criterion.forward(prediction, data.target);
    totalLoss += loss;
    
    // Check accuracy
    const predictedClass = prediction[0] > prediction[1] ? 0 : 1;
    const targetClass = data.target[0] > data.target[1] ? 0 : 1;
    if (predictedClass === targetClass) {
      correct++;
    }
    
    // Backward pass
    const gradLoss = criterion.backward(prediction, data.target);
    model.backward(data.input, gradLoss);
    
    // Update parameters
    const params = model.parameters();
    for (let i = 0; i < params.parameters.length; i++) {
      const param = params.parameters[i];
      const grad = params.gradParameters[i];
      
      // Flatten and update all parameters
      const updateArray = (p, g) => {
        if (Array.isArray(p)) {
          for (let j = 0; j < p.length; j++) {
            if (Array.isArray(p[j])) {
              updateArray(p[j], g[j]);
            } else {
              // Check for NaN in gradients
              if (!isNaN(g[j]) && isFinite(g[j])) {
                p[j] -= learningRate * g[j];
              }
            }
          }
        }
      };
      
      updateArray(param, grad);
    }
    
    // Zero gradients
    model.zeroGradParameters();
  }
  
  // Print progress
  const avgLoss = totalLoss / trainingData.length;
  const accuracy = (correct / trainingData.length * 100).toFixed(1);
  
  if ((epoch + 1) % 5 === 0) {
    console.log(`Epoch ${epoch + 1}: Loss = ${avgLoss.toFixed(4)}, Accuracy = ${accuracy}%`);
  }
}

console.log('\n=== Testing ESNNM Model ===\n');

// Test on new samples
model.evaluate();
const testSamples = [
  { pattern: 0, name: 'Rising' },
  { pattern: 1, name: 'Oscillating' },
  { pattern: 0, name: 'Rising' },
  { pattern: 1, name: 'Oscillating' }
];

console.log('Classification Results:\n');

for (let i = 0; i < testSamples.length; i++) {
  const testSample = testSamples[i];
  const sequence = generateTemporalPattern(testSample.pattern, 10);
  
  model.resetStates();
  const prediction = model.forward(sequence);
  
  const predictedClass = prediction[0] > prediction[1] ? 0 : 1;
  const confidence = Math.max(...prediction);
  const className = predictedClass === 0 ? 'Rising' : 'Oscillating';
  const isCorrect = className === testSample.name;
  
  console.log(`Sample ${i + 1}: True=${testSample.name.padEnd(12)} | ` +
              `Predicted=${className.padEnd(12)} | ` +
              `Confidence=${confidence.toFixed(4)} | ` +
              `${isCorrect ? '✓' : '✗'}`);
}

console.log('\n=== Internal State Analysis ===\n');

// Analyze internal states
model.resetStates();
const testSequence = generateTemporalPattern(0, 10);
model.forward(testSequence);

const internalStates = model.getInternalStates();
console.log('Reservoir state dimension:', internalStates.reservoir.length);
console.log('Number of membranes:', internalStates.membranes.length);
console.log('Objects per membrane:', internalStates.membranes[0].length);

console.log('\nReservoir state sample (first 5 neurons):');
for (let i = 0; i < 5; i++) {
  console.log(`  Neuron ${i}: ${internalStates.reservoir[i].toFixed(4)}`);
}

console.log('\nMembrane states (first membrane, first 5 object types):');
for (let i = 0; i < 5; i++) {
  console.log(`  Object ${i}: ${internalStates.membranes[0][i].toFixed(4)}`);
}

console.log('\n✓ ESNNM example complete!');
console.log('\n' + '='.repeat(60));
console.log('Key Features Demonstrated:');
console.log('='.repeat(60));
console.log('✓ Reservoir Computing: Temporal dynamics extraction');
console.log('✓ Membrane Computing: Symbolic-numeric P-system computation');
console.log('✓ Neuro-Symbolic Integration: Best of both paradigms');
console.log('✓ Feature Embedding: Learnable execution contexts');
console.log('✓ End-to-end Training: Gradient-based learning');
console.log('='.repeat(60));
