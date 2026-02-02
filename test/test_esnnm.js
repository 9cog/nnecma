/**
 * Tests for ESNNM (Echo-State-Neural-Network-Membrane) modules
 */

const nn = require('../src/index');

// Test colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const RESET = '\x1b[0m';

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`${GREEN}✓${RESET} ${message}`);
    passed++;
  } else {
    console.log(`${RED}✗${RESET} ${message}`);
    failed++;
  }
}

function assertClose(a, b, tolerance, message) {
  const diff = Math.abs(a - b);
  assert(diff < tolerance, `${message} (diff: ${diff})`);
}

console.log('\n=== Testing ESNNM Modules ===\n');

// Test 1: ReservoirLayer instantiation
console.log('Testing ReservoirLayer instantiation...');
const reservoir = new nn.ReservoirLayer(3, 10, {
  spectralRadius: 0.9,
  leakRate: 0.8,
  sparsity: 0.1
});
assert(reservoir instanceof nn.Module, 'ReservoirLayer is instance of Module');
assert(reservoir.inputSize === 3, 'ReservoirLayer input size is correct');
assert(reservoir.reservoirSize === 10, 'ReservoirLayer reservoir size is correct');
assert(reservoir.spectralRadius === 0.9, 'ReservoirLayer spectral radius is correct');

// Test 2: ReservoirLayer forward pass (single sample)
console.log('\nTesting ReservoirLayer forward pass (single sample)...');
reservoir.resetState();
const resInput = [0.5, -0.3, 0.8];
const resOutput = reservoir.forward(resInput);
assert(Array.isArray(resOutput), 'ReservoirLayer output is array');
assert(resOutput.length === 10, 'ReservoirLayer output has correct size');

// Test 3: ReservoirLayer forward pass (sequence)
console.log('\nTesting ReservoirLayer forward pass (sequence)...');
reservoir.resetState();
const resSequence = [
  [0.5, -0.3, 0.8],
  [0.2, 0.4, -0.1],
  [-0.5, 0.6, 0.3]
];
const resSeqOutput = reservoir.forward(resSequence);
assert(Array.isArray(resSeqOutput), 'ReservoirLayer sequence output is array');
assert(resSeqOutput.length === 3, 'ReservoirLayer sequence output has correct length');
assert(Array.isArray(resSeqOutput[0]), 'ReservoirLayer sequence elements are arrays');
assert(resSeqOutput[0].length === 10, 'ReservoirLayer sequence elements have correct size');

// Test 4: ReservoirLayer state management
console.log('\nTesting ReservoirLayer state management...');
reservoir.resetState();
const state1 = reservoir.getState();
reservoir.forward([1, 1, 1]);
const state2 = reservoir.getState();
let statesAreDifferent = false;
for (let i = 0; i < state1.length; i++) {
  if (Math.abs(state1[i] - state2[i]) > 0.001) {
    statesAreDifferent = true;
    break;
  }
}
assert(statesAreDifferent, 'ReservoirLayer state changes after forward pass');
reservoir.resetState();
const state3 = reservoir.getState();
const allZeros = state3.every(v => Math.abs(v) < 0.001);
assert(allZeros, 'ReservoirLayer state resets to zero');

// Test 5: ReservoirLayer backward pass
console.log('\nTesting ReservoirLayer backward pass...');
reservoir.resetState();
const resInput2 = [0.5, -0.3, 0.8];
const resOutput2 = reservoir.forward(resInput2);
const resGradOutput = new Array(10).fill(0.1);
const resGradInput = reservoir.backward(resInput2, resGradOutput);
assert(Array.isArray(resGradInput), 'ReservoirLayer backward returns gradInput');
assert(resGradInput.length === 3, 'ReservoirLayer gradInput has correct size');

// Test 6: MembraneLayer instantiation
console.log('\nTesting MembraneLayer instantiation...');
const membrane = new nn.MembraneLayer(5, 3, {
  objectTypes: 8,
  ruleComplexity: 4,
  communicationRate: 0.5
});
assert(membrane instanceof nn.Module, 'MembraneLayer is instance of Module');
assert(membrane.inputSize === 5, 'MembraneLayer input size is correct');
assert(membrane.numMembranes === 3, 'MembraneLayer number of membranes is correct');
assert(membrane.objectTypes === 8, 'MembraneLayer object types is correct');

// Test 7: MembraneLayer forward pass
console.log('\nTesting MembraneLayer forward pass...');
const memInput = [0.5, -0.3, 0.8, 0.2, -0.1];
const memOutput = membrane.forward(memInput);
assert(Array.isArray(memOutput), 'MembraneLayer output is array');
const expectedSize = 3 * 8; // numMembranes * objectTypes
assert(memOutput.length === expectedSize, 'MembraneLayer output has correct size');

// Test 8: MembraneLayer backward pass
console.log('\nTesting MembraneLayer backward pass...');
const memGradOutput = new Array(24).fill(0.1); // 3 * 8
const memGradInput = membrane.backward(memInput, memGradOutput);
assert(Array.isArray(memGradInput), 'MembraneLayer backward returns gradInput');
assert(memGradInput.length === 5, 'MembraneLayer gradInput has correct size');

// Test 9: MembraneLayer state management
console.log('\nTesting MembraneLayer state management...');
membrane.resetStates();
const memStates1 = membrane.getStates();
assert(memStates1.length === 3, 'MembraneLayer has correct number of membrane states');
assert(memStates1[0].length === 8, 'Each membrane state has correct size');

// Test 10: MembraneLayer parameters
console.log('\nTesting MembraneLayer parameters...');
const memParams = membrane.parameters();
assert(memParams.parameters.length > 0, 'MembraneLayer has parameters');
assert(memParams.gradParameters.length > 0, 'MembraneLayer has gradient parameters');
assert(memParams.parameters.length === memParams.gradParameters.length, 
       'Parameters and gradients have matching lengths');

// Test 11: ESNNMContainer instantiation
console.log('\nTesting ESNNMContainer instantiation...');
const esnnm = new nn.ESNNMContainer(3, 10, 2, 5, {
  reservoir: {
    spectralRadius: 0.9,
    leakRate: 0.8
  },
  membrane: {
    objectTypes: 8,
    ruleComplexity: 4
  }
});
assert(esnnm instanceof nn.Module, 'ESNNMContainer is instance of Module');
assert(esnnm.inputSize === 3, 'ESNNMContainer input size is correct');
assert(esnnm.outputSize === 5, 'ESNNMContainer output size is correct');

// Test 12: ESNNMContainer forward pass (single sample)
console.log('\nTesting ESNNMContainer forward pass (single sample)...');
esnnm.resetStates();
const esnnmInput = [0.5, -0.3, 0.8];
const esnnmOutput = esnnm.forward(esnnmInput);
assert(Array.isArray(esnnmOutput), 'ESNNMContainer output is array');
assert(esnnmOutput.length === 5, 'ESNNMContainer output has correct size');

// Test 13: ESNNMContainer forward pass (sequence)
console.log('\nTesting ESNNMContainer forward pass (sequence)...');
esnnm.resetStates();
const esnnmSequence = [
  [0.5, -0.3, 0.8],
  [0.2, 0.4, -0.1],
  [-0.5, 0.6, 0.3]
];
const esnnmSeqOutput = esnnm.forward(esnnmSequence);
assert(Array.isArray(esnnmSeqOutput), 'ESNNMContainer sequence output is array');
assert(esnnmSeqOutput.length === 5, 'ESNNMContainer sequence output has correct size');

// Test 14: ESNNMContainer backward pass
console.log('\nTesting ESNNMContainer backward pass...');
esnnm.resetStates();
const esnnmInput2 = [0.5, -0.3, 0.8];
const esnnmOutput2 = esnnm.forward(esnnmInput2);
const esnnmGradOutput = new Array(5).fill(0.1);
const esnnmGradInput = esnnm.backward(esnnmInput2, esnnmGradOutput);
assert(Array.isArray(esnnmGradInput), 'ESNNMContainer backward returns gradInput');
assert(esnnmGradInput.length === 3, 'ESNNMContainer gradInput has correct size');

// Test 15: ESNNMContainer parameters
console.log('\nTesting ESNNMContainer parameters...');
const esnnmParams = esnnm.parameters();
assert(esnnmParams.parameters.length > 0, 'ESNNMContainer has parameters');
assert(esnnmParams.gradParameters.length > 0, 'ESNNMContainer has gradient parameters');

// Test 16: ESNNMContainer state management
console.log('\nTesting ESNNMContainer state management...');
esnnm.resetStates();
const internalStates1 = esnnm.getInternalStates();
assert(Array.isArray(internalStates1.reservoir), 'ESNNMContainer has reservoir state');
assert(Array.isArray(internalStates1.membranes), 'ESNNMContainer has membrane states');
esnnm.forward([1, 1, 1]);
const internalStates2 = esnnm.getInternalStates();
let reservoirStateChanged = false;
for (let i = 0; i < internalStates1.reservoir.length; i++) {
  if (Math.abs(internalStates1.reservoir[i] - internalStates2.reservoir[i]) > 0.001) {
    reservoirStateChanged = true;
    break;
  }
}
assert(reservoirStateChanged, 'ESNNMContainer internal states change after forward pass');

// Test 17: ESNNMContainer training simulation
console.log('\nTesting ESNNMContainer training simulation...');
const esnnmModel = new nn.ESNNMContainer(2, 8, 2, 1);
const criterion = new nn.MSECriterion();
esnnmModel.resetStates();

const trainInput = [0.5, -0.5];
const trainTarget = [1.0];

// Forward pass
const prediction = esnnmModel.forward(trainInput);
const loss = criterion.forward(prediction, trainTarget);
assert(loss >= 0, 'Training loss is non-negative');

// Backward pass
const gradLoss = criterion.backward(prediction, trainTarget);
const gradInput = esnnmModel.backward(trainInput, gradLoss);
assert(Array.isArray(gradInput), 'ESNNMContainer training backward returns gradient');

// Get parameters
const trainParams = esnnmModel.parameters();
assert(trainParams.parameters.length > 0, 'ESNNMContainer has trainable parameters');

// Test 18: ESNNMContainer mode switching
console.log('\nTesting ESNNMContainer mode switching...');
esnnmModel.train();
assert(esnnmModel.training === true, 'ESNNMContainer can be set to training mode');
esnnmModel.evaluate();
assert(esnnmModel.training === false, 'ESNNMContainer can be set to evaluation mode');

// Test 19: Integration with Sequential (if needed)
console.log('\nTesting ESNNM with Sequential wrapper...');
const model = new nn.Sequential(
  new nn.ESNNMContainer(2, 8, 2, 4),
  new nn.ReLU(),
  new nn.Linear(4, 1)
);
const seqInput = [0.5, -0.5];
const seqOutput = model.forward(seqInput);
assert(Array.isArray(seqOutput), 'Sequential with ESNNM produces output');
assert(seqOutput.length === 1, 'Sequential with ESNNM has correct output size');

// Test 20: Zero gradients
console.log('\nTesting gradient zeroing...');
esnnmModel.zeroGradParameters();
const params = esnnmModel.parameters();
let allGradsZero = true;
for (const grad of params.gradParameters) {
  if (Array.isArray(grad)) {
    for (let i = 0; i < grad.length; i++) {
      if (Array.isArray(grad[i])) {
        for (let j = 0; j < grad[i].length; j++) {
          if (Math.abs(grad[i][j]) > 1e-10) {
            allGradsZero = false;
            break;
          }
        }
      } else {
        if (Math.abs(grad[i]) > 1e-10) {
          allGradsZero = false;
          break;
        }
      }
    }
  }
}
assert(allGradsZero, 'All gradients are zeroed after zeroGradParameters()');

// Summary
console.log('\n' + '='.repeat(50));
console.log(`Tests passed: ${GREEN}${passed}${RESET}`);
console.log(`Tests failed: ${failed > 0 ? RED : GREEN}${failed}${RESET}`);
console.log('='.repeat(50) + '\n');

process.exit(failed > 0 ? 1 : 0);
