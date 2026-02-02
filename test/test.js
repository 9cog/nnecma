/**
 * Basic tests for nn.ecma
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

console.log('\n=== Testing nn.ecma ===\n');

// Test 1: Module instantiation
console.log('Testing module instantiation...');
const linear = new nn.Linear(3, 2);
assert(linear instanceof nn.Module, 'Linear is instance of Module');
assert(linear.inputSize === 3, 'Linear input size is correct');
assert(linear.outputSize === 2, 'Linear output size is correct');

// Test 2: Linear forward pass
console.log('\nTesting Linear forward pass...');
const input = [1, 2, 3];
const output = linear.forward(input);
assert(Array.isArray(output), 'Linear output is array');
assert(output.length === 2, 'Linear output has correct size');

// Test 3: ReLU activation
console.log('\nTesting ReLU activation...');
const relu = new nn.ReLU();
const reluInput = [-1, 0, 1, 2];
const reluOutput = relu.forward(reluInput);
assert(reluOutput[0] === 0, 'ReLU: negative becomes 0');
assert(reluOutput[1] === 0, 'ReLU: zero stays 0');
assert(reluOutput[2] === 1, 'ReLU: positive stays positive (1)');
assert(reluOutput[3] === 2, 'ReLU: positive stays positive (2)');

// Test 4: Sigmoid activation
console.log('\nTesting Sigmoid activation...');
const sigmoid = new nn.Sigmoid();
const sigmoidInput = [0];
const sigmoidOutput = sigmoid.forward(sigmoidInput);
assertClose(sigmoidOutput[0], 0.5, 0.001, 'Sigmoid(0) ≈ 0.5');

// Test 5: Tanh activation
console.log('\nTesting Tanh activation...');
const tanh = new nn.Tanh();
const tanhInput = [0];
const tanhOutput = tanh.forward(tanhInput);
assertClose(tanhOutput[0], 0, 0.001, 'Tanh(0) ≈ 0');

// Test 6: Sequential container
console.log('\nTesting Sequential container...');
const model = new nn.Sequential(
  new nn.Linear(3, 4),
  new nn.ReLU(),
  new nn.Linear(4, 2)
);
const seqInput = [1, 0, -1];
const seqOutput = model.forward(seqInput);
assert(Array.isArray(seqOutput), 'Sequential output is array');
assert(seqOutput.length === 2, 'Sequential output has correct size');

// Test 7: MSE Criterion
console.log('\nTesting MSE Criterion...');
const mse = new nn.MSECriterion();
const predicted = [1, 2, 3];
const target = [1, 2, 3];
const loss = mse.forward(predicted, target);
assertClose(loss, 0, 0.001, 'MSE loss is 0 for identical predictions');

const predicted2 = [1, 2];
const target2 = [0, 0];
const loss2 = mse.forward(predicted2, target2);
assertClose(loss2, 2.5, 0.001, 'MSE loss is correct: (1^2 + 2^2) / 2 = 2.5');

// Test 8: CrossEntropy Criterion
console.log('\nTesting CrossEntropy Criterion...');
const ce = new nn.CrossEntropyCriterion();
const logits = [1, 2, 3];
const targetClass = 2;
const ceLoss = ce.forward(logits, targetClass);
assert(ceLoss > 0, 'CrossEntropy loss is positive');

// Test 9: Backward pass for Linear
console.log('\nTesting backward pass for Linear...');
const linearBwd = new nn.Linear(2, 2, false); // no bias for simplicity
linearBwd.weight = [[1, 0], [0, 1]]; // identity
const bwdInput = [1, 2];
const bwdOutput = linearBwd.forward(bwdInput);
const gradOutput = [1, 1];
const gradInput = linearBwd.backward(bwdInput, gradOutput);
assert(Array.isArray(gradInput), 'Backward pass returns gradInput');
assert(gradInput.length === 2, 'GradInput has correct size');

// Test 10: End-to-end training step simulation
console.log('\nTesting end-to-end training simulation...');
const net = new nn.Sequential(
  new nn.Linear(2, 3),
  new nn.ReLU(),
  new nn.Linear(3, 1)
);
const criterion = new nn.MSECriterion();

const trainInput = [0.5, -0.5];
const trainTarget = [1.0];

// Forward pass
const prediction = net.forward(trainInput);
const trainLoss = criterion.forward(prediction, trainTarget);
assert(trainLoss >= 0, 'Training loss is non-negative');

// Backward pass
const gradLoss = criterion.backward(prediction, trainTarget);
const gradNetInput = net.backward(trainInput, gradLoss);
assert(Array.isArray(gradNetInput), 'Network backward returns gradient');

// Test parameters
const params = net.parameters();
assert(params.parameters.length > 0, 'Network has parameters');
assert(params.gradParameters.length > 0, 'Network has gradient parameters');

// Summary
console.log('\n' + '='.repeat(50));
console.log(`Tests passed: ${GREEN}${passed}${RESET}`);
console.log(`Tests failed: ${failed > 0 ? RED : GREEN}${failed}${RESET}`);
console.log('='.repeat(50) + '\n');

process.exit(failed > 0 ? 1 : 0);
