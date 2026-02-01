/**
 * Simple Neural Network Example
 * 
 * Demonstrates basic usage of nn.ecma to create and use a neural network
 */

const nn = require('../src/index');

console.log('=== Simple Neural Network Example ===\n');

// Create a simple feedforward network
const model = new nn.Sequential(
  new nn.Linear(4, 8),     // Input: 4, Hidden: 8
  new nn.ReLU(),           // Activation
  new nn.Linear(8, 3),     // Hidden: 8, Output: 3
  new nn.Sigmoid()         // Output activation
);

console.log('Network architecture:');
console.log(model.toString());
console.log();

// Example input
const input = [0.5, -0.3, 0.8, 0.2];
console.log('Input:', input);

// Forward pass
const output = model.forward(input);
console.log('Output:', output);
console.log();

// Get network parameters
const params = model.parameters();
console.log('Number of parameter tensors:', params.parameters.length);
console.log('Number of gradient tensors:', params.gradParameters.length);
