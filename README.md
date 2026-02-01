# nn.ecma

A Neural Network library for ECMAScript (JavaScript), inspired by [torch/nn](https://github.com/torch/nn).

## Overview

nn.ecma is a modular neural network library that provides a simple and flexible way to build and train neural networks in JavaScript. It follows the design principles of the original Torch nn library, implementing a Module-based architecture where every layer, activation function, and network is a module that can be easily composed.

## Features

- **Modular Architecture**: Every component is a `Module` with consistent `forward()` and `backward()` methods
- **Flexible Composition**: Build complex networks by combining simple modules
- **Pure JavaScript**: No external dependencies, works in Node.js and browsers
- **Easy to Extend**: Add new layers and loss functions by extending the base classes

## Installation

```bash
npm install nnecma
```

Or use it directly:

```javascript
const nn = require('./src/index');
```

## Core Components

### Base Classes

- **Module**: Abstract base class for all neural network modules
- **Criterion**: Abstract base class for all loss functions

### Layers

- **Linear**: Fully connected (dense) layer with optional bias
- **ReLU**: Rectified Linear Unit activation
- **Sigmoid**: Sigmoid activation function
- **Tanh**: Hyperbolic tangent activation function

### Containers

- **Sequential**: Chains modules in sequence, feeding output of one to input of next

### Loss Functions (Criterions)

- **MSECriterion**: Mean Squared Error loss
- **CrossEntropyCriterion**: Cross Entropy loss with softmax

## Quick Start

### Building a Simple Neural Network

```javascript
const nn = require('nnecma');

// Create a simple feedforward network
const model = new nn.Sequential(
  new nn.Linear(10, 20),    // Input: 10, Hidden: 20
  new nn.ReLU(),            // Activation
  new nn.Linear(20, 5),     // Hidden: 20, Output: 5
  new nn.Sigmoid()          // Output activation
);

// Forward pass
const input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const output = model.forward(input);
console.log('Output:', output);
```

### Training a Network

```javascript
const nn = require('nnecma');

// Create model
const model = new nn.Sequential(
  new nn.Linear(2, 4),
  new nn.Tanh(),
  new nn.Linear(4, 1)
);

// Create loss function
const criterion = new nn.MSECriterion();

// Training data
const input = [0.5, -0.3];
const target = [1.0];

// Forward pass
const prediction = model.forward(input);
const loss = criterion.forward(prediction, target);
console.log('Loss:', loss);

// Backward pass
const gradLoss = criterion.backward(prediction, target);
const gradInput = model.backward(input, gradLoss);

// Get parameters for optimization
const params = model.parameters();

// Manual SGD update
const learningRate = 0.01;
for (let i = 0; i < params.parameters.length; i++) {
  const param = params.parameters[i];
  const grad = params.gradParameters[i];
  
  if (Array.isArray(param[0])) {
    for (let j = 0; j < param.length; j++) {
      for (let k = 0; k < param[j].length; k++) {
        param[j][k] -= learningRate * grad[j][k];
      }
    }
  } else {
    for (let j = 0; j < param.length; j++) {
      param[j] -= learningRate * grad[j];
    }
  }
}

// Zero gradients for next iteration
model.zeroGradParameters();
```

## API Reference

### Module

Base class for all neural network modules.

#### Methods

- `forward(input)`: Performs forward pass, returns output
- `backward(input, gradOutput)`: Performs backward pass, returns gradient w.r.t. input
- `parameters()`: Returns object with `parameters` and `gradParameters` arrays
- `train()`: Sets module to training mode
- `evaluate()`: Sets module to evaluation mode
- `zeroGradParameters()`: Zeros all gradient parameters

### Linear(inputSize, outputSize, bias=true)

Fully connected layer: `y = xW^T + b`

### ReLU()

Applies element-wise: `f(x) = max(0, x)`

### Sigmoid()

Applies element-wise: `f(x) = 1 / (1 + exp(-x))`

### Tanh()

Applies element-wise: `f(x) = tanh(x)`

### Sequential(...modules)

Container that chains modules in sequence.

### MSECriterion()

Mean Squared Error loss

### CrossEntropyCriterion()

Cross Entropy loss with softmax

## Examples

See the [examples](./examples) directory for more usage examples.

## License

MIT

## Acknowledgments

This library is inspired by [torch/nn](https://github.com/torch/nn), the neural network package for Torch7.
