# nn.ecma

A Neural Network library for ECMAScript (JavaScript), inspired by [torch/nn](https://github.com/torch/nn), **now extended with ESNNM (Echo-State-Neural-Network-Membrane) capabilities** for neuro-symbolic computing.

## Overview

nn.ecma is a modular neural network library that provides a simple and flexible way to build and train neural networks in JavaScript. It follows the design principles of the original Torch nn library, implementing a Module-based architecture where every layer, activation function, and network is a module that can be easily composed.

**NEW**: The library now includes advanced neuro-symbolic computing capabilities through ESNNM, combining reservoir computing (echo-state networks) with membrane computing (P-systems) for processing temporal sequences and symbolic-numeric computation.

## Features

### Core Features
- **Modular Architecture**: Every component is a `Module` with consistent `forward()` and `backward()` methods
- **Flexible Composition**: Build complex networks by combining simple modules
- **Pure JavaScript**: No external dependencies, works in Node.js and browsers
- **Easy to Extend**: Add new layers and loss functions by extending the base classes

### ESNNM Features (NEW)
- **Reservoir Computing**: Echo-state networks for temporal pattern processing with fixed random dynamics
- **Membrane Computing**: P-system inspired symbolic-numeric computation with learnable evolution rules
- **Neuro-Symbolic Integration**: Combine subsymbolic (neural) and symbolic (membrane) computation
- **Feature Embedding**: Learnable execution contexts for rich representation learning
- **Temporal Sequence Processing**: Native support for time series and sequential data

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
- **ReservoirLayer** (NEW): Echo-state network layer for reservoir computing
- **MembraneLayer** (NEW): P-system membrane computing layer

### Containers

- **Sequential**: Chains modules in sequence, feeding output of one to input of next
- **ESNNMContainer** (NEW): Integrated Echo-State-Neural-Network-Membrane architecture

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

## ESNNM: Neuro-Symbolic Computing

The library now includes ESNNM (Echo-State-Neural-Network-Membrane), a novel architecture that combines reservoir computing with membrane computing for neuro-symbolic AI.

### Quick Start with ESNNM

```javascript
const nn = require('nnecma');

// Create an integrated ESNNM model
const model = new nn.ESNNMContainer(
  3,   // inputSize
  20,  // reservoirSize
  3,   // numMembranes
  2,   // outputSize
  {
    reservoir: {
      spectralRadius: 0.9,  // Controls memory
      leakRate: 0.7,        // Controls dynamics
      sparsity: 0.15        // Connectivity
    },
    membrane: {
      objectTypes: 8,       // Symbolic objects
      ruleComplexity: 6,    // Evolution rules
      communicationRate: 0.4 // Inter-membrane transfer
    }
  }
);

// Process temporal sequence
const sequence = [
  [0.5, -0.3, 0.8],
  [0.2, 0.4, -0.1],
  [-0.5, 0.6, 0.3]
];

model.resetStates();  // Reset between sequences
const output = model.forward(sequence);
console.log('Output:', output);
```

### When to Use ESNNM

ESNNM is particularly suited for:

- **Temporal Pattern Recognition**: Time series classification, sequence prediction
- **Symbolic-Numeric Tasks**: Problems requiring both neural and symbolic reasoning
- **Fast Training**: Reservoir weights are fixed, only readout/membrane parameters trained
- **Memory-Dependent Tasks**: The reservoir provides rich memory of past inputs
- **Structured Computation**: Membrane P-systems enable rule-based transformations

### ESNNM Architecture

1. **ReservoirLayer**: Processes input through fixed random recurrent dynamics
   - Temporal feature extraction
   - No training required (fixed weights)
   - Configurable spectral radius for memory control

2. **MembraneLayer**: Symbolic-numeric computation via P-systems
   - Multiple membrane compartments
   - Fuzzy object multisets (differentiable)
   - Learnable evolution rules
   - Inter-membrane communication

3. **Readout Layer**: Trainable linear projection to output space

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

### ReservoirLayer(inputSize, reservoirSize, options)

Echo-state network layer for reservoir computing. Fixed random weights provide rich temporal dynamics.

**Options**:
- `spectralRadius` (default: 0.9): Controls reservoir memory (closer to 1 = longer memory)
- `leakRate` (default: 1.0): Leak rate for state updates (lower = slower dynamics)
- `inputScale` (default: 1.0): Scaling for input weights
- `sparsity` (default: 0.1): Fraction of non-zero reservoir connections

**Methods**:
- `resetState()`: Reset reservoir state (call between sequences)
- `getState()`: Get current reservoir state
- `setState(state)`: Set reservoir state

### MembraneLayer(inputSize, numMembranes, options)

P-system membrane computing layer with learnable evolution rules.

**Options**:
- `objectTypes` (default: 16): Number of distinct symbolic objects
- `ruleComplexity` (default: 8): Number of evolution rules
- `communicationRate` (default: 0.5): Rate of inter-membrane object transfer
- `fuzzyness` (default: 0.9): Degree of fuzzy (continuous) object counts

**Methods**:
- `resetStates()`: Reset all membrane states
- `getStates()`: Get current membrane states
- `setStates(states)`: Set membrane states

### ESNNMContainer(inputSize, reservoirSize, numMembranes, outputSize, options)

Integrated Echo-State-Neural-Network-Membrane architecture combining reservoir computing and membrane computing.

**Options**:
- `reservoir`: Options object for ReservoirLayer (see above)
- `membrane`: Options object for MembraneLayer (see above)

**Methods**:
- `resetStates()`: Reset both reservoir and membrane states
- `getInternalStates()`: Get both reservoir and membrane states
- `setInternalStates(states)`: Set internal states

## Examples

See the [examples](./examples) directory for more usage examples:

- `simple_network.js` - Basic feedforward network
- `xor_problem.js` - Training on XOR problem
- `reservoir_temporal.js` - Temporal pattern recognition with reservoir computing (NEW)
- `esnnm_integrated.js` - Integrated neuro-symbolic computing with ESNNM (NEW)

## License

MIT

## Acknowledgments

This library is inspired by [torch/nn](https://github.com/torch/nn), the neural network package for Torch7.

The ESNNM extension incorporates concepts from:
- **Reservoir Computing**: Echo State Networks (Jaeger, 2001)
- **Membrane Computing**: P-systems (Păun, 2000)
- **Neuro-Symbolic AI**: Integration of neural and symbolic computation
