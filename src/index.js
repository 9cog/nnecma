/**
 * nn.ecma - Neural Network library for ECMAScript
 * 
 * A JavaScript implementation inspired by torch/nn
 */

// Base classes
const Module = require('./Module');
const Criterion = require('./Criterion');

// Layers
const Linear = require('./modules/Linear');
const ReLU = require('./modules/ReLU');
const Sigmoid = require('./modules/Sigmoid');
const Tanh = require('./modules/Tanh');

// Containers
const Sequential = require('./containers/Sequential');

// Criterions (Loss functions)
const MSECriterion = require('./criterions/MSECriterion');
const CrossEntropyCriterion = require('./criterions/CrossEntropyCriterion');

module.exports = {
  // Base classes
  Module,
  Criterion,
  
  // Layers
  Linear,
  ReLU,
  Sigmoid,
  Tanh,
  
  // Containers
  Sequential,
  
  // Criterions
  MSECriterion,
  CrossEntropyCriterion,
};
