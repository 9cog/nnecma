const Module = require('./Module');

/**
 * Criterion - Base class for all loss functions
 */
class Criterion extends Module {
  constructor() {
    super();
    this.loss = 0;
    this.gradInput = null;
  }

  /**
   * Computes the loss
   * @param {Array|Number} input - The predicted output
   * @param {Array|Number} target - The target output
   * @returns {Number} The loss value
   */
  forward(input, target) {
    throw new Error('forward() must be implemented by subclass');
  }

  /**
   * Computes the gradient of the loss w.r.t. the input
   * @param {Array|Number} input - The predicted output
   * @param {Array|Number} target - The target output
   * @returns {Array|Number} The gradient
   */
  backward(input, target) {
    throw new Error('backward() must be implemented by subclass');
  }
}

module.exports = Criterion;
