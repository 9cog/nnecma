/**
 * Module - Base class for all neural network modules
 * 
 * This is the abstract base class for all neural network modules in nn.ecma.
 * All layers, containers, and networks should inherit from this class.
 */
class Module {
  constructor() {
    this.output = null;
    this.gradInput = null;
    this.parameters_ = [];
    this.gradParameters_ = [];
    this.training = true;
  }

  /**
   * Forward pass - computes output given input
   * @param {Array|Number} input - The input to the module
   * @returns {Array|Number} The output of the module
   */
  forward(input) {
    throw new Error('forward() must be implemented by subclass');
  }

  /**
   * Backward pass - computes gradients
   * @param {Array|Number} input - The original input
   * @param {Array|Number} gradOutput - Gradient of loss w.r.t. output
   * @returns {Array|Number} Gradient of loss w.r.t. input
   */
  backward(input, gradOutput) {
    this.updateGradInput(input, gradOutput);
    this.accGradParameters(input, gradOutput);
    return this.gradInput;
  }

  /**
   * Updates the gradient with respect to the input
   * @param {Array|Number} input - The input
   * @param {Array|Number} gradOutput - Gradient of loss w.r.t. output
   */
  updateGradInput(input, gradOutput) {
    throw new Error('updateGradInput() must be implemented by subclass');
  }

  /**
   * Accumulates the gradient with respect to the parameters
   * @param {Array|Number} input - The input
   * @param {Array|Number} gradOutput - Gradient of loss w.r.t. output
   */
  accGradParameters(input, gradOutput) {
    // Default implementation does nothing (for modules without parameters)
  }

  /**
   * Returns all parameters of the module
   * @returns {Object} Object containing parameters and gradParameters arrays
   */
  parameters() {
    return {
      parameters: this.parameters_,
      gradParameters: this.gradParameters_
    };
  }

  /**
   * Sets the module to training mode
   * @returns {Module} this module
   */
  train() {
    this.training = true;
    return this;
  }

  /**
   * Sets the module to evaluation mode
   * @returns {Module} this module
   */
  evaluate() {
    this.training = false;
    return this;
  }

  /**
   * Zeros all gradients
   */
  zeroGradParameters() {
    const params = this.parameters();
    params.gradParameters.forEach(grad => {
      if (Array.isArray(grad)) {
        for (let i = 0; i < grad.length; i++) {
          if (Array.isArray(grad[i])) {
            for (let j = 0; j < grad[i].length; j++) {
              grad[i][j] = 0;
            }
          } else {
            grad[i] = 0;
          }
        }
      }
    });
  }

  /**
   * Clones the module (shallow copy)
   * @returns {Module} A new module instance
   */
  clone() {
    const clone = Object.create(Object.getPrototypeOf(this));
    Object.assign(clone, this);
    return clone;
  }

  /**
   * Returns string representation of the module
   * @returns {String} String representation
   */
  toString() {
    return `${this.constructor.name}`;
  }
}

module.exports = Module;
