const Module = require('../Module');

/**
 * Tanh - Hyperbolic tangent activation function
 * 
 * Applies the element-wise function: f(x) = tanh(x)
 */
class Tanh extends Module {
  constructor() {
    super();
  }

  forward(input) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      this.output = input.map(row => 
        Array.isArray(row) ? row.map(x => Math.tanh(x)) : Math.tanh(row)
      );
    } else {
      this.output = input.map(x => Math.tanh(x));
    }

    return this.output;
  }

  updateGradInput(input, gradOutput) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      this.gradInput = this.output.map((row, i) => 
        Array.isArray(row) 
          ? row.map((y, j) => gradOutput[i][j] * (1 - y * y))
          : gradOutput[i] * (1 - row * row)
      );
    } else {
      this.gradInput = this.output.map((y, i) => gradOutput[i] * (1 - y * y));
    }

    return this.gradInput;
  }

  toString() {
    return 'Tanh';
  }
}

module.exports = Tanh;
