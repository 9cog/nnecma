const Module = require('../Module');

/**
 * Sigmoid - Sigmoid activation function
 * 
 * Applies the element-wise function: f(x) = 1 / (1 + exp(-x))
 */
class Sigmoid extends Module {
  constructor() {
    super();
  }

  _sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  forward(input) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      this.output = input.map(row => 
        Array.isArray(row) ? row.map(x => this._sigmoid(x)) : this._sigmoid(row)
      );
    } else {
      this.output = input.map(x => this._sigmoid(x));
    }

    return this.output;
  }

  updateGradInput(input, gradOutput) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      this.gradInput = this.output.map((row, i) => 
        Array.isArray(row) 
          ? row.map((y, j) => gradOutput[i][j] * y * (1 - y))
          : gradOutput[i] * row * (1 - row)
      );
    } else {
      this.gradInput = this.output.map((y, i) => gradOutput[i] * y * (1 - y));
    }

    return this.gradInput;
  }

  toString() {
    return 'Sigmoid';
  }
}

module.exports = Sigmoid;
