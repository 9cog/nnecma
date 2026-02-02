const Module = require('../Module');

/**
 * ReLU - Rectified Linear Unit activation
 * 
 * Applies the element-wise function: f(x) = max(0, x)
 */
class ReLU extends Module {
  constructor() {
    super();
  }

  forward(input) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      this.output = input.map(row => 
        Array.isArray(row) ? row.map(x => Math.max(0, x)) : Math.max(0, row)
      );
    } else {
      this.output = input.map(x => Math.max(0, x));
    }

    return this.output;
  }

  updateGradInput(input, gradOutput) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      this.gradInput = input.map((row, i) => 
        Array.isArray(row) 
          ? row.map((x, j) => x > 0 ? gradOutput[i][j] : 0)
          : (row > 0 ? gradOutput[i] : 0)
      );
    } else {
      this.gradInput = input.map((x, i) => x > 0 ? gradOutput[i] : 0);
    }

    return this.gradInput;
  }

  toString() {
    return 'ReLU';
  }
}

module.exports = ReLU;
