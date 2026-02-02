const Criterion = require('../Criterion');

/**
 * MSECriterion - Mean Squared Error loss
 * 
 * Computes the mean squared error between input and target.
 * Loss = (1/n) * sum((input - target)^2)
 */
class MSECriterion extends Criterion {
  constructor() {
    super();
  }

  forward(input, target) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      let totalLoss = 0;
      const batchSize = input.length;
      
      for (let b = 0; b < batchSize; b++) {
        const inputRow = input[b];
        const targetRow = target[b];
        
        for (let i = 0; i < inputRow.length; i++) {
          const diff = inputRow[i] - targetRow[i];
          totalLoss += diff * diff;
        }
      }
      
      this.loss = totalLoss / (batchSize * input[0].length);
    } else {
      let totalLoss = 0;
      for (let i = 0; i < input.length; i++) {
        const diff = input[i] - target[i];
        totalLoss += diff * diff;
      }
      this.loss = totalLoss / input.length;
    }

    return this.loss;
  }

  backward(input, target) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      const batchSize = input.length;
      const size = input[0].length;
      const norm = 2.0 / (batchSize * size);
      
      this.gradInput = input.map((inputRow, b) => 
        inputRow.map((val, i) => norm * (val - target[b][i]))
      );
    } else {
      const norm = 2.0 / input.length;
      this.gradInput = input.map((val, i) => norm * (val - target[i]));
    }

    return this.gradInput;
  }

  toString() {
    return 'MSECriterion';
  }
}

module.exports = MSECriterion;
