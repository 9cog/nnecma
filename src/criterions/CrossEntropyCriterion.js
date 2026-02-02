const Criterion = require('../Criterion');

/**
 * CrossEntropyCriterion - Cross Entropy loss
 * 
 * Computes the cross entropy loss between input (logits or probabilities) and target.
 * This implementation assumes input is logits and applies softmax internally.
 */
class CrossEntropyCriterion extends Criterion {
  constructor() {
    super();
    this.logSoftmaxOutput = null;
  }

  _logSoftmax(input) {
    const max = Math.max(...input);
    const exps = input.map(x => Math.exp(x - max));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return input.map(x => x - max - Math.log(sumExps));
  }

  _softmax(input) {
    const max = Math.max(...input);
    const exps = input.map(x => Math.exp(x - max));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
  }

  forward(input, target) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      const batchSize = input.length;
      let totalLoss = 0;
      this.logSoftmaxOutput = [];
      
      for (let b = 0; b < batchSize; b++) {
        const logProbs = this._logSoftmax(input[b]);
        this.logSoftmaxOutput.push(logProbs);
        
        // target can be either class index or one-hot vector
        if (typeof target[b] === 'number') {
          totalLoss -= logProbs[target[b]];
        } else {
          for (let i = 0; i < target[b].length; i++) {
            totalLoss -= target[b][i] * logProbs[i];
          }
        }
      }
      
      this.loss = totalLoss / batchSize;
    } else {
      const logProbs = this._logSoftmax(input);
      this.logSoftmaxOutput = logProbs;
      
      if (typeof target === 'number') {
        this.loss = -logProbs[target];
      } else {
        let loss = 0;
        for (let i = 0; i < target.length; i++) {
          loss -= target[i] * logProbs[i];
        }
        this.loss = loss;
      }
    }

    return this.loss;
  }

  backward(input, target) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      const batchSize = input.length;
      this.gradInput = [];
      
      for (let b = 0; b < batchSize; b++) {
        const probs = this._softmax(input[b]);
        const grad = [...probs];
        
        // Subtract target
        if (typeof target[b] === 'number') {
          grad[target[b]] -= 1;
        } else {
          for (let i = 0; i < target[b].length; i++) {
            grad[i] -= target[b][i];
          }
        }
        
        // Normalize by batch size
        for (let i = 0; i < grad.length; i++) {
          grad[i] /= batchSize;
        }
        
        this.gradInput.push(grad);
      }
    } else {
      const probs = this._softmax(input);
      this.gradInput = [...probs];
      
      if (typeof target === 'number') {
        this.gradInput[target] -= 1;
      } else {
        for (let i = 0; i < target.length; i++) {
          this.gradInput[i] -= target[i];
        }
      }
    }

    return this.gradInput;
  }

  toString() {
    return 'CrossEntropyCriterion';
  }
}

module.exports = CrossEntropyCriterion;
