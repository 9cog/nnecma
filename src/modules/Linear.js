const Module = require('../Module');

/**
 * Linear - Fully connected layer (affine transformation)
 * 
 * Applies a linear transformation to the incoming data: y = xW^T + b
 * 
 * @param {Number} inputSize - Size of input features
 * @param {Number} outputSize - Size of output features
 * @param {Boolean} bias - Whether to include bias term (default: true)
 */
class Linear extends Module {
  constructor(inputSize, outputSize, bias = true) {
    super();
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.hasBias = bias;

    // Initialize weights with Xavier/Glorot initialization
    this.weight = this._initializeWeights(outputSize, inputSize);
    this.gradWeight = this._zeros(outputSize, inputSize);

    if (this.hasBias) {
      this.bias = this._zeros(outputSize);
      this.gradBias = this._zeros(outputSize);
    }

    // Register parameters
    this.parameters_ = [this.weight];
    this.gradParameters_ = [this.gradWeight];
    
    if (this.hasBias) {
      this.parameters_.push(this.bias);
      this.gradParameters_.push(this.gradBias);
    }
  }

  _initializeWeights(rows, cols) {
    const stdv = 1.0 / Math.sqrt(cols);
    const weights = new Array(rows);
    for (let i = 0; i < rows; i++) {
      weights[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        weights[i][j] = (Math.random() * 2 - 1) * stdv;
      }
    }
    return weights;
  }

  _zeros(rows, cols = null) {
    if (cols === null) {
      return new Array(rows).fill(0);
    }
    const result = new Array(rows);
    for (let i = 0; i < rows; i++) {
      result[i] = new Array(cols).fill(0);
    }
    return result;
  }

  forward(input) {
    // input: [batchSize, inputSize] or [inputSize]
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      const batchSize = input.length;
      this.output = new Array(batchSize);
      
      for (let b = 0; b < batchSize; b++) {
        this.output[b] = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; i++) {
          let sum = 0;
          for (let j = 0; j < this.inputSize; j++) {
            sum += input[b][j] * this.weight[i][j];
          }
          if (this.hasBias) {
            sum += this.bias[i];
          }
          this.output[b][i] = sum;
        }
      }
    } else {
      this.output = new Array(this.outputSize);
      for (let i = 0; i < this.outputSize; i++) {
        let sum = 0;
        for (let j = 0; j < this.inputSize; j++) {
          sum += input[j] * this.weight[i][j];
        }
        if (this.hasBias) {
          sum += this.bias[i];
        }
        this.output[i] = sum;
      }
    }

    return this.output;
  }

  updateGradInput(input, gradOutput) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      const batchSize = input.length;
      this.gradInput = new Array(batchSize);
      
      for (let b = 0; b < batchSize; b++) {
        this.gradInput[b] = new Array(this.inputSize);
        for (let j = 0; j < this.inputSize; j++) {
          let sum = 0;
          for (let i = 0; i < this.outputSize; i++) {
            sum += gradOutput[b][i] * this.weight[i][j];
          }
          this.gradInput[b][j] = sum;
        }
      }
    } else {
      this.gradInput = new Array(this.inputSize);
      for (let j = 0; j < this.inputSize; j++) {
        let sum = 0;
        for (let i = 0; i < this.outputSize; i++) {
          sum += gradOutput[i] * this.weight[i][j];
        }
        this.gradInput[j] = sum;
      }
    }

    return this.gradInput;
  }

  accGradParameters(input, gradOutput) {
    const isBatch = Array.isArray(input[0]);
    
    if (isBatch) {
      const batchSize = input.length;
      
      for (let b = 0; b < batchSize; b++) {
        for (let i = 0; i < this.outputSize; i++) {
          for (let j = 0; j < this.inputSize; j++) {
            this.gradWeight[i][j] += gradOutput[b][i] * input[b][j];
          }
          if (this.hasBias) {
            this.gradBias[i] += gradOutput[b][i];
          }
        }
      }
    } else {
      for (let i = 0; i < this.outputSize; i++) {
        for (let j = 0; j < this.inputSize; j++) {
          this.gradWeight[i][j] += gradOutput[i] * input[j];
        }
        if (this.hasBias) {
          this.gradBias[i] += gradOutput[i];
        }
      }
    }
  }

  toString() {
    return `Linear(${this.inputSize}, ${this.outputSize})`;
  }
}

module.exports = Linear;
