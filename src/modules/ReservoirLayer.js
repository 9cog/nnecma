const Module = require('../Module');

/**
 * ReservoirLayer - Echo State Network (ESN) reservoir for reservoir computing
 * 
 * Implements a fixed random recurrent layer with dynamical reservoir properties.
 * The reservoir weights are NOT trained - only the readout layer is trained.
 * 
 * Key properties:
 * - Fixed random input weights (W_in)
 * - Fixed random recurrent weights (W_res) with controlled spectral radius
 * - Leak rate for leaky integration
 * - Sparse connectivity patterns
 * 
 * @param {Number} inputSize - Size of input features
 * @param {Number} reservoirSize - Size of reservoir (hidden state)
 * @param {Object} options - Configuration options
 *   - spectralRadius: Spectral radius of recurrent matrix (default: 0.9)
 *   - leakRate: Leak rate for state update (default: 1.0, no leak)
 *   - inputScale: Scaling factor for input weights (default: 1.0)
 *   - sparsity: Sparsity of reservoir connections (default: 0.1)
 *   - seed: Random seed for reproducibility (default: null)
 */
class ReservoirLayer extends Module {
  constructor(inputSize, reservoirSize, options = {}) {
    super();
    this.inputSize = inputSize;
    this.reservoirSize = reservoirSize;
    
    // Configuration parameters
    this.spectralRadius = options.spectralRadius || 0.9;
    this.leakRate = options.leakRate || 1.0;
    this.inputScale = options.inputScale || 1.0;
    this.sparsity = options.sparsity || 0.1;
    this.seed = options.seed || null;
    
    // Initialize reservoir state
    this.reservoirState = this._zeros(reservoirSize);
    this.prevReservoirState = this._zeros(reservoirSize);
    
    // Initialize fixed random weights
    this._initializeWeights();
    
    // No learnable parameters for basic reservoir
    this.parameters_ = [];
    this.gradParameters_ = [];
  }
  
  _zeros(size) {
    return new Array(size).fill(0);
  }
  
  _random() {
    // Simple random number generator (could be seeded)
    return Math.random();
  }
  
  _initializeWeights() {
    // Initialize input weights (W_in): [reservoirSize x inputSize]
    this.inputWeight = new Array(this.reservoirSize);
    for (let i = 0; i < this.reservoirSize; i++) {
      this.inputWeight[i] = new Array(this.inputSize);
      for (let j = 0; j < this.inputSize; j++) {
        this.inputWeight[i][j] = (this._random() * 2 - 1) * this.inputScale;
      }
    }
    
    // Initialize reservoir weights (W_res): [reservoirSize x reservoirSize]
    // with controlled sparsity
    this.reservoirWeight = new Array(this.reservoirSize);
    for (let i = 0; i < this.reservoirSize; i++) {
      this.reservoirWeight[i] = new Array(this.reservoirSize);
      for (let j = 0; j < this.reservoirSize; j++) {
        if (this._random() < this.sparsity) {
          this.reservoirWeight[i][j] = this._random() * 2 - 1;
        } else {
          this.reservoirWeight[i][j] = 0;
        }
      }
    }
    
    // Scale reservoir weights to desired spectral radius
    this._scaleReservoirWeights();
    
    // Initialize bias
    this.bias = new Array(this.reservoirSize);
    for (let i = 0; i < this.reservoirSize; i++) {
      this.bias[i] = (this._random() * 2 - 1) * 0.1;
    }
  }
  
  _scaleReservoirWeights() {
    // Approximate spectral radius using power iteration
    // This is a simplified version - for production, use proper eigenvalue computation
    const maxIterations = 100;
    const tolerance = 1e-6;
    
    let v = new Array(this.reservoirSize);
    for (let i = 0; i < this.reservoirSize; i++) {
      v[i] = this._random();
    }
    
    let eigenvalue = 0;
    for (let iter = 0; iter < maxIterations; iter++) {
      // v_new = W * v
      const vNew = new Array(this.reservoirSize);
      for (let i = 0; i < this.reservoirSize; i++) {
        vNew[i] = 0;
        for (let j = 0; j < this.reservoirSize; j++) {
          vNew[i] += this.reservoirWeight[i][j] * v[j];
        }
      }
      
      // Compute norm
      let norm = 0;
      for (let i = 0; i < this.reservoirSize; i++) {
        norm += vNew[i] * vNew[i];
      }
      norm = Math.sqrt(norm);
      
      // Normalize
      for (let i = 0; i < this.reservoirSize; i++) {
        v[i] = vNew[i] / norm;
      }
      
      // Check convergence
      if (Math.abs(norm - eigenvalue) < tolerance) {
        eigenvalue = norm;
        break;
      }
      eigenvalue = norm;
    }
    
    // Scale weights to desired spectral radius
    const currentSpectralRadius = eigenvalue;
    const scale = this.spectralRadius / currentSpectralRadius;
    
    for (let i = 0; i < this.reservoirSize; i++) {
      for (let j = 0; j < this.reservoirSize; j++) {
        this.reservoirWeight[i][j] *= scale;
      }
    }
  }
  
  forward(input) {
    // input: [inputSize] (single sample) or [sequenceLength, inputSize] (sequence)
    const isSequence = Array.isArray(input[0]);
    
    if (isSequence) {
      // Process sequence
      const sequenceLength = input.length;
      this.output = new Array(sequenceLength);
      
      for (let t = 0; t < sequenceLength; t++) {
        this.output[t] = this._updateState(input[t]);
      }
    } else {
      // Single time step
      this.output = this._updateState(input);
    }
    
    return this.output;
  }
  
  _updateState(input) {
    // Store previous state for backward pass
    this.prevReservoirState = [...this.reservoirState];
    
    // Compute new state: x(t+1) = (1-α)*x(t) + α*tanh(W_in*u(t) + W_res*x(t) + b)
    // where α is the leak rate
    const newState = new Array(this.reservoirSize);
    
    for (let i = 0; i < this.reservoirSize; i++) {
      // Input contribution
      let sum = 0;
      for (let j = 0; j < this.inputSize; j++) {
        sum += this.inputWeight[i][j] * input[j];
      }
      
      // Recurrent contribution
      for (let j = 0; j < this.reservoirSize; j++) {
        sum += this.reservoirWeight[i][j] * this.reservoirState[j];
      }
      
      // Bias
      sum += this.bias[i];
      
      // Apply activation (tanh) and leak rate
      const activation = Math.tanh(sum);
      newState[i] = (1 - this.leakRate) * this.reservoirState[i] + this.leakRate * activation;
    }
    
    // Update reservoir state
    this.reservoirState = newState;
    
    return [...this.reservoirState];
  }
  
  updateGradInput(input, gradOutput) {
    // For reservoir computing, we typically don't backpropagate through the reservoir
    // The reservoir weights are fixed. Only the readout layer is trained.
    // However, we provide gradient computation for integration with other modules
    
    const isSequence = Array.isArray(input[0]);
    
    if (isSequence) {
      const sequenceLength = input.length;
      this.gradInput = new Array(sequenceLength);
      
      for (let t = 0; t < sequenceLength; t++) {
        this.gradInput[t] = this._computeGradInput(input[t], gradOutput[t]);
      }
    } else {
      this.gradInput = this._computeGradInput(input, gradOutput);
    }
    
    return this.gradInput;
  }
  
  _computeGradInput(input, gradOutput) {
    // Gradient w.r.t. input: dL/du = W_in^T * diag(1-tanh^2(z)) * dL/dx * α
    // This is a simplified version that doesn't fully account for temporal dependencies
    
    const gradInput = new Array(this.inputSize);
    for (let j = 0; j < this.inputSize; j++) {
      gradInput[j] = 0;
      for (let i = 0; i < this.reservoirSize; i++) {
        // Simplified gradient computation
        gradInput[j] += this.inputWeight[i][j] * gradOutput[i] * this.leakRate;
      }
    }
    
    return gradInput;
  }
  
  /**
   * Reset the reservoir state (call between sequences)
   */
  resetState() {
    this.reservoirState = this._zeros(this.reservoirSize);
    this.prevReservoirState = this._zeros(this.reservoirSize);
  }
  
  /**
   * Get current reservoir state
   */
  getState() {
    return [...this.reservoirState];
  }
  
  /**
   * Set reservoir state (for initialization or continuation)
   */
  setState(state) {
    if (state.length !== this.reservoirSize) {
      throw new Error('State size mismatch');
    }
    this.reservoirState = [...state];
  }
  
  toString() {
    return `ReservoirLayer(${this.inputSize}, ${this.reservoirSize}, ` +
           `spectralRadius=${this.spectralRadius}, leakRate=${this.leakRate})`;
  }
}

module.exports = ReservoirLayer;
