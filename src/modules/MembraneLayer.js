const Module = require('../Module');

// Helper function for sigmoid
function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * MembraneLayer - P-System Membrane Computing Layer
 * 
 * Implements a differentiable P-system with membrane structure for symbolic-numeric computing.
 * Uses fuzzy/continuous object multisets to enable gradient-based learning.
 * 
 * P-systems consist of:
 * - Membrane structure (hierarchical compartments)
 * - Object multisets (feature representations)
 * - Evolution rules (rewriting rules with rates)
 * 
 * @param {Number} inputSize - Size of input features
 * @param {Number} numMembranes - Number of membrane compartments
 * @param {Object} options - Configuration options
 *   - objectTypes: Number of distinct object types (default: 16)
 *   - ruleComplexity: Number of evolution rules (default: 8)
 *   - hierarchy: Membrane hierarchy structure (default: flat)
 *   - communicationRate: Rate of inter-membrane communication (default: 0.5)
 *   - fuzzyness: Degree of fuzzy object counts (default: 0.9)
 */
class MembraneLayer extends Module {
  constructor(inputSize, numMembranes, options = {}) {
    super();
    this.inputSize = inputSize;
    this.numMembranes = numMembranes;
    
    // Configuration
    this.objectTypes = options.objectTypes || 16;
    this.ruleComplexity = options.ruleComplexity || 8;
    this.communicationRate = options.communicationRate || 0.5;
    this.fuzzyness = options.fuzzyness || 0.9;
    
    // Membrane hierarchy (default: all siblings under root)
    this.hierarchy = options.hierarchy || this._createFlatHierarchy();
    
    // Initialize membrane states (object multisets)
    this.membraneStates = new Array(numMembranes);
    for (let i = 0; i < numMembranes; i++) {
      this.membraneStates[i] = this._zeros(this.objectTypes);
    }
    
    // Initialize learnable parameters
    this._initializeParameters();
  }
  
  _zeros(size) {
    return new Array(size).fill(0);
  }
  
  _random() {
    return Math.random();
  }
  
  _createFlatHierarchy() {
    // All membranes are siblings under a root membrane
    return {
      type: 'flat',
      root: 0,
      children: Array.from({length: this.numMembranes}, (_, i) => i)
    };
  }
  
  _initializeParameters() {
    // Input encoding: map input features to object multisets
    // [numMembranes * objectTypes, inputSize]
    this.inputEncoder = new Array(this.numMembranes * this.objectTypes);
    this.gradInputEncoder = new Array(this.numMembranes * this.objectTypes);
    
    for (let i = 0; i < this.numMembranes * this.objectTypes; i++) {
      this.inputEncoder[i] = new Array(this.inputSize);
      this.gradInputEncoder[i] = new Array(this.inputSize);
      const stdv = 1.0 / Math.sqrt(this.inputSize);
      for (let j = 0; j < this.inputSize; j++) {
        this.inputEncoder[i][j] = (this._random() * 2 - 1) * stdv;
        this.gradInputEncoder[i][j] = 0;
      }
    }
    
    // Evolution rules: [ruleComplexity, objectTypes] -> [objectTypes]
    // Each rule transforms object multiset probabilistically
    this.ruleWeights = new Array(this.ruleComplexity);
    this.gradRuleWeights = new Array(this.ruleComplexity);
    
    for (let r = 0; r < this.ruleComplexity; r++) {
      this.ruleWeights[r] = new Array(this.objectTypes);
      this.gradRuleWeights[r] = new Array(this.objectTypes);
      for (let i = 0; i < this.objectTypes; i++) {
        this.ruleWeights[r][i] = new Array(this.objectTypes);
        this.gradRuleWeights[r][i] = new Array(this.objectTypes);
        const stdv = 1.0 / Math.sqrt(this.objectTypes);
        for (let j = 0; j < this.objectTypes; j++) {
          this.ruleWeights[r][i][j] = (this._random() * 2 - 1) * stdv;
          this.gradRuleWeights[r][i][j] = 0;
        }
      }
    }
    
    // Communication weights: [numMembranes, numMembranes, objectTypes]
    // Controls object transfer between membranes
    this.communicationWeights = new Array(this.numMembranes);
    this.gradCommunicationWeights = new Array(this.numMembranes);
    
    for (let i = 0; i < this.numMembranes; i++) {
      this.communicationWeights[i] = new Array(this.numMembranes);
      this.gradCommunicationWeights[i] = new Array(this.numMembranes);
      for (let j = 0; j < this.numMembranes; j++) {
        this.communicationWeights[i][j] = new Array(this.objectTypes);
        this.gradCommunicationWeights[i][j] = new Array(this.objectTypes);
        for (let k = 0; k < this.objectTypes; k++) {
          // Initialize with small positive values for communication
          this.communicationWeights[i][j][k] = this._random() * 0.1;
          this.gradCommunicationWeights[i][j][k] = 0;
        }
      }
    }
    
    // Output projection: map object multisets to output features
    const outputSize = this.numMembranes * this.objectTypes;
    this.outputProjection = new Array(outputSize);
    this.gradOutputProjection = new Array(outputSize);
    
    for (let i = 0; i < outputSize; i++) {
      this.outputProjection[i] = (this._random() * 2 - 1) * 0.1;
      this.gradOutputProjection[i] = 0;
    }
    
    // Register parameters
    this.parameters_ = [
      this.inputEncoder,
      this.ruleWeights,
      this.communicationWeights,
      this.outputProjection
    ];
    
    this.gradParameters_ = [
      this.gradInputEncoder,
      this.gradRuleWeights,
      this.gradCommunicationWeights,
      this.gradOutputProjection
    ];
  }
  
  forward(input) {
    // Phase 1: Encode input into object multisets
    this._encodeInput(input);
    
    // Phase 2: Apply evolution rules (P-system computation)
    this._applyEvolutionRules();
    
    // Phase 3: Inter-membrane communication
    this._communicateBetweenMembranes();
    
    // Phase 4: Project to output space
    this.output = this._projectOutput();
    
    return this.output;
  }
  
  _encodeInput(input) {
    // Map input features to object multisets in each membrane
    // Using learned encoding matrix
    
    for (let m = 0; m < this.numMembranes; m++) {
      for (let o = 0; o < this.objectTypes; o++) {
        const encoderIdx = m * this.objectTypes + o;
        let sum = 0;
        for (let i = 0; i < this.inputSize; i++) {
          sum += this.inputEncoder[encoderIdx][i] * input[i];
        }
        // Use sigmoid to bound object counts [0, 1] (fuzzy multiset)
        this.membraneStates[m][o] = 1.0 / (1.0 + Math.exp(-sum));
      }
    }
  }
  
  _applyEvolutionRules() {
    // Apply evolution rules to transform object multisets
    // Each rule is applied with a certain probability/weight
    
    this.preRuleStates = this.membraneStates.map(state => [...state]);
    
    for (let m = 0; m < this.numMembranes; m++) {
      const newState = this._zeros(this.objectTypes);
      
      // Apply each rule and accumulate results
      for (let r = 0; r < this.ruleComplexity; r++) {
        // Rule application: transform current multiset
        for (let i = 0; i < this.objectTypes; i++) {
          let ruleOutput = 0;
          for (let j = 0; j < this.objectTypes; j++) {
            ruleOutput += this.ruleWeights[r][i][j] * this.membraneStates[m][j];
          }
          // Accumulate with fuzzy weighting
          newState[i] += Math.tanh(ruleOutput) * this.fuzzyness / this.ruleComplexity;
        }
      }
      
      // Weighted combination with original state (fuzzy evolution)
      for (let i = 0; i < this.objectTypes; i++) {
        this.membraneStates[m][i] = (1 - this.fuzzyness) * this.membraneStates[m][i] + 
                                     this.fuzzyness * newState[i];
      }
    }
  }
  
  _communicateBetweenMembranes() {
    // Objects can move between membranes based on communication weights
    
    this.preCommunicationStates = this.membraneStates.map(state => [...state]);
    const transfers = new Array(this.numMembranes);
    
    for (let i = 0; i < this.numMembranes; i++) {
      transfers[i] = this._zeros(this.objectTypes);
    }
    
    // Calculate transfers
    for (let i = 0; i < this.numMembranes; i++) {
      for (let j = 0; j < this.numMembranes; j++) {
        if (i !== j) {
          for (let o = 0; o < this.objectTypes; o++) {
            // Communication strength based on learned weights
            const commStrength = sigmoid(this.communicationWeights[i][j][o]);
            const transfer = this.membraneStates[i][o] * commStrength * this.communicationRate;
            
            transfers[i][o] -= transfer;
            transfers[j][o] += transfer;
          }
        }
      }
    }
    
    // Apply transfers
    for (let i = 0; i < this.numMembranes; i++) {
      for (let o = 0; o < this.objectTypes; o++) {
        this.membraneStates[i][o] += transfers[i][o];
        // Clamp to valid range
        this.membraneStates[i][o] = Math.max(0, Math.min(1, this.membraneStates[i][o]));
      }
    }
  }
  
  _projectOutput() {
    // Flatten membrane states and project to output
    const flattenedState = [];
    for (let m = 0; m < this.numMembranes; m++) {
      for (let o = 0; o < this.objectTypes; o++) {
        flattenedState.push(this.membraneStates[m][o]);
      }
    }
    
    // Weighted projection
    const output = [];
    const outputSize = flattenedState.length;
    for (let i = 0; i < outputSize; i++) {
      output.push(flattenedState[i] * this.outputProjection[i]);
    }
    
    return output;
  }
  
  updateGradInput(input, gradOutput) {
    // Backpropagation through membrane computing
    // This is a simplified version - full BPTT through P-systems is complex
    
    // Gradient w.r.t. input through input encoder
    this.gradInput = this._zeros(this.inputSize);
    
    // Backprop through output projection
    const gradFlattened = new Array(this.numMembranes * this.objectTypes);
    for (let i = 0; i < gradFlattened.length; i++) {
      gradFlattened[i] = gradOutput[i] * this.outputProjection[i];
    }
    
    // Distribute gradients to membrane states
    const gradMembraneStates = new Array(this.numMembranes);
    for (let m = 0; m < this.numMembranes; m++) {
      gradMembraneStates[m] = new Array(this.objectTypes);
      for (let o = 0; o < this.objectTypes; o++) {
        const flatIdx = m * this.objectTypes + o;
        gradMembraneStates[m][o] = gradFlattened[flatIdx];
      }
    }
    
    // Backprop through input encoder
    for (let m = 0; m < this.numMembranes; m++) {
      for (let o = 0; o < this.objectTypes; o++) {
        const encoderIdx = m * this.objectTypes + o;
        const state = this.membraneStates[m][o];
        // Gradient of sigmoid
        const sigmoidGrad = state * (1 - state);
        const grad = gradMembraneStates[m][o] * sigmoidGrad;
        
        for (let i = 0; i < this.inputSize; i++) {
          this.gradInput[i] += this.inputEncoder[encoderIdx][i] * grad;
        }
      }
    }
    
    return this.gradInput;
  }
  
  accGradParameters(input, gradOutput) {
    // Accumulate gradients for learnable parameters
    
    // Gradient for output projection
    for (let i = 0; i < this.outputProjection.length; i++) {
      const flatIdx = i;
      const m = Math.floor(flatIdx / this.objectTypes);
      const o = flatIdx % this.objectTypes;
      this.gradOutputProjection[i] += gradOutput[i] * this.membraneStates[m][o];
    }
    
    // Gradient for input encoder (simplified)
    for (let m = 0; m < this.numMembranes; m++) {
      for (let o = 0; o < this.objectTypes; o++) {
        const encoderIdx = m * this.objectTypes + o;
        const state = this.membraneStates[m][o];
        const sigmoidGrad = state * (1 - state);
        
        // Backprop through encoder
        for (let i = 0; i < this.inputSize; i++) {
          const flatIdx = m * this.objectTypes + o;
          this.gradInputEncoder[encoderIdx][i] += input[i] * sigmoidGrad * gradOutput[flatIdx];
        }
      }
    }
    
    // Gradients for rule weights and communication weights would require
    // storing intermediate states and more complex backprop
    // This is a simplified version for demonstration
  }
  
  /**
   * Reset membrane states (call between independent computations)
   */
  resetStates() {
    for (let i = 0; i < this.numMembranes; i++) {
      this.membraneStates[i] = this._zeros(this.objectTypes);
    }
  }
  
  /**
   * Get current membrane states
   */
  getStates() {
    return this.membraneStates.map(state => [...state]);
  }
  
  /**
   * Set membrane states (for initialization)
   */
  setStates(states) {
    if (states.length !== this.numMembranes) {
      throw new Error('Number of membrane states mismatch');
    }
    this.membraneStates = states.map(state => [...state]);
  }
  
  toString() {
    return `MembraneLayer(${this.inputSize}, ${this.numMembranes} membranes, ` +
           `${this.objectTypes} object types)`;
  }
}

module.exports = MembraneLayer;
