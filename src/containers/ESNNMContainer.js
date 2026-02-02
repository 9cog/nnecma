const Module = require('../Module');

/**
 * ESNNMContainer - Echo-State-Neural-Network-Membrane Container
 * 
 * Combines reservoir computing (echo-state networks) with membrane computing (P-systems)
 * to create a neuro-symbolic computing architecture.
 * 
 * Architecture:
 * 1. Reservoir Layer: Temporal dynamics and feature extraction
 * 2. Membrane Layer: Symbolic-numeric computation via P-systems
 * 3. Readout Layer: Trainable linear projection to output
 * 
 * The reservoir provides rich temporal representations, the membrane layer
 * performs structured symbolic computation, and the readout layer learns
 * to extract relevant features for the task.
 * 
 * @param {Number} inputSize - Size of input features
 * @param {Number} reservoirSize - Size of reservoir state
 * @param {Number} numMembranes - Number of membrane compartments
 * @param {Number} outputSize - Size of output
 * @param {Object} options - Configuration options for reservoir and membrane layers
 */
class ESNNMContainer extends Module {
  constructor(inputSize, reservoirSize, numMembranes, outputSize, options = {}) {
    super();
    this.inputSize = inputSize;
    this.reservoirSize = reservoirSize;
    this.numMembranes = numMembranes;
    this.outputSize = outputSize;
    
    // Lazy loading of components to avoid circular dependencies
    const ReservoirLayer = require('../modules/ReservoirLayer');
    const MembraneLayer = require('../modules/MembraneLayer');
    const Linear = require('../modules/Linear');
    
    // Extract options for each component
    const reservoirOptions = options.reservoir || {};
    const membraneOptions = options.membrane || {};
    
    // Create reservoir layer
    this.reservoir = new ReservoirLayer(inputSize, reservoirSize, reservoirOptions);
    
    // Create membrane layer (takes reservoir output as input)
    this.membrane = new MembraneLayer(
      reservoirSize, 
      numMembranes, 
      membraneOptions
    );
    
    // Create readout layer (trainable)
    // Takes membrane output and projects to final output
    const membraneOutputSize = numMembranes * (membraneOptions.objectTypes || 16);
    this.readout = new Linear(membraneOutputSize, outputSize);
    
    // Aggregate parameters (only from trainable layers)
    this._aggregateParameters();
  }
  
  _aggregateParameters() {
    // Only membrane and readout layers have trainable parameters
    // Reservoir weights are fixed
    
    this.parameters_ = [];
    this.gradParameters_ = [];
    
    // Add membrane parameters
    const membraneParams = this.membrane.parameters();
    this.parameters_.push(...membraneParams.parameters);
    this.gradParameters_.push(...membraneParams.gradParameters);
    
    // Add readout parameters
    const readoutParams = this.readout.parameters();
    this.parameters_.push(...readoutParams.parameters);
    this.gradParameters_.push(...readoutParams.gradParameters);
  }
  
  forward(input) {
    // input can be:
    // - Single sample: [inputSize]
    // - Sequence: [sequenceLength, inputSize]
    
    // Phase 1: Reservoir processing (temporal dynamics)
    const reservoirOutput = this.reservoir.forward(input);
    
    // Phase 2: Membrane computing (symbolic processing)
    // For sequences, process the final state or aggregate
    let membraneInput;
    if (Array.isArray(reservoirOutput[0])) {
      // Sequence output from reservoir - use final state
      membraneInput = reservoirOutput[reservoirOutput.length - 1];
    } else {
      // Single sample
      membraneInput = reservoirOutput;
    }
    
    const membraneOutput = this.membrane.forward(membraneInput);
    
    // Phase 3: Readout (learned projection)
    this.output = this.readout.forward(membraneOutput);
    
    // Store intermediate outputs for backward pass
    this._intermediates = {
      reservoirOutput: reservoirOutput,
      membraneInput: membraneInput,
      membraneOutput: membraneOutput
    };
    
    return this.output;
  }
  
  backward(input, gradOutput) {
    // Backward pass through the ESNNM architecture
    
    // Phase 1: Backward through readout
    const gradMembraneOutput = this.readout.backward(
      this._intermediates.membraneOutput, 
      gradOutput
    );
    
    // Phase 2: Backward through membrane layer
    const gradMembraneInput = this.membrane.backward(
      this._intermediates.membraneInput,
      gradMembraneOutput
    );
    
    // Phase 3: Backward through reservoir
    // For sequences, we need to handle the gradient properly
    let gradReservoirInput;
    if (Array.isArray(this._intermediates.reservoirOutput[0])) {
      // Sequence: gradient only flows through final state
      const sequenceLength = this._intermediates.reservoirOutput.length;
      const gradReservoirOutput = new Array(sequenceLength);
      for (let t = 0; t < sequenceLength - 1; t++) {
        gradReservoirOutput[t] = new Array(this.reservoirSize).fill(0);
      }
      gradReservoirOutput[sequenceLength - 1] = gradMembraneInput;
      
      gradReservoirInput = this.reservoir.backward(input, gradReservoirOutput);
    } else {
      // Single sample
      gradReservoirInput = this.reservoir.backward(input, gradMembraneInput);
    }
    
    this.gradInput = gradReservoirInput;
    return this.gradInput;
  }
  
  updateGradInput(input, gradOutput) {
    // Called by backward() - implemented in backward()
    return this.backward(input, gradOutput);
  }
  
  accGradParameters(input, gradOutput) {
    // Accumulation is handled by individual layers in backward()
    // This is called automatically by Module.backward()
  }
  
  parameters() {
    // Return aggregated parameters from trainable layers
    return {
      parameters: this.parameters_,
      gradParameters: this.gradParameters_
    };
  }
  
  train() {
    super.train();
    this.reservoir.train();
    this.membrane.train();
    this.readout.train();
    return this;
  }
  
  evaluate() {
    super.evaluate();
    this.reservoir.evaluate();
    this.membrane.evaluate();
    this.readout.evaluate();
    return this;
  }
  
  zeroGradParameters() {
    this.membrane.zeroGradParameters();
    this.readout.zeroGradParameters();
  }
  
  /**
   * Reset internal states of reservoir and membrane layers
   * Call this between independent sequences or batches
   */
  resetStates() {
    this.reservoir.resetState();
    this.membrane.resetStates();
  }
  
  /**
   * Get the current internal states
   */
  getInternalStates() {
    return {
      reservoir: this.reservoir.getState(),
      membranes: this.membrane.getStates()
    };
  }
  
  /**
   * Set the internal states
   */
  setInternalStates(states) {
    if (states.reservoir) {
      this.reservoir.setState(states.reservoir);
    }
    if (states.membranes) {
      this.membrane.setStates(states.membranes);
    }
  }
  
  toString() {
    return `ESNNMContainer(\n` +
           `  Reservoir: ${this.reservoir.toString()}\n` +
           `  Membrane: ${this.membrane.toString()}\n` +
           `  Readout: ${this.readout.toString()}\n` +
           `)`;
  }
}

module.exports = ESNNMContainer;
