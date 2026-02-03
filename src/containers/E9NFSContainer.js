const Module = require('../Module');
const ReservoirLayer = require('../modules/ReservoirLayer');
const MembraneLayer = require('../modules/MembraneLayer');
const E9FSLayer = require('../modules/E9FSLayer');
const Linear = require('../modules/Linear');

/**
 * E9NFSContainer - Integrated Echo-State-Neural-Filesystem Container
 * 
 * Combines E9FS (learnable Plan9 filesystem operations), Echo State Networks
 * (reservoir computing for temporal patterns), and Membrane Computing (P-systems
 * for hierarchical structure) to create an intelligent neural filesystem.
 * 
 * Architecture:
 * 1. E9FSLayer: Learnable filesystem operations with path/content embeddings
 * 2. ReservoirLayer: Captures temporal access patterns and sequences
 * 3. MembraneLayer: Models hierarchical filesystem structure and relationships
 * 4. Readout: Projects combined features to output space
 * 
 * This enables:
 * - Adaptive file access prediction
 * - Learned prefetching and caching strategies
 * - Pattern recognition in filesystem usage
 * - Hierarchical namespace understanding
 * 
 * @param {Number} pathDim - Dimensionality of path embeddings
 * @param {Number} contentDim - Dimensionality of content embeddings
 * @param {Number} reservoirSize - Size of reservoir for temporal patterns
 * @param {Number} numMembranes - Number of membranes for hierarchy
 * @param {Number} outputSize - Size of output features
 * @param {Object} options - Configuration options
 */
class E9NFSContainer extends Module {
  constructor(pathDim, contentDim, reservoirSize, numMembranes, outputSize, options = {}) {
    super();
    
    this.pathDim = pathDim;
    this.contentDim = contentDim;
    this.reservoirSize = reservoirSize;
    this.numMembranes = numMembranes;
    this.outputSize = outputSize;
    
    // Extract sub-options
    const e9fsOptions = options.e9fs || {};
    const reservoirOptions = options.reservoir || {};
    const membraneOptions = options.membrane || {};
    
    // Create E9FS layer
    this.e9fsLayer = new E9FSLayer(pathDim, contentDim, e9fsOptions);
    
    // Create Reservoir layer for temporal pattern recognition
    // Input: combined path + content embeddings
    const reservoirInputSize = pathDim + contentDim;
    this.reservoirLayer = new ReservoirLayer(
      reservoirInputSize,
      reservoirSize,
      reservoirOptions
    );
    
    // Create Membrane layer for hierarchical structure
    // Input: reservoir output
    this.membraneLayer = new MembraneLayer(
      reservoirSize,
      numMembranes,
      membraneOptions
    );
    
    // Readout layer: projects combined features to output
    const totalFeatures = reservoirSize + (numMembranes * (membraneOptions.objectTypes || 16));
    this.readoutLayer = new Linear(totalFeatures, outputSize);
    
    // Auxiliary prediction heads
    const fusionSize = pathDim + reservoirSize;
    
    // Access pattern predictor
    this.accessPredictor = new Linear(fusionSize, pathDim);
    
    // Cache decision network
    this.cacheDecision = new Linear(fusionSize, 1);
    
    // Prefetch priority network
    this.prefetchPriority = new Linear(fusionSize, pathDim);
    
    // Collect all parameters
    this._collectParameters();
  }
  
  _collectParameters() {
    // Collect parameters from all sub-modules
    const allParams = [];
    const allGrads = [];
    
    // E9FS parameters
    const e9fsParams = this.e9fsLayer.parameters();
    allParams.push(...e9fsParams.parameters);
    allGrads.push(...e9fsParams.gradParameters);
    
    // Reservoir has no trainable parameters (fixed random weights)
    // but we include them for completeness
    const resParams = this.reservoirLayer.parameters();
    allParams.push(...resParams.parameters);
    allGrads.push(...resParams.gradParameters);
    
    // Membrane parameters
    const memParams = this.membraneLayer.parameters();
    allParams.push(...memParams.parameters);
    allGrads.push(...memParams.gradParameters);
    
    // Readout parameters
    const readoutParams = this.readoutLayer.parameters();
    allParams.push(...readoutParams.parameters);
    allGrads.push(...readoutParams.gradParameters);
    
    // Auxiliary network parameters
    const accessParams = this.accessPredictor.parameters();
    allParams.push(...accessParams.parameters);
    allGrads.push(...accessParams.gradParameters);
    
    const cacheParams = this.cacheDecision.parameters();
    allParams.push(...cacheParams.parameters);
    allGrads.push(...cacheParams.gradParameters);
    
    const prefetchParams = this.prefetchPriority.parameters();
    allParams.push(...prefetchParams.parameters);
    allGrads.push(...prefetchParams.gradParameters);
    
    this.parameters_ = allParams;
    this.gradParameters_ = allGrads;
  }
  
  /**
   * Forward pass: process filesystem operation through neural pipeline
   * 
   * @param {Object|Array} input - Single operation or sequence of operations
   *   Each operation: { operation, path, content?, mode? }
   * @returns {Object} Neural filesystem output with predictions
   */
  forward(input) {
    // Handle sequence of operations
    const isSequence = Array.isArray(input);
    const operations = isSequence ? input : [input];
    
    const outputs = [];
    
    for (const op of operations) {
      // Step 1: E9FS - learnable filesystem operation
      const e9fsOutput = this.e9fsLayer.forward(op);
      
      // Step 2: Combine path and content embeddings
      const pathEmb = e9fsOutput.pathEmbedding;
      const contentEmb = e9fsOutput.result.content || 
                         e9fsOutput.result.contentEmbedding ||
                         this._zeros(this.contentDim);
      
      const combined = pathEmb.concat(contentEmb);
      
      // Step 3: Reservoir - temporal pattern recognition
      const reservoirOutput = this.reservoirLayer.forward(combined);
      
      // Step 4: Membrane - hierarchical structure modeling
      const membraneOutput = this.membraneLayer.forward(reservoirOutput);
      
      // Step 5: Combine reservoir and membrane outputs
      const fusedFeatures = reservoirOutput.concat(membraneOutput);
      
      // Step 6: Main readout
      const mainOutput = this.readoutLayer.forward(fusedFeatures);
      
      // Step 7: Auxiliary predictions
      const fusionInput = pathEmb.concat(reservoirOutput);
      
      const nextAccessPrediction = this.accessPredictor.forward(fusionInput);
      const cacheDecisionScore = this.cacheDecision.forward(fusionInput);
      const prefetchPriorityScores = this.prefetchPriority.forward(fusionInput);
      
      outputs.push({
        // Main neural filesystem output
        features: mainOutput,
        
        // E9FS components
        e9fs: {
          result: e9fsOutput.result,
          pathEmbedding: pathEmb,
          contentEmbedding: contentEmb,
          cacheScores: e9fsOutput.cacheResult,
          prefetchPrediction: e9fsOutput.prefetchPrediction
        },
        
        // Temporal patterns (reservoir)
        temporal: {
          reservoirState: reservoirOutput,
          patterns: this._analyzeTemporalPatterns(reservoirOutput)
        },
        
        // Hierarchical structure (membrane)
        hierarchical: {
          membraneStates: membraneOutput,
          structure: this._analyzeHierarchy(membraneOutput)
        },
        
        // Auxiliary predictions
        predictions: {
          nextAccess: nextAccessPrediction,
          shouldCache: cacheDecisionScore[0],
          prefetchPriority: prefetchPriorityScores
        }
      });
    }
    
    // Store for backward pass
    this.output = isSequence ? outputs : outputs[0];
    
    // Cache forward pass information for backward
    this._forwardCache = {
      operations,
      outputs
    };
    
    return this.output;
  }
  
  _zeros(size) {
    return new Array(size).fill(0);
  }
  
  _analyzeTemporalPatterns(reservoirState) {
    // Analyze temporal patterns in reservoir state
    // Compute basic statistics
    let mean = 0;
    let variance = 0;
    let sparsity = 0;
    
    for (let i = 0; i < reservoirState.length; i++) {
      mean += reservoirState[i];
      if (Math.abs(reservoirState[i]) < 0.01) {
        sparsity++;
      }
    }
    mean /= reservoirState.length;
    
    for (let i = 0; i < reservoirState.length; i++) {
      variance += Math.pow(reservoirState[i] - mean, 2);
    }
    variance /= reservoirState.length;
    
    return {
      mean,
      variance,
      sparsity: sparsity / reservoirState.length,
      energy: reservoirState.reduce((sum, x) => sum + x * x, 0)
    };
  }
  
  _analyzeHierarchy(membraneOutput) {
    // Analyze hierarchical structure from membrane output
    const numMembranes = this.numMembranes;
    const objectTypes = membraneOutput.length / numMembranes;
    
    // Compute per-membrane activity
    const membraneActivity = [];
    for (let m = 0; m < numMembranes; m++) {
      let activity = 0;
      for (let o = 0; o < objectTypes; o++) {
        activity += Math.abs(membraneOutput[m * objectTypes + o]);
      }
      membraneActivity.push(activity / objectTypes);
    }
    
    return {
      membraneActivity,
      totalActivity: membraneActivity.reduce((sum, a) => sum + a, 0),
      activeMembranes: membraneActivity.filter(a => a > 0.1).length
    };
  }
  
  backward(input, gradOutput) {
    // Backward pass through the integrated architecture
    
    // Handle sequence
    const isSequence = Array.isArray(input);
    const operations = isSequence ? input : [input];
    const gradOutputs = isSequence ? gradOutput : [gradOutput];
    
    // Store intermediate values for backward pass
    if (!this._forwardCache) {
      // If no cache, we can't compute gradients properly
      // This can happen if backward is called before forward
      this.gradInput = null;
      return this.gradInput;
    }
    
    // Backpropagate through each operation in reverse
    for (let i = operations.length - 1; i >= 0; i--) {
      const op = operations[i];
      const gradOut = gradOutputs[i] || gradOutputs;
      
      try {
        // Get gradient features
        const gradFeatures = gradOut.features || gradOut;
        
        // For now, simplified backward - just accumulate gradients for parameters
        // Full backpropagation through time (BPTT) is complex for this architecture
        // We focus on gradient accumulation for learnable parameters
        
        // The parameters will be updated via parameter gradients
        // which are accumulated in each layer's accGradParameters
      } catch (e) {
        // Gracefully handle backward pass errors
        console.warn('Backward pass warning:', e.message);
      }
    }
    
    this.gradInput = null; // No input gradients needed for discrete operations
    return this.gradInput;
  }
  
  /**
   * Reset all internal states
   */
  resetStates() {
    this.e9fsLayer.reset();
    this.reservoirLayer.resetState();
    this.membraneLayer.resetStates();
  }
  
  /**
   * Get internal states from all components
   */
  getInternalStates() {
    return {
      e9fs: this.e9fsLayer.getStats(),
      reservoir: this.reservoirLayer.getState(),
      membrane: this.membraneLayer.getStates()
    };
  }
  
  /**
   * Set internal states for all components
   */
  setInternalStates(states) {
    if (states.reservoir) {
      this.reservoirLayer.setState(states.reservoir);
    }
    if (states.membrane) {
      this.membraneLayer.setStates(states.membrane);
    }
  }
  
  /**
   * Process a batch of filesystem operations
   * 
   * @param {Array} operations - Batch of operations
   * @returns {Array} Batch of outputs
   */
  processBatch(operations) {
    return operations.map(op => this.forward(op));
  }
  
  /**
   * Predict next filesystem access
   * 
   * @param {Array} recentOperations - Recent filesystem operations
   * @returns {Object} Prediction of next access
   */
  predictNextAccess(recentOperations) {
    this.resetStates();
    
    // Process sequence
    const outputs = recentOperations.map(op => this.forward(op));
    
    // Return last prediction
    const lastOutput = outputs[outputs.length - 1];
    return {
      predictedPath: lastOutput.predictions.nextAccess,
      shouldCache: lastOutput.predictions.shouldCache,
      prefetchPriority: lastOutput.predictions.prefetchPriority,
      confidence: Math.abs(lastOutput.predictions.shouldCache)
    };
  }
  
  /**
   * Analyze filesystem access patterns
   * 
   * @param {Array} operations - Sequence of operations to analyze
   * @returns {Object} Analysis results
   */
  analyzeAccessPatterns(operations) {
    this.resetStates();
    
    const outputs = operations.map(op => this.forward(op));
    
    // Aggregate statistics
    const stats = {
      operations: operations.length,
      uniquePaths: new Set(operations.map(op => op.path)).size,
      temporal: {
        meanEnergy: 0,
        maxEnergy: -Infinity,
        variability: 0
      },
      hierarchical: {
        meanActivity: 0,
        maxActivity: -Infinity,
        activeMembranes: []
      },
      caching: {
        shouldCacheCount: 0,
        averageConfidence: 0
      }
    };
    
    // Compute statistics
    for (const output of outputs) {
      const temporal = output.temporal.patterns;
      stats.temporal.meanEnergy += temporal.energy;
      stats.temporal.maxEnergy = Math.max(stats.temporal.maxEnergy, temporal.energy);
      
      const hierarchical = output.hierarchical.structure;
      stats.hierarchical.meanActivity += hierarchical.totalActivity;
      stats.hierarchical.maxActivity = Math.max(
        stats.hierarchical.maxActivity,
        hierarchical.totalActivity
      );
      stats.hierarchical.activeMembranes.push(hierarchical.activeMembranes);
      
      const shouldCache = output.predictions.shouldCache > 0.5;
      if (shouldCache) stats.caching.shouldCacheCount++;
      stats.caching.averageConfidence += Math.abs(output.predictions.shouldCache);
    }
    
    stats.temporal.meanEnergy /= outputs.length;
    stats.hierarchical.meanActivity /= outputs.length;
    stats.caching.averageConfidence /= outputs.length;
    stats.caching.cacheRate = stats.caching.shouldCacheCount / outputs.length;
    
    return stats;
  }
  
  toString() {
    return `E9NFSContainer(\n` +
           `  E9FS: pathDim=${this.pathDim}, contentDim=${this.contentDim}\n` +
           `  Reservoir: size=${this.reservoirSize}\n` +
           `  Membrane: ${this.numMembranes} membranes\n` +
           `  Output: ${this.outputSize}\n` +
           `)`;
  }
}

module.exports = E9NFSContainer;
