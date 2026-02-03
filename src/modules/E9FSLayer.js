const Module = require('../Module');

/**
 * E9FSLayer - Learnable Plan9 Filesystem Operations Layer
 * 
 * Implements differentiable filesystem operations inspired by Plan9's design principles.
 * Makes filesystem operations learnable through neural network parameters, enabling:
 * - Adaptive path resolution and navigation
 * - Learned file content representation and embedding
 * - Optimized filesystem access patterns
 * - Predictive prefetching and caching
 * 
 * Plan9 filesystem concepts implemented:
 * - Everything is a file (uniform interface)
 * - Hierarchical namespace
 * - File descriptors and operations (open, read, write, close, stat)
 * - Dynamic path resolution
 * 
 * @param {Number} pathDim - Dimensionality of path embeddings
 * @param {Number} contentDim - Dimensionality of file content embeddings
 * @param {Object} options - Configuration options
 *   - maxDepth: Maximum directory depth (default: 8)
 *   - numOperations: Number of filesystem operation types (default: 5)
 *   - cacheSize: Size of learned cache (default: 32)
 *   - adaptiveRate: Learning rate for adaptive operations (default: 0.1)
 */
class E9FSLayer extends Module {
  constructor(pathDim, contentDim, options = {}) {
    super();
    this.pathDim = pathDim;
    this.contentDim = contentDim;
    
    // Configuration
    this.maxDepth = options.maxDepth || 8;
    this.numOperations = options.numOperations || 5; // open, read, write, close, stat
    this.cacheSize = options.cacheSize || 32;
    this.adaptiveRate = options.adaptiveRate || 0.1;
    
    // Filesystem state
    this.currentPath = null;
    this.openFiles = new Map(); // file descriptor -> file info
    this.fileCache = new Map(); // path -> content embedding
    this.accessHistory = []; // history of file accesses
    
    // Initialize learnable parameters
    this._initializeParameters();
  }
  
  _zeros(size) {
    return new Array(size).fill(0);
  }
  
  _random() {
    return Math.random();
  }
  
  _initializeParameters() {
    // Path encoder: converts path strings to embeddings
    // [pathDim, maxDepth * 256] - encoding each directory component
    this.pathEncoder = new Array(this.pathDim);
    this.gradPathEncoder = new Array(this.pathDim);
    
    for (let i = 0; i < this.pathDim; i++) {
      this.pathEncoder[i] = new Array(256); // ASCII character encoding
      this.gradPathEncoder[i] = new Array(256);
      const stdv = 1.0 / Math.sqrt(256);
      for (let j = 0; j < 256; j++) {
        this.pathEncoder[i][j] = (this._random() * 2 - 1) * stdv;
        this.gradPathEncoder[i][j] = 0;
      }
    }
    
    // Content encoder: embeds file content into learned representation
    // [contentDim, 256] - byte-level encoding
    this.contentEncoder = new Array(this.contentDim);
    this.gradContentEncoder = new Array(this.contentDim);
    
    for (let i = 0; i < this.contentDim; i++) {
      this.contentEncoder[i] = new Array(256);
      this.gradContentEncoder[i] = new Array(256);
      const stdv = 1.0 / Math.sqrt(256);
      for (let j = 0; j < 256; j++) {
        this.contentEncoder[i][j] = (this._random() * 2 - 1) * stdv;
        this.gradContentEncoder[i][j] = 0;
      }
    }
    
    // Operation embeddings: learned representations for each filesystem operation
    // [numOperations, pathDim]
    this.operationEmbeddings = new Array(this.numOperations);
    this.gradOperationEmbeddings = new Array(this.numOperations);
    
    for (let i = 0; i < this.numOperations; i++) {
      this.operationEmbeddings[i] = new Array(this.pathDim);
      this.gradOperationEmbeddings[i] = new Array(this.pathDim);
      const stdv = 1.0 / Math.sqrt(this.pathDim);
      for (let j = 0; j < this.pathDim; j++) {
        this.operationEmbeddings[i][j] = (this._random() * 2 - 1) * stdv;
        this.gradOperationEmbeddings[i][j] = 0;
      }
    }
    
    // Cache attention weights: learns what to cache based on access patterns
    // [cacheSize, pathDim + contentDim]
    this.cacheAttention = new Array(this.cacheSize);
    this.gradCacheAttention = new Array(this.cacheSize);
    
    for (let i = 0; i < this.cacheSize; i++) {
      this.cacheAttention[i] = new Array(this.pathDim + this.contentDim);
      this.gradCacheAttention[i] = new Array(this.pathDim + this.contentDim);
      const stdv = 1.0 / Math.sqrt(this.pathDim + this.contentDim);
      for (let j = 0; j < this.pathDim + this.contentDim; j++) {
        this.cacheAttention[i][j] = (this._random() * 2 - 1) * stdv;
        this.gradCacheAttention[i][j] = 0;
      }
    }
    
    // Prefetch predictor: predicts next file access
    // [pathDim, pathDim * 2] - based on current path and history
    this.prefetchPredictor = new Array(this.pathDim);
    this.gradPrefetchPredictor = new Array(this.pathDim);
    
    for (let i = 0; i < this.pathDim; i++) {
      this.prefetchPredictor[i] = new Array(this.pathDim * 2);
      this.gradPrefetchPredictor[i] = new Array(this.pathDim * 2);
      const stdv = 1.0 / Math.sqrt(this.pathDim * 2);
      for (let j = 0; j < this.pathDim * 2; j++) {
        this.prefetchPredictor[i][j] = (this._random() * 2 - 1) * stdv;
        this.gradPrefetchPredictor[i][j] = 0;
      }
    }
    
    // Hierarchical structure weights: models directory hierarchy
    // [maxDepth, pathDim]
    this.hierarchyWeights = new Array(this.maxDepth);
    this.gradHierarchyWeights = new Array(this.maxDepth);
    
    for (let i = 0; i < this.maxDepth; i++) {
      this.hierarchyWeights[i] = new Array(this.pathDim);
      this.gradHierarchyWeights[i] = new Array(this.pathDim);
      const stdv = 1.0 / Math.sqrt(this.pathDim);
      for (let j = 0; j < this.pathDim; j++) {
        this.hierarchyWeights[i][j] = (this._random() * 2 - 1) * stdv;
        this.gradHierarchyWeights[i][j] = 0;
      }
    }
    
    // Register parameters
    this.parameters_ = [
      this.pathEncoder,
      this.contentEncoder,
      this.operationEmbeddings,
      this.cacheAttention,
      this.prefetchPredictor,
      this.hierarchyWeights
    ];
    
    this.gradParameters_ = [
      this.gradPathEncoder,
      this.gradContentEncoder,
      this.gradOperationEmbeddings,
      this.gradCacheAttention,
      this.gradPrefetchPredictor,
      this.gradHierarchyWeights
    ];
  }
  
  /**
   * Forward pass: performs filesystem operation
   * @param {Object} input - Filesystem operation request
   *   - operation: 'open'|'read'|'write'|'close'|'stat'
   *   - path: file path string
   *   - content: file content (for write operations)
   *   - mode: operation mode flags
   * @returns {Object} Operation result with learned features
   */
  forward(input) {
    const { operation, path, content, mode } = input;
    
    // Encode path to embedding
    const pathEmbedding = this._encodePath(path);
    
    // Get operation embedding
    const opIndex = this._operationToIndex(operation);
    const opEmbedding = this.operationEmbeddings[opIndex];
    
    // Combine path and operation
    const combinedEmbedding = this._combineEmbeddings(pathEmbedding, opEmbedding);
    
    // Apply hierarchical structure understanding
    const structuredEmbedding = this._applyHierarchy(pathEmbedding, path);
    
    // Check cache and compute cache attention
    const cacheResult = this._computeCacheAttention(pathEmbedding);
    
    // Predict next access for prefetching
    const prefetchPrediction = this._predictNextAccess(pathEmbedding);
    
    // Execute operation based on learned parameters
    let result;
    switch (operation) {
      case 'open':
        result = this._learnedOpen(path, pathEmbedding, mode);
        break;
      case 'read':
        result = this._learnedRead(path, pathEmbedding);
        break;
      case 'write':
        result = this._learnedWrite(path, pathEmbedding, content);
        break;
      case 'close':
        result = this._learnedClose(path, pathEmbedding);
        break;
      case 'stat':
        result = this._learnedStat(path, pathEmbedding);
        break;
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }
    
    // Update access history
    this.accessHistory.push({
      path,
      operation,
      embedding: pathEmbedding,
      timestamp: Date.now()
    });
    
    // Maintain history window
    if (this.accessHistory.length > 100) {
      this.accessHistory.shift();
    }
    
    // Store output for backward pass
    this.output = {
      result,
      pathEmbedding,
      opEmbedding,
      combinedEmbedding,
      structuredEmbedding,
      cacheResult,
      prefetchPrediction
    };
    
    return this.output;
  }
  
  _encodePath(path) {
    // Encode path string into learned embedding
    const embedding = this._zeros(this.pathDim);
    
    if (!path) return embedding;
    
    // Process each character in path
    for (let i = 0; i < Math.min(path.length, 256); i++) {
      const charCode = path.charCodeAt(i) % 256;
      for (let j = 0; j < this.pathDim; j++) {
        embedding[j] += this.pathEncoder[j][charCode];
      }
    }
    
    // Normalize
    let norm = 0;
    for (let i = 0; i < this.pathDim; i++) {
      norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm) + 1e-8;
    
    for (let i = 0; i < this.pathDim; i++) {
      embedding[i] /= norm;
    }
    
    return embedding;
  }
  
  _encodeContent(content) {
    // Encode file content into learned embedding
    const embedding = this._zeros(this.contentDim);
    
    if (!content) return embedding;
    
    // Convert content to bytes and encode
    const bytes = typeof content === 'string' 
      ? content.split('').map(c => c.charCodeAt(0) % 256)
      : content;
    
    for (let i = 0; i < Math.min(bytes.length, 1024); i++) {
      const byte = bytes[i];
      for (let j = 0; j < this.contentDim; j++) {
        embedding[j] += this.contentEncoder[j][byte];
      }
    }
    
    // Normalize
    let norm = 0;
    for (let i = 0; i < this.contentDim; i++) {
      norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm) + 1e-8;
    
    for (let i = 0; i < this.contentDim; i++) {
      embedding[i] /= norm;
    }
    
    return embedding;
  }
  
  _operationToIndex(operation) {
    const ops = ['open', 'read', 'write', 'close', 'stat'];
    const idx = ops.indexOf(operation);
    return idx >= 0 ? idx : 0;
  }
  
  _combineEmbeddings(embedding1, embedding2) {
    const combined = new Array(embedding1.length);
    for (let i = 0; i < embedding1.length; i++) {
      combined[i] = Math.tanh(embedding1[i] + embedding2[i]);
    }
    return combined;
  }
  
  _applyHierarchy(pathEmbedding, path) {
    // Apply hierarchical understanding based on path depth
    const depth = Math.min(
      (path || '').split('/').filter(s => s.length > 0).length,
      this.maxDepth - 1
    );
    
    const structured = new Array(this.pathDim);
    for (let i = 0; i < this.pathDim; i++) {
      structured[i] = pathEmbedding[i] * this.hierarchyWeights[depth][i];
    }
    
    return structured;
  }
  
  _computeCacheAttention(pathEmbedding) {
    // Compute attention scores for cache entries
    const scores = new Array(this.cacheSize);
    
    for (let i = 0; i < this.cacheSize; i++) {
      let score = 0;
      for (let j = 0; j < this.pathDim; j++) {
        score += this.cacheAttention[i][j] * pathEmbedding[j];
      }
      scores[i] = 1.0 / (1.0 + Math.exp(-score)); // sigmoid
    }
    
    return scores;
  }
  
  _predictNextAccess(pathEmbedding) {
    // Predict next file access based on current path and history
    let historyEmbedding = this._zeros(this.pathDim);
    
    if (this.accessHistory.length > 0) {
      const recent = this.accessHistory.slice(-5);
      for (const access of recent) {
        for (let i = 0; i < this.pathDim; i++) {
          historyEmbedding[i] += access.embedding[i] / recent.length;
        }
      }
    }
    
    // Concatenate current and history
    const input = pathEmbedding.concat(historyEmbedding);
    
    // Predict next path
    const prediction = this._zeros(this.pathDim);
    for (let i = 0; i < this.pathDim; i++) {
      for (let j = 0; j < this.pathDim * 2; j++) {
        prediction[i] += this.prefetchPredictor[i][j] * input[j];
      }
      prediction[i] = Math.tanh(prediction[i]);
    }
    
    return prediction;
  }
  
  _learnedOpen(path, pathEmbedding, mode) {
    // Learned open operation
    const fd = this.openFiles.size + 1;
    this.openFiles.set(fd, {
      path,
      embedding: pathEmbedding,
      mode: mode || 'r',
      offset: 0
    });
    
    return {
      success: true,
      fd,
      embedding: pathEmbedding
    };
  }
  
  _learnedRead(path, pathEmbedding) {
    // Learned read operation - returns content embedding
    let contentEmbedding;
    
    if (this.fileCache.has(path)) {
      contentEmbedding = this.fileCache.get(path);
    } else {
      // Generate learned representation
      contentEmbedding = this._zeros(this.contentDim);
      for (let i = 0; i < this.contentDim; i++) {
        for (let j = 0; j < this.pathDim; j++) {
          contentEmbedding[i] += pathEmbedding[j] * (i / this.contentDim);
        }
        contentEmbedding[i] = Math.tanh(contentEmbedding[i]);
      }
      
      this.fileCache.set(path, contentEmbedding);
    }
    
    return {
      success: true,
      content: contentEmbedding,
      embedding: pathEmbedding
    };
  }
  
  _learnedWrite(path, pathEmbedding, content) {
    // Learned write operation
    const contentEmbedding = this._encodeContent(content);
    
    // Update cache
    this.fileCache.set(path, contentEmbedding);
    
    return {
      success: true,
      written: content ? content.length : 0,
      embedding: pathEmbedding,
      contentEmbedding
    };
  }
  
  _learnedClose(path, pathEmbedding) {
    // Learned close operation
    let closed = false;
    
    for (const [fd, info] of this.openFiles.entries()) {
      if (info.path === path) {
        this.openFiles.delete(fd);
        closed = true;
        break;
      }
    }
    
    return {
      success: true,
      closed,
      embedding: pathEmbedding
    };
  }
  
  _learnedStat(path, pathEmbedding) {
    // Learned stat operation - returns file metadata as learned features
    const depthScore = (path || '').split('/').filter(s => s.length > 0).length / this.maxDepth;
    const cacheScore = this.fileCache.has(path) ? 1.0 : 0.0;
    
    return {
      success: true,
      path,
      depth: depthScore,
      cached: cacheScore,
      embedding: pathEmbedding
    };
  }
  
  updateGradInput(input, gradOutput) {
    // Backpropagate gradients through filesystem operations
    // This allows learning optimal filesystem access patterns
    
    this.gradInput = {
      path: null,
      operation: null
    };
    
    // For now, gradients are accumulated in parameter gradients
    // Input gradients are not typically needed for discrete operations
    
    return this.gradInput;
  }
  
  accGradParameters(input, gradOutput) {
    // Accumulate gradients for learnable parameters
    // This is called during backward pass to update parameter gradients
    
    if (!this.output) return;
    
    const { pathEmbedding } = this.output;
    const { path } = input;
    
    // Gradient for path encoder (simplified)
    if (path) {
      for (let i = 0; i < Math.min(path.length, 256); i++) {
        const charCode = path.charCodeAt(i) % 256;
        for (let j = 0; j < this.pathDim; j++) {
          if (gradOutput.pathEmbedding) {
            this.gradPathEncoder[j][charCode] += gradOutput.pathEmbedding[j];
          }
        }
      }
    }
    
    // Gradients for other parameters would be computed based on
    // specific loss functions for filesystem optimization
  }
  
  /**
   * Reset filesystem state
   */
  reset() {
    this.currentPath = null;
    this.openFiles.clear();
    this.fileCache.clear();
    this.accessHistory = [];
  }
  
  /**
   * Get filesystem statistics
   */
  getStats() {
    return {
      openFiles: this.openFiles.size,
      cachedFiles: this.fileCache.size,
      historyLength: this.accessHistory.length
    };
  }
  
  toString() {
    return `E9FSLayer(pathDim=${this.pathDim}, contentDim=${this.contentDim}, ` +
           `cache=${this.cacheSize}, depth=${this.maxDepth})`;
  }
}

module.exports = E9FSLayer;
