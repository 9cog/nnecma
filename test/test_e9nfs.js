/**
 * Tests for E9NFS - Neural Filesystem Operations
 */

const nn = require('../src/index');

// Test colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const BLUE = '\x1b[34m';
const RESET = '\x1b[0m';

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`${GREEN}✓${RESET} ${message}`);
    passed++;
  } else {
    console.log(`${RED}✗${RESET} ${message}`);
    failed++;
  }
}

function assertClose(a, b, tolerance, message) {
  const diff = Math.abs(a - b);
  assert(diff < tolerance, `${message} (diff: ${diff})`);
}

console.log('\n=== Testing E9NFS - Neural Filesystem Operations ===\n');

// Test 1: E9FSLayer instantiation
console.log(`${BLUE}Testing E9FSLayer instantiation...${RESET}`);
const pathDim = 16;
const contentDim = 32;
const e9fs = new nn.E9FSLayer(pathDim, contentDim, {
  maxDepth: 8,
  cacheSize: 16
});

assert(e9fs instanceof nn.Module, 'E9FSLayer is instance of Module');
assert(e9fs.pathDim === pathDim, 'E9FSLayer has correct pathDim');
assert(e9fs.contentDim === contentDim, 'E9FSLayer has correct contentDim');
assert(e9fs.maxDepth === 8, 'E9FSLayer has correct maxDepth');
assert(e9fs.cacheSize === 16, 'E9FSLayer has correct cacheSize');

// Test 2: E9FSLayer open operation
console.log(`\n${BLUE}Testing E9FSLayer open operation...${RESET}`);
const openInput = {
  operation: 'open',
  path: '/usr/local/bin/node',
  mode: 'r'
};
const openOutput = e9fs.forward(openInput);

assert(openOutput.result !== undefined, 'Open operation returns result');
assert(openOutput.result.success === true, 'Open operation succeeds');
assert(openOutput.result.fd > 0, 'Open operation returns file descriptor');
assert(openOutput.pathEmbedding !== undefined, 'Open operation returns path embedding');
assert(Array.isArray(openOutput.pathEmbedding), 'Path embedding is array');
assert(openOutput.pathEmbedding.length === pathDim, 'Path embedding has correct dimension');

// Test 3: E9FSLayer read operation
console.log(`\n${BLUE}Testing E9FSLayer read operation...${RESET}`);
const readInput = {
  operation: 'read',
  path: '/usr/local/bin/node'
};
const readOutput = e9fs.forward(readInput);

assert(readOutput.result !== undefined, 'Read operation returns result');
assert(readOutput.result.success === true, 'Read operation succeeds');
assert(readOutput.result.content !== undefined, 'Read operation returns content');
assert(Array.isArray(readOutput.result.content), 'Content is array (embedding)');
assert(readOutput.result.content.length === contentDim, 'Content embedding has correct dimension');

// Test 4: E9FSLayer write operation
console.log(`\n${BLUE}Testing E9FSLayer write operation...${RESET}`);
const writeInput = {
  operation: 'write',
  path: '/tmp/test.txt',
  content: 'Hello, E9NFS!'
};
const writeOutput = e9fs.forward(writeInput);

assert(writeOutput.result !== undefined, 'Write operation returns result');
assert(writeOutput.result.success === true, 'Write operation succeeds');
assert(writeOutput.result.written > 0, 'Write operation reports bytes written');
assert(writeOutput.result.contentEmbedding !== undefined, 'Write operation creates content embedding');

// Test 5: E9FSLayer close operation
console.log(`\n${BLUE}Testing E9FSLayer close operation...${RESET}`);
const closeInput = {
  operation: 'close',
  path: '/usr/local/bin/node'
};
const closeOutput = e9fs.forward(closeInput);

assert(closeOutput.result !== undefined, 'Close operation returns result');
assert(closeOutput.result.success === true, 'Close operation succeeds');

// Test 6: E9FSLayer stat operation
console.log(`\n${BLUE}Testing E9FSLayer stat operation...${RESET}`);
const statInput = {
  operation: 'stat',
  path: '/usr/local/bin/node'
};
const statOutput = e9fs.forward(statInput);

assert(statOutput.result !== undefined, 'Stat operation returns result');
assert(statOutput.result.success === true, 'Stat operation succeeds');
assert(statOutput.result.depth !== undefined, 'Stat returns depth score');
assert(statOutput.result.cached !== undefined, 'Stat returns cache status');

// Test 7: E9FSLayer caching
console.log(`\n${BLUE}Testing E9FSLayer caching...${RESET}`);
const stats1 = e9fs.getStats();
assert(stats1.cachedFiles >= 0, 'Can get cache statistics');

// Read same file twice to test caching
const read1 = e9fs.forward({ operation: 'read', path: '/test/file1.txt' });
const read2 = e9fs.forward({ operation: 'read', path: '/test/file1.txt' });

const stats2 = e9fs.getStats();
assert(stats2.cachedFiles >= stats1.cachedFiles, 'Cache grows with reads');

// Test 8: E9FSLayer access history
console.log(`\n${BLUE}Testing E9FSLayer access history...${RESET}`);
e9fs.reset();
const initialStats = e9fs.getStats();
assert(initialStats.historyLength === 0, 'History is empty after reset');

e9fs.forward({ operation: 'open', path: '/file1.txt' });
e9fs.forward({ operation: 'read', path: '/file1.txt' });
e9fs.forward({ operation: 'close', path: '/file1.txt' });

const afterStats = e9fs.getStats();
assert(afterStats.historyLength === 3, 'History tracks operations');

// Test 9: E9FSLayer prefetch prediction
console.log(`\n${BLUE}Testing E9FSLayer prefetch prediction...${RESET}`);
const prefetchOutput = e9fs.forward({ operation: 'read', path: '/home/user/docs/file.txt' });
assert(prefetchOutput.prefetchPrediction !== undefined, 'Prefetch prediction is generated');
assert(Array.isArray(prefetchOutput.prefetchPrediction), 'Prefetch prediction is array');
assert(prefetchOutput.prefetchPrediction.length === pathDim, 'Prefetch prediction has correct dimension');

// Test 10: E9NFSContainer instantiation
console.log(`\n${BLUE}Testing E9NFSContainer instantiation...${RESET}`);
const reservoirSize = 20;
const numMembranes = 3;
const outputSize = 10;

const e9nfs = new nn.E9NFSContainer(
  pathDim,
  contentDim,
  reservoirSize,
  numMembranes,
  outputSize,
  {
    reservoir: {
      spectralRadius: 0.9,
      leakRate: 0.7,
      sparsity: 0.15
    },
    membrane: {
      objectTypes: 8,
      ruleComplexity: 4
    }
  }
);

assert(e9nfs instanceof nn.Module, 'E9NFSContainer is instance of Module');
assert(e9nfs.pathDim === pathDim, 'E9NFSContainer has correct pathDim');
assert(e9nfs.contentDim === contentDim, 'E9NFSContainer has correct contentDim');
assert(e9nfs.reservoirSize === reservoirSize, 'E9NFSContainer has correct reservoirSize');
assert(e9nfs.numMembranes === numMembranes, 'E9NFSContainer has correct numMembranes');

// Test 11: E9NFSContainer forward pass
console.log(`\n${BLUE}Testing E9NFSContainer forward pass...${RESET}`);
const nfsInput = {
  operation: 'read',
  path: '/usr/local/lib/module.js',
  mode: 'r'
};

const nfsOutput = e9nfs.forward(nfsInput);

assert(nfsOutput !== undefined, 'E9NFSContainer forward returns output');
assert(nfsOutput.features !== undefined, 'Output contains features');
assert(Array.isArray(nfsOutput.features), 'Features is array');
assert(nfsOutput.features.length === outputSize, 'Features has correct output size');

assert(nfsOutput.e9fs !== undefined, 'Output contains E9FS component');
assert(nfsOutput.temporal !== undefined, 'Output contains temporal component');
assert(nfsOutput.hierarchical !== undefined, 'Output contains hierarchical component');
assert(nfsOutput.predictions !== undefined, 'Output contains predictions');

// Test 12: E9NFSContainer auxiliary predictions
console.log(`\n${BLUE}Testing E9NFSContainer auxiliary predictions...${RESET}`);
assert(nfsOutput.predictions.nextAccess !== undefined, 'Has next access prediction');
assert(Array.isArray(nfsOutput.predictions.nextAccess), 'Next access is array');
assert(nfsOutput.predictions.nextAccess.length === pathDim, 'Next access has correct dimension');

assert(typeof nfsOutput.predictions.shouldCache === 'number', 'Has cache decision');
assert(nfsOutput.predictions.prefetchPriority !== undefined, 'Has prefetch priority');

// Test 13: E9NFSContainer sequence processing
console.log(`\n${BLUE}Testing E9NFSContainer sequence processing...${RESET}`);
const sequence = [
  { operation: 'open', path: '/home/user/file1.txt', mode: 'r' },
  { operation: 'read', path: '/home/user/file1.txt' },
  { operation: 'close', path: '/home/user/file1.txt' },
  { operation: 'open', path: '/home/user/file2.txt', mode: 'r' },
  { operation: 'read', path: '/home/user/file2.txt' }
];

e9nfs.resetStates();
const seqOutput = e9nfs.forward(sequence);

assert(Array.isArray(seqOutput), 'Sequence processing returns array');
assert(seqOutput.length === sequence.length, 'Output length matches input length');

for (let i = 0; i < seqOutput.length; i++) {
  assert(seqOutput[i].features !== undefined, `Output ${i} has features`);
  assert(seqOutput[i].predictions !== undefined, `Output ${i} has predictions`);
}

// Test 14: E9NFSContainer predictNextAccess
console.log(`\n${BLUE}Testing E9NFSContainer predictNextAccess...${RESET}`);
const recentOps = [
  { operation: 'read', path: '/var/log/app.log' },
  { operation: 'read', path: '/var/log/error.log' },
  { operation: 'read', path: '/var/log/access.log' }
];

const prediction = e9nfs.predictNextAccess(recentOps);

assert(prediction !== undefined, 'Prediction is generated');
assert(prediction.predictedPath !== undefined, 'Has predicted path');
assert(prediction.shouldCache !== undefined, 'Has cache decision');
assert(prediction.prefetchPriority !== undefined, 'Has prefetch priority');
assert(prediction.confidence !== undefined, 'Has confidence score');
assert(typeof prediction.confidence === 'number', 'Confidence is number');

// Test 15: E9NFSContainer analyzeAccessPatterns
console.log(`\n${BLUE}Testing E9NFSContainer analyzeAccessPatterns...${RESET}`);
const opsToAnalyze = [
  { operation: 'open', path: '/data/file1.dat', mode: 'r' },
  { operation: 'read', path: '/data/file1.dat' },
  { operation: 'close', path: '/data/file1.dat' },
  { operation: 'open', path: '/data/file2.dat', mode: 'r' },
  { operation: 'read', path: '/data/file2.dat' },
  { operation: 'close', path: '/data/file2.dat' },
  { operation: 'open', path: '/data/file1.dat', mode: 'r' },
  { operation: 'read', path: '/data/file1.dat' }
];

const analysis = e9nfs.analyzeAccessPatterns(opsToAnalyze);

assert(analysis !== undefined, 'Analysis is generated');
assert(analysis.operations === opsToAnalyze.length, 'Analysis tracks operation count');
assert(analysis.uniquePaths > 0, 'Analysis tracks unique paths');
assert(analysis.temporal !== undefined, 'Analysis includes temporal patterns');
assert(analysis.hierarchical !== undefined, 'Analysis includes hierarchical structure');
assert(analysis.caching !== undefined, 'Analysis includes caching statistics');
assert(typeof analysis.caching.cacheRate === 'number', 'Cache rate is number');
assert(analysis.caching.cacheRate >= 0 && analysis.caching.cacheRate <= 1, 'Cache rate is valid probability');

// Test 16: E9NFSContainer state management
console.log(`\n${BLUE}Testing E9NFSContainer state management...${RESET}`);
e9nfs.resetStates();
const initialStates = e9nfs.getInternalStates();

assert(initialStates !== undefined, 'Can get internal states');
assert(initialStates.e9fs !== undefined, 'States include E9FS');
assert(initialStates.reservoir !== undefined, 'States include reservoir');
assert(initialStates.membrane !== undefined, 'States include membrane');

// Process some operations
e9nfs.forward({ operation: 'read', path: '/test/state.txt' });

const afterStates = e9nfs.getInternalStates();
assert(afterStates.e9fs.historyLength > 0, 'E9FS state changes after operations');

// Test 17: E9NFSContainer parameters
console.log(`\n${BLUE}Testing E9NFSContainer parameters...${RESET}`);
const params = e9nfs.parameters();

assert(params !== undefined, 'Can get parameters');
assert(params.parameters !== undefined, 'Has parameters array');
assert(params.gradParameters !== undefined, 'Has gradient parameters array');
assert(params.parameters.length > 0, 'Has learnable parameters');
assert(params.parameters.length === params.gradParameters.length, 'Parameters and gradients match');

// Test 18: E9NFSContainer backward pass
console.log(`\n${BLUE}Testing E9NFSContainer backward pass...${RESET}`);
e9nfs.resetStates();
const forwardOut = e9nfs.forward({ operation: 'read', path: '/test/grad.txt' });

// Create dummy gradient
const gradOut = {
  features: new Array(outputSize).fill(0.1)
};

try {
  const gradInput = e9nfs.backward(
    { operation: 'read', path: '/test/grad.txt' },
    gradOut
  );
  assert(true, 'Backward pass completes without error');
} catch (e) {
  assert(false, `Backward pass error: ${e.message}`);
}

// Test 19: E9NFSContainer processBatch
console.log(`\n${BLUE}Testing E9NFSContainer processBatch...${RESET}`);
const batch = [
  { operation: 'read', path: '/batch/file1.txt' },
  { operation: 'read', path: '/batch/file2.txt' },
  { operation: 'read', path: '/batch/file3.txt' }
];

e9nfs.resetStates();
const batchOutput = e9nfs.processBatch(batch);

assert(Array.isArray(batchOutput), 'Batch output is array');
assert(batchOutput.length === batch.length, 'Batch output length matches input');

for (let i = 0; i < batchOutput.length; i++) {
  assert(batchOutput[i].features !== undefined, `Batch output ${i} has features`);
}

// Test 20: E9NFSContainer temporal patterns
console.log(`\n${BLUE}Testing E9NFSContainer temporal pattern recognition...${RESET}`);
const temporalSequence = [
  { operation: 'read', path: '/logs/2024-01-01.log' },
  { operation: 'read', path: '/logs/2024-01-02.log' },
  { operation: 'read', path: '/logs/2024-01-03.log' }
];

e9nfs.resetStates();
const tempOutput = e9nfs.forward(temporalSequence);

assert(Array.isArray(tempOutput), 'Temporal sequence returns outputs');
for (let i = 0; i < tempOutput.length; i++) {
  const patterns = tempOutput[i].temporal.patterns;
  assert(patterns !== undefined, `Output ${i} has temporal patterns`);
  assert(typeof patterns.energy === 'number', `Output ${i} has energy metric`);
  assert(typeof patterns.sparsity === 'number', `Output ${i} has sparsity metric`);
}

// Test 21: E9FSLayer hierarchical path encoding
console.log(`\n${BLUE}Testing E9FSLayer hierarchical path encoding...${RESET}`);
const shallowPath = { operation: 'stat', path: '/home' };
const deepPath = { operation: 'stat', path: '/home/user/docs/project/src/main.js' };

const shallowOut = e9fs.forward(shallowPath);
const deepOut = e9fs.forward(deepPath);

assert(shallowOut.structuredEmbedding !== undefined, 'Shallow path has structured embedding');
assert(deepOut.structuredEmbedding !== undefined, 'Deep path has structured embedding');
assert(shallowOut.result.depth < deepOut.result.depth, 'Deep path has higher depth score');

// Test 22: Integration test - Learn from access pattern
console.log(`\n${BLUE}Testing E9NFS learning from access patterns...${RESET}`);
e9nfs.resetStates();
const learningSequence = [
  { operation: 'open', path: '/project/src/index.js', mode: 'r' },
  { operation: 'read', path: '/project/src/index.js' },
  { operation: 'open', path: '/project/src/utils.js', mode: 'r' },
  { operation: 'read', path: '/project/src/utils.js' },
  { operation: 'open', path: '/project/src/config.js', mode: 'r' },
  { operation: 'read', path: '/project/src/config.js' }
];

// Process sequence multiple times to "learn" pattern
for (let i = 0; i < 3; i++) {
  e9nfs.forward(learningSequence);
}

// Now predict
const finalPrediction = e9nfs.predictNextAccess([
  { operation: 'open', path: '/project/src/index.js', mode: 'r' },
  { operation: 'read', path: '/project/src/index.js' }
]);

assert(finalPrediction !== undefined, 'Learning produces prediction');
assert(finalPrediction.predictedPath !== undefined, 'Prediction includes path');
assert(finalPrediction.confidence >= 0, 'Confidence is non-negative');

console.log('\n' + '='.repeat(50));
console.log(`${GREEN}Tests passed: ${passed}${RESET}`);
console.log(`${RED}Tests failed: ${failed}${RESET}`);
console.log('='.repeat(50) + '\n');

// Exit with appropriate code
process.exit(failed > 0 ? 1 : 0);
