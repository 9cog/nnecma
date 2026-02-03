/**
 * E9NFS Example - Neural Filesystem with Adaptive Access Patterns
 * 
 * This example demonstrates how E9NFS (Echo-State-Neural-Filesystem) can learn
 * and predict filesystem access patterns using a combination of:
 * - E9FS: Learnable Plan9 filesystem operations
 * - Reservoir Computing: Temporal pattern recognition
 * - Membrane Computing: Hierarchical structure modeling
 */

const nn = require('../src/index');

console.log('='.repeat(70));
console.log('E9NFS - Neural Filesystem with Adaptive Access Patterns');
console.log('='.repeat(70));
console.log();

// Configuration
const pathDim = 32;        // Dimension of path embeddings
const contentDim = 64;     // Dimension of content embeddings
const reservoirSize = 50;  // Size of echo-state reservoir
const numMembranes = 4;    // Number of membrane compartments
const outputSize = 16;     // Output feature dimension

console.log('Configuration:');
console.log(`  Path Dimension: ${pathDim}`);
console.log(`  Content Dimension: ${contentDim}`);
console.log(`  Reservoir Size: ${reservoirSize}`);
console.log(`  Membranes: ${numMembranes}`);
console.log(`  Output Size: ${outputSize}`);
console.log();

// Create E9NFS model
console.log('Creating E9NFS model...');
const e9nfs = new nn.E9NFSContainer(
  pathDim,
  contentDim,
  reservoirSize,
  numMembranes,
  outputSize,
  {
    e9fs: {
      maxDepth: 10,
      cacheSize: 32,
      adaptiveRate: 0.1
    },
    reservoir: {
      spectralRadius: 0.95,  // High memory for temporal patterns
      leakRate: 0.8,         // Moderate dynamics
      sparsity: 0.15         // Sparse connectivity
    },
    membrane: {
      objectTypes: 12,       // Symbolic objects for hierarchy
      ruleComplexity: 6,     // Evolution rules
      communicationRate: 0.4 // Inter-membrane transfer
    }
  }
);

console.log('Model created:');
console.log(e9nfs.toString());
console.log();

// Simulate a development workflow with file access patterns
console.log('='.repeat(70));
console.log('Scenario 1: Development Workflow - Learning Access Patterns');
console.log('='.repeat(70));
console.log();

const devWorkflow = [
  // Developer starts working on a feature
  { operation: 'open', path: '/project/src/index.js', mode: 'r' },
  { operation: 'read', path: '/project/src/index.js' },
  
  // Checks utilities
  { operation: 'open', path: '/project/src/utils/helper.js', mode: 'r' },
  { operation: 'read', path: '/project/src/utils/helper.js' },
  
  // Checks config
  { operation: 'open', path: '/project/config/app.json', mode: 'r' },
  { operation: 'read', path: '/project/config/app.json' },
  
  // Goes back to main file
  { operation: 'open', path: '/project/src/index.js', mode: 'r' },
  { operation: 'read', path: '/project/src/index.js' },
  
  // Writes changes
  { operation: 'write', path: '/project/src/index.js', content: 'Updated code...' },
  { operation: 'close', path: '/project/src/index.js' }
];

console.log('Processing development workflow...');
e9nfs.resetStates();

for (let i = 0; i < devWorkflow.length; i++) {
  const op = devWorkflow[i];
  const output = e9nfs.forward(op);
  
  console.log(`\nStep ${i + 1}: ${op.operation} ${op.path}`);
  console.log(`  Cache Decision: ${output.predictions.shouldCache > 0.5 ? 'YES' : 'NO'} (${output.predictions.shouldCache.toFixed(3)})`);
  console.log(`  Temporal Energy: ${output.temporal.patterns.energy.toFixed(3)}`);
  console.log(`  Active Membranes: ${output.hierarchical.structure.activeMembranes}`);
}

console.log();

// Analyze the access pattern
console.log('='.repeat(70));
console.log('Analysis: Access Pattern Statistics');
console.log('='.repeat(70));
console.log();

const analysis = e9nfs.analyzeAccessPatterns(devWorkflow);

console.log('Pattern Analysis Results:');
console.log(`  Total Operations: ${analysis.operations}`);
console.log(`  Unique Paths: ${analysis.uniquePaths}`);
console.log(`  Cache Rate: ${(analysis.caching.cacheRate * 100).toFixed(1)}%`);
console.log(`  Average Cache Confidence: ${analysis.caching.averageConfidence.toFixed(3)}`);
console.log(`  Mean Temporal Energy: ${analysis.temporal.meanEnergy.toFixed(3)}`);
console.log(`  Max Temporal Energy: ${analysis.temporal.maxEnergy.toFixed(3)}`);
console.log(`  Mean Hierarchical Activity: ${analysis.hierarchical.meanActivity.toFixed(3)}`);
console.log();

// Prediction: What file will be accessed next?
console.log('='.repeat(70));
console.log('Scenario 2: Predictive Prefetching');
console.log('='.repeat(70));
console.log();

const recentAccess = [
  { operation: 'read', path: '/project/src/index.js' },
  { operation: 'read', path: '/project/src/utils/helper.js' }
];

console.log('Given recent access pattern:');
recentAccess.forEach((op, i) => {
  console.log(`  ${i + 1}. ${op.operation} ${op.path}`);
});

const prediction = e9nfs.predictNextAccess(recentAccess);

console.log('\nPrediction for next access:');
console.log(`  Predicted Path (embedding): [${prediction.predictedPath.slice(0, 5).map(x => x.toFixed(3)).join(', ')}...]`);
console.log(`  Should Cache: ${prediction.shouldCache > 0.5 ? 'YES' : 'NO'}`);
console.log(`  Confidence: ${prediction.confidence.toFixed(3)}`);
console.log(`  Prefetch Priority (top 5): [${prediction.prefetchPriority.slice(0, 5).map(x => x.toFixed(3)).join(', ')}...]`);
console.log();

// Log file access pattern - typical server scenario
console.log('='.repeat(70));
console.log('Scenario 3: Log File Monitoring Pattern');
console.log('='.repeat(70));
console.log();

const logPattern = [
  { operation: 'open', path: '/var/log/app/2024-01-15.log', mode: 'r' },
  { operation: 'read', path: '/var/log/app/2024-01-15.log' },
  { operation: 'stat', path: '/var/log/app/2024-01-15.log' },
  { operation: 'close', path: '/var/log/app/2024-01-15.log' },
  
  { operation: 'open', path: '/var/log/app/2024-01-16.log', mode: 'r' },
  { operation: 'read', path: '/var/log/app/2024-01-16.log' },
  { operation: 'stat', path: '/var/log/app/2024-01-16.log' },
  { operation: 'close', path: '/var/log/app/2024-01-16.log' },
  
  { operation: 'open', path: '/var/log/error/2024-01-15.log', mode: 'r' },
  { operation: 'read', path: '/var/log/error/2024-01-15.log' },
  { operation: 'close', path: '/var/log/error/2024-01-15.log' }
];

console.log('Processing log monitoring pattern...');
e9nfs.resetStates();

const logOutputs = logPattern.map(op => e9nfs.forward(op));

console.log('\nTemporal Pattern Evolution:');
logOutputs.forEach((output, i) => {
  const op = logPattern[i];
  console.log(`  ${i + 1}. ${op.operation.padEnd(6)} ${op.path}`);
  console.log(`     Energy: ${output.temporal.patterns.energy.toFixed(3)}, ` +
              `Sparsity: ${output.temporal.patterns.sparsity.toFixed(3)}, ` +
              `Variance: ${output.temporal.patterns.variance.toFixed(3)}`);
});

console.log();

// Hierarchical structure analysis
console.log('='.repeat(70));
console.log('Scenario 4: Hierarchical Directory Navigation');
console.log('='.repeat(70));
console.log();

const hierarchicalAccess = [
  { operation: 'stat', path: '/home' },
  { operation: 'stat', path: '/home/user' },
  { operation: 'stat', path: '/home/user/documents' },
  { operation: 'stat', path: '/home/user/documents/project' },
  { operation: 'stat', path: '/home/user/documents/project/src' },
  { operation: 'stat', path: '/home/user/documents/project/src/main.js' }
];

console.log('Analyzing hierarchical navigation:');
e9nfs.resetStates();

hierarchicalAccess.forEach(op => {
  const output = e9nfs.forward(op);
  const depth = op.path.split('/').filter(s => s.length > 0).length;
  
  console.log(`\nDepth ${depth}: ${op.path}`);
  console.log(`  Membrane Activity: [${output.hierarchical.structure.membraneActivity.map(a => a.toFixed(2)).join(', ')}]`);
  console.log(`  Active Membranes: ${output.hierarchical.structure.activeMembranes}`);
  console.log(`  Total Activity: ${output.hierarchical.structure.totalActivity.toFixed(3)}`);
});

console.log();

// Batch processing
console.log('='.repeat(70));
console.log('Scenario 5: Batch File Processing');
console.log('='.repeat(70));
console.log();

const batchFiles = [
  { operation: 'read', path: '/data/input/file001.dat' },
  { operation: 'read', path: '/data/input/file002.dat' },
  { operation: 'read', path: '/data/input/file003.dat' },
  { operation: 'read', path: '/data/input/file004.dat' },
  { operation: 'read', path: '/data/input/file005.dat' }
];

console.log('Processing batch of files...');
e9nfs.resetStates();

const batchResults = e9nfs.processBatch(batchFiles);

console.log(`\nBatch Results (${batchResults.length} files):`);
batchResults.forEach((result, i) => {
  console.log(`  File ${i + 1}: Cache=${result.predictions.shouldCache.toFixed(3)}, ` +
              `Energy=${result.temporal.patterns.energy.toFixed(3)}`);
});

// Calculate average cache recommendation
const avgCache = batchResults.reduce((sum, r) => sum + r.predictions.shouldCache, 0) / batchResults.length;
console.log(`\nAverage Cache Recommendation: ${avgCache.toFixed(3)}`);
console.log();

// State inspection
console.log('='.repeat(70));
console.log('Internal State Inspection');
console.log('='.repeat(70));
console.log();

const states = e9nfs.getInternalStates();

console.log('E9FS State:');
console.log(`  Open Files: ${states.e9fs.openFiles}`);
console.log(`  Cached Files: ${states.e9fs.cachedFiles}`);
console.log(`  History Length: ${states.e9fs.historyLength}`);

console.log('\nReservoir State:');
const reservoirSample = states.reservoir.slice(0, 10);
console.log(`  Sample (first 10): [${reservoirSample.map(x => x.toFixed(3)).join(', ')}...]`);
console.log(`  State Size: ${states.reservoir.length}`);

console.log('\nMembrane States:');
states.membrane.forEach((membrane, i) => {
  const sum = membrane.reduce((s, v) => s + Math.abs(v), 0);
  console.log(`  Membrane ${i + 1} Activity: ${sum.toFixed(3)}`);
});

console.log();

// Summary
console.log('='.repeat(70));
console.log('Summary');
console.log('='.repeat(70));
console.log();

console.log('E9NFS successfully demonstrated:');
console.log('  ✓ Learnable filesystem operations (E9FS)');
console.log('  ✓ Temporal pattern recognition (Reservoir)');
console.log('  ✓ Hierarchical structure modeling (Membrane)');
console.log('  ✓ Predictive prefetching and caching');
console.log('  ✓ Adaptive access pattern learning');
console.log('  ✓ Batch processing capabilities');
console.log();

console.log('The neural filesystem combines:');
console.log('  • Plan9 filesystem concepts (everything is a file)');
console.log('  • Echo State Networks (temporal dynamics)');
console.log('  • Membrane Computing (P-systems for hierarchy)');
console.log('  • Deep learning (learnable parameters)');
console.log();

console.log('Applications:');
console.log('  • Intelligent file caching systems');
console.log('  • Predictive filesystem prefetching');
console.log('  • Adaptive storage optimization');
console.log('  • Pattern-based access control');
console.log('  • Smart file navigation assistants');
console.log();

console.log('='.repeat(70));
console.log('Example completed successfully!');
console.log('='.repeat(70));
