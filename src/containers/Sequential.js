const Module = require('../Module');

/**
 * Sequential - Container for chaining modules in sequence
 * 
 * Feeds the output of one module to the input of the next module in a sequential manner.
 */
class Sequential extends Module {
  constructor(...modules) {
    super();
    this.modules = modules.length === 1 && Array.isArray(modules[0]) ? modules[0] : modules;
  }

  /**
   * Add a module to the sequence
   * @param {Module} module - Module to add
   * @returns {Sequential} this
   */
  add(module) {
    this.modules.push(module);
    return this;
  }

  forward(input) {
    let output = input;
    for (const module of this.modules) {
      output = module.forward(output);
    }
    this.output = output;
    return this.output;
  }

  backward(input, gradOutput) {
    let currentGradOutput = gradOutput;
    let currentInput = input;

    // Store intermediate outputs for backward pass
    const outputs = [input];
    for (const module of this.modules) {
      outputs.push(module.output);
    }

    // Backward pass through modules in reverse
    for (let i = this.modules.length - 1; i >= 0; i--) {
      currentGradOutput = this.modules[i].backward(outputs[i], currentGradOutput);
    }

    this.gradInput = currentGradOutput;
    return this.gradInput;
  }

  updateGradInput(input, gradOutput) {
    return this.backward(input, gradOutput);
  }

  parameters() {
    const params = [];
    const gradParams = [];

    for (const module of this.modules) {
      const moduleParams = module.parameters();
      params.push(...moduleParams.parameters);
      gradParams.push(...moduleParams.gradParameters);
    }

    return {
      parameters: params,
      gradParameters: gradParams
    };
  }

  train() {
    super.train();
    for (const module of this.modules) {
      module.train();
    }
    return this;
  }

  evaluate() {
    super.evaluate();
    for (const module of this.modules) {
      module.evaluate();
    }
    return this;
  }

  zeroGradParameters() {
    for (const module of this.modules) {
      module.zeroGradParameters();
    }
  }

  toString() {
    const moduleStrings = this.modules.map((m, i) => `  (${i}): ${m.toString()}`).join('\n');
    return `Sequential {\n${moduleStrings}\n}`;
  }
}

module.exports = Sequential;
