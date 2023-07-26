const tf = require('@tensorflow/tfjs-node');

// Define the time series data and parameters
const givenValues = [10,20,30];
const targetValues = [20,40,60];

// Prepare the input data for training the model
const inputSequence = tf.tensor1d(givenValues); // Normalize the input data

const outputSequence = tf.tensor1d(targetValues); // Normalize the output data

const emaFactor = 0.9;

// Build and train the neural network model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 100000, inputShape: [1], activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Output layer

model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

const targetLossThreshold = 0.1;
let loss = Number.MAX_VALUE;

async function waitTraining() {
  while (loss >= targetLossThreshold) {
    const history = await model.fit(inputSequence, outputSequence, { epochs: 100, batchSize: 1, verbose: 0 });
    loss = history.history.loss[0];
    console.log(`Accuracy Loss: ${loss}`);
  }
}

async function predict(inp) {
  await waitTraining();

  const predictValues = inp; // Normalize the input for prediction
  const final = [];
  let emaValue = predictValues[0];
  for (let i = 0; i < predictValues.length; i++) {
    const nextValueTensor = model.predict(tf.tensor2d([[predictValues[i]]]));
    const nextValue = Math.round(nextValueTensor.dataSync()[0]); // Denormalize the output
    emaValue = emaFactor * nextValue + (1 - emaFactor) * emaValue;
    final.push(nextValue);
  }
  console.log("Final:", final);
}


predict([16, 35]).catch(err => {
    console.error("Error training the model:", err);
  });
