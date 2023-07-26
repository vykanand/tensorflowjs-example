const tf = require('@tensorflow/tfjs-node');
// const tf = require('@tensorflow/tfjs-node-gpu'); // Use tfjs-node-gpu instead of tfjs-node


// Define the time series data and parameters
const givenValues = [10,20,30,60];
const numPredictions = 1; // Number of future values to predict

// Prepare the input data for training the model
const inputSequence = tf.tensor1d(givenValues);
const targetValues = [20,30,40,70];
const outputSequence = tf.tensor1d(targetValues);

const emaFactor = 0.9

// Build and train the linear regression model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 100000, inputShape: [1],activation: 'relu' })); // Add 16 neurons
model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Output layer


model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

const targetLossThreshold = 1;
let loss = Number.MAX_VALUE;

async function train() {
  while (loss >= targetLossThreshold) {
    const history = await model.fit(inputSequence, outputSequence, { epochs: 100, batchSize: 1, verbose: 0 });
    loss = history.history.loss[0];
    console.log(`Accuracy Loss: ${loss}`);
  }

  // Now you can use the trained model to predict future values
  // const lastInputValue = inputSequence.dataSync()[inputSequence.size - 1];
  // const nextValueTensor = model.predict(tf.tensor2d([[32]]));
  // const nextValue = Math.round(nextValueTensor.dataSync()[0]);
  // console.log(nextValue);
  
  const predictValues = [35,28];
  const final = [];
  let emaValue = predictValues[0];
  for (let i = 0; i < predictValues.length; i++) {
  const nextValueTensor = model.predict(tf.tensor2d([[predictValues[i]]]));
  const nextValue = Math.round(nextValueTensor.dataSync()[0]);
  emaValue = emaFactor * nextValue + (1 - emaFactor) * emaValue;
  // final.push(Math.round(emaValue));
  final.push(nextValue)
  }
  console.log("Final :", final);

  // const futureValues = [...givenValues, ...predictedValues];
  // console.log("Future Values:", futureValues);
}

train()
  .catch(err => {
    console.error("Error training the model:", err);
  });
