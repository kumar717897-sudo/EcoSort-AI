import ort from 'onnxruntime-node';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

const CLASS_NAMES = [
  'battery',
  'biological',
  'cardboard',
  'glass',
  'metal',
  'paper',
  'plastic',
  'trash',
];

async function runTest() {
  const modelPath = path.resolve('public/models/wastewise-yolo.onnx');
  console.log('Loading model from:', modelPath);
  const session = await ort.InferenceSession.create(modelPath);

  console.log('Input names:', session.inputNames);
  console.log('Output names:', session.outputNames);

  const samples = [
    { name: 'plastic_bottle.jpg', expectedCategory: 'Plastic' },
    { name: 'cardboard.jpg', expectedCategory: 'Paper/Cardboard' },
    { name: 'fruit_scraps.jpg', expectedCategory: 'Organic fruit/vegetable scraps' },
    { name: 'person.jpg', expectedCategory: 'Uncertain/Unsupported' },
    { name: 'alarm_clock.jpg', expectedCategory: 'Uncertain/Unsupported' },
  ];

  for (const sample of samples) {
    const imgPath = path.resolve('public/samples', sample.name);
    // Resize to 224x224 and get raw RGB values
    const { data, info } = await sharp(imgPath)
      .resize(224, 224, { fit: 'fill' })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Format: 1 x 3 x 224 x 224 Float32Array normalized to [0, 1]
    const float32Data = new Float32Array(1 * 3 * 224 * 224);
    const pixelCount = 224 * 224;

    for (let i = 0; i < pixelCount; i++) {
      float32Data[i] = data[i * 3 + 0] / 255.0; // R
      float32Data[pixelCount + i] = data[i * 3 + 1] / 255.0; // G
      float32Data[pixelCount * 2 + i] = data[i * 3 + 2] / 255.0; // B
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
    const inputName = session.inputNames[0];
    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;

    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];
    const rawScores = Array.from(outputTensor.data as Float32Array);

    // Apply softmax if raw scores are logits, or check if already softmax
    const maxVal = Math.max(...rawScores);
    const expScores = rawScores.map(v => Math.exp(v - maxVal));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map(v => v / sumExp);

    const classScores = CLASS_NAMES.map((cls, idx) => ({
      class: cls,
      raw: rawScores[idx],
      prob: probs[idx],
    })).sort((a, b) => b.prob - a.prob);

    console.log(`\n========================================`);
    console.log(`Test: ${sample.name} (Expected: ${sample.expectedCategory})`);
    console.log(`Top 3 Predictions:`);
    for (let j = 0; j < 3; j++) {
      console.log(`  ${j + 1}. ${classScores[j].class}: ${(classScores[j].prob * 100).toFixed(2)}% (raw: ${classScores[j].raw.toFixed(4)})`);
    }

    // Waste mapping logic:
    // Supported categories:
    // 1. Paper/Cardboard: cardboard (idx 2), paper (idx 5)
    // 2. Plastic: plastic (idx 6)
    // 3. Organic fruit/vegetable scraps: biological (idx 1)
    //
    // Non-waste / unsupported classes:
    // battery, glass, metal, trash
    // Also, if confidence < 60% or margin between top1 and top2 is low, uncertain.
  }
}

runTest().catch(console.error);
