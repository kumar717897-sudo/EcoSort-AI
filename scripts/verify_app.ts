import fs from 'fs';
import path from 'path';
import ort from 'onnxruntime-node';
import sharp from 'sharp';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const ONNX_CLASSES = [
  'battery',
  'biological',
  'cardboard',
  'glass',
  'metal',
  'paper',
  'plastic',
  'trash',
];

async function runComprehensiveVerification() {
  console.log('=== EcoSort AI Comprehensive Verification ===\n');

  // 1. Verify Model Weights existence
  const modelPath = path.resolve('public/models/wastewise-yolo.onnx');
  const exists = fs.existsSync(modelPath);
  const stats = fs.statSync(modelPath);
  console.log(`[CHECK 1] Model File: ${modelPath}`);
  console.log(`          Exists: ${exists}, Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);

  // 2. Load ONNX Session
  const session = await ort.InferenceSession.create(modelPath);
  console.log(`[CHECK 2] ONNX Session Initialized. Input: ${session.inputNames}, Output: ${session.outputNames}\n`);

  const samples = [
    { filename: 'plastic_bottle.jpg', expected: 'Plastic' },
    { filename: 'cardboard.jpg', expected: 'Paper/Cardboard' },
    { filename: 'fruit_scraps.jpg', expected: 'Organic fruit/vegetable scraps' },
    { filename: 'person.jpg', expected: 'Uncertain/Unsupported' },
    { filename: 'alarm_clock.jpg', expected: 'Uncertain/Unsupported' },
  ];

  console.log('=== Testing WasteWise YOLOv8n-cls (Dedicated Waste Model) ===');
  for (const sample of samples) {
    const imgPath = path.resolve('public/samples', sample.filename);
    const { data } = await sharp(imgPath)
      .resize(224, 224, { fit: 'fill' })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixelCount = 224 * 224;
    const float32Data = new Float32Array(1 * 3 * 224 * 224);
    for (let i = 0; i < pixelCount; i++) {
      float32Data[i] = data[i * 3 + 0] / 255.0;
      float32Data[pixelCount + i] = data[i * 3 + 1] / 255.0;
      float32Data[pixelCount * 2 + i] = data[i * 3 + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
    const results = await session.run({ [session.inputNames[0]]: tensor });
    const rawScores = Array.from(results[session.outputNames[0]].data as Float32Array);

    const predictions = ONNX_CLASSES.map((className, idx) => ({
      className,
      score: rawScores[idx],
    })).sort((a, b) => b.score - a.score);

    const top1 = predictions[0];
    const top2 = predictions[1];
    console.log(`Sample: ${sample.filename} (Expected: ${sample.expected})`);
    console.log(`  Top-1: ${top1.className} (${(top1.score * 100).toFixed(2)}%)`);
    console.log(`  Top-2: ${top2.className} (${(top2.score * 100).toFixed(2)}%)`);
  }

  console.log('\n=== Testing Gemini 3.8 Flash Vision (Cloud Free Tier) ===');
  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
  });

  for (const sample of samples) {
    const imgPath = path.resolve('public/samples', sample.filename);
    const base64 = fs.readFileSync(imgPath).toString('base64');

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: {
        parts: [
          { inlineData: { mimeType: 'image/jpeg', data: base64 } },
          {
            text: `Classify for waste sorting into one of: 'paper_cardboard', 'plastic', 'organic', or 'uncertain_unsupported'.
If the image shows a person, animal, electronic device, furniture, or non-recyclable item, choose 'uncertain_unsupported'.`,
          },
        ],
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            category: { type: Type.STRING },
            confidence: { type: Type.NUMBER },
            detectedItem: { type: Type.STRING },
          },
          required: ['category', 'confidence', 'detectedItem'],
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    console.log(`Sample: ${sample.filename}`);
    console.log(`  Category: ${parsed.category}, Confidence: ${(parsed.confidence * 100).toFixed(1)}%, Detected: "${parsed.detectedItem}"`);
  }

  console.log('\n=== Verification Completed Successfully ===');
}

runComprehensiveVerification().catch(console.error);
