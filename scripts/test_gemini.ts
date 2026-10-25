import { GoogleGenAI, Type } from '@google/genai';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

const samples = [
  'plastic_bottle.jpg',
  'cardboard.jpg',
  'fruit_scraps.jpg',
  'person.jpg',
  'alarm_clock.jpg',
];

async function testGemini() {
  console.log('Testing Gemini 3.8 Flash vision on the 5 samples...');
  for (const filename of samples) {
    const filePath = path.resolve('public/samples', filename);
    const data = fs.readFileSync(filePath);
    const base64 = data.toString('base64');

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: base64,
            },
          },
          {
            text: `Analyze this image for waste sorting. We only support three target categories:
1. 'paper_cardboard' (Paper or cardboard waste)
2. 'plastic' (Plastic bottles, containers, bags, packaging)
3. 'organic' (Fruit/vegetable scraps, food peels, organic compostable matter)

If the image is NOT one of these three waste types, or if it is a person, animal, electronic device (alarm clock), or non-waste object, or unclear, classify as 'uncertain_unsupported'.

Return a JSON object with:
- category: "paper_cardboard" | "plastic" | "organic" | "uncertain_unsupported"
- confidence: number between 0 and 1
- detectedItem: brief name of what is in the image
- reason: brief 1-sentence reason
`,
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
            reason: { type: Type.STRING },
          },
          required: ['category', 'confidence', 'detectedItem', 'reason'],
        },
      },
    });

    console.log(`\nSample: ${filename}`);
    console.log(response.text);
  }
}

testGemini().catch(console.error);
