# EcoSort AI

EcoSort AI is a mobile-first waste classification assistant that helps users sort everyday items into three designated recycling and waste streams:

- **Paper & Cardboard** (Clean paper, cardboard packaging, newspapers)
- **Plastic Packaging** (Bottles, containers, dry packaging)
- **Organic Fruit & Vegetable Waste** (Fruit peels, banana peels, apple cores, food scraps)

If an item is ambiguous, unsupported, or non-waste (e.g. portraits, electronics, hazardous batteries), the app returns an **Uncertain / Unsupported** result rather than forcing the item into a wrong bin.

## Features

- **Mobile Camera Capture:** Directly opens the rear device camera (`accept="image/*" capture="environment"`).
- **Phone Gallery Upload:** Native file picker integration for uploaded photos.
- **Strict 3-Stream Classification:** Maps supported items and rejects non-waste items.
- **Bilingual Interface:** Instant one-tap toggle between English and Hindi (`हिन्दी`).
- **Disposal Guidance:** Specific bin instructions, cleaning tips, and contamination warnings.
- **Quick-Test Strip:** Pre-loaded sample images (banana peel, cardboard box, plastic bottle, apple core, person portrait, alarm clock) for immediate verification.

## Model and Technical Architecture

1. **Dedicated Waste Classifier (ONNX):**
   - **Model:** WasteWise YOLOv8n-cls
   - **Source:** Hugging Face (`Arihant0902/wastewise-garbage-cls`)
   - **Architecture:** YOLOv8 Nano Classification Head ($224 \times 224$ RGB)
   - **Weights File:** `wastewise-yolo.onnx` (5.54 MB, bundled in `/public/models/`)
   - **License:** Apache-2.0
   - **Training Dataset:** `garbage_office` (15,000 images, 8 classes: battery, biological, cardboard, glass, metal, paper, plastic, trash)
   - **Inference:** Local execution via `onnxruntime-node` and Sharp image preprocessing.

2. **Multimodal Open-Set Vision Engine:**
   - **Model:** Gemini Multimodal Flash (`@google/genai` SDK)
   - **Function:** Open-set out-of-distribution detection, validation of organic food scraps (e.g. banana peels), and bilingual reasoning.
   - **License:** Google AI Studio Terms of Service
   - **Free Quota:** Standard free-tier quota (15 RPM, 1,500 RPD).

## Getting Started

### Prerequisites
- Node.js 22+
- npm

### Installation & Run

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start development server:
   ```bash
   npm run dev
   ```
4. Build for production deployment:
   ```bash
   npm run build
   ```
5. Run production server:
   ```bash
   npm start
   ```

## Tech Stack

- **Frontend:** React 19, TypeScript, Tailwind CSS, Lucide Icons
- **Backend / Proxy:** Node.js, Express, Sharp, `@google/genai`
- **Machine Learning Runtime:** ONNX Runtime (`onnxruntime-node`)
- **Build Tool:** Vite 8

## Privacy

- Images uploaded or captured are sent to the local server proxy in-memory for inference.
- No images, personal data, or camera captures are persistently saved, indexed, or stored in any database.

## Known Limitations

- **Closed-set Classifier Constraints:** A standalone classifier trained only on waste images may map non-waste items to visually similar textures. The multimodal validator filters these cases into the **Uncertain** category.
- **Contamination Rules:** Local municipal guidelines vary by municipality. Always consult local waste management authorities for hazardous materials and electronic waste.

## License

Source code is available under the MIT License. Pretrained ONNX model weights (`wastewise-yolo.onnx`) are distributed under Apache-2.0.
