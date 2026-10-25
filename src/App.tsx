import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera,
  Upload, 
  Trash2, 
  Leaf, 
  FileText, 
  AlertTriangle, 
  CheckCircle2, 
  RefreshCw, 
  Globe, 
  Cpu, 
  Info, 
  WifiOff,
  Sparkles,
  HelpCircle,
  ExternalLink
} from 'lucide-react';

interface Prediction {
  label: string;
  percentage: number;
}

interface GuidanceItem {
  title: string;
  bin: string;
  guidance: string;
  doNotMix: string;
  badgeColor: string;
}

interface ClassificationResult {
  category: 'paper_cardboard' | 'plastic' | 'organic' | 'uncertain_unsupported';
  confidence: number;
  detectedItem: string;
  reasonEn: string;
  reasonHi: string;
  modelName: string;
  modelSource: string;
  license: string;
  allPredictions?: Prediction[];
  guidance: {
    en: GuidanceItem;
    hi: GuidanceItem;
  };
}

interface SampleImage {
  id: string;
  title: string;
  filename: string;
  url: string;
  expected: string;
  badge: string;
}

interface ModelInfo {
  primaryModel: {
    name: string;
    source: string;
    architecture: string;
    format: string;
    license: string;
    dataset: string;
    cost: string;
    quotaLimits: string;
    knownLimitations: string;
  };
  secondaryModel: {
    name: string;
    source: string;
    architecture: string;
    license: string;
    cost: string;
    quotaLimits: string;
    knownStrengths: string;
  };
}

export default function App() {
  const [lang, setLang] = useState<'en' | 'hi'>('en');
  const [modelChoice, setModelChoice] = useState<'gemini' | 'wastewise'>('gemini');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedImageName, setSelectedImageName] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [showModelModal, setShowModelModal] = useState<boolean>(false);
  const [isOffline, setIsOffline] = useState<boolean>(!navigator.onLine);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Fetch sample images list
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => setSamples(data.samples || []))
      .catch((err) => console.error('Failed to load samples', err));

    // Fetch model info
    fetch('/api/model-info')
      .then((res) => res.json())
      .then((data) => setModelInfo(data))
      .catch((err) => console.error('Failed to load model info', err));

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Handle Gallery File Selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError(lang === 'en' ? 'Please upload a valid image file (JPG, PNG, WebP).' : 'कृपया वैध फोटो फाइल चुनें (JPG, PNG, WebP)।');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const base64 = event.target?.result as string;
      setSelectedImage(base64);
      setSelectedImageName(file.name);
      classifyImage(base64, modelChoice);
    };
    reader.onerror = () => {
      setError(lang === 'en' ? 'Failed to read image from device.' : 'डिवाइस से फोटो लोड करने में त्रुटि हुई।');
    };
    reader.readAsDataURL(file);
  };

  // Run Real Inference
  const classifyImage = async (base64Data: string, modelToUse: 'wastewise' | 'gemini') => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64: base64Data,
          modelChoice: modelToUse,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server returned error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error('Classification request failed:', err);
      if (!navigator.onLine) {
        setError(lang === 'en' 
          ? 'Network is offline. Please check your internet connection.' 
          : 'इंटरनेट कनेक्शन उपलब्ध नहीं है। कृपया नेटवर्क जांचें।');
      } else {
        setError(err.message || (lang === 'en' ? 'Inference failed. Please try again.' : 'वर्गीकरण विफल रहा। कृपया पुनः प्रयास करें।'));
      }
    } finally {
      setLoading(false);
    }
  };

  // Load sample image
  const handleSelectSample = async (sample: SampleImage) => {
    setError(null);
    setSelectedImageName(sample.title);
    setLoading(true);
    try {
      const res = await fetch(sample.url);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        setSelectedImage(base64);
        classifyImage(base64, modelChoice);
      };
      reader.readAsDataURL(blob);
    } catch (err: any) {
      setError(lang === 'en' ? 'Failed to load sample image.' : 'सैंपल फोटो लोड करने में त्रुटि।');
      setLoading(false);
    }
  };

  // Switch model and re-classify if image is present
  const handleModelChange = (newModel: 'wastewise' | 'gemini') => {
    setModelChoice(newModel);
    if (selectedImage) {
      classifyImage(selectedImage, newModel);
    }
  };

  // Reset
  const handleReset = () => {
    setSelectedImage(null);
    setSelectedImageName('');
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (cameraInputRef.current) {
      cameraInputRef.current.value = '';
    }
  };

  const currentGuidance = result?.guidance ? result.guidance[lang] : null;

  return (
    <div className="max-w-md mx-auto min-h-screen bg-slate-50 flex flex-col shadow-sm">
      {/* Offline Alert Banner */}
      {isOffline && (
        <div className="bg-amber-500 text-white text-xs px-4 py-2 font-medium flex items-center justify-between sticky top-0 z-50 shadow-sm">
          <div className="flex items-center gap-2">
            <WifiOff className="w-4 h-4 shrink-0" />
            <span>{lang === 'en' ? 'Offline Mode detected. Reconnect for cloud inference.' : 'ऑफ़लाइन मोड। क्लाउड वर्गीकरण हेतु इंटरनेट कनेक्ट करें।'}</span>
          </div>
        </div>
      )}

      {/* Header Bar */}
      <header className="bg-emerald-700 text-white px-4 py-3.5 sticky top-0 z-40 flex items-center justify-between shadow-md">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-800/80 flex items-center justify-center border border-emerald-500/40">
            <Leaf className="w-5 h-5 text-emerald-300" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight leading-tight">EcoSort AI</h1>
            <p className="text-[11px] text-emerald-100 font-medium leading-none">
              {lang === 'en' ? 'Waste Segregation Assistant' : 'कचरा पृथक्करण सहायक'}
            </p>
          </div>
        </div>

        {/* Language & Info Buttons */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setLang(lang === 'en' ? 'hi' : 'en')}
            className="flex items-center gap-1 text-xs bg-emerald-800/80 hover:bg-emerald-800 text-emerald-100 px-2.5 py-1.5 rounded-md border border-emerald-500/30 font-medium transition-colors"
            title="Toggle Language"
            aria-label="Toggle Language"
          >
            <Globe className="w-3.5 h-3.5" />
            <span>{lang === 'en' ? 'हिन्दी' : 'English'}</span>
          </button>

          <button
            onClick={() => setShowModelModal(true)}
            className="p-1.5 bg-emerald-800/80 hover:bg-emerald-800 text-emerald-100 rounded-md border border-emerald-500/30"
            title="Model Info & Transparency"
            aria-label="Model Information"
          >
            <Info className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-4 flex flex-col gap-4">
        {/* Model Selection Selector */}
        <div className="bg-white rounded-xl border border-slate-200 p-3 shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-emerald-600" />
              <span className="text-xs font-semibold text-slate-700">
                {lang === 'en' ? 'Classification Engine' : 'वर्गीकरण इंजन'}
              </span>
            </div>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 font-medium border border-emerald-200">
              {modelChoice === 'wastewise' ? '100% Free / On-Device ONNX' : 'Free Tier Cloud API'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => handleModelChange('wastewise')}
              className={`text-left p-2.5 rounded-lg border text-xs transition-all ${
                modelChoice === 'wastewise'
                  ? 'bg-emerald-50/70 border-emerald-500 text-emerald-950 font-semibold ring-1 ring-emerald-500'
                  : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
              }`}
            >
              <div className="font-semibold text-[11px] text-slate-800 flex items-center gap-1">
                <span>WasteWise YOLOv8</span>
              </div>
              <p className="text-[10px] text-slate-500 mt-0.5 leading-tight">
                {lang === 'en' ? 'Waste-trained ONNX (15k images)' : 'अपशिष्ट-प्रशिक्षित मॉडल (15k फोटो)'}
              </p>
            </button>

            <button
              onClick={() => handleModelChange('gemini')}
              className={`text-left p-2.5 rounded-lg border text-xs transition-all ${
                modelChoice === 'gemini'
                  ? 'bg-emerald-50/70 border-emerald-500 text-emerald-950 font-semibold ring-1 ring-emerald-500'
                  : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
              }`}
            >
              <div className="font-semibold text-[11px] text-slate-800 flex items-center gap-1">
                <span>Gemini 3.8 Flash</span>
                <Sparkles className="w-2.5 h-2.5 text-amber-500" />
              </div>
              <p className="text-[10px] text-slate-500 mt-0.5 leading-tight">
                {lang === 'en' ? 'Open-set cloud validator' : 'ओपन-सेट क्लाउड विश्लेषक'}
              </p>
            </button>
          </div>
        </div>

        {/* Supported Categories Badge List */}
        <div className="bg-slate-100/70 rounded-xl p-2.5 border border-slate-200 text-[11px]">
          <p className="text-slate-500 font-medium mb-1.5 flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            <span>{lang === 'en' ? 'Supported Categories (Strictly 3 Streams):' : 'समर्थित कचरा श्रेणियां (केवल 3 श्रेणियां):'}</span>
          </p>
          <div className="flex flex-wrap gap-1.5">
            <span className="px-2 py-0.5 bg-blue-100 text-blue-800 rounded-md font-medium border border-blue-200">
              📦 {lang === 'en' ? 'Paper/Cardboard' : 'कागज / गत्ता'}
            </span>
            <span className="px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded-md font-medium border border-emerald-200">
              🧴 {lang === 'en' ? 'Plastic Packaging' : 'प्लास्टिक पैकेजिंग'}
            </span>
            <span className="px-2 py-0.5 bg-green-100 text-green-800 rounded-md font-medium border border-green-200">
              🍎 {lang === 'en' ? 'Organic Scraps' : 'जैविक फल/सब्जी'}
            </span>
            <span className="px-2 py-0.5 bg-amber-100 text-amber-900 rounded-md font-medium border border-amber-200">
              ⚠️ {lang === 'en' ? 'Others → Uncertain' : 'अन्य → अनिश्चित'}
            </span>
          </div>
        </div>

        {/* Image Upload & Camera Capture Area */}
        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm text-center">
          {/* Gallery file input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            className="hidden"
            id="gallery-file-upload"
          />

          {/* Direct Camera capture input */}
          <input
            type="file"
            ref={cameraInputRef}
            onChange={handleFileChange}
            accept="image/*"
            capture="environment"
            className="hidden"
            id="camera-file-capture"
          />

          {!selectedImage ? (
            <div className="border-2 border-dashed border-emerald-300 bg-emerald-50/40 rounded-xl p-5 flex flex-col items-center justify-center transition-colors">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-700 shadow-xs">
                  <Camera className="w-6 h-6" />
                </div>
                <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-700 shadow-xs">
                  <Upload className="w-6 h-6" />
                </div>
              </div>
              <span className="text-sm font-semibold text-slate-800 mb-1">
                {lang === 'en' ? 'Take Photo or Choose from Gallery' : 'फोटो खींचें या गैलरी से चुनें'}
              </span>
              <span className="text-xs text-slate-500 max-w-[260px] mb-4">
                {lang === 'en' 
                  ? 'Capture waste with camera or upload from gallery to identify recycling stream' 
                  : 'कैमरे से फोटो लें या गैलरी से अपलोड कर सही कचरा श्रेणी जानें'}
              </span>

              {/* Action Buttons: Take Photo (Camera) alongside Gallery Upload */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-[320px]">
                <button
                  type="button"
                  onClick={() => cameraInputRef.current?.click()}
                  className="bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white text-xs font-semibold py-2.5 px-3 rounded-lg shadow-sm flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                >
                  <Camera className="w-4 h-4 shrink-0" />
                  <span>{lang === 'en' ? 'Take Photo' : 'कैमरे से फोटो लें'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-white hover:bg-slate-50 active:bg-slate-100 text-slate-700 border border-slate-300 text-xs font-semibold py-2.5 px-3 rounded-lg shadow-xs flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                >
                  <Upload className="w-4 h-4 shrink-0 text-emerald-600" />
                  <span>{lang === 'en' ? 'Browse Gallery' : 'गैलरी से चुनें'}</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center">
              <div className="relative w-full max-h-56 rounded-xl overflow-hidden bg-slate-100 border border-slate-200 flex items-center justify-center">
                <img
                  src={selectedImage}
                  alt="Uploaded waste item"
                  className="max-h-56 w-auto object-contain mx-auto"
                />
                {loading && (
                  <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-xs flex flex-col items-center justify-center text-white p-4">
                    <RefreshCw className="w-8 h-8 animate-spin text-emerald-400 mb-2" />
                    <span className="text-xs font-semibold">
                      {lang === 'en' ? 'Running Model Inference...' : 'मॉडल विश्लेषण जारी है...'}
                    </span>
                    <span className="text-[10px] text-slate-300 mt-1">
                      {modelChoice === 'wastewise' ? 'ONNX YOLOv8n-cls (224x224 RGB)' : 'Gemini 3.8 Flash Vision'}
                    </span>
                  </div>
                )}
              </div>

              <div className="flex items-center justify-between w-full mt-3 px-1">
                <span className="text-xs text-slate-600 truncate max-w-[150px] font-medium">
                  {selectedImageName}
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => cameraInputRef.current?.click()}
                    disabled={loading}
                    className="text-xs text-emerald-700 hover:text-emerald-800 font-medium px-2 py-1 bg-emerald-50 rounded border border-emerald-200 flex items-center gap-1 cursor-pointer"
                    title={lang === 'en' ? 'Retake Photo' : 'पुनः फोटो लें'}
                  >
                    <Camera className="w-3.5 h-3.5" />
                    <span>{lang === 'en' ? 'Retake' : 'कैमरा'}</span>
                  </button>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading}
                    className="text-xs text-emerald-700 hover:text-emerald-800 font-medium px-2 py-1 bg-emerald-50 rounded border border-emerald-200 flex items-center gap-1 cursor-pointer"
                    title={lang === 'en' ? 'Change Image from Gallery' : 'गैलरी से बदलें'}
                  >
                    <Upload className="w-3.5 h-3.5" />
                    <span>{lang === 'en' ? 'Gallery' : 'गैलरी'}</span>
                  </button>
                  <button
                    onClick={handleReset}
                    disabled={loading}
                    className="text-xs text-rose-600 hover:text-rose-700 font-medium p-1 hover:bg-rose-50 rounded cursor-pointer"
                    title="Remove Image"
                    aria-label="Remove Image"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Quick Test Samples (Specifically provided for rapid testing on Android phone) */}
          <div className="mt-4 pt-3 border-t border-slate-100 text-left">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-700">
                {lang === 'en' ? 'Quick Test with Sample Images:' : 'सैंपल फोटो से तुरंत जांचें:'}
              </span>
              <span className="text-[10px] text-slate-400 font-normal">
                {lang === 'en' ? 'Tap any sample' : 'क्लिक करके टेस्ट करें'}
              </span>
            </div>
            <div className="grid grid-cols-5 gap-1.5">
              {samples.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleSelectSample(s)}
                  disabled={loading}
                  className="group flex flex-col items-center bg-slate-50 hover:bg-emerald-50 border border-slate-200 hover:border-emerald-300 rounded-lg p-1 transition-all text-center"
                >
                  <img
                    src={s.url}
                    alt={s.title}
                    className="w-11 h-11 object-cover rounded-md border border-slate-200 mb-1"
                  />
                  <span className="text-[9px] text-slate-600 group-hover:text-emerald-800 font-medium leading-tight truncate w-full">
                    {s.title.split(' ')[0]}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Error State Banner */}
        {error && (
          <div className="bg-rose-50 border border-rose-200 rounded-xl p-3.5 flex items-start gap-3 text-rose-900 text-xs">
            <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold text-rose-800">
                {lang === 'en' ? 'Classification Error' : 'वर्गीकरण में त्रुटि'}
              </p>
              <p className="mt-0.5 text-rose-700 leading-relaxed">{error}</p>
              {selectedImage && (
                <button
                  onClick={() => classifyImage(selectedImage, modelChoice)}
                  className="mt-2 inline-flex items-center gap-1.5 text-xs font-semibold text-rose-700 bg-rose-100 hover:bg-rose-200 px-2.5 py-1 rounded-md"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>{lang === 'en' ? 'Retry Classification' : 'पुनः प्रयास करें'}</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* Result Card */}
        {result && currentGuidance && (
          <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-sm flex flex-col">
            {/* Header category badge banner */}
            <div
              className={`p-4 border-b ${
                result.category === 'paper_cardboard'
                  ? 'bg-blue-50/80 border-blue-200'
                  : result.category === 'plastic'
                  ? 'bg-emerald-50/80 border-emerald-200'
                  : result.category === 'organic'
                  ? 'bg-green-50/80 border-green-200'
                  : 'bg-amber-50/80 border-amber-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  {lang === 'en' ? 'Classification Result' : 'वर्गीकरण परिणाम'}
                </span>
                <span
                  className={`text-xs px-2.5 py-0.5 rounded-full font-bold ${
                    result.category === 'paper_cardboard'
                      ? 'bg-blue-600 text-white'
                      : result.category === 'plastic'
                      ? 'bg-emerald-600 text-white'
                      : result.category === 'organic'
                      ? 'bg-green-600 text-white'
                      : 'bg-amber-600 text-white'
                  }`}
                >
                  {result.category === 'uncertain_unsupported'
                    ? lang === 'en' ? 'Uncertain / Unsupported' : 'असमर्थित / अनिश्चित'
                    : currentGuidance.title}
                </span>
              </div>

              <div className="mt-2">
                <h3 className="text-base font-bold text-slate-900 flex items-center gap-1.5">
                  {result.category === 'uncertain_unsupported' ? (
                    <AlertTriangle className="w-5 h-5 text-amber-600" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                  )}
                  <span>{currentGuidance.title}</span>
                </h3>
                <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                  {lang === 'en' ? result.reasonEn : result.reasonHi}
                </p>
              </div>

              {/* Confidence Score Bar */}
              <div className="mt-3 pt-2.5 border-t border-slate-200/60 flex items-center justify-between text-xs">
                <div className="flex items-center gap-1.5 text-slate-600">
                  <span className="font-medium">{lang === 'en' ? 'Confidence:' : 'विश्वसनीयता:'}</span>
                  <span className="font-bold text-slate-800">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-28 bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      result.confidence >= 0.70 ? 'bg-emerald-600' : 'bg-amber-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(5, result.confidence * 100))}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Disposal Guidance Section */}
            <div className="p-4 flex flex-col gap-3.5">
              {/* Disposal Bin Target */}
              <div className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-50 border border-slate-200">
                <div className="w-9 h-9 rounded-lg bg-emerald-100 flex items-center justify-center text-emerald-800 shrink-0">
                  <Trash2 className="w-5 h-5" />
                </div>
                <div>
                  <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">
                    {lang === 'en' ? 'Designated Bin / Stream' : 'उचित डिब्बा (कचरा पात्र)'}
                  </span>
                  <p className="text-xs font-bold text-slate-900 mt-0.5">
                    {currentGuidance.bin}
                  </p>
                </div>
              </div>

              {/* Instructions */}
              <div>
                <h4 className="text-xs font-bold text-slate-800 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  <span>{lang === 'en' ? 'Correct Disposal Instructions' : 'सही निस्तारण निर्देश'}</span>
                </h4>
                <p className="text-xs text-slate-600 leading-relaxed bg-slate-50/80 p-2.5 rounded-lg border border-slate-100">
                  {currentGuidance.guidance}
                </p>
              </div>

              {/* What NOT to mix */}
              <div>
                <h4 className="text-xs font-bold text-amber-900 mb-1 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                  <span>{lang === 'en' ? 'Do NOT Mix (Contamination Warning)' : 'यह सामग्री न मिलाएं (सावधानी)'}</span>
                </h4>
                <p className="text-xs text-amber-900/90 leading-relaxed bg-amber-50/70 p-2.5 rounded-lg border border-amber-200/80">
                  {currentGuidance.doNotMix}
                </p>
              </div>

              {/* If model provides all predictions (WasteWise ONNX), show raw breakdown */}
              {result.allPredictions && result.allPredictions.length > 0 && (
                <div className="pt-2 border-t border-slate-100">
                  <span className="text-[11px] font-semibold text-slate-500 mb-1.5 block">
                    {lang === 'en' ? 'Raw Model Predictions (WasteWise 8-Class):' : 'मॉडल का वर्गवार स्कोर (8 श्रेणियां):'}
                  </span>
                  <div className="grid grid-cols-2 gap-1 text-[10px]">
                    {result.allPredictions.map((pred) => (
                      <div
                        key={pred.label}
                        className={`px-2 py-1 rounded flex justify-between items-center ${
                          pred.percentage > 15 ? 'bg-emerald-50 text-emerald-900 font-semibold' : 'bg-slate-50 text-slate-600'
                        }`}
                      >
                        <span className="capitalize">{pred.label}</span>
                        <span>{pred.percentage.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Attribution / License Footer */}
              <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-400">
                <span>Model: {result.modelName.split(' ')[0]}</span>
                <span>License: {result.license}</span>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Model Transparency & Technical Details Modal */}
      {showModelModal && modelInfo && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs z-50 flex items-end sm:items-center justify-center p-3">
          <div className="bg-white rounded-2xl max-w-md w-full max-h-[85vh] overflow-y-auto p-5 text-slate-800 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-emerald-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  {lang === 'en' ? 'Model Documentation & Cost' : 'मॉडल विवरण व लागत'}
                </h3>
              </div>
              <button
                onClick={() => setShowModelModal(false)}
                className="text-slate-400 hover:text-slate-600 p-1 rounded-md text-sm font-bold"
              >
                ✕
              </button>
            </div>

            {/* Model 1: WasteWise YOLOv8 */}
            <div className="bg-slate-50 rounded-xl p-3.5 border border-slate-200 text-xs flex flex-col gap-1.5">
              <span className="font-bold text-emerald-800 text-[13px]">
                {modelInfo.primaryModel.name}
              </span>
              <p className="text-slate-600">
                <strong className="text-slate-700">Source:</strong> {modelInfo.primaryModel.source}
              </p>
              <p className="text-slate-600">
                <strong className="text-slate-700">Architecture:</strong> {modelInfo.primaryModel.architecture}
              </p>
              <p className="text-slate-600">
                <strong className="text-slate-700">Weights File:</strong> {modelInfo.primaryModel.format}
              </p>
              <p className="text-slate-600">
                <strong className="text-slate-700">License:</strong> {modelInfo.primaryModel.license}
              </p>
              <p className="text-slate-600">
                <strong className="text-slate-700">Training Dataset:</strong> {modelInfo.primaryModel.dataset}
              </p>
              <p className="text-slate-600">
                <strong className="text-slate-700">Cost & Quotas:</strong> {modelInfo.primaryModel.cost} ({modelInfo.primaryModel.quotaLimits})
              </p>
              <div className="mt-1 p-2 bg-amber-50 rounded border border-amber-200 text-[11px] text-amber-900">
                <strong>Known Limitation:</strong> {modelInfo.primaryModel.knownLimitations}
              </div>
            </div>

            {/* Model 2: Gemini 3.8 Flash Vision */}
            <div className="bg-slate-50 rounded-xl p-3.5 border border-slate-200 text-xs flex flex-col gap-1.5">
              <span className="font-bold text-blue-800 text-[13px]">
                {modelInfo.secondaryModel.name}
              </span>
              <p className="text-slate-600">
                <strong className="text-slate-700">Source:</strong> {modelInfo.secondaryModel.source}
              </p>
              <p className="text-slate-600">
                <strong className="text-slate-700">Cost:</strong> {modelInfo.secondaryModel.cost}
              </p>
              <p className="text-slate-600">
                <strong className="text-slate-700">Free Quota:</strong> {modelInfo.secondaryModel.quotaLimits}
              </p>
              <div className="mt-1 p-2 bg-emerald-50 rounded border border-emerald-200 text-[11px] text-emerald-900">
                <strong>Open-Set Strength:</strong> {modelInfo.secondaryModel.knownStrengths}
              </div>
            </div>

            <button
              onClick={() => setShowModelModal(false)}
              className="w-full py-2.5 bg-emerald-700 text-white font-semibold text-xs rounded-xl shadow-xs hover:bg-emerald-800"
            >
              {lang === 'en' ? 'Close Transparency Report' : 'विवरण बंद करें'}
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
