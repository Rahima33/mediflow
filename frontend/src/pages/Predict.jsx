import { useRef, useState } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { runTriage } from "../api/triage";

export default function Predict() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const reportRef = useRef(null);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await runTriage(file);
      setResult(data);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          "Something went wrong analyzing this image. Confirm the backend is running and try again."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleDownload() {
    if (!reportRef.current) return;
    const canvas = await html2canvas(reportRef.current, {
      backgroundColor: "#0a0e1a",
      scale: 2,
    });
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF("p", "pt", "a4");
    const pageWidth = pdf.internal.pageSize.getWidth();
    const imgWidth = pageWidth - 40;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    pdf.addImage(imgData, "PNG", 20, 20, imgWidth, imgHeight);
    pdf.save(`mediflow-report-${Date.now()}.pdf`);
  }

  const isPneumonia = result?.prediction === "PNEUMONIA";

  return (
    <section className="max-w-4xl mx-auto px-6 pt-24 pb-24">
      <h1 className="text-3xl font-bold tracking-tight mb-4">
        Analyze an X-ray
      </h1>
      <p className="text-slate-400 mb-10">
        Upload a chest X-ray image. The result is a screening aid, not a
        diagnosis — clinician review is always required.
      </p>

      {!result && (
        <div className="bg-slate-900/50 border border-dashed border-slate-700 rounded-xl p-10 text-center">
          <input
            id="file-input"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <label htmlFor="file-input" className="cursor-pointer">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected X-ray preview"
                className="max-h-64 mx-auto rounded-lg mb-4"
              />
            ) : (
              <p className="text-slate-400 mb-2">
                Click to choose an X-ray image
              </p>
            )}
            <span className="text-sm text-teal-400">
              {file ? "Choose a different file" : "Browse files"}
            </span>
          </label>

          {file && (
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="block mx-auto mt-6 bg-gradient-to-r from-blue-500 to-teal-500 px-8 py-3 rounded-full font-medium disabled:opacity-50"
            >
              {loading ? "Analyzing…" : "Run Analysis"}
            </button>
          )}

          {error && (
            <p className="text-red-400 text-sm mt-4">{error}</p>
          )}
        </div>
      )}

      {result && (
        <>
          <div ref={reportRef} className="bg-[#0a0e1a] p-2">
            <div className="flex items-center justify-between mb-6">
              <span
                className={`px-4 py-1.5 rounded-full text-sm font-semibold ${
                  isPneumonia
                    ? "bg-amber-500/15 text-amber-400"
                    : "bg-teal-500/15 text-teal-400"
                }`}
              >
                {result.prediction}
              </span>
              <span className="text-slate-400 text-sm">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-8">
              <div>
                <p className="text-xs text-slate-500 mb-2">Original</p>
                <img
                  src={previewUrl}
                  alt="Original X-ray"
                  className="rounded-lg w-full border border-slate-800"
                />
              </div>
              <div>
                <p className="text-xs text-slate-500 mb-2">
                  Grad-CAM Explainability
                </p>
                <img
                  src={`data:image/png;base64,${result.gradcam_base64}`}
                  alt="Grad-CAM heatmap"
                  className="rounded-lg w-full border border-slate-800"
                />
              </div>
            </div>

            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-5">
              <div>
                <h3 className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                  Finding
                </h3>
                <p className="text-sm leading-relaxed">{result.finding}</p>
              </div>
              <div>
                <h3 className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                  Clinical Context
                </h3>
                <p className="text-sm leading-relaxed text-slate-300">
                  {result.clinical_context}
                </p>
              </div>
              <div>
                <h3 className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                  Next Steps
                </h3>
                <p className="text-sm leading-relaxed text-slate-300">
                  {result.next_steps}
                </p>
              </div>
              <div className="border-t border-slate-800 pt-4">
                <p className="text-xs text-slate-500 leading-relaxed">
                  {result.disclaimer}
                </p>
              </div>
            </div>
          </div>

          <div className="flex gap-4 mt-8">
            <button
              onClick={handleDownload}
              className="bg-gradient-to-r from-blue-500 to-teal-500 px-6 py-2.5 rounded-full text-sm font-medium"
            >
              Download Report (PDF)
            </button>
            <button
              onClick={() => {
                setResult(null);
                setFile(null);
                setPreviewUrl(null);
              }}
              className="border border-slate-700 px-6 py-2.5 rounded-full text-sm font-medium text-slate-300 hover:border-slate-500"
            >
              Analyze Another
            </button>
          </div>
        </>
      )}
    </section>
  );
}
