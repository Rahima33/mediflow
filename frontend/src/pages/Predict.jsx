import { useEffect, useRef, useState } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { runTriage } from "../api/triage";

export default function Predict() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const reportRef = useRef(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function handleFile(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose a chest X-ray image file.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
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

  function resetCase() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setResult(null);
    setFile(null);
    setPreviewUrl(null);
    setError(null);
  }

  const isPneumonia = result?.prediction === "PNEUMONIA";
  const confidencePct = result ? (result.confidence * 100).toFixed(1) : null;
  const reviewStatus = result?.needs_review
    ? "Human review required"
    : "Report generated";
  const reviewTone = result?.needs_review
    ? "border-amber-500/30 bg-amber-500/10 text-amber-200"
    : "border-teal-500/30 bg-teal-500/10 text-teal-200";

  return (
    <section className="max-w-6xl mx-auto px-6 pt-16 pb-24">
      <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-teal-300">
            Triage workspace
          </p>
          <h1 className="mt-3 text-3xl font-bold tracking-tight">
            Analyze a chest X-ray
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            Upload a frontal chest X-ray for model classification, Grad-CAM
            review, and a structured screening report. This is not a diagnosis;
            clinician review is always required.
          </p>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/70 px-4 py-3 text-xs text-slate-400">
          <span className="font-semibold text-slate-200">Safety gate:</span>{" "}
          low-confidence cases are flagged for human review.
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="space-y-4">
          <div
            className={`rounded-lg border p-5 transition-colors ${
              isDragging
                ? "border-teal-400 bg-teal-500/10"
                : "border-slate-800 bg-slate-900/45"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold">Image input</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Drag an image here or choose a file to begin.
                </p>
              </div>
              {file && (
                <button
                  type="button"
                  onClick={resetCase}
                  className="rounded-md border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 hover:border-slate-500"
                >
                  Clear
                </button>
              )}
            </div>

            <input
              id="file-input"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />

            <label
              htmlFor="file-input"
              className="block cursor-pointer rounded-lg border border-dashed border-slate-700 bg-slate-950/50 p-4 text-center hover:border-slate-500"
            >
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Selected X-ray preview"
                  className="mx-auto max-h-[360px] w-full rounded-md object-contain"
                />
              ) : (
                <div className="flex min-h-[280px] flex-col items-center justify-center">
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-md border border-slate-700 text-xl text-teal-300">
                    +
                  </div>
                  <p className="text-sm font-medium text-slate-200">
                    Choose chest X-ray image
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    PNG, JPG, or JPEG
                  </p>
                </div>
              )}
            </label>

            {file && (
              <div className="mt-4 rounded-md border border-slate-800 bg-slate-950/60 p-3">
                <p className="truncate text-sm font-medium text-slate-200">
                  {file.name}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB |{" "}
                  {file.type || "image file"}
                </p>
              </div>
            )}

            {error && (
              <p className="mt-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                {error}
              </p>
            )}

            <button
              onClick={handleSubmit}
              disabled={!file || loading}
              className="mt-5 w-full rounded-md bg-teal-500 px-5 py-3 text-sm font-semibold text-slate-950 transition-colors hover:bg-teal-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
            >
              {loading ? "Analyzing..." : result ? "Run analysis again" : "Run analysis"}
            </button>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/35 p-5">
            <h2 className="text-base font-semibold">Review checklist</h2>
            <div className="mt-4 space-y-3 text-sm text-slate-400">
              <p>Confirm image quality and patient positioning before relying on output.</p>
              <p>Use Grad-CAM only as a reasoning aid, not proof of pathology.</p>
              <p>Escalate urgent symptoms or discordant findings to clinician review.</p>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/45 p-5">
          {!result ? (
            <div className="flex min-h-[620px] flex-col justify-center text-center">
              <p className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">
                Awaiting analysis
              </p>
              <h2 className="mt-3 text-2xl font-semibold">Results will appear here</h2>
              <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-slate-400">
                After analysis, this panel will show prediction status,
                confidence, original imaging, Grad-CAM, and report sections.
              </p>
            </div>
          ) : (
            <>
              <div ref={reportRef} className="bg-[#0a0e1a] p-2">
                <div className="mb-5 grid gap-3 sm:grid-cols-3">
                  <div
                    className={`rounded-lg border p-4 ${
                      isPneumonia
                        ? "border-amber-500/30 bg-amber-500/10"
                        : "border-teal-500/30 bg-teal-500/10"
                    }`}
                  >
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      Prediction
                    </p>
                    <p
                      className={`mt-2 text-xl font-bold ${
                        isPneumonia ? "text-amber-200" : "text-teal-200"
                      }`}
                    >
                      {result.prediction}
                    </p>
                  </div>

                  <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      Confidence
                    </p>
                    <p className="mt-2 text-xl font-bold text-white">
                      {confidencePct}%
                    </p>
                  </div>

                  <div className={`rounded-lg border p-4 ${reviewTone}`}>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      Status
                    </p>
                    <p className="mt-2 text-base font-semibold">
                      {reviewStatus}
                    </p>
                  </div>
                </div>

                <div className="mb-6 grid gap-4 sm:grid-cols-2">
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                      Original
                    </p>
                    <img
                      src={previewUrl}
                      alt="Original X-ray"
                      className="aspect-square w-full rounded-md border border-slate-800 object-contain"
                    />
                  </div>
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                      Grad-CAM
                    </p>
                    <img
                      src={`data:image/png;base64,${result.gradcam_base64}`}
                      alt="Grad-CAM heatmap"
                      className="aspect-square w-full rounded-md border border-slate-800 object-contain"
                    />
                  </div>
                </div>

                {result.needs_review && (
                  <div className="mb-5 rounded-md border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-100">
                    This case was flagged by the safety gate. Treat the report
                    as a screening note and route the image for human review.
                  </div>
                )}

                <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-950/50 p-5">
                  <section>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Finding
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-slate-100">
                      {result.finding}
                    </p>
                  </section>
                  <section>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Clinical context
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-slate-300">
                      {result.clinical_context}
                    </p>
                  </section>
                  <section>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Next steps
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-slate-300">
                      {result.next_steps}
                    </p>
                  </section>
                  <section className="border-t border-slate-800 pt-4">
                    <p className="text-xs leading-relaxed text-slate-500">
                      {result.disclaimer}
                    </p>
                  </section>
                </div>
              </div>

              <div className="mt-5 flex flex-col gap-3 sm:flex-row">
                <button
                  onClick={handleDownload}
                  className="rounded-md bg-teal-500 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-teal-400"
                >
                  Download report PDF
                </button>
                <button
                  onClick={resetCase}
                  className="rounded-md border border-slate-700 px-5 py-2.5 text-sm font-medium text-slate-300 hover:border-slate-500"
                >
                  Analyze another
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
