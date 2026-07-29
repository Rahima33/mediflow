const steps = [
  {
    title: "Upload",
    desc: "A chest X-ray image is submitted through the Predict tab.",
  },
  {
    title: "Preprocessing",
    desc: "A pretrained segmentation model crops the image to the lung-field bounding box, removing peripheral content that was found to bias earlier model versions.",
  },
  {
    title: "DenseNet121 Classification",
    desc: "An xrv-pretrained DenseNet121 backbone predicts NORMAL or PNEUMONIA with a confidence score and 89% accuracy.",
  },
  {
    title: "Grad-CAM Explainability",
    desc: "Gradient-weighted class activation mapping generates a heatmap showing which image regions drove the prediction.",
  },
  {
    title: "Confidence Gate",
    desc: "Predictions below a fixed confidence threshold are routed directly to a human-review flag, bypassing report generation.",
  },
  {
    title: "Guideline Retrieval (CRAG)",
    desc: "Relevant clinical guideline passages are retrieved from a vector knowledge base, then graded by a language model for genuine relevance before use.",
  },
  {
    title: "Report Generation",
    desc: "If retrieval is graded sufficient, a structured report is generated: finding, clinical context, next steps, and a standing disclaimer.",
  },
];

export default function HowItWorks() {
  return (
    <section className="max-w-3xl mx-auto px-6 pt-24 pb-24">
      <h1 className="text-3xl font-bold tracking-tight mb-4">How It Works</h1>
      <p className="text-slate-400 mb-16">
        Every case moves through the same pipeline. Two independent safety
        gates — confidence and retrieval sufficiency — can route a case to
        human review instead of generating a report.
      </p>

      <div className="relative border-l border-slate-800 pl-8 space-y-10">
        {steps.map((s, i) => (
          <div key={s.title} className="relative">
            <span className="absolute -left-[41px] top-0 w-4 h-4 rounded-full bg-gradient-to-r from-blue-400 to-teal-400" />
            <h3 className="font-semibold">{s.title}</h3>
            <p className="text-slate-400 text-sm mt-1">{s.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
