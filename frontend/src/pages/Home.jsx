import { Link } from "react-router-dom";

const features = [
  {
    title: "DenseNet121 Classification",
    desc: "Frozen-backbone CNN, transfer learning, 89% test accuracy on validated chest X-ray data.",
  },
  {
    title: "Grad-CAM Explainability",
    desc: "Every prediction shows exactly which image regions drove the model's decision.",
  },
  {
    title: "CRAG-Grounded Reports",
    desc: "Clinical guideline retrieval, graded for relevance before grounding any report.",
  },
];

export default function Home() {
  return (
    <>
      <section className="max-w-4xl mx-auto text-center px-6 pt-32 pb-20">
        <h1 className="text-5xl font-bold tracking-tight leading-tight">
          AI-Assisted Chest X-ray{" "}
          <span className="bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent">
            Triage
          </span>
        </h1>
        <p className="text-slate-400 mt-6 text-lg max-w-2xl mx-auto">
          A transparent, explainable pipeline for pneumonia screening —
          classification, visual reasoning, and clinically-grounded reporting
          in one flow.
        </p>
        <Link
          to="/predict"
          className="inline-block mt-8 bg-gradient-to-r from-blue-500 to-teal-500 text-white px-8 py-3 rounded-full font-medium hover:opacity-90 transition-opacity"
        >
          Analyze an X-ray →
        </Link>
      </section>

      <section className="max-w-5xl mx-auto px-6 pb-24 grid md:grid-cols-3 gap-6">
        {features.map((f) => (
          <div
            key={f.title}
            className="bg-slate-900/50 border border-slate-800 rounded-xl p-6"
          >
            <h3 className="font-semibold mb-2">{f.title}</h3>
            <p className="text-slate-400 text-sm">{f.desc}</p>
          </div>
        ))}
      </section>
    </>
  );
}
