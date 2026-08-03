const stats = [
  { label: "Accuracy", value: "89%" },
  { label: "Macro F1", value: "0.88" },
  { label: "NORMAL Recall", value: "86%" },
  { label: "PNEUMONIA Recall", value: "91%" },
];

export default function AboutModel() {
  return (
    <section className="max-w-3xl mx-auto px-6 pt-24 pb-24">
      <h1 className="text-3xl font-bold tracking-tight mb-4">
        About the Model
      </h1>
      <p className="text-slate-400 mb-12">
        The deployed checkpoint is best_model_xrv_backbone.pth: an
        X-ray-pretrained DenseNet121 backbone from torchxrayvision, fine-tuned
        for binary chest X-ray triage on true bounding-box lung crops. It uses
        single-channel XRV normalization in the live API path.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {stats.map((s) => (
          <div
            key={s.label}
            className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 text-center"
          >
            <div className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent">
              {s.value}
            </div>
            <div className="text-xs text-slate-500 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      <h2 className="text-xl font-semibold mb-3">What was tried, and why</h2>
      <ul className="text-slate-400 text-sm space-y-3 mb-12 list-disc pl-5">
        <li>
          Class-weighted loss and a weighted sampler were required to correct
          a baseline model that had collapsed to predicting PNEUMONIA by
          default due to class imbalance.
        </li>
        <li>
          A frozen ImageNet-pretrained backbone outperformed a standard
          freeze-then-unfreeze strategy on this dataset size, likely because a
          shared learning rate across an unfrozen backbone caused catastrophic
          forgetting of useful pretrained features.
        </li>
        <li>
          Grad-CAM analysis revealed shortcut learning in ImageNet-backbone
          runs: several misclassified cases relied on chest-wall and shoulder
          regions rather than lung tissue, a failure mode not visible from
          accuracy alone.
        </li>
        <li>
          True lung-field bounding-box crops reduced peripheral bias more
          effectively than masking non-lung pixels to black, and improved the
          reported validation metrics.
        </li>
        <li>
          One persistent false-positive case still showed peripheral attention
          after true-crop preprocessing. Switching from an ImageNet backbone to
          an X-ray-pretrained torchxrayvision DenseNet121 resolved that case,
          indicating the shortcut came from the natural-image backbone rather
          than the crop pipeline.
        </li>
      </ul>

      <h2 className="text-xl font-semibold mb-3">Known limitations</h2>
      <ul className="text-slate-400 text-sm space-y-3 list-disc pl-5">
        <li>
          The XRV backbone resolved the previously audited shortcut-attention
          examples, but the full test set still needs a complete Grad-CAM audit.
        </li>
        <li>
          Trained on a public binary pneumonia dataset, not a clinically
          validated or regulator-reviewed dataset.
        </li>
        <li>
          High confidence does not guarantee a correct or well-localized
          prediction. This tool is a screening aid, not a diagnosis.
        </li>
      </ul>
    </section>
  );
}