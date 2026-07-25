import Nav from "./Nav";

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-[#0a0e1a] text-white">
      <Nav />
      <main>{children}</main>
      <footer className="border-t border-slate-800 mt-24">
        <div className="max-w-6xl mx-auto px-6 py-8 text-sm text-slate-500">
          MediFlow is a portfolio research project. Not a diagnostic device.
          Not for clinical use.
        </div>
      </footer>
    </div>
  );
}
