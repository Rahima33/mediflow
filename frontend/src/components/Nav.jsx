import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Home" },
  { to: "/how-it-works", label: "How It Works" },
  { to: "/about-model", label: "About the Model" },
  { to: "/predict", label: "Predict" },
];

export default function Nav() {
  return (
    <nav className="border-b border-slate-800 bg-[#0a0e1a]/95 backdrop-blur sticky top-0 z-50">
      <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4">
        <span className="text-white font-semibold text-lg tracking-tight">
          MediFlow
        </span>
        <div className="flex gap-8">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `text-sm transition-colors ${
                  isActive ? "text-teal-400" : "text-slate-400 hover:text-white"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
}
