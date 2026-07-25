import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import HowItWorks from "./pages/HowItWorks";
import AboutModel from "./pages/AboutModel";
import Predict from "./pages/Predict";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/about-model" element={<AboutModel />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
