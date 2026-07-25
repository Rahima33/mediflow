import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function runTriage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await axios.post(`${API_URL}/triage`, formData);
  return data;
}
