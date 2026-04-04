const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function analyzeLogs(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    body: form
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Server error");
  }

  return res.json();
}
