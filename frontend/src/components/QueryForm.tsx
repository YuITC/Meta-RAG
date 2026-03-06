"use client";

type Props = { onSubmit: (query: string) => void; loading: boolean };

export default function QueryForm({ onSubmit, loading }: Props) {
  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const query = (form.elements.namedItem("query") as HTMLTextAreaElement).value.trim();
    if (query) onSubmit(query);
  }

  return (
    <form onSubmit={handleSubmit}>
      <textarea
        name="query"
        rows={3}
        placeholder="Enter your research question..."
        disabled={loading}
        style={{
          width: "100%",
          padding: "12px 16px",
          background: "#1a1a1a",
          border: "1px solid #333",
          borderRadius: 8,
          color: "#e8e8e8",
          fontSize: 15,
          resize: "vertical",
          outline: "none",
        }}
      />
      <button
        type="submit"
        disabled={loading}
        style={{
          marginTop: 12,
          padding: "10px 24px",
          background: loading ? "#333" : "#2563eb",
          color: "#fff",
          border: "none",
          borderRadius: 6,
          fontSize: 14,
          fontWeight: 600,
          cursor: loading ? "not-allowed" : "pointer",
        }}
      >
        {loading ? "Researching..." : "Ask"}
      </button>
    </form>
  );
}
