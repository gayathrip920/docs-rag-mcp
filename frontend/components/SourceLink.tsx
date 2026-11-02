import { safeHref } from "@/lib/api";

export default function SourceLink({ url, children }: { url: string; children: React.ReactNode }) {
  const href = safeHref(url);
  if (!href) return <span>{children}</span>;
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-600 hover:underline dark:text-blue-400"
    >
      {children}
    </a>
  );
}
