import { Fragment } from "react";

// Shows an answer as plain text. Numbers like [1] become links to the source list.
// Nothing here uses raw HTML, so text from the model cannot run as a script.

function withCitations(text: string, sourceCount: number) {
  const pieces = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
  return pieces.map((piece, i) => {
    const match = piece.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
    if (!match) return <Fragment key={i}>{piece}</Fragment>;
    const numbers = match[1].split(",").map((n) => parseInt(n.trim(), 10));
    return (
      <Fragment key={i}>
        [
        {numbers.map((n, j) => (
          <Fragment key={j}>
            {j > 0 && ", "}
            {n >= 1 && n <= sourceCount ? (
              <a href={`#source-${n}`} className="text-blue-600 hover:underline dark:text-blue-400">
                {n}
              </a>
            ) : (
              n
            )}
          </Fragment>
        ))}
        ]
      </Fragment>
    );
  });
}

export default function AnswerText({ text, sourceCount }: { text: string; sourceCount: number }) {
  // Odd-numbered parts are code blocks (the text between ``` marks).
  const parts = text.split("```");
  return (
    <div className="space-y-3 leading-relaxed">
      {parts.map((part, i) => {
        if (i % 2 === 1) {
          const code = part.replace(/^[a-zA-Z0-9_+-]*\n/, "");
          return (
            <pre
              key={i}
              className="overflow-x-auto rounded-md bg-slate-100 p-3 text-sm dark:bg-slate-900"
            >
              <code>{code.trimEnd()}</code>
            </pre>
          );
        }
        return part.trim() ? (
          <p key={i} className="whitespace-pre-wrap">
            {withCitations(part.trim(), sourceCount)}
          </p>
        ) : null;
      })}
    </div>
  );
}
