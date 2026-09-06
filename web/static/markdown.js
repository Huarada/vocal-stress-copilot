// Minimal, dependency-free Markdown renderer for the Analyst Agent's chat replies.
//
// The chat panel previously used `textContent`, so the model's Markdown (**bold**,
// `code`, bullet lists) reached the reviewer as literal asterisks and backticks.
//
// SECURITY: the rendered text is LLM output, and that model is fed transcripts of
// whatever a candidate said out loud — it is not trusted content. HTML is therefore
// escaped FIRST and Markdown applied to the escaped text, so any tags the model emits
// become visible characters and never live markup. The only real HTML in the output is
// the tags this function itself generates.
//
// Supports the subset the agent actually emits: headings, **bold**, *italic*, `code`,
// bullet and numbered lists, paragraphs. Anything else degrades to plain text.

(function (root) {
  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderInline(text) {
    // Code spans are extracted to placeholders first so that emphasis markers *inside*
    // code (e.g. `a ** b`) can't pair with markers outside it.
    const codeSpans = [];
    let working = text.replace(/`([^`]+)`/g, function (_, code) {
      codeSpans.push(code);
      return "\u0000CODE" + (codeSpans.length - 1) + "\u0000";
    });

    working = working
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

    return working.replace(/\u0000CODE(\d+)\u0000/g, function (_, index) {
      return "<code>" + codeSpans[Number(index)] + "</code>";
    });
  }

  function renderMarkdown(raw) {
    if (!raw) return "";

    const lines = escapeHtml(raw).split(/\r?\n/);
    const out = [];
    let paragraph = [];
    let listTag = null;

    function flushParagraph() {
      if (paragraph.length) {
        out.push("<p>" + renderInline(paragraph.join(" ")) + "</p>");
        paragraph = [];
      }
    }

    function closeList() {
      if (listTag) {
        out.push("</" + listTag + ">");
        listTag = null;
      }
    }

    function openList(tag) {
      if (listTag !== tag) {
        closeList();
        out.push("<" + tag + ">");
        listTag = tag;
      }
    }

    for (const line of lines) {
      const trimmed = line.trim();

      if (!trimmed) {
        flushParagraph();
        closeList();
        continue;
      }

      const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        flushParagraph();
        closeList();
        out.push("<h4>" + renderInline(heading[2]) + "</h4>");
        continue;
      }

      const bullet = trimmed.match(/^[*-]\s+(.*)$/);
      if (bullet) {
        flushParagraph();
        openList("ul");
        out.push("<li>" + renderInline(bullet[1]) + "</li>");
        continue;
      }

      const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/);
      if (numbered) {
        flushParagraph();
        openList("ol");
        out.push("<li>" + renderInline(numbered[1]) + "</li>");
        continue;
      }

      closeList();
      paragraph.push(trimmed);
    }

    flushParagraph();
    closeList();
    return out.join("");
  }

  root.renderMarkdown = renderMarkdown;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { renderMarkdown, escapeHtml };
  }
})(typeof window !== "undefined" ? window : globalThis);
