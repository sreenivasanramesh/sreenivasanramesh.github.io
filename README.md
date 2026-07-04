# vasan.dev — personal site

Static, no build step. `index.html` + `css/style.css` + `js/main.js`.

## Preview locally

```sh
python3 -m http.server 8321
# → http://localhost:8321
```

## Deploy

Any static host works as-is:

- **GitHub Pages** — push to a repo, Settings → Pages → deploy from branch.
- **Netlify / Cloudflare Pages** — drag-and-drop the folder or connect the repo. No build command, publish directory = root.

## Notes

- Fonts: Archivo (variable) + Instrument Serif via Google Fonts.
- All animation respects `prefers-reduced-motion`.
- Content edits: everything lives in `index.html`; colors/typography in the
  `:root` block of `css/style.css`.
- Design decisions: `docs/superpowers/specs/2026-07-04-personal-site-design.md`.
