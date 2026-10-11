# personal site


## Preview locally

```sh
python3 -m http.server 8321
# → http://localhost:8321
```


## How it's put together

- The concept is a contact sheet. Type is Redaction (display), Switzer (text) and Departure Mono (labels), self-hosted in `assets/fonts/`. Grease red is the one accent.
- The hero print is a small WebGL shader in `js/main.js` (develop-in, grain, halation). Without WebGL it falls back to a plain image.
- Stickers come in two print runs, `assets/stickers/riso-*` and `photo-*`. Each visit gets one at random, and the hero line is a 50/50 pick between two drafts. Force either with `?set=riso|photo` and `?line=1|2`.
- Modes: Daylight, B&W (`.silver`) and Safelight (`.safe`) live as CSS variables in `:root`. The choice is saved in `localStorage`, and dark-mode systems default to Safelight.
- Showcase media lives in `assets/work/`. Edit the `WORK` list in `js/main.js` to change frames, copy or modal media.
- Motion effects respects `prefers-reduced-motion`.

## Easter eggs

Type `wololo`. Type `how do you turn this on`. Four steady steps/clicks on the hero (or click the worm sticker). Find Waldo in a print. Ride the worm in the footer.
