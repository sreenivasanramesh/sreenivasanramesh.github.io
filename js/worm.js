/* Arrakis Runner: a side-scrolling worm-runner easter egg.
   Printed like the rest of the site: paper, ink and grease red, with riso grain.
   Jump the rocks, eat the crew, grab a thumper to call in more crew.
   Opens from the footer worm or frame 06A; Esc or × closes. */

(() => {
  "use strict";

  const dlg = document.getElementById("worm-dialog");
  const canvas = document.getElementById("worm-canvas");
  const opener = document.getElementById("shai");
  if (!dlg || !canvas) return;

  const ctx = canvas.getContext("2d");
  const W = canvas.width;   // 720
  const H = canvas.height;  // 220
  const GROUND = 178;
  const scoreEl = document.getElementById("worm-score");
  const hiEl = document.getElementById("worm-hi");
  const muteBtn = document.getElementById("worm-mute");
  const closeBtn = document.getElementById("worm-close");

  /* ---------- palette: read from the page so every lighting mode applies ---------- */
  const parseColor = (str) => {
    if (!str) return [0, 0, 0];
    str = str.trim();
    const mHex = /^#?([0-9a-f]{6})$/i.exec(str);
    if (mHex) {
      const v = parseInt(mHex[1], 16);
      return [v >> 16, (v >> 8) & 255, v & 255];
    }
    const mRgb = /^rgba?\s*\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(str);
    if (mRgb) return [Math.round(+mRgb[1]), Math.round(+mRgb[2]), Math.round(+mRgb[3])];
    return [0, 0, 0];
  };
  const mix = (a, b, t) => `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")})`;
  let C = {};
  const readPalette = () => {
    const cs = getComputedStyle(document.documentElement);
    const paper = parseColor(cs.getPropertyValue("--paper")), ink = parseColor(cs.getPropertyValue("--ink"));
    let red = parseColor(cs.getPropertyValue("--red"));
    const isSilver = document.documentElement.classList.contains("silver");
    const isSafe = document.documentElement.classList.contains("safe");
    if (isSilver) {
      // in B&W / silver gelatin mode, map red accent to a rich silver tone
      red = [80, 80, 80];
    } else if (isSafe) {
      // safelight bright red accent (avoid waiting on 1.4s CSS transition)
      red = [255, 90, 60];
    }
    const dark = (paper[0] + paper[1] + paper[2]) / 3 < 100;
    C = {
      bg: mix(paper, ink, 0), sun: mix(red, paper, 0.08), duneFar: mix(paper, ink, 0.07), duneNear: mix(paper, ink, 0.12),
      mesa: mix(paper, ink, 0.1), groundTop: mix(ink, paper, 0), ground: mix(paper, ink, 0.05), speck: mix(paper, ink, 0.22),
      worm: mix(red, paper, 0), wormHi: mix(red, paper, 0.35), wormLo: mix(red, ink, 0.35), seam: mix(red, ink, 0.5),
      maw: mix(ink, paper, 0), teeth: mix(paper, ink, 0), rider: mix(ink, paper, 0), riderFace: mix(paper, red, 0.2),
      rock: mix(ink, paper, 0.12), rockHi: mix(ink, paper, 0.4), rockLo: mix(ink, paper, 0), spire: mix(ink, paper, 0.25), spireLit: mix(ink, paper, 0.45),
      person: mix(ink, paper, 0.05), visor: mix(paper, red, 0.4), text: mix(ink, paper, 0), dim: mix(ink, paper, 0.45), red: mix(red, paper, 0),
      dust: mix(paper, ink, 0.35), dark,
    };
  };

  /* riso grain: one noise tile, multiplied (or screened in the dark) at a fresh offset each frame */
  const grain = document.createElement("canvas");
  grain.width = grain.height = 256;
  {
    const g = grain.getContext("2d"), d = g.createImageData(256, 256);
    for (let i = 0; i < d.data.length; i += 4) {
      const v = 236 + Math.random() * 19;
      d.data[i] = d.data[i + 1] = d.data[i + 2] = v;
      d.data[i + 3] = 255;
    }
    g.putImageData(d, 0, 0);
  }

  /* ---------- state ---------- */
  let open = false, raf = null;
  let state, speed, score, frames, worm, obstacles, people, thumpers, particles, nextSpawn, shake, frenzy;
  let hi = parseInt(localStorage.getItem("vr-worm-hi") || "0", 10);
  const reset = () => {
    state = "ready"; speed = 5; score = 0; frames = 0; shake = 0; frenzy = 0;
    worm = { y: GROUND, vy: 0, grounded: true, hist: new Array(64).fill(GROUND), gulp: 0, holding: false };
    obstacles = []; people = []; thumpers = []; particles = []; nextSpawn = 70;
  };
  reset();

  /* ---------- sound ---------- */
  let muted = localStorage.getItem("vr-worm-mute") === "1";
  const music = new Audio("assets/arrakis.mp3");
  music.loop = true; music.volume = 0.32; music.preload = "none"; music.muted = muted;
  const paintMute = () => { muteBtn.textContent = muted ? "Sound [off]" : "Sound [on]"; };
  paintMute();
  muteBtn.addEventListener("click", () => {
    muted = !muted;
    localStorage.setItem("vr-worm-mute", muted ? "1" : "0");
    music.muted = muted;
    paintMute();
  });
  let actx;
  const blip = (type, f0, f1, dur, vol) => {
    if (muted) return;
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === "suspended") actx.resume();
      const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
      o.type = type;
      o.frequency.setValueAtTime(f0, t);
      o.frequency.exponentialRampToValueAtTime(Math.max(f1, 1), t + dur);
      g.gain.setValueAtTime(vol, t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g); g.connect(actx.destination);
      o.start(t); o.stop(t + dur + 0.02);
    } catch (e) { /* silence is acceptable on arrakis */ }
  };

  /* ---------- background, generated once ---------- */
  const rnd = (a, b) => a + Math.random() * (b - a);
  const mkDunes = (minH, maxH) => {
    const pts = [];
    let x = -40;
    while (x < W * 2 + 80) { pts.push({ x, h: rnd(minH, maxH), w: rnd(70, 150) }); x += pts[pts.length - 1].w; }
    return pts;
  };
  const dunesFar = mkDunes(18, 44), dunesNear = mkDunes(8, 26);
  const mesas = Array.from({ length: 5 }, (_, i) => ({ x: i * 320 + rnd(0, 120), w: rnd(60, 130), h: rnd(34, 60), top: rnd(0.5, 0.8) }));
  let bgScroll = 0;

  /* ---------- spawning ---------- */
  const spawn = () => {
    const roll = Math.random();
    if (frenzy > 0 ? roll < 0.65 : roll < 0.22) {
      people.push({ x: W + 20, w: 10, h: 16, step: 0 });
    } else if (frenzy <= 0 && score > 120 && roll < 0.29 && !thumpers.length) {
      thumpers.push({ x: W + 20, t: 0 });
    } else if (roll < 0.55) {
      obstacles.push({ x: W + 20, w: Math.round(rnd(14, 24)), h: Math.round(rnd(14, 22)), type: "rock" });
    } else if (roll < 0.78) {
      obstacles.push({ x: W + 20, w: Math.round(rnd(26, 40)), h: Math.round(rnd(24, 34)), type: "spire" });
    } else {
      const w1 = Math.round(rnd(12, 18));
      obstacles.push({ x: W + 20, w: w1, h: Math.round(rnd(12, 18)), type: "rock" });
      obstacles.push({ x: W + 26 + w1, w: Math.round(rnd(12, 20)), h: Math.round(rnd(16, 24)), type: "rock" });
    }
    nextSpawn = Math.round(rnd(52, Math.max(66, 132 - speed * 6)) * (frenzy > 0 ? 0.6 : 1));
  };

  /* ---------- input ---------- */
  const startMusic = () => music.play().catch(() => {});
  const jump = () => {
    if (state === "ready") { state = "run"; startMusic(); return; }
    if (state === "over") { reset(); state = "run"; startMusic(); return; }
    if (worm.grounded) {
      worm.vy = -10.6; worm.grounded = false; worm.holding = true;
      blip("sine", 95, 150, 0.14, 0.08);
    }
  };
  const onKey = (e) => {
    if (!open) return;
    if (e.key === " " || e.key === "ArrowUp") { e.preventDefault(); if (!e.repeat) jump(); }
  };
  const onKeyUp = (e) => { if (e.key === " " || e.key === "ArrowUp") worm.holding = false; };

  const hit = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

  /* ---------- update ---------- */
  const update = () => {
    frames++;
    if (state === "run") {
      bgScroll += speed;
      speed = Math.min(15, speed + 0.002);
      score += speed * 0.032;
      if (frenzy > 0) frenzy--;

      if (!worm.grounded) {
        worm.vy += worm.holding && worm.vy < 0 ? 0.4 : 0.62;
        worm.y += worm.vy;
        if (worm.y >= GROUND) {
          worm.y = GROUND; worm.vy = 0; worm.grounded = true;
          for (let i = 0; i < 6; i++) particles.push({ x: rnd(70, 110), y: GROUND + 2, vx: rnd(-2, 2), vy: rnd(-2.5, -0.5), life: 14, c: C.wormLo });
        }
      } else {
        worm.y = GROUND + Math.sin(frames * 0.25) * 1.5;
      }

      if (--nextSpawn <= 0) spawn();
      obstacles.forEach((o) => (o.x -= speed));
      people.forEach((p) => { p.x -= speed + 0.6; p.step += 0.25; });
      thumpers.forEach((t) => { t.x -= speed; t.t++; });
      obstacles = obstacles.filter((o) => o.x + o.w > -10);
      people = people.filter((p) => p.x + p.w > -10);
      thumpers = thumpers.filter((t) => t.x > -20);

      const head = { x: 76, y: worm.y - 40, w: 36, h: 40 };
      for (const o of obstacles) {
        if (hit(head, { x: o.x + 3, y: GROUND - o.h + 3, w: o.w - 6, h: o.h - 3 })) {
          state = "over"; shake = 10;
          hi = Math.max(hi, Math.floor(score));
          localStorage.setItem("vr-worm-hi", String(hi));
          blip("sawtooth", 130, 35, 0.5, 0.09);
        }
      }
      for (const p of people) {
        if (!p.eaten && head.x < p.x + p.w && head.x + head.w > p.x && worm.y > GROUND - 26) {
          p.eaten = true; score += 50; worm.gulp = 12;
          blip("square", 320, 110, 0.16, 0.06);
          for (let i = 0; i < 10; i++) particles.push({ x: p.x + 5, y: GROUND - 8, vx: rnd(-2.5, 2.5), vy: rnd(-4, -1), life: 18, c: C.person });
        }
      }
      people = people.filter((p) => !p.eaten);
      for (const t of thumpers) {
        if (!t.got && head.x < t.x + 8 && head.x + head.w > t.x - 4) {
          t.got = true; frenzy = 360; score += 25;
          blip("triangle", 60, 220, 0.4, 0.1);
        }
      }
      thumpers = thumpers.filter((t) => !t.got);
    }
    worm.hist.unshift(worm.y);
    worm.hist.length = 64;
    if (worm.gulp > 0) worm.gulp--;
    particles.forEach((pt) => { pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.25; pt.life--; });
    particles = particles.filter((pt) => pt.life > 0);
    if (shake > 0) shake--;
  };

  /* ---------- draw ---------- */
  const px = (x, y, w, h, c) => { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h)); };
  const drawDunes = (pts, parallax, color, base) => {
    ctx.fillStyle = color;
    const off = (bgScroll * parallax) % (W + 80);
    pts.forEach((d) => {
      const x = (((d.x - off) % (W * 2 + 80)) + W * 2 + 80) % (W * 2 + 80) - 40;
      if (x > W + 40) return;
      ctx.beginPath();
      ctx.moveTo(x, base);
      ctx.quadraticCurveTo(x + d.w / 2, base - d.h, x + d.w, base);
      ctx.fill();
    });
  };

  const drawWorm = () => {
    for (let i = 6; i >= 1; i--) {
      const segY = worm.hist[i * 6];
      const sw = 22 - i * 1.8, sh = 26 - i * 2.6, sx = 88 - i * 14;
      px(sx - sw / 2 + 2, segY - sh - 2, sw - 4, 2, C.worm);
      px(sx - sw / 2, segY - sh, sw, sh, C.worm);
      px(sx - sw / 2 + 2, segY - sh, sw - 4, 3, C.wormHi);
      px(sx - sw / 2, segY - 5, sw, 5, C.wormLo);
      px(sx - sw / 2 + 1, segY - sh + Math.round(sh * 0.4), sw - 2, 2, C.wormLo);
      px(sx + sw / 2 - 3, segY - sh + 3, 1, sh - 6, C.seam);
    }
    const hy = worm.y, mouth = worm.gulp > 0 ? 10 : 4;
    px(88, hy - 30, 24, 30, C.worm);
    px(90, hy - 32, 20, 2, C.wormHi);
    px(88, hy - 30, 24, 3, C.wormHi);
    px(88, hy - 6, 24, 6, C.wormLo);
    const mawY = hy - 24 - mouth / 2, mawH = 18 + mouth;
    px(102, mawY, 12, mawH, C.maw);
    for (let t = 0; t < 4; t++) {
      px(102 + t * 3, mawY, 2, 4 - (t % 2), C.teeth);
      px(102 + t * 3, mawY + mawH - (4 - (t % 2)), 2, 4 - (t % 2), C.teeth);
    }
    px(101, mawY - 2, 13, 2, C.wormLo);
    px(101, mawY + mawH, 13, 2, C.wormLo);
    // rider, hooked to the crown
    const ry = worm.hist[6] - 26;
    ctx.strokeStyle = C.dim; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(76, ry - 2); ctx.quadraticCurveTo(88, ry - 10, 98, hy - 30); ctx.stroke();
    const flap = Math.sin(frames * 0.45) > 0 ? 1 : 0;
    px(62 - flap * 3, ry - 5 + flap * 2, 8, 3, C.rider);
    px(64 - flap * 2, ry - 8 + flap, 7, 3, C.rider);
    px(69, ry - 12, 7, 7, C.rider);
    px(71, ry - 10, 3, 3, C.riderFace);
    px(68, ry - 5, 9, 8, C.rider);
    px(76, ry - 4, 3, 2, C.rider);
    if (worm.grounded && state === "run" && frames % 3 === 0) {
      particles.push({ x: rnd(60, 80), y: GROUND + 1, vx: rnd(-3, -1), vy: rnd(-1.8, -0.3), life: 12, c: C.dust });
    }
  };

  const draw = () => {
    ctx.save();
    if (shake > 0) ctx.translate(rnd(-3, 3), rnd(-3, 3));
    px(0, 0, W, H, C.bg);

    // one sun, printed in red, slightly off-register
    ctx.fillStyle = C.sun;
    ctx.beginPath(); ctx.arc(590, 52, 22, 0, 7); ctx.fill();
    ctx.strokeStyle = C.text; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(592, 54, 22, 0, 7); ctx.stroke();

    ctx.fillStyle = C.mesa;
    mesas.forEach((m) => {
      const mx = (((m.x - bgScroll * 0.12) % (W + 400)) + W + 400) % (W + 400) - 200;
      if (mx > W + 20) return;
      ctx.beginPath();
      ctx.moveTo(mx, GROUND + 2);
      ctx.lineTo(mx + m.w * (1 - m.top) * 0.5, GROUND - m.h);
      ctx.lineTo(mx + m.w * (1 + m.top) * 0.5, GROUND - m.h);
      ctx.lineTo(mx + m.w, GROUND + 2);
      ctx.fill();
    });
    drawDunes(dunesFar, 0.25, C.duneFar, GROUND + 2);
    drawDunes(dunesNear, 0.55, C.duneNear, GROUND + 2);

    px(0, GROUND, W, 2, C.groundTop);
    px(0, GROUND + 2, W, H - GROUND, C.ground);
    for (let i = 0; i < 14; i++) {
      const gx = (((i * 61 - bgScroll) % W) + W) % W;
      px(gx, GROUND + 8 + (i % 3) * 8, 6, 1.5, C.speck);
    }

    obstacles.forEach((o) => {
      if (o.type === "rock") {
        px(o.x + 2, GROUND - o.h - 2, o.w - 4, 2, C.rock);
        px(o.x, GROUND - o.h, o.w, o.h, C.rock);
        px(o.x + 1, GROUND - o.h, Math.max(3, o.w * 0.4), 3, C.rockHi);
        px(o.x + o.w - 5, GROUND - o.h + 3, 4, o.h - 3, C.rockLo);
      } else {
        ctx.fillStyle = C.spire;
        ctx.beginPath(); ctx.moveTo(o.x, GROUND + 1); ctx.lineTo(o.x + o.w * 0.5, GROUND - o.h); ctx.lineTo(o.x + o.w, GROUND + 1); ctx.fill();
        ctx.fillStyle = C.spireLit;
        ctx.beginPath(); ctx.moveTo(o.x, GROUND + 1); ctx.lineTo(o.x + o.w * 0.5, GROUND - o.h); ctx.lineTo(o.x + o.w * 0.6, GROUND + 1); ctx.fill();
      }
    });

    // thumpers: a stake that pulses rings into the sand
    thumpers.forEach((t) => {
      px(t.x, GROUND - 22, 3, 22, C.text);
      px(t.x - 3, GROUND - 24, 9, 4, C.red);
      const r = (t.t % 30) / 30;
      ctx.strokeStyle = C.red; ctx.globalAlpha = 1 - r;
      ctx.beginPath(); ctx.ellipse(t.x + 1, GROUND, 6 + r * 22, 2 + r * 4, 0, 0, 7); ctx.stroke();
      ctx.globalAlpha = 1;
    });

    people.forEach((p) => {
      const leg = Math.sin(p.step * 4) > 0 ? 1 : -1;
      px(p.x + 2, GROUND - p.h, 6, 5, C.person);
      px(p.x + 3, GROUND - p.h + 1, 4, 2, C.visor);
      px(p.x + 1, GROUND - p.h + 5, 8, 7, C.person);
      px(p.x - 1, GROUND - p.h + 6, 2, 5, C.person);
      px(p.x + 2 + leg, GROUND - 4, 2, 4, C.person);
      px(p.x + 6 - leg, GROUND - 4, 2, 4, C.person);
    });

    drawWorm();
    particles.forEach((pt) => px(pt.x, pt.y, 2, 2, pt.c));

    ctx.textAlign = "center";
    ctx.font = '11px "Departure Mono", ui-monospace, monospace';
    if (state === "ready") {
      ctx.fillStyle = C.text;
      ctx.fillText("PRESS SPACE OR TAP TO WAKE THE WORM", W / 2, 96);
    } else if (state === "over") {
      ctx.fillStyle = C.red;
      ctx.font = '16px "Departure Mono", ui-monospace, monospace';
      ctx.fillText("THE DESERT CLAIMS YOU", W / 2, 88);
      ctx.fillStyle = C.text;
      ctx.font = '11px "Departure Mono", ui-monospace, monospace';
      ctx.fillText(`SCORE ${String(Math.floor(score)).padStart(4, "0")}  ·  SPACE TO RIDE AGAIN`, W / 2, 110);
    } else if (frenzy > 0) {
      ctx.fillStyle = C.red;
      ctx.textAlign = "left";
      ctx.fillText("THUMPER · THE CREW IS RUNNING", 16, 24);
    }

    ctx.globalCompositeOperation = C.dark ? "screen" : "multiply";
    ctx.globalAlpha = C.dark ? 0.06 : 0.7;
    const gx = -Math.floor(Math.random() * 256), gy = -Math.floor(Math.random() * 256);
    for (let x = gx; x < W; x += 256) for (let y = gy; y < H; y += 256) ctx.drawImage(grain, x, y);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    ctx.restore();
  };

  const loop = () => {
    update();
    draw();
    scoreEl.textContent = String(Math.floor(score)).padStart(4, "0");
    hiEl.textContent = "HI " + String(hi).padStart(4, "0");
    raf = requestAnimationFrame(loop);
  };

  /* ---------- open / close ---------- */
  let returnTo = opener;
  const openGame = (from) => {
    if (open) return;
    returnTo = from && from.focus ? from : opener;
    readPalette();
    open = true;
    reset();
    dlg.showModal();
    closeBtn.focus();
    raf = requestAnimationFrame(loop);
  };
  const close = () => {
    if (!open) return;
    open = false;
    music.pause();
    music.currentTime = 0;
    cancelAnimationFrame(raf);
    if (dlg.open) dlg.close();
    if (returnTo) returnTo.focus({ preventScroll: true });
  };

  if (opener) opener.addEventListener("click", () => openGame(opener));
  closeBtn.addEventListener("click", close);
  dlg.addEventListener("cancel", (e) => { e.preventDefault(); close(); });
  dlg.addEventListener("close", close);
  // the canvas is the jump button; clicking the backdrop leaves
  dlg.addEventListener("pointerdown", (e) => {
    if (e.target === canvas) { e.preventDefault(); jump(); return; }
    if (e.target === dlg) close();
  });
  dlg.addEventListener("pointerup", () => { worm.holding = false; });
  addEventListener("keydown", onKey);
  addEventListener("keyup", onKeyUp);
  new MutationObserver(() => { if (open) readPalette(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["class", "style"] });

  /* handle for frame 06A and automated checks */
  window.__worm = {
    open: openGame,
    close,
    state: () => ({ state, speed, score, y: worm.y, obstacles: obstacles.length, people: people.length, thumpers: thumpers.length, frenzy }),
    jump,
    forceThumper: () => thumpers.push({ x: 140, t: 0 }),
  };
})();
