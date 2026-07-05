/* Arrakis Runner: a side-scrolling worm-runner easter egg.
   Original pixel art drawn in code. Jump the rocks, eat the crew.
   Opens from the sandworm on the footer line; ESC or × closes. */

(() => {
  "use strict";

  const modal = document.querySelector(".worm-modal");
  const opener = document.querySelector(".shai");
  const canvas = document.querySelector(".worm-canvas");
  if (!modal || !opener || !canvas) return;

  const ctx = canvas.getContext("2d");
  const W = canvas.width;   // 720
  const H = canvas.height;  // 220
  const GROUND = 178;

  const scoreEl = modal.querySelector("[data-worm-score]");
  const hiEl = modal.querySelector("[data-worm-hi]");
  const closeBtn = modal.querySelector(".worm-modal__close");

  /* palette (site-native night desert) */
  const C = {
    bg: "#0e0e0c",
    duneFar: "#171209",
    duneNear: "#211709",
    groundTop: "#c07a3a",
    ground: "#171310",
    worm: "#c07a3a",
    wormHi: "#d99a55",
    wormLo: "#8f5722",
    maw: "#14100a",
    teeth: "#f4f1e9",
    rider: "#3d2f1e",
    riderFace: "#eecfa8",
    rock: "#4a463e",
    rockHi: "#6b665c",
    person: "#eae7df",
    text: "#eae7df",
    dim: "#8f8c83",
  };
  const accent = () =>
    getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#ff4d24";

  /* ---------- state ---------- */
  let open = false;
  let raf = null;
  let state, speed, score, hi, frames, worm, obstacles, people, particles, nextSpawn, shake;
  hi = parseInt(localStorage.getItem("vr-worm-hi") || "0", 10);

  const reset = () => {
    state = "ready"; // ready | run | over
    speed = 4.4;
    score = 0;
    frames = 0;
    shake = 0;
    worm = { y: GROUND, vy: 0, grounded: true, hist: new Array(64).fill(GROUND), gulp: 0, holding: false };
    obstacles = [];
    people = [];
    particles = [];
    nextSpawn = 70;
  };
  reset();

  /* ---------- tiny synth ---------- */
  let actx;
  const blip = (type, f0, f1, dur, vol) => {
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === "suspended") actx.resume();
      const t = actx.currentTime;
      const o = actx.createOscillator();
      o.type = type;
      o.frequency.setValueAtTime(f0, t);
      o.frequency.exponentialRampToValueAtTime(Math.max(f1, 1), t + dur);
      const g = actx.createGain();
      g.gain.setValueAtTime(vol, t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g);
      g.connect(actx.destination);
      o.start(t);
      o.stop(t + dur + 0.02);
    } catch (e) { /* silence is acceptable on arrakis */ }
  };

  /* ---------- background, generated once ---------- */
  const rnd = (a, b) => a + Math.random() * (b - a);
  const stars = Array.from({ length: 42 }, () => ({ x: rnd(0, W), y: rnd(6, 90), tw: rnd(0, 6.28) }));
  const mkDunes = (n, minH, maxH) => {
    const pts = [];
    let x = -40;
    while (x < W * 2 + 80) {
      pts.push({ x, h: rnd(minH, maxH), w: rnd(70, 150) });
      x += pts[pts.length - 1].w;
    }
    return pts;
  };
  const dunesFar = mkDunes(0, 18, 44);
  const dunesNear = mkDunes(0, 8, 26);
  let bgScroll = 0;

  /* ---------- spawning ---------- */
  const spawn = () => {
    const roll = Math.random();
    if (roll < 0.22) {
      // a harvester crew member: food, not hazard
      people.push({ x: W + 20, w: 10, h: 16, step: 0 });
    } else if (roll < 0.5) {
      obstacles.push({ x: W + 20, w: Math.round(rnd(14, 24)), h: Math.round(rnd(14, 22)), type: "rock" });
    } else if (roll < 0.75) {
      obstacles.push({ x: W + 20, w: Math.round(rnd(26, 40)), h: Math.round(rnd(24, 34)), type: "spire" });
    } else {
      const w1 = Math.round(rnd(12, 18));
      obstacles.push({ x: W + 20, w: w1, h: Math.round(rnd(12, 18)), type: "rock" });
      obstacles.push({ x: W + 26 + w1, w: Math.round(rnd(12, 20)), h: Math.round(rnd(16, 24)), type: "rock" });
    }
    nextSpawn = Math.round(rnd(52, Math.max(66, 132 - speed * 6)));
  };

  /* ---------- input ---------- */
  const jump = () => {
    if (state === "ready") { state = "run"; return; }
    if (state === "over") { reset(); state = "run"; return; }
    if (worm.grounded) {
      worm.vy = -10.6;
      worm.grounded = false;
      worm.holding = true;
      blip("sine", 95, 150, 0.14, 0.08);
    }
  };
  const onKey = (e) => {
    if (!open) return;
    if (e.key === " " || e.key === "ArrowUp") {
      e.preventDefault();
      if (!e.repeat) jump();
    } else if (e.key === "Escape") {
      close();
    }
  };
  const onKeyUp = (e) => {
    if (e.key === " " || e.key === "ArrowUp") worm.holding = false;
  };

  /* ---------- update ---------- */
  const update = () => {
    frames++;
    bgScroll += speed;

    if (state === "run") {
      speed = Math.min(11, speed + 0.0013);
      score += speed * 0.032;

      // physics with variable jump height
      if (!worm.grounded) {
        worm.vy += worm.holding && worm.vy < 0 ? 0.4 : 0.62;
        worm.y += worm.vy;
        if (worm.y >= GROUND) {
          worm.y = GROUND;
          worm.vy = 0;
          worm.grounded = true;
          for (let i = 0; i < 6; i++) {
            particles.push({ x: rnd(70, 110), y: GROUND + 2, vx: rnd(-2, 2), vy: rnd(-2.5, -0.5), life: 14, c: C.wormLo });
          }
        }
      } else {
        worm.y = GROUND + Math.sin(frames * 0.25) * 1.5; // idle undulation
      }

      if (--nextSpawn <= 0) spawn();

      obstacles.forEach((o) => (o.x -= speed));
      people.forEach((p) => { p.x -= speed + 0.6; p.step += 0.25; });
      obstacles = obstacles.filter((o) => o.x + o.w > -10);
      people = people.filter((p) => p.x + p.w > -10);

      // collision boxes: head + first body segment
      const headBox = { x: 76, y: worm.y - 40, w: 36, h: 40 };
      for (const o of obstacles) {
        const ob = { x: o.x + 3, y: GROUND - o.h + 3, w: o.w - 6, h: o.h - 3 };
        if (headBox.x < ob.x + ob.w && headBox.x + headBox.w > ob.x &&
            headBox.y < ob.y + ob.h && headBox.y + headBox.h > ob.y) {
          state = "over";
          shake = 10;
          hi = Math.max(hi, Math.floor(score));
          localStorage.setItem("vr-worm-hi", String(hi));
          blip("sawtooth", 130, 35, 0.5, 0.09);
        }
      }
      for (const p of people) {
        if (p.eaten) continue;
        if (headBox.x < p.x + p.w && headBox.x + headBox.w > p.x && worm.y > GROUND - 26) {
          p.eaten = true;
          score += 50;
          worm.gulp = 12;
          blip("square", 320, 110, 0.16, 0.06);
          for (let i = 0; i < 10; i++) {
            particles.push({ x: p.x + 5, y: GROUND - 8, vx: rnd(-2.5, 2.5), vy: rnd(-4, -1), life: 18, c: C.person });
          }
        }
      }
      people = people.filter((p) => !p.eaten);
    }

    // worm body follows the head with a delay
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
      const x = ((d.x - off) % (W * 2 + 80) + W * 2 + 80) % (W * 2 + 80) - 40;
      if (x > W + 40) return;
      ctx.beginPath();
      ctx.moveTo(x, base);
      ctx.quadraticCurveTo(x + d.w / 2, base - d.h, x + d.w, base);
      ctx.fill();
    });
  };

  const drawWorm = () => {
    // body segments, tail to head
    for (let i = 5; i >= 1; i--) {
      const segY = worm.hist[i * 7];
      const sw = 20 - i * 1.6;
      const sh = 22 - i * 2.4;
      const sx = 88 - i * 15;
      px(sx - sw / 2, segY - sh, sw, sh, C.worm);
      px(sx - sw / 2, segY - sh, sw, 3, C.wormHi);
      px(sx - sw / 2, segY - 4, sw, 4, C.wormLo);
      px(sx - sw / 2 + 2, segY - sh + 5, sw - 4, 1, C.wormLo); // ring line
    }
    // head
    const hy = worm.y;
    const mouthOpen = worm.gulp > 0 ? 8 : 4;
    px(88, hy - 26, 22, 26, C.worm);
    px(88, hy - 26, 22, 3, C.wormHi);
    px(88, hy - 5, 22, 5, C.wormLo);
    // maw, facing forward
    px(104, hy - 20 - mouthOpen / 2, 10, 14 + mouthOpen, C.maw);
    for (let t = 0; t < 3; t++) {
      px(104 + t * 3, hy - 20 - mouthOpen / 2, 2, 3, C.teeth);
      px(104 + t * 3, hy - 8 + mouthOpen / 2, 2, 3, C.teeth);
    }
    // rider on the second segment
    const ry = worm.hist[7] - 22;
    px(70, ry - 10, 6, 6, C.rider);            // hood
    px(72, ry - 8, 3, 3, C.riderFace);         // face
    px(69, ry - 4, 8, 7, C.rider);             // body
    const flap = Math.sin(frames * 0.4) > 0 ? 1 : 0;
    px(64 - flap * 2, ry - 6 + flap, 5, 3, "#5a4325"); // cape
    // sand spray while grounded
    if (worm.grounded && state === "run" && frames % 3 === 0) {
      particles.push({ x: rnd(64, 80), y: GROUND + 1, vx: rnd(-3, -1), vy: rnd(-1.5, -0.2), life: 10, c: C.wormLo });
    }
  };

  const draw = () => {
    ctx.save();
    if (shake > 0) ctx.translate(rnd(-3, 3), rnd(-3, 3));

    px(0, 0, W, H, C.bg);

    // two moons, sparse stars
    ctx.fillStyle = C.dim;
    stars.forEach((s) => {
      if (Math.sin(frames * 0.02 + s.tw) > -0.4) ctx.fillRect(s.x, s.y, 1.5, 1.5);
    });
    ctx.fillStyle = "#dcd6c6";
    ctx.beginPath(); ctx.arc(573, 47, 13, 0, 7); ctx.fill();
    ctx.fillStyle = C.bg;
    ctx.beginPath(); ctx.arc(568, 43, 11, 0, 7); ctx.fill();
    ctx.fillStyle = "#8a857a";
    ctx.beginPath(); ctx.arc(632, 30, 5, 0, 7); ctx.fill();

    drawDunes(dunesFar, 0.25, C.duneFar, GROUND + 2);
    drawDunes(dunesNear, 0.55, C.duneNear, GROUND + 2);

    // ground
    px(0, GROUND, W, 2, C.groundTop);
    px(0, GROUND + 2, W, H - GROUND, C.ground);
    for (let i = 0; i < 14; i++) {
      const gx = ((i * 61 - bgScroll * 1.0) % W + W) % W;
      px(gx, GROUND + 8 + (i % 3) * 8, 6, 1.5, "#2b2118");
    }

    // obstacles
    obstacles.forEach((o) => {
      if (o.type === "rock") {
        px(o.x, GROUND - o.h, o.w, o.h, C.rock);
        px(o.x + 2, GROUND - o.h, o.w - 6, 3, C.rockHi);
        px(o.x + o.w - 5, GROUND - o.h + 4, 3, o.h - 6, "#3a362f");
      } else {
        ctx.fillStyle = C.duneNear;
        ctx.beginPath();
        ctx.moveTo(o.x, GROUND + 1);
        ctx.lineTo(o.x + o.w * 0.5, GROUND - o.h);
        ctx.lineTo(o.x + o.w, GROUND + 1);
        ctx.fill();
        px(o.x + o.w * 0.42, GROUND - o.h, o.w * 0.16, 3, C.groundTop);
      }
    });

    // crew members
    people.forEach((p) => {
      const leg = Math.sin(p.step * 4) > 0 ? 1 : -1;
      px(p.x + 2, GROUND - p.h, 6, 5, p.eaten ? C.dim : C.person);       // head/helmet
      px(p.x + 3, GROUND - p.h + 1, 3, 2, "#14100a");                    // visor
      px(p.x + 1, GROUND - p.h + 5, 8, 7, C.person);                     // body
      px(p.x + 2 + leg, GROUND - 4, 2, 4, C.person);                     // legs
      px(p.x + 6 - leg, GROUND - 4, 2, 4, C.person);
    });

    drawWorm();

    // particles
    particles.forEach((pt) => px(pt.x, pt.y, 2, 2, pt.c));

    // HUD text
    ctx.fillStyle = C.dim;
    ctx.font = "700 11px ui-monospace, monospace";
    ctx.textAlign = "left";
    if (state === "ready") {
      ctx.fillStyle = C.text;
      ctx.textAlign = "center";
      ctx.fillText("PRESS SPACE OR TAP TO WAKE THE WORM", W / 2, 92);
    } else if (state === "over") {
      ctx.textAlign = "center";
      ctx.fillStyle = accent();
      ctx.font = "700 16px ui-monospace, monospace";
      ctx.fillText("THE DESERT CLAIMS YOU", W / 2, 84);
      ctx.fillStyle = C.text;
      ctx.font = "700 11px ui-monospace, monospace";
      ctx.fillText(`SCORE ${String(Math.floor(score)).padStart(4, "0")}   ·   SPACE TO RIDE AGAIN`, W / 2, 106);
    }

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
  const openGame = () => {
    open = true;
    reset();
    modal.hidden = false;
    document.documentElement.style.overflow = "hidden";
    closeBtn.focus();
    raf = requestAnimationFrame(loop);
  };
  const close = () => {
    open = false;
    modal.hidden = true;
    document.documentElement.style.overflow = "";
    cancelAnimationFrame(raf);
    opener.focus();
  };

  opener.addEventListener("click", openGame);
  closeBtn.addEventListener("click", close);
  modal.addEventListener("pointerdown", (e) => {
    if (e.target === modal) close();          // click the backdrop to leave
  });
  canvas.addEventListener("pointerdown", (e) => { e.preventDefault(); jump(); });
  canvas.addEventListener("pointerup", () => { worm.holding = false; });
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", onKeyUp);

  /* debug handle for automated checks */
  window.__worm = {
    open: openGame,
    close,
    state: () => ({ state, speed, score, y: worm.y, obstacles: obstacles.length, people: people.length }),
    jump,
    forceRock: () => obstacles.push({ x: 140, w: 20, h: 20, type: "rock" }),
    forcePerson: () => people.push({ x: 120, w: 10, h: 16, step: 0 }),
  };
})();
