const fs = require("fs");
const path = require("path");

const root = process.cwd();
const dist = path.join(root, "dist");

const entries = [
  ["index.html", "index.html"],
  ["style.css", "style.css"],
  ["app.js", "app.js"],
  ["data", "data"],
];

function copyEntry(from, to) {
  const source = path.join(root, from);
  const target = path.join(dist, to);

  if (!fs.existsSync(source)) {
    throw new Error(`Missing static asset: ${from}`);
  }

  fs.cpSync(source, target, { recursive: true });
}

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });

entries.forEach(([from, to]) => copyEntry(from, to));

console.log(`Built static site in ${path.relative(root, dist)}`);
