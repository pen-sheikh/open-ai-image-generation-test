import "dotenv/config";
import fs from "fs";
import sharp from "sharp";
import { EDIT_MODEL, editWithMask, writePreview } from "./edit-core.js";

const USAGE = `Usage: node edit.js <image> "<prompt>" (--box x,y,w,h | --mask mask.png) [--out edited.png] [--preview]
  --box      pixel rectangle to edit (x,y = top-left corner)
  --mask     PNG where TRANSPARENT pixels = edit, opaque pixels = keep
  --preview  only write preview.png showing the edit area in red; no API call`;

const [imagePath, prompt, ...rest] = process.argv.slice(2);
const opt = (name) => {
  const i = rest.indexOf(name);
  return i === -1 ? undefined : rest[i + 1];
};

function fail(message) {
  throw new Error(`${message}\n\n${USAGE}`);
}

async function main() {
  if (!imagePath || !prompt) fail("Missing image or prompt.");
  if (!fs.existsSync(imagePath)) fail(`Image not found: ${imagePath}`);

  const { width, height } = await sharp(imagePath).metadata();
  const box = opt("--box");
  const maskFile = opt("--mask");

  let maskPng;
  if (box) {
    const [x, y, w, h] = box.split(",").map(Number);
    if ([x, y, w, h].some(Number.isNaN)) fail(`--box needs 4 numbers, got: ${box}`);
    if (x >= width || y >= height || x + w <= 0 || y + h <= 0) {
      fail(`--box ${box} is outside the image, which is only ${width}x${height}.`);
    }
    const px = Buffer.alloc(width * height * 4, 255);
    for (let row = Math.max(y, 0); row < Math.min(y + h, height); row++) {
      for (let col = Math.max(x, 0); col < Math.min(x + w, width); col++) {
        px[(row * width + col) * 4 + 3] = 0;
      }
    }
    maskPng = await sharp(px, { raw: { width, height, channels: 4 } }).png().toBuffer();
  } else if (maskFile) {
    if (!fs.existsSync(maskFile)) fail(`Mask not found: ${maskFile}`);
    maskPng = await sharp(maskFile).resize(width, height, { fit: "fill" }).ensureAlpha().png().toBuffer();
  } else {
    fail("Give an edit area with --box or --mask.");
  }

  const alpha = await sharp(maskPng).extractChannel("alpha").raw().toBuffer();
  const editPixels = alpha.filter((a) => a < 128).length;
  if (editPixels === 0) {
    fail("The mask has no transparent pixels, so there is nothing to edit (a normal photo is fully opaque).");
  }
  console.log(`Edit area: ${((editPixels / alpha.length) * 100).toFixed(1)}% of ${width}x${height}`);

  if (rest.includes("--preview")) {
    await writePreview(imagePath, maskPng, "preview.png");
    console.log("Saved preview.png (edit area tinted red). No API call made.");
    return;
  }

  console.log(`Editing ${imagePath} with ${EDIT_MODEL}...`);
  const outPath = opt("--out") ?? "edited.png";
  await editWithMask({ imagePath, maskPng, prompt, outPath });
  console.log(`Saved ${outPath}`);
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exitCode = 1;
});
