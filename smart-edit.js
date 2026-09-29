import "dotenv/config";
import fs from "fs";
import OpenAI from "openai";
import sharp from "sharp";
import { pipeline, Sam2Model, AutoProcessor, RawImage } from "@huggingface/transformers";
import { EDIT_MODEL, editWithMask, writePreview } from "./edit-core.js";

const USAGE = `Usage: node smart-edit.js <image> "<what to change>" [--out edited.png] [--pick N] [--keep "a,b"] [--preview]
  Example: node smart-edit.js room.jpg "make the rug on the floor red"
  --pick     use the Nth match when the object appears more than once (default 1 = best match)
  --keep     extra objects to protect, comma separated (objects touching the target are found automatically)
  --preview  save preview.png (red = will be edited, green = protected); no image edit call`;

const PARSE_MODEL = "gpt-5.4-mini";
const DETECTOR = "onnx-community/grounding-dino-tiny-ONNX";
const SEGMENTER = "onnx-community/sam2.1-hiera-base-plus-ONNX";
const DETECT_THRESHOLD = 0.3;
// Blur sigma used to grow the edit mask ~12px past the object so edges, shadows and reflections blend.
const GROW_SIGMA = 5;
// Protected objects are grown ~1px so their anti-aliased edges are kept too.
const PROTECT_SIGMA = 0.5;

const [imagePath, request, ...rest] = process.argv.slice(2);
const opt = (name) => {
  const i = rest.indexOf(name);
  return i === -1 ? undefined : rest[i + 1];
};

function fail(message) {
  throw new Error(`${message}\n\n${USAGE}`);
}

async function parseRequest(text) {
  const thumb = await sharp(imagePath).resize(1024, 1024, { fit: "inside" }).jpeg().toBuffer();
  const res = await new OpenAI().chat.completions.create({
    model: PARSE_MODEL,
    messages: [
      {
        role: "system",
        content:
          "You convert a photo edit request into JSON, looking at the photo. " +
          "target: the single object to change, as a short plain noun phrase an object detector understands " +
          "(e.g. 'car', 'rug', 'neon sign', 'white car'); keep words that tell similar objects apart, drop words about the change itself. " +
          "edit_prompt: a clear instruction for an image editing model describing only the change to that object " +
          "(e.g. 'Replace the rug with a plain red wool rug of the same size, lying flat on the floor'). " +
          "keep: list every distinct object visible in the photo other than the target (furniture, tables, chairs, people, animals, " +
          "vehicles, poles, plants, decor), at most 15, as short detector-friendly noun phrases with a distinguishing detail " +
          "(e.g. 'round brass table', 'yellow sofa'). Include small objects near or on the target. " +
          "Do not list the target itself, other objects of the same kind as the target, or surfaces (floor, wall, ceiling, road, ground, sky).",
      },
      {
        role: "user",
        content: [
          { type: "text", text },
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${thumb.toString("base64")}` } },
        ],
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "edit_request",
        strict: true,
        schema: {
          type: "object",
          properties: {
            target: { type: "string" },
            edit_prompt: { type: "string" },
            keep: { type: "array", items: { type: "string" } },
          },
          required: ["target", "edit_prompt", "keep"],
          additionalProperties: false,
        },
      },
    },
  });
  return JSON.parse(res.choices[0].message.content);
}

async function loadModels() {
  const [detector, sam, processor] = await Promise.all([
    pipeline("zero-shot-object-detection", DETECTOR, { dtype: "fp32" }),
    Sam2Model.from_pretrained(SEGMENTER, { dtype: "fp32" }),
    AutoProcessor.from_pretrained(SEGMENTER),
  ]);
  return { detector, sam, processor };
}

async function detect(models, img, label) {
  // Grounding DINO expects period-terminated phrases.
  return models.detector(img.raw, [`${label}.`], { threshold: DETECT_THRESHOLD });
}

// Runs SAM on a padded crop around the box so thin parts (legs, poles, wires) get more model resolution.
async function segment(models, img, box) {
  const { width: W, height: H } = img;
  const pad = 0.15 * Math.max(box.xmax - box.xmin, box.ymax - box.ymin);
  const x0 = Math.max(0, Math.floor(box.xmin - pad));
  const y0 = Math.max(0, Math.floor(box.ymin - pad));
  const cw = Math.min(W, Math.ceil(box.xmax + pad)) - x0;
  const ch = Math.min(H, Math.ceil(box.ymax + pad)) - y0;
  const crop = await sharp(img.data, { raw: { width: W, height: H, channels: 3 } })
    .extract({ left: x0, top: y0, width: cw, height: ch })
    .raw()
    .toBuffer();

  const inputs = await models.processor(new RawImage(new Uint8ClampedArray(crop), cw, ch, 3), {
    input_boxes: [[[box.xmin - x0, box.ymin - y0, box.xmax - x0, box.ymax - y0]]],
  });
  const out = await models.sam(inputs);
  const [masks] = await models.processor.post_process_masks(out.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
  const scores = Array.from(out.iou_scores.data);
  const best = scores.indexOf(Math.max(...scores));
  const m = masks.data.subarray(best * cw * ch, (best + 1) * cw * ch);

  const full = new Uint8Array(W * H);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) full[(y + y0) * W + x + x0] = m[y * cw + x] ? 1 : 0;
  }
  return full;
}

async function grow(mask, sigma, W, H) {
  const out = await sharp(Buffer.from(mask.map((v) => (v ? 255 : 0))), { raw: { width: W, height: H, channels: 1 } })
    .blur(sigma)
    .threshold(10)
    .extractChannel(0)
    .raw()
    .toBuffer();
  return Uint8Array.from(out, (v) => (v ? 1 : 0));
}

function boxInsideFraction(inner, outer) {
  const w = Math.max(0, Math.min(inner.xmax, outer.xmax) - Math.max(inner.xmin, outer.xmin));
  const h = Math.max(0, Math.min(inner.ymax, outer.ymax) - Math.max(inner.ymin, outer.ymin));
  return (w * h) / ((inner.xmax - inner.xmin) * (inner.ymax - inner.ymin));
}

function pickOverlapping(found, targetBox) {
  return found.reduce((best, d) =>
    boxInsideFraction(d.box, targetBox) > boxInsideFraction(best.box, targetBox) ? d : best,
  );
}

async function main() {
  if (!imagePath || !request) fail("Missing image or request.");
  if (!fs.existsSync(imagePath)) fail(`Image not found: ${imagePath}`);
  const pick = Number(opt("--pick") ?? 1);

  const { data, info } = await sharp(imagePath).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const W = info.width, H = info.height;
  const img = { data, width: W, height: H, raw: new RawImage(new Uint8ClampedArray(data), W, H, 3) };

  console.log(`1/4 Understanding request with ${PARSE_MODEL}...`);
  const parsed = await parseRequest(request);
  const keepLabels = [...new Set([...parsed.keep, ...(opt("--keep")?.split(",").map((s) => s.trim()) ?? [])])];
  console.log(`  target: "${parsed.target}"\n  edit:   "${parsed.edit_prompt}"\n  keep:   ${JSON.stringify(keepLabels)}`);

  console.log("2/4 Loading local models and finding objects...");
  const models = await loadModels();
  const found = await detect(models, img, parsed.target);
  if (found.length === 0) fail(`Could not find "${parsed.target}" in the image. Try naming it differently.`);
  found.forEach((d, i) => {
    const { xmin, ymin, xmax, ymax } = d.box;
    console.log(`  ${parsed.target} ${i + 1}: score ${d.score.toFixed(2)}, box ${xmin},${ymin} → ${xmax},${ymax}`);
  });
  if (pick > found.length) fail(`--pick ${pick} requested, but only ${found.length} match(es) found.`);
  const targetBox = found[pick - 1].box;

  console.log("3/4 Outlining objects...");
  const targetMask = await segment(models, img, targetBox);
  const edit = await grow(targetMask, GROW_SIGMA, W, H);
  const protect = new Uint8Array(W * H);
  const protectedLabels = [];
  for (const label of keepLabels) {
    const hits = await detect(models, img, label);
    if (hits.length === 0) {
      console.log(`  keep "${label}": not found, skipped`);
      continue;
    }
    const { box } = pickOverlapping(hits, targetBox);
    if (boxInsideFraction(box, targetBox) === 0) {
      console.log(`  keep "${label}": does not touch the target, skipped`);
      continue;
    }
    const keepRaw = await segment(models, img, box);
    let keepArea = 0, sharedWithTarget = 0;
    for (let i = 0; i < keepRaw.length; i++) {
      keepArea += keepRaw[i];
      sharedWithTarget += keepRaw[i] & targetMask[i];
    }
    if (sharedWithTarget > 0.3 * keepArea) {
      console.log(`  keep "${label}": mostly the target itself, skipped`);
      continue;
    }
    const keepMask = await grow(keepRaw, PROTECT_SIGMA, W, H);
    if (!keepMask.some((v, i) => v && edit[i])) {
      console.log(`  keep "${label}": does not touch the target, skipped`);
      continue;
    }
    for (let i = 0; i < protect.length; i++) protect[i] |= keepMask[i];
    protectedLabels.push(label);

    // An object resting on the target hides part of it; the target shows through gaps
    // (e.g. between table legs), so the whole area under the object joins the edit.
    const onTarget = boxInsideFraction(box, targetBox) > 0.8;
    if (onTarget) {
      for (let y = Math.floor(box.ymin); y < Math.ceil(box.ymax); y++) {
        for (let x = Math.floor(box.xmin); x < Math.ceil(box.xmax); x++) edit[y * W + x] = 1;
      }
    }
    console.log(`  keep "${label}": protected${onTarget ? ", gaps under it will be edited" : ""}`);
  }

  const rgba = Buffer.alloc(W * H * 4, 255);
  let editPixels = 0;
  for (let i = 0; i < edit.length; i++) {
    if (edit[i] && !protect[i]) {
      rgba[i * 4 + 3] = 0;
      editPixels++;
    }
  }
  const maskPng = await sharp(rgba, { raw: { width: W, height: H, channels: 4 } }).png().toBuffer();
  console.log(`  edit area: ${((editPixels / (W * H)) * 100).toFixed(1)}% of ${W}x${H}`);

  if (rest.includes("--preview")) {
    await writePreview(imagePath, maskPng, "preview.png", protect);
    console.log("Saved preview.png (red = will be edited, green = protected). No image edit call made.");
    return;
  }

  console.log(`4/4 Editing with ${EDIT_MODEL}...`);
  const outPath = opt("--out") ?? "edited.png";
  const keepNote = protectedLabels.length ? ` Keep the ${protectedLabels.join(" and the ")} exactly as they are.` : "";
  await editWithMask({
    imagePath,
    maskPng,
    protect,
    prompt: `${parsed.edit_prompt}${keepNote} Match the original photo's lighting, perspective, shadows and style.`,
    outPath,
  });
  console.log(`Saved ${outPath}`);
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exitCode = 1;
});
