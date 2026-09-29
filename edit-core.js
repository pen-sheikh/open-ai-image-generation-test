import OpenAI, { toFile } from "openai";
import sharp from "sharp";

export const EDIT_MODEL = "gpt-image-2.5-sunburst";

// maskPng is an RGBA PNG the same size as the image: alpha 0 = edit, alpha 255 = keep.
// protect (optional) is a Uint8Array of width*height; 1 = pixel is always restored exactly, with no feathering.

export async function writePreview(imagePath, maskPng, outPath, protect) {
  const { width, height } = await sharp(imagePath).metadata();
  const alpha = await sharp(maskPng).extractChannel("alpha").raw().toBuffer();
  const px = await sharp(imagePath).removeAlpha().raw().toBuffer();
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] < 128) {
      px[i * 3] = (px[i * 3] + 255) / 2;
      px[i * 3 + 1] /= 2;
      px[i * 3 + 2] /= 2;
    } else if (protect?.[i]) {
      px[i * 3] /= 2;
      px[i * 3 + 1] = (px[i * 3 + 1] + 255) / 2;
      px[i * 3 + 2] /= 2;
    }
  }
  await sharp(px, { raw: { width, height, channels: 3 } }).png().toFile(outPath);
}

export async function editWithMask({ imagePath, maskPng, prompt, outPath, protect }) {
  const { width, height } = await sharp(imagePath).metadata();

  const res = await new OpenAI().images.edit({
    model: EDIT_MODEL,
    image: await toFile(await sharp(imagePath).png().toBuffer(), "image.png", { type: "image/png" }),
    mask: await toFile(maskPng, "mask.png", { type: "image/png" }),
    prompt,
    quality: "high",
  });

  // The model treats the mask as a hint and can subtly redraw the whole frame,
  // so paste the original pixels back everywhere outside the mask (feathered edge).
  // Protected pixels count as "edit" for the feather, so the blend only happens at the outer edge
  // of the edit area and no old pixels bleed out around protected objects (they are restored below).
  const { data: alpha, info } = await sharp(maskPng).extractChannel("alpha").raw().toBuffer({ resolveWithObject: true });
  if (protect) for (let i = 0; i < alpha.length; i++) if (protect[i]) alpha[i] = 0;
  const keep = await sharp(alpha, { raw: { width: info.width, height: info.height, channels: 1 } })
    .blur(3)
    .extractChannel(0)
    .raw()
    .toBuffer();
  const orig = await sharp(imagePath).removeAlpha().raw().toBuffer();
  const out = await sharp(Buffer.from(res.data[0].b64_json, "base64"))
    .resize(width, height, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();
  for (let i = 0; i < keep.length; i++) {
    const a = protect?.[i] ? 1 : keep[i] / 255;
    for (let c = 0; c < 3; c++) {
      out[i * 3 + c] = orig[i * 3 + c] * a + out[i * 3 + c] * (1 - a);
    }
  }
  await sharp(out, { raw: { width, height, channels: 3 } }).png().toFile(outPath);
}
