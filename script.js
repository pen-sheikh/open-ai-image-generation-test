import "dotenv/config";
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const result = await client.images.generate({
//   model: "gpt-image-2",
  model: "gpt-image-2.5-sunburst", // for editing precision 
//   model: "gpt-image-2.5-flare", //  for fast, high-quality everyday image generation
  prompt:
    "A cinematic photograph of a futuristic city in Tokyo at night, neon lights, rain, highly detailed",
});

const image = Buffer.from(result.data[0].b64_json, "base64");

fs.writeFileSync("generated.png", image);

console.log("Image saved!");
