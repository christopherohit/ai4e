import sharp from 'sharp';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../../', import.meta.url);
const covers = new URL('../src/assets/covers/', import.meta.url);
const sources = {
  'cnn-explainer': 'tools/kien-thuc/cnn-explainer/public/assets/figures/preview.png',
  'transformer-explainer': 'tools/kien-thuc/transformer-explainer/static/preview/teaser.png',
  'diffusion-explainer': 'tools/kien-thuc/diffusion-explainer/assets/preview/de_crownjewel.png',
  ganlab: 'tools/kien-thuc/ganlab/ganlab-teaser.png',
  dodrio: 'https://i.imgur.com/wA5wudI.png',
  fairvis: 'tools/du-an/FairVis/teaser.png',
  viscuit: 'tools/du-an/VisCUIT/assets/logo/thumbnail.png',
  // The repository README references this teaser; it has no local raster preview.
  energyvis: 'https://i.imgur.com/gqqnF3d.png',
};
await mkdir(covers, { recursive: true });
for (const [slug, source] of Object.entries(sources)) {
  let input;
  if (source.startsWith('https:')) {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`${slug}: HTTP ${response.status}`);
    input = Buffer.from(await response.arrayBuffer());
  } else input = fileURLToPath(new URL(source, root));
  const output = fileURLToPath(new URL(`${slug}.webp`, covers));
  await sharp(input).resize({ width: 1600, withoutEnlargement: true }).webp({ quality: 85 }).toFile(output);
  const meta = await sharp(output).metadata();
  if (!meta.width || meta.width > 1600) throw new Error(`${slug}: invalid cover dimensions`);
  console.log(`${slug}: ${meta.width}x${meta.height}`);
}
await writeFile(new URL('.handoff/survey.json', root), JSON.stringify({
  surveyedAt: new Date().toISOString(),
  note: 'Recovered cover survey from repository files; EnergyVis teaser URL comes from its README.',
  thumbnailCandidates: sources,
}, null, 2) + '\n');
