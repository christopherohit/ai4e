import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { basename } from 'node:path';
import { TOPICS, TOPIC_SLUGS } from './lib/topics';
import { validToolName } from '../../tools/native-tools.mjs';

/** Category slugs — keep in sync with CATEGORIES in src/lib/site.ts. */
export const CATEGORY_SLUGS = ['kien-thuc', 'du-an'] as const;

const nonEmpty = () => z.string().trim().min(1);

const posts = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/posts', generateId: ({ entry }) => basename(entry, '.mdx') }),
  schema: ({ image }) =>
    z.object({
      /** Vietnamese title. */
      title: nonEmpty(),
      /** Short Vietnamese summary used on cards, meta description and search. */
      description: nonEmpty().max(200),
      category: z.enum(CATEGORY_SLUGS),
      topic: z.enum(TOPIC_SLUGS),
      /** Vietnamese display names, e.g. ["CNN", "Thị giác máy tính"]. */
      tags: z.array(nonEmpty()).min(1),
      /** Slug of the built viz under /viz/<slug>/ (optional for text-only posts). */
      viz: z
        .string()
        .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, 'viz phải là slug dạng kebab-case')
        .optional(),
      /** Iframe height in px. */
      vizHeight: z.number().int().min(200).max(4000).default(820),
      /** Cover image in src/assets/covers/. */
      cover: image(),
      /** Display order within the category (ascending). */
      order: z.number(),
      date: z.coerce.date(),
    }).refine(data => TOPICS[data.topic].category === data.category, { message: 'Chuyên mục không thuộc nhóm nội dung đã chọn.', path: ['topic'] })
      .refine(data => validToolName(data.category, data.viz), { message: 'Công cụ kiến thức mới phải có hậu tố -explainer.', path: ['viz'] }),
});

export const collections = { posts };
