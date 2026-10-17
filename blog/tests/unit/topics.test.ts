import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { categoryTopics, TOPICS } from '../../src/lib/topics';
import recipes from '../../scripts/viz-recipes.mjs';
import { nativeTools, validToolName } from '../../../tools/native-tools.mjs';

describe('project structure', () => {
  it('requires the knowledge suffix without changing article URLs or legacy GAN Lab', () => {
    expect(validToolName('kien-thuc', 'nerf')).toBe(false);
    expect(validToolName('kien-thuc', 'future-topic')).toBe(false);
    expect(validToolName('kien-thuc', 'future-topic-explainer')).toBe(true);
    expect(validToolName('kien-thuc', 'ganlab')).toBe(true);
    expect(validToolName('kien-thuc', undefined)).toBe(true);
    expect(validToolName('du-an', 'synctalk')).toBe(true);
  });
  it('keeps native visualization entries in tools with knowledge naming conventions', () => {
    expect(Object.keys(nativeTools)).toHaveLength(9);
    for (const [slug, tool] of Object.entries(nativeTools)) {
      expect(tool.source).toBe(`tools/${tool.category}/${slug}`);
      if (tool.category === 'kien-thuc') expect(slug).toMatch(/-explainer$/);
      expect(existsSync(resolve('..', tool.source, 'Tool.astro'))).toBe(true);
      expect(existsSync(resolve('..', tool.source, 'render.ts'))).toBe(true);
      const articles = readdirSync(resolve('src/content/posts', tool.category), { recursive: true }).map(String);
      const article = articles.find(file => file.endsWith(`/${tool.article}.mdx`));
      expect(article).toBeDefined();
      expect(readFileSync(resolve('src/content/posts', tool.category, article!), 'utf8')).toContain(`viz: ${slug}`);
      expect(readFileSync(resolve('src/integrations/LocalTool.astro'), 'utf8')).toContain(`${tool.source}/Tool.astro`);
    }
  });
  it('separates knowledge topics from project topics', () => {
    expect(categoryTopics('kien-thuc').map(topic => topic.slug)).toEqual(['nen-tang', 'thi-giac-may-tinh', 'xu-ly-ngon-ngu', 'mo-hinh-sinh']);
    expect(categoryTopics('du-an').map(topic => topic.slug)).toEqual(['giai-thich-mo-hinh', 'danh-gia-thien-lech', 'nang-luong-ai', 'nhan-vat-so', 'agent-ky-thuat']);
  });

  it('keeps all eight source recipes resolvable under tools', () => {
    expect(Object.values(recipes)).toHaveLength(8);
    for (const recipe of Object.values(recipes)) {
      expect(recipe.src).toMatch(/^tools\/(kien-thuc|du-an)\//);
      expect(existsSync(resolve('..', recipe.src))).toBe(true);
    }
  });

  it('stores every article under its category/topic with a unique public slug', () => {
    const directory = resolve('src/content/posts');
    const articles = readdirSync(directory, { recursive: true }).map(String).filter(path => path.endsWith('.mdx'));
    expect(articles).toHaveLength(17);
    const slugs = new Set<string>();
    for (const file of articles) {
      const [category, topic, filename] = file.split('/');
      expect(TOPICS[topic as keyof typeof TOPICS].category).toBe(category);
      const text = readFileSync(join(directory, file), 'utf8');
      expect(text).toContain(`category: ${category}`);
      expect(text).toContain(`topic: ${topic}`);
      expect(slugs.has(filename)).toBe(false);
      slugs.add(filename);
    }
  });
});
