import { getCollection, render } from 'astro:content';
import type { APIRoute } from 'astro';
import { postHref, sortPosts } from '../lib/posts';
import { stripMdx, type SearchDoc } from '../lib/search';

const BODY_LIMIT = 4000;

export const GET: APIRoute = async () => {
  const posts = sortPosts(await getCollection('posts'));
  const docs: SearchDoc[] = await Promise.all(
    posts.map(async (post) => {
      const { headings } = await render(post);
      return {
        id: post.id,
        url: postHref(post),
        title: post.data.title,
        description: post.data.description,
        category: post.data.category,
        tags: [...post.data.tags],
        headings: headings.filter((heading) => heading.depth <= 3).map((heading) => heading.text),
        body: stripMdx(post.body ?? '').slice(0, BODY_LIMIT),
      };
    }),
  );
  return new Response(JSON.stringify(docs), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
};
