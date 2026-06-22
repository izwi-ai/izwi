import { routes, type VercelConfig } from '@vercel/config/v1';

const docsOrigin =
  process.env.MINTLIFY_DOCS_ORIGIN ??
  (process.env.MINTLIFY_SUBDOMAIN
    ? `https://${process.env.MINTLIFY_SUBDOMAIN}.mintlify.dev`
    : undefined);

if (!docsOrigin) {
  throw new Error(
    'Set MINTLIFY_DOCS_ORIGIN or MINTLIFY_SUBDOMAIN before deploying docs rewrites.'
  );
}

const mintlifyDocsBase = `${docsOrigin.replace(/\/+$/, '')}/docs`;

export const config: VercelConfig = {
  rewrites: [
    routes.rewrite('/docs', mintlifyDocsBase),
    routes.rewrite('/docs/:match*', `${mintlifyDocsBase}/:match*`)
  ]
};
